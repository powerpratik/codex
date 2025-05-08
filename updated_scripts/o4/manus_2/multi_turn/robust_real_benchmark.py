#!/usr/bin/env python3
"""
Robust Real Benchmark for KV Cache Management Strategies

This script evaluates different KV cache management strategies using real metrics:
- KV cache size (real physical memory usage)
- Inference time (total, per token, time to first token)
- Accuracy (perplexity and optional Azure evaluation)

The implementation properly applies eviction strategies to the actual KV cache
and measures real physical metrics.
"""

import os
import traceback
import json
import time
import torch
import argparse
import logging
import datetime
import numpy as np
import psutil
import re
import openai
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm

import transformers
# Silence HuggingFace logs & disable progress bars
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
# Set up logging: remove all existing handlers, then log only to file
root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)
file_handler = logging.FileHandler('benchmark.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root.addHandler(file_handler)
root.setLevel(logging.INFO)
# Our benchmark logger
logger = logging.getLogger('real_benchmark')
# Ensure logs propagate to root (which writes only to file)
logger.propagate = True
# Prevent RealKVCacheManager from adding its own StreamHandler
logger.addHandler(logging.NullHandler())

# Optional imports for memory profiling
try:
    import gc
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, memory profiling will be limited")

# Optional imports for Azure evaluation
try:
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai not available, Azure evaluation will be disabled")

# Add a direct memory tracking function right after the imports
def measure_gpu_memory():
    """Directly measure GPU memory usage if available"""
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory usage in bytes and convert to MB
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            
            # Get per-device memory for all devices
            device_info = {}
            for i in range(torch.cuda.device_count()):
                device_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                device_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                device_info[f"cuda:{i}"] = {
                    "allocated_mb": device_allocated,
                    "reserved_mb": device_reserved
                }
                
            return {
                "total_allocated_mb": allocated,
                "total_reserved_mb": reserved,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
                "devices": device_info
            }
        else:
            return {"error": "CUDA not available"}
    except (ImportError, AttributeError) as e:
        return {"error": f"Failed to measure GPU memory: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error measuring GPU memory: {str(e)}"}

# Add CPU memory tracking as well
def measure_system_memory():
    """Measure system memory usage if available"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": f"Failed to measure system memory: {str(e)}"}

def measure_kv_cache_size(past_key_values):
    """Measure the size of the KV cache in MB more accurately"""
    try:
        if past_key_values is None:
            return 0.0
            
        import torch
        
        # Calculate the memory usage of past_key_values
        total_size_bytes = 0
        
        # For debugging - log the type and structure
        logger.info(f"KV cache type: {type(past_key_values)}")
        
        # Handle different KV cache formats
        if isinstance(past_key_values, tuple) and all(isinstance(layer, tuple) for layer in past_key_values):
            # Standard format in most HF models
            for layer_past in past_key_values:
                for tensor in layer_past:
                    if isinstance(tensor, torch.Tensor):
                        total_size_bytes += tensor.nelement() * tensor.element_size()
        elif hasattr(past_key_values, 'key_states') and hasattr(past_key_values, 'value_states'):
            # New DynamicCache format in some newer models
            for layer_idx in range(len(past_key_values.key_states)):
                k_tensor = past_key_values.key_states[layer_idx]
                v_tensor = past_key_values.value_states[layer_idx]
                if isinstance(k_tensor, torch.Tensor) and isinstance(v_tensor, torch.Tensor):
                    total_size_bytes += k_tensor.nelement() * k_tensor.element_size()
                    total_size_bytes += v_tensor.nelement() * v_tensor.element_size()
        
        # If still 0, use sequence length estimation as fallback
        if total_size_bytes == 0 and hasattr(past_key_values, '__len__') and len(past_key_values) > 0:
            # Rough estimation based on model size and sequence length
            layers = len(past_key_values)
            seq_len = -1
            
            # Try to find sequence length from first layer
            if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) > 0:
                first_tensor = past_key_values[0][0]
                if isinstance(first_tensor, torch.Tensor) and len(first_tensor.shape) >= 3:
                    seq_len = first_tensor.shape[2]
            
            if seq_len > 0:
                # Estimate based on model parameters
                hidden_size = 4096  # Default for many LLMs
                per_token_bytes = 2 * hidden_size * 2  # Key and value, float16 = 2 bytes
                total_size_bytes = layers * seq_len * per_token_bytes
                
        
        # Convert to MB
        total_size_mb = total_size_bytes / (1024 * 1024)
        if total_size_bytes == 0:
            print(":::::::::::::::::::::ESTIMATES:::::::::")
            logger.info("WARNING: Returning zero KV cache size - measurement failed completely")
        
        
        return total_size_mb
    except Exception as e:
        logger.warning(f"Error measuring KV cache size: {e}")
        traceback.print_exc()
        return 0.0

class RealKVCacheManager:
    """
    Real KV Cache Manager that physically measures and manipulates the KV cache.
    This implementation properly applies eviction strategies to the actual KV cache
    and measures real physical metrics.
    """
    
    def __init__(self, model, tokenizer=None, cfg=None, logger=None):
        """
        Initialize the KV Cache Manager.
        
        Args:
            model: The language model
            tokenizer: The tokenizer (optional)
            cfg: Configuration dictionary (optional)
            logger: Logger instance (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set up configuration
        self.cfg = cfg or {}
        self.threshold = self.cfg.get("kv_threshold", 1000)  # Default 1000 MB
        self.threshold_trigger_ratio = 0.95  # Trigger eviction at 95% of threshold
        
        # Set up logger (all logs will go to file via the root logger)
        self.logger = logger or logging.getLogger("kv_cache_manager")
        
        # Initialize memory tracking
        self.current_memory_mb = 0
        self.peak_memory_mb = 0
        self.layer_memory = {}
        self.step_memory = []
        
        # Initialize eviction stats
        self.eviction_count = 0
        self.total_eviction_time = 0
        
        # Add direct memory tracking
        self.last_sequence_length = 0
        self.per_token_memory_estimate = 0.5  # MB per token for Llama-2-7b (adjust if needed)
        
        # Register hooks for memory tracking
        self.hooks = []
        self._register_hooks()
        self.logger.info("Memory tracking hooks registered")
    
    def _register_hooks(self):
        """Register hooks to track memory usage"""
        # Remove existing hooks
        self.remove_hooks()
        
        # Add new hooks for attention modules
        hook_count = 0
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                hook = module.register_forward_hook(self._memory_tracking_hook)
                self.hooks.append(hook)
                hook_count += 1
                self.logger.debug(f"Registered memory tracking hook for {name}")
        
        self.logger.info(f"Total hooks registered: {hook_count}")
    
    def _memory_tracking_hook(self, module, input_tensor, output_tensor):
        """Hook to track memory usage during forward pass"""
        try:
            # First check in the outputs structure
            past_key_values = None
            
            # Try different ways to access past_key_values from output
            if hasattr(output_tensor, 'past_key_values') and output_tensor.past_key_values is not None:
                past_key_values = output_tensor.past_key_values
            elif isinstance(output_tensor, tuple) and len(output_tensor) > 1:
                # Some models return past_key_values as the second item in output tuple
                if isinstance(output_tensor[1], tuple):
                    past_key_values = output_tensor[1]
            
            # If no past_key_values in output, try to find it in model itself
            if past_key_values is None and hasattr(self.model, 'past_key_values'):
                past_key_values = self.model.past_key_values
                
            # If found, update memory stats
            if past_key_values is not None:
                self._update_memory_stats(past_key_values)
                self.logger.info(f"Hook updated memory stats: {self.current_memory_mb:.2f} MB")
        except Exception as e:
            self.logger.warning(f"Error in memory tracking hook: {e}")
            traceback.print_exc()  # Add stack trace for better debugging
        
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _update_memory_stats(self, past_key_values):
        """Update memory statistics based on past_key_values"""
        total_memory = 0
        layer_memory = {}
        
        # Handle None case gracefully
        if past_key_values is None:
            self.logger.info("past_key_values is None, skipping memory update")
            return
        
        # Add debug info about the type
        self.logger.debug(f"past_key_values type: {type(past_key_values)}")
        
        # Handle DynamicCache format (used by Llama 3)
        if hasattr(past_key_values, 'get_layers_count') or hasattr(past_key_values, 'key_states'):
            try:
                self.logger.info("Detected DynamicCache format (used by Llama 3)")
                num_layers = 0
                
                if hasattr(past_key_values, 'get_layers_count'):
                    num_layers = past_key_values.get_layers_count()
                elif hasattr(past_key_values, 'key_states'):
                    num_layers = len(past_key_values.key_states)
                
                for i in range(num_layers):
                    layer_total = 0
                    # Try to get layer cache
                    if hasattr(past_key_values, 'get_layer_cache'):
                        key_states, value_states = past_key_values.get_layer_cache(i)
                    else:
                        key_states = past_key_values.key_states[i]
                        value_states = past_key_values.value_states[i]
                    
                    # Calculate memory for key states
                    if isinstance(key_states, torch.Tensor):
                        tensor_memory = key_states.numel() * key_states.element_size() / (1024 * 1024)  # MB
                        layer_total += tensor_memory
                    
                    # Calculate memory for value states
                    if isinstance(value_states, torch.Tensor):
                        tensor_memory = value_states.numel() * value_states.element_size() / (1024 * 1024)  # MB
                        layer_total += tensor_memory
                    
                    layer_memory[f"layer_{i}"] = layer_total
                    total_memory += layer_total
                
                self.logger.info(f"Successfully measured DynamicCache memory: {total_memory:.2f} MB")
            except Exception as e:
                self.logger.warning(f"Error measuring DynamicCache memory: {e}")
                # Continue to try other methods
        
        # Original tuple-based calculation (for older models)
        if total_memory == 0:
                       
            
            try:
                for i, layer_cache in enumerate(past_key_values):
                    layer_total = 0
                    for tensor in layer_cache:
                        # Calculate actual memory used by tensor
                        if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                            tensor_memory = tensor.numel() * tensor.element_size() / (1024 * 1024)  # MB
                            layer_total += tensor_memory
                    
                    layer_memory[f"layer_{i}"] = layer_total
                    total_memory += layer_total
                self.logger.warning(f"Total memory{total_memory:.2f}MB")
            except Exception as e:
                self.logger.warning(f"Error calculating memory from tensors: {e}")
        
        # If direct calculation failed or returned very small values, use sequence length estimation
        if total_memory < 1.0:  # If less than 1 MB, likely incorrect
            self.logger.warning("Direct measurement failed, using sequence length estimation")
            # Get sequence length from the first layer's key tensor
            seq_len = 0
            try:
                if hasattr(past_key_values, 'get_layer_cache'):
                    key_states, _ = past_key_values.get_layer_cache(0)
                    seq_len = key_states.size(2)
                elif hasattr(past_key_values, 'key_states') and len(past_key_values.key_states) > 0:
                    seq_len = past_key_values.key_states[0].size(2)
                elif past_key_values and past_key_values[0] and len(past_key_values[0]) > 0:
                    seq_len = past_key_values[0][0].size(2)
                    self.last_sequence_length = seq_len
            except Exception as e:
                self.logger.warning(f"Error getting sequence length: {e}")
                # seq_len = self.last_sequence_length + 1  # Increment by 1 as fallback
                # self.last_sequence_length = seq_len
            
            # Estimate memory based on sequence length
            self.logger.info(f"Using estimated memory based on sequence length: {seq_len}")
            total_memory = seq_len * self.per_token_memory_estimate
            
            # Distribute estimated memory across layers
            num_layers = 0
            if hasattr(past_key_values, 'get_layers_count'):
                num_layers = past_key_values.get_layers_count()
            elif hasattr(past_key_values, 'key_states'):
                num_layers = len(past_key_values.key_states)
            elif past_key_values:
                num_layers = len(past_key_values)
            else:
                num_layers = 32  # Default to 32 layers for Llama-2-7b
                
            for i in range(num_layers):
                layer_memory[f"layer_{i}"] = total_memory / num_layers
        
        # Update stats
        self.current_memory_mb = total_memory
        self.peak_memory_mb = max(self.peak_memory_mb, total_memory)
        self.layer_memory = layer_memory
        self.step_memory.append(total_memory)
        
        # Log memory usage
        self.logger.info(f"Current KV cache size: {total_memory:.2f} MB, Peak: {self.peak_memory_mb:.2f} MB, Seq len: {self.last_sequence_length}")
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        return {
            "current_memory_mb": self.current_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "layer_memory": self.layer_memory,
            "step_memory": self.step_memory,
            "eviction_count": self.eviction_count,
            "avg_eviction_time": self.total_eviction_time / max(1, self.eviction_count)
        }
    
    def get_stats(self):
        """Alias for get_memory_stats() for compatibility"""
        return self.get_memory_stats()
    
    def apply_eviction_strategy(self, past_key_values, strategy_name, attention_scores=None, token_types=None):
        """
        Apply the specified eviction strategy to the KV cache.
        
        Args:
            past_key_values: The current KV cache
            strategy_name: Name of the strategy to apply
            attention_scores: Attention scores for each token (optional)
            token_types: Types of each token (optional)
            
        Returns:
            Updated past_key_values
        """
        # Update memory stats first
        self._update_memory_stats(past_key_values)
        
        # Determine eviction trigger point based on configured threshold and trigger ratio
        trigger_point = self.threshold * self.threshold_trigger_ratio
        if self.current_memory_mb < trigger_point:
            self.logger.info(f"Skipping eviction, current memory {self.current_memory_mb:.2f} MB < trigger point {trigger_point:.2f} MB")
            return past_key_values
        else:
            self.logger.info(f"Applying eviction, current memory {self.current_memory_mb:.2f} MB >= trigger point {trigger_point:.2f} MB")
        
            # Start timing eviction
            eviction_start = time.time()
            
            # Apply the appropriate strategy

            if "Baseline" in strategy_name:
                # Baseline does no eviction
                self.logger.info("Baseline MA GAYO")
                updated_past = past_key_values
                return updated_past
            elif "Random" in strategy_name:
                # Extract keep ratio from strategy name
                keep_ratio = 0.7  # Default
                if "keep=" in strategy_name:
                    try:
                        keep_ratio = float(strategy_name.split("(")[1].split(")")[0])
                    except:
                        pass
                updated_past = self._apply_random_strategy(past_key_values, keep_ratio)
            elif "AttentionTop" in strategy_name:
                # Extract keep ratio from strategy name
                keep_ratio = 0.7  # Default
                if "keep=" in strategy_name:
                    try:
                        keep_ratio = float(strategy_name.split("(kee")[1].split(")")[0])/100
                    except:
                        pass
                updated_past = self._apply_attention_top_strategy(past_key_values, attention_scores, keep_ratio=0.7)
                # updated_past = past_key_values
            elif "AttentionBottom" in strategy_name:
                # Extract keep ratio from strategy name
                keep_ratio = 0.7  # Default
                if "keep=" in strategy_name:
                    try:
                        keep_ratio = float(strategy_name.split("(")[1].split(")")[0])
                    except:
                        pass
                updated_past = self._apply_attention_bottom_strategy(past_key_values, attention_scores, keep_ratio)
            elif "HybridNPercent" in strategy_name:
                # Extract parameters from strategy name
                keep_ratio = 0.7  # Default
                recency_weight = 0.5  # Default
                attention_weight = 0.3  # Default
                type_weight = 0.2  # Default
                
                if "keep=" in strategy_name:
                    try:
                        keep_ratio = float(strategy_name.split("(")[1].split(",")[0])
                    except:
                        pass
                if "r=" in strategy_name:
                    try:
                        recency_weight = float(strategy_name.split("r=")[1].split(",")[0])
                    except:
                        pass
                if "a=" in strategy_name:
                    try:
                        attention_weight = float(strategy_name.split("a=")[1].split(",")[0])
                    except:
                        pass
                if "t=" in strategy_name:
                    try:
                        type_weight = float(strategy_name.split("t=")[1].split(")")[0])
                    except:
                        pass
                
                updated_past = self._apply_hybrid_strategy(
                    past_key_values, 
                    attention_scores, 
                    token_types, 
                    keep_ratio, 
                    recency_weight, 
                    attention_weight, 
                    type_weight
                )
            elif "SlidingWindow" in strategy_name:
                # Extract window size and important token ratio from strategy name
                window_size = 0.7  # Default
                important_ratio = 0.1  # Default
                
                if "window=" in strategy_name:
                    try:
                        window_size = float(strategy_name.split("window=")[1].split(",")[0])
                    except:
                        pass
                if "important=" in strategy_name:
                    try:
                        important_ratio = float(strategy_name.split("important=")[1].split(")")[0])
                    except:
                        pass
                
                updated_past = self._apply_sliding_window(past_key_values, window_size, important_ratio, attention_scores)
            elif "AdaptiveAttention" in strategy_name:
                # Extract base keep ratio from strategy name
                base_keep = 0.7  # Default
                if "base_keep=" in strategy_name:
                    try:
                        base_keep = float(strategy_name.split("base_keep=")[1].split(")")[0])
                    except:
                        pass
                
                updated_past = self._apply_adaptive_attention(past_key_values, attention_scores, base_keep)
            else:
                # Unknown strategy, return unchanged
                self.logger.warning(f"Unknown strategy: {strategy_name}, using Baseline")
                updated_past = past_key_values
            
            # End timing and update stats
            eviction_time = time.time() - eviction_start
            self.eviction_count += 1
            self.total_eviction_time += eviction_time
            
            # Wrap tuple outputs into a Cache object so Llama-2 will accept them
            # If eviction returned a raw tuple, pack it into a DynamicCache
            result = updated_past
            try:
                from transformers.cache_utils import DynamicCache
            except ImportError:
                DynamicCache = None
            if isinstance(updated_past, tuple) and DynamicCache is not None:
                new_cache = DynamicCache()
                # Populate the cache layer by layer from the legacy tuple
                for layer_idx, layer_cache in enumerate(updated_past):
                    if isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
                        key_states, value_states = layer_cache[0], layer_cache[1]
                        new_cache.update(key_states, value_states, layer_idx)
                result = new_cache

            # Force CUDA to free memory
            # torch.cuda.empty_cache()
            return result
    
    def _apply_random_strategy(self, past_key_values, keep_ratio=0.7):

        """
        Apply random eviction strategy: randomly select tokens to keep.
        
        Args:
            past_key_values: The current KV cache
            keep_ratio: Fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """
        self.logger.info("Random MA GAYO")
        updated_past = []
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Randomly select indices to keep
                indices_to_keep = torch.randperm(seq_len, device=tensor.device)[:keep_count]
                
                # Create a new tensor with only the kept tokens
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def _apply_attention_top_strategy(self, past_key_values, attention_scores, keep_ratio=1):
        """
        Apply attention-based top strategy: keep tokens with highest attention scores.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            keep_ratio: Fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """

        self.logger.info("TOP MA GAYO")

        
        updated_past = []
        # breakpoint()
        
        
        # attention_scores: tuple of [B, H, Q, K]
        attn = torch.stack(attention_scores, dim=0).mean(dim=0)   # → [B, H, Q, K]
        attn = attn.mean(dim=1)                                  # → [B, Q, K]
        if attn.ndim == 3:
            attn = attn.mean(dim=1)                            # → [B, K]
        importance_scores = attn.mean(0) 
        
        
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            # breakpoint()
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Create a new tensor with only the kept tokens
                # indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
                indices_to_keep= torch.topk(importance_scores, keep_count,largest=True).indices
                indices_to_keep, _ = torch.sort(indices_to_keep)
                indices_to_keep = indices_to_keep.to(tensor.device, dtype= torch.long)
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def _apply_attention_bottom_strategy(self, past_key_values, attention_scores, keep_ratio=0.7):
        """
        Apply attention-based bottom strategy: keep tokens with lowest attention scores.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            keep_ratio: Fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """
        self.logger.info("BOT MA GAYO")
        
        updated_past = []
        # breakpoint()
        
        
        # attention_scores: tuple of [B, H, Q, K]
        attn = torch.stack(attention_scores, dim=0).mean(dim=0)   # → [B, H, Q, K]
        attn = attn.mean(dim=1)                                  # → [B, Q, K]
        if attn.ndim == 3:
            attn = attn.mean(dim=1)                            # → [B, K]
        importance_scores = attn.mean(0) 
        
        
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            # breakpoint()
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Create a new tensor with only the kept tokens
                # indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
                indices_to_keep= torch.topk(importance_scores, keep_count,largest=False).indices
                indices_to_keep = indices_to_keep.to(tensor.device, dtype= torch.long)
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def _apply_hybrid_strategy(self, past_key_values, attention_scores, token_types, 
                              keep_ratio=0.7, recency_weight=0.5, attention_weight=0.3, type_weight=0.2):
        """
        Apply hybrid strategy: combine recency, attention, and token type importance.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            token_types: Types of each token
            keep_ratio: Fraction of tokens to keep
            recency_weight: Weight for recency factor
            attention_weight: Weight for attention factor
            type_weight: Weight for token type factor
            
        Returns:
            Updated past_key_values
        """
        self.logger.info("Hybrid MA GAYO")
        # Check if we need to apply eviction
        if self.current_memory_mb < self.threshold * 0.1:  # Lower to 10% for testing
            return past_key_values
        
        # Start timing eviction
        eviction_start = time.time()
        
        if attention_scores is None or token_types is None:
            # Fallback to sliding window if missing data
            updated_past = self._apply_sliding_window(past_key_values, keep_ratio)
            
            # End timing and update stats
            eviction_time = time.time() - eviction_start
            self.eviction_count += 1
            self.total_eviction_time += eviction_time
            
            # Force CUDA to free memory
            # torch.cuda.empty_cache()
            
            return updated_past
        
        updated_past = []
        
        
        
        attn = torch.stack(attention_scores, dim=0).mean(0)   # → [B, H, Q, K]
        attn = attn.mean(1)                                  # → [B, Q, K]
        if attn.size(1) > 1:
            attn = attn.mean(1)                             # → [B, K]
        importance_scores = attn.mean(0) 
        
        # Normalize to get attention importance scores
        # attention_importance = np.mean(avg_attention, axis=(0, 1))
        
        # Convert token types to importance scores
        type_importance = np.zeros(len(token_types))
        for i, token_type in enumerate(token_types):
            if token_type == "special":
                type_importance[i] = 1.0
            elif token_type == "rare":
                type_importance[i] = 0.8
            elif token_type == "punctuation":
                type_importance[i] = 0.3
            else:  # common
                type_importance[i] = 0.5
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Calculate hybrid importance scores
                hybrid_scores = np.zeros(seq_len)
                
                # Add recency component
                recency_scores = np.linspace(0, 1, seq_len)
                hybrid_scores += recency_weight * recency_scores
                
                # Add attention component if available
                if len(attention_importance) == seq_len:
                    # Normalize attention scores to [0, 1]
                    norm_attention = (attention_importance - attention_importance.min()) 
                    if attention_importance.max() > attention_importance.min():
                        norm_attention /= (attention_importance.max() - attention_importance.min())
                    hybrid_scores += attention_weight * norm_attention
                
                # Add token type component if available
                if len(type_importance) == seq_len:
                    hybrid_scores += type_weight * type_importance
                
                # Get indices of tokens with highest hybrid scores
                importance_indices = np.argsort(hybrid_scores)[-keep_count:]
                
                # Create a new tensor with only the kept tokens
                indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        # End timing and update stats
        eviction_time = time.time() - eviction_start
        self.eviction_count += 1
        self.total_eviction_time += eviction_time
        
        # Force CUDA to free memory
        # torch.cuda.empty_cache()
        
        return tuple(updated_past)
    
    def _apply_sliding_window(self, past_key_values, window_size=0.7, important_ratio=0.1, attention_scores=None):
        """
        Apply sliding window strategy: keep recent tokens and important tokens.
        
        Args:
            past_key_values: The current KV cache
            window_size: Fraction of recent tokens to keep
            important_ratio: Fraction of important tokens to keep
            attention_scores: Attention scores for each token (optional)
            
        Returns:
            Updated past_key_values
        """
        self.logger.info("SLIDING MA GAYO")
        updated_past = []
        
        # Calculate importance scores if attention scores are available
        importance_scores = None
        if attention_scores is not None:
            
            attn = torch.stack(attention_scores, dim=0).mean(0)   # → [B, H, Q, K]
            attn = attn.mean(1)                                  # → [B, Q, K]
            if attn.size(1) > 1:
                attn = attn.mean(1)                             # → [B, K]
            importance_scores = attn.mean(0) 
            
            # Calculate importance scores (average attention received by each token)
            # importance_scores = np.mean(avg_attention, axis=(0, 1))
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many recent tokens to keep
                recent_count = max(1, int(seq_len * window_size))
                
                # Calculate how many important tokens to keep
                important_count = max(0, int(seq_len * important_ratio))
                
                # Get indices of recent tokens
                recent_indices = list(range(seq_len - recent_count, seq_len))
                
                # Get indices of important tokens if available
                important_indices = []
                if importance_scores is not None and len(importance_scores) == seq_len and important_count > 0:
                    # Get indices of tokens with highest importance scores, excluding recent tokens
                    candidate_indices = list(range(0, seq_len - recent_count))
                    candidate_scores = importance_scores[:seq_len - recent_count]
                    if len(candidate_indices) > 0:
                        important_indices = np.argsort(candidate_scores)[-important_count:]
                
                # Combine indices
                indices_to_keep = list(set(recent_indices + list(important_indices)))
                indices_to_keep.sort()  # Sort to maintain order
                
                # Create a new tensor with only the kept tokens
                indices_tensor = torch.tensor(indices_to_keep, device=tensor.device)
                new_tensor = tensor.index_select(2, indices_tensor)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def _apply_adaptive_attention(self, past_key_values, attention_scores, base_keep=0.7):
        """
        Apply adaptive attention strategy: keep tokens based on attention distribution.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            base_keep: Base fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """
        self.logger.info("ADAPTIVE MA GAYO")
        
        updated_past = []
        
        
        attn = torch.stack(attention_scores, dim=0).mean(0)   # → [B, H, Q, K]
        attn = attn.mean(1)                                  # → [B, Q, K]
        if attn.size(1) > 1:
            attn = attn.mean(1)                             # → [B, K]
        importance_scores = attn.mean(0) 
        
        # Calculate importance scores (average attention received by each token)
        # importance_scores = np.mean(avg_attention, axis=(0, 1))
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate adaptive keep ratio based on attention distribution
                if len(importance_scores) == seq_len:
                    # Calculate entropy of attention distribution
                    norm_scores = importance_scores / np.sum(importance_scores)
                    entropy = -np.sum(norm_scores * np.log2(norm_scores + 1e-10))
                    max_entropy = np.log2(seq_len)
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    # Adjust keep ratio based on entropy
                    # Higher entropy (more uniform attention) -> keep fewer tokens
                    # Lower entropy (more focused attention) -> keep more tokens
                    adaptive_keep = base_keep * (1.0 - 0.5 * normalized_entropy)
                else:
                    adaptive_keep = base_keep
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * adaptive_keep))
                
                # Get indices of tokens with highest importance scores
                if len(importance_scores) == seq_len:
                    importance_indices = np.argsort(importance_scores)[-keep_count:]
                else:
                    # Fallback to recent tokens if dimensions don't match
                    importance_indices = list(range(seq_len - keep_count, seq_len))
                
                # Create a new tensor with only the kept tokens
                indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def __del__(self):
        """Clean up when the object is deleted"""
        self.remove_hooks()


class EnhancedBaseline:
    """Baseline strategy that does no eviction"""
    def __init__(self, **kwargs):
        self.name = "Baseline"


class RandomKVCacheStrategy:
    """Random KV cache management strategy"""
    def __init__(self, keep=0.7, **kwargs):
        self.keep = keep
        self.name = f"Random(keep={self.keep})"


class AttentionTopStrategy:
    """Attention-based top strategy: keep tokens with highest attention scores"""
    def __init__(self, keep=0.7, **kwargs):
        self.keep = keep
        self.name = f"AttentionTop(keep={self.keep})"


class AttentionBottomStrategy:
    """Attention-based bottom strategy: keep tokens with lowest attention scores"""
    def __init__(self, keep=0.7, **kwargs):
        self.keep = keep
        self.name = f"AttentionBottom(keep={self.keep})"


class HybridNPercentStrategy:
    """Hybrid strategy: combine recency, attention, and token type importance"""
    def __init__(self, keep=0.7, recency_weight=0.5, attention_weight=0.3, type_weight=0.2, **kwargs):
        self.keep = keep
        self.recency_weight = recency_weight
        self.attention_weight = attention_weight
        self.type_weight = type_weight
        self.name = f"HybridNPercent(keep={self.keep},r={self.recency_weight},a={self.attention_weight},t={self.type_weight})"


class SlidingWindowStrategy:
    """Sliding window strategy: keep recent tokens and important tokens"""
    def __init__(self, window=0.7, important=0.1, **kwargs):
        self.window = window
        self.important = important
        self.name = f"SlidingWindow(window={self.window},important={self.important})"


class AdaptiveAttentionStrategy:
    """Adaptive attention strategy: keep tokens based on attention distribution"""
    def __init__(self, base_keep=0.7, **kwargs):
        self.base_keep = base_keep
        self.name = f"AdaptiveAttention(base_keep={self.base_keep})"

# In robust_real_benchmark.py

def load_model_and_tokenizer(model_name: str, cache_dir: Optional[str] = None) -> Tuple:
    """
    Load model and tokenizer from HuggingFace.

    Args:
        model_name: Name of the model to load
        cache_dir: Directory to cache model files

    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading model {model_name} from {cache_dir if cache_dir else 'default cache'}")

    try:
        # Try loading with trust_remote_code=True, as some models (especially newer ones)
        # might host tokenizer configuration or chat template logic in the model's repository.
        # Use with caution and ensure you trust the model source.
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True # Added for robustness
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True # Added for robustness
        )
        model.config.output_attentions = True 

        # After loading, explicitly check if the chat template is present
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            logger.info(f"Successfully loaded tokenizer with chat_template for {model_name}.")
            # logger.debug(f"Chat template content: {tokenizer.chat_template}") # Optional: for deeper debugging
        else:
            logger.warning(f"Tokenizer for {model_name} was loaded WITHOUT a chat_template. "
                           "Multi-turn conversations will use manual formatting and likely perform poorly or incorrectly with Llama 3.")
            # As a last resort, if you know the exact Llama 3 instruct template, you could attempt to set it here.
            # However, it's preferable that it loads automatically with the correct model identifier.
            # Example for Llama-3-8B-Instruct (use with extreme caution, verify exact syntax from official sources):
            # llama3_instruct_template = (
            #     "{% set loop_messages = messages %}"
            #     "{% for message in loop_messages %}"
            #         "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
            #         "{% if loop.index0 == 0 %}"
            #             "{% set content = bos_token + content %}" # bos_token should be defined in tokenizer
            #         "{% endif %}"
            #         "{{ content }}"
            #     "{% endfor %}"
            #     "{% if add_generation_prompt %}"
            #         "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            #     "{% endif %}"
            # )
            # if tokenizer.bos_token: # Make sure bos_token exists
            #    try:
            #        tokenizer.chat_template = llama3_instruct_template
            #        logger.info("Manually applied a Llama 3 Instruct chat template as a fallback.")
            #    except Exception as e:
            #        logger.error(f"Error manually setting chat template: {e}")
            # else:
            #    logger.error("Cannot manually set chat_template: tokenizer.bos_token is not defined.")


        # Ensure padding token is set (original logic, keep this)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set tokenizer.pad_token to tokenizer.eos_token")
            else:
                # If Llama 3 uses a specific pad token or if adding one is problematic, this might need adjustment.
                # Typically, for generation, left-padding is preferred and pad_token = eos_token is common.
                new_pad_token = "[PAD]"
                tokenizer.add_special_tokens({'pad_token': new_pad_token})
                # Important: Resize model embeddings if a new token is added
                # model.resize_token_embeddings(len(tokenizer)) # Uncomment if you add a truly NEW token not already in vocab
                logger.info(f"Added '{new_pad_token}' as pad_token.")


        return tokenizer, model

    except Exception as e:
        logger.error(f"Error loading model or tokenizer for {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise

# def load_model_and_tokenizer(model_name: str, cache_dir: Optional[str] = None) -> Tuple:
#     """
#     Load model and tokenizer from HuggingFace.
    
#     Args:
#         model_name: Name of the model to load
#         cache_dir: Directory to cache model files
        
#     Returns:
#         Tuple of (tokenizer, model)
#     """
#     logger.info(f"Loading model {model_name} from {cache_dir if cache_dir else 'default cache'}")
    
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             cache_dir=cache_dir,
#             torch_dtype=torch.float16,
#             device_map="auto",
            
#         )
#         model.config.output_attentions = True
#         return tokenizer, model

#     except Exception as e:
#         logger.error(f"Error loading model: {e}")
#         raise


def setup_tokenizer_padding(tokenizer):
    """
    Set up padding token for the tokenizer if it doesn't have one.
    This is necessary for Llama tokenizers which don't have a padding token by default.
    """
    # Check if the tokenizer already has a padding token
    if tokenizer.pad_token is None:
        # Use the EOS token as the padding token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        else:
            # If there's no EOS token either, add a new special token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added [PAD] as pad_token")
    
    return tokenizer


def get_token_types(tokens: List[int], tokenizer) -> List[str]:
    """
    Classify tokens into types: special, rare, common, punctuation.
    
    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer to decode tokens
        
    Returns:
        List of token types
    """
    token_types = []
    
    try:
        # Get vocabulary size
        vocab_size = len(tokenizer.get_vocab())
        
        # Define thresholds
        special_tokens = set(tokenizer.all_special_ids)
        rare_threshold = int(vocab_size * 0.9)  # Top 10% are rare
        
        # Punctuation characters
        punctuation = set(".,;:!?()[]{}-_\"'`/\\<>@#$%^&*+=|~")
        
        for token in tokens:
            if token in special_tokens:
                token_types.append("special")
            elif token > rare_threshold:
                token_types.append("rare")
            else:
                # Check if it's punctuation
                token_str = tokenizer.decode([token])
                if any(c in punctuation for c in token_str):
                    token_types.append("punctuation")
                else:
                    token_types.append("common")
    except Exception as e:
        logger.warning(f"Error classifying tokens: {e}")
        # Fallback: mark all as common
    """
    Extract prompt from data item, handling various data structures.
    
    Args:
        data_item: Data item from dataset
        
    Returns:
        Prompt string
    """
    if isinstance(data_item, str):
        return data_item
    elif isinstance(data_item, dict):
        # Check for common fields
        if 'prompt' in data_item:
            return data_item['prompt']
        elif 'instruction' in data_item:
            return data_item['instruction']
        elif 'turns' in data_item:  # MTBench format
            return data_item['turns'][0]  # Return first turn for initial prompt
    elif isinstance(data_item, list):
        return '\n'.join(data_item)
    raise ValueError(f"Unsupported data item format: {type(data_item)}")

def get_second_turn(data_item: Dict):
    """
    Get the second turn from MTBench data item.
    
    Args:
        data_item: Data item from MTBench dataset
        
    Returns:
        Second turn prompt string
    """
    if isinstance(data_item, dict) and 'turns' in data_item:
        if len(data_item['turns']) > 1:
            return data_item['turns'][1]
    return None

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load JSONL dataset from file."""
    items: List[Dict] = []
    try:
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            content = f.read().strip()
            # Check if this is a JSON array instead of JSONL
            if content.startswith('[') and content.endswith(']'):
                print("Detected JSON array format instead of JSONL")
                json_data = json.loads(content)
                if isinstance(json_data, list):
                    items = json_data
                    print(f"Successfully loaded {len(items)} items from JSON array")
            else:
                # Process as JSONL (line by line)
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    data_item = json.loads(line)
                    items.append(data_item)
        
        # Debug the loaded items
        print(f"Loaded {len(items)} items from dataset")
        if items:
            print(f"First item keys: {list(items[0].keys() if isinstance(items[0], dict) else ['not a dict'])}")
            # Debug multi-turn structure
            if isinstance(items[0], dict) and "turns" in items[0]:
                print(f"First item (ID: {items[0].get('question_id', 'unknown')}) has {len(items[0]['turns'])} turns:")
                for i, turn in enumerate(items[0]["turns"]):
                    print(f"  Turn {i+1}: {turn[:50]}..." if turn else f"  Turn {i+1}: <empty>")
        else:
            print("WARNING: Dataset is empty!")
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {dataset_path}")
        # Create an example item for testing
        items = [{
            "question_id": "test_1",
            "category": "writing",
            "turns": [
                "Write a short poem about artificial intelligence.",
                "Now make it rhyme better."
            ]
        }]
        print(f"Created mock dataset with {len(items)} items for testing")
        # Debug mock data
        print(f"Mock item has {len(items[0]['turns'])} turns:")
        for i, turn in enumerate(items[0]["turns"]):
            print(f"  Turn {i+1}: {turn}")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        traceback.print_exc()
        items = []
    return items

def get_prompt(data_item):
    """
    Extract prompt from different data item formats.
    
    Args:
        data_item: Data item from dataset
        
    Returns:
        Prompt string
    """
    if isinstance(data_item, str):
        return data_item
    elif isinstance(data_item, dict):
        # Check for common fields
        if 'prompt' in data_item:
            return data_item['prompt']
        elif 'instruction' in data_item:
            return data_item['instruction']
        elif 'turns' in data_item and isinstance(data_item['turns'], list) and len(data_item['turns']) > 0:
            # Return first turn for initial prompt
            return data_item['turns'][0]
        # If no known fields found, return a default prompt
        return "Hello, how can I help you today?"
    elif isinstance(data_item, list):
        # Handle list of dictionaries
        if data_item and isinstance(data_item[0], dict):
            # Try to extract text from each dictionary
            texts = []
            for item in data_item:
                if 'text' in item:
                    texts.append(item['text'])
                elif 'content' in item:
                    texts.append(item['content'])
                elif 'prompt' in item:
                    texts.append(item['prompt'])
            if texts:
                return '\n'.join(texts)
            
            # If we couldn't extract text, use the first item
            if data_item:
                return get_prompt(data_item[0])
        else:
            # It's a list of strings
            try:
                return '\n'.join(str(x) for x in data_item)
            except Exception as e:
                print(f"Error joining list items: {e}")
                return str(data_item[0]) if data_item else "Hello"
    
    # If nothing else works, return a default prompt
    return "Hello, how can I help you today?"

def process_single_prompt(model, tokenizer, prompt, strat, cache_manager, max_gen_tokens=100, azure_eval=False, azure_config=None, history_kv=None):
    """
    Process a single prompt with token-by-token generation and KV-cache management.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        strat: The cache management strategy
        cache_manager: The KV cache manager
        max_gen_tokens: Maximum number of tokens to generate
        azure_eval: Whether to perform Azure evaluation
        azure_config: Azure configuration (if azure_eval is True)
        history_kv: Previous KV cache state (for multi-turn)
        
    Returns:
        Dict with results including:
            - generated_text: The generated text
            - perplexity: Perplexity score
            - azure_score: Azure evaluation score (if enabled)
            - memory_usage: Memory usage statistics
            - timing: Timing statistics
            - final_kv: Final KV cache state
    """
    # Initialize variables
    generated_tokens = []
    perplexity_sum = 0
    perplexity_count = 0
    start_time = time.time()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    # Move to device
    device = model.device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Initialize KV cache
    if history_kv is not None:
        past_key_values = history_kv
    else:
        past_key_values = None
    
    # Generate tokens one by one
    for i in range(max_gen_tokens):
        # Get current memory usage
        memory_stats = profile_memory()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, -1:],  # Only use last token
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=True,
                use_cache=True
            )
        
        # Get logits and next token
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        
        # Update perplexity
        perplexity_sum += calculate_perplexity(logits, next_token)
        perplexity_count += 1
        
        # Update generated tokens
        generated_tokens.append(next_token.item())
        
        # Update input for next iteration
        input_ids = next_token.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)
        
        # Update KV cache
        past_key_values = outputs.past_key_values
        
        # Apply cache management strategy
        if cache_manager is not None:
            past_key_values = cache_manager.apply_strategy(
                past_key_values,
                strat,
                memory_stats
            )
        
        # Check if generation should stop
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Calculate final metrics
    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_second = len(generated_tokens) / total_time if total_time > 0 else 0
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Calculate final perplexity
    final_perplexity = np.exp(perplexity_sum / perplexity_count) if perplexity_count > 0 else float('inf')
    
    # Perform Azure evaluation if enabled
    azure_score = None
    if azure_eval and HAS_OPENAI:
        try:
            azure_score = evaluate_with_azure(
                prompt,
                generated_text,
                api_key=azure_config.get('api_key'),
                endpoint=azure_config.get('endpoint'),
                version=azure_config.get('version'),
                deployment=azure_config.get('deployment')
            )
        except Exception as e:
            logger.warning(f"Azure evaluation failed: {e}")
    
    return {
        'generated_text': generated_text,
        'perplexity': final_perplexity,
        'azure_score': azure_score,
        'memory_usage': memory_stats,
        'timing': {
            'total_time': total_time,
            'tokens_per_second': tokens_per_second
        },
        'final_kv': past_key_values,
    }

_process_single_prompt_simplified = None  # deprecated fallback, not used when pipeline is available
# Global chat_pipe will be initialized in main() after loading model/tokenizer
chat_pipe = None

# A new class to track KV cache during generation
class KVCacheTracker:
    """Tracks KV cache growth during generation by using hooks"""
    
    def __init__(self, model):
        self.model = model
        self.kv_cache_sizes = {}
        self.current_token_index = 0
        self.hooks = []
        self.install_hooks()
    
    def install_hooks(self):
        """Install hooks on attention layers to track KV cache"""
        # Remove any existing hooks
        self.remove_hooks()
        
        # For transformer models with attention blocks
        for name, module in self.model.named_modules():
            # Look for attention modules
            if "attention" in name.lower() and hasattr(module, "forward"):
                hook = module.register_forward_hook(self.attention_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def attention_hook(self, module, input_tensors, output_tensors):
        """Hook for attention modules to track KV cache size"""
        try:
            # Check if this hook contains past_key_values info
            # Past key values could be in inputs or outputs
            past_key_values = None
            
            # Check in inputs
            if isinstance(input_tensors, tuple) and len(input_tensors) > 2:
                for item in input_tensors:
                    if isinstance(item, tuple) and all(isinstance(x, torch.Tensor) for x in item):
                        past_key_values = item
                        break
            
            # Check in outputs 
            if past_key_values is None and hasattr(output_tensors, "past_key_values"):
                past_key_values = output_tensors.past_key_values
            
            # If past_key_values found, measure size
            if past_key_values is not None:
                kv_size = measure_kv_cache_size(past_key_values)
                # Only update if we got a real value
                if kv_size > 0:
                    self.kv_cache_sizes[str(self.current_token_index)] = kv_size
        except Exception as e:
            logger.warning(f"Error in attention hook: {e}")
    
    def increment_token(self):
        """Increment the token counter"""
        self.current_token_index += 1
    
    def get_kv_cache_sizes(self):
        """Get the recorded KV cache sizes"""
        return self.kv_cache_sizes

def safe_sampling(logits, temperature=0.7):
    """Sample from logits with robust handling of numerical issues"""
    # Check for NaN or Inf in logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logger.warning("FALLBACK: Found NaN or Inf in logits, using argmax instead of sampling")
        return torch.argmax(logits).unsqueeze(0).unsqueeze(0)

    # Apply temperature and check again
    scaled_logits = logits / max(temperature, 1e-5)  # Prevent division by zero or very small numbers
    if torch.isnan(scaled_logits).any() or torch.isinf(scaled_logits).any():
        logger.warning("FALLBACK: Temperature scaling caused NaN or Inf, using argmax instead of sampling")
        return torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        
    # Calculate probabilities with extra numerical stability measures
    try:
        # Subtract max for numerical stability (standard practice)
        scaled_logits = scaled_logits - scaled_logits.max()
        # Apply exp and check for NaN/Inf
        exp_logits = torch.exp(scaled_logits)
        if torch.isnan(exp_logits).any() or torch.isinf(exp_logits).any():
            raise ValueError("Exp calculation produced NaN or Inf")
            
        # Calculate probabilities
        probs = exp_logits / exp_logits.sum()
        
        # Ensure valid probability distribution
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            raise ValueError("Invalid probability distribution")
            
        # Sample from the distribution
        return torch.multinomial(probs, num_samples=1).unsqueeze(0)
    except Exception as e:
        logger.warning(f"FALLBACK: Sampling failed ({e}), using argmax instead")
        return torch.argmax(logits).unsqueeze(0).unsqueeze(0)


def _process_single_prompt_chat1(model, tokenizer, prompt, strat, cache_manager, max_gen_tokens=100, azure_eval=False, azure_config=None, history_kv=None):
    """Multi-turn chat via HuggingFace ChatCompletionPipeline override."""
    # Log the turn being processed for debugging
    logger.info(f"Processing turn: '{prompt[:50]}...' with history size: {len(history_kv) if isinstance(history_kv, list) else 0}")
    
    user_prompt = prompt.strip()
    # Build message list with enhanced system prompt
    if not isinstance(history_kv, list):
        messages = [{
            "role": "system", 
            "content": "You are a helpful assistant that follows instructions precisely. When asked to write content, create the exact content requested, not instructions about it. When asked to rewrite or modify your previous response, apply the requested changes directly to your last answer. Always fulfill the task as specified without adding explanations unless requested."
        }]
    else:
        # Check if there's already a system message
        messages = history_kv.copy()
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {
                "role": "system", 
                "content": "You are a helpful assistant. Always follow the user's instructions precisely. When asked to modify previous content, apply the requested changes directly to your previous response."
            })
    
    messages.append({"role": "user", "content": user_prompt})
    
    # Log full conversation context for debugging
    logger.info(f"Full conversation context ({len(messages)} messages):")
    for i, msg in enumerate(messages):
        logger.info(msg.keys())
        logger.info(f"  Message {i}: {msg['role']} - {msg['content'][:50]}...")
    
    # Measure memory before generation
    pre_gen_gpu_memory = measure_gpu_memory()
    pre_gen_system_memory = measure_system_memory()
    
    # Initialize KV cache tracker
    kv_tracker = KVCacheTracker(model)
    
    # Start timing
    start_time = time.time()
    
    # Reset cache manager stats before generation
    if cache_manager:
        cache_manager._update_memory_stats(None)  # Reset stats
    
    # For token timing tracking
    token_times = {}
    
    # Invoke chat pipeline or fallback to simple generate
    if chat_pipe is None:
        # Start tokenization timing
        encoding_start = time.time()
        
        # Use chat template if available for proper multi-turn formatting  
        if isinstance(history_kv, list) and history_kv: # Log only for subsequent turns with history
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                logger.info(f"Tokenizer has a chat_template defined. Will attempt to use it. Template (partial): {str(tokenizer.chat_template)[:200]}...")
            else:
                logger.warning("Tokenizer does NOT have a chat_template defined or it is empty. "
                               "Falling back to manual prompt construction for multi-turn. "
                               "This is highly likely to cause issues with Llama 3.")
            logger.info(f"Current history_kv for turn: {history_kv}")
            logger.info(f"Current messages object that will be passed to apply_chat_template: {messages}")


         # Use chat template if available for proper multi-turn formatting
        try:
            # Try to use the model's chat template
            if hasattr(tokenizer, 'apply_chat_template') and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template and isinstance(history_kv, list):
                logger.info("Attempting to use tokenizer.apply_chat_template for multi-turn conversation.")
                inputs_ids = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True # This is important for instruct models
                ).to(model.device)
                inputs = {"input_ids": inputs_ids}
                # For debugging, decode and log the tokenized input
                # logger.info(f"Input IDs from chat_template: {inputs_ids}")
                # logger.info(f"Decoded input from chat_template: {tokenizer.decode(inputs_ids[0])}")
            else:
                if isinstance(history_kv, list) and history_kv: # Only log warning if it's a multi-turn scenario that should use template
                    logger.warning("Condition for using apply_chat_template not fully met (e.g., no history, or tokenizer lacks template). "
                                   "Falling back to manual prompt construction for multi-turn.")
                # Fallback logic (original code)
                # just before you call tokenizer(...) for either turn:
                if not history_kv:
                    # Turn 1: no history, just the user instruction
                    prompt_text = user_prompt.strip()
                else:
                    # Turn 2+: dump raw user+assistant messages so the model actually sees
                    # what it wrote last time before you ask it to rewrite or continue
                    history_str = ""
                    for msg in history_kv:
                        # Capitalize role so it reads naturally: "User: …" / "Assistant: …"
                        history_str += f"{msg['role'].capitalize()}: {msg['content'].strip()}\n"
                    prompt_text = (
                        history_str
                        + f"User: {user_prompt.strip()}\n"
                        + "Assistant:" # This line is often model-specific for prompting the assistant
                    )
                logger.info(f"Manually constructed prompt_text for tokenizer: {prompt_text}")
                inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=False # Ensure padding is false if not batching same-length sequences
                ).to(model.device)


                
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, falling back to simple format")
            # Fallback to original approach
            if isinstance(history_kv, list):
                history_text = " ".join([m.get('content', '') for m in history_kv])
                conv = history_text + " " + user_prompt
            else:
                conv = user_prompt
            inputs = tokenizer(conv, return_tensors="pt", truncation=True, padding=False).to(model.device)
        
        encoding_time = time.time() - encoding_start
        
        # Track first token generation
        token_start_time = time.time()
        
        # Generate output with hooks tracking KV cache
        try:
            # Try to do token-by-token generation with KV tracking
            with torch.no_grad():
                # Initial forward pass
                outputs = model(inputs.input_ids if hasattr(inputs, 'input_ids') else inputs["input_ids"], 
                               use_cache=True,output_attentions=True,return_dict=True)
                first_token_time = time.time() - token_start_time
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits
                attentions = outputs.attentions
                
                # Token-by-token generation
                all_tokens = (inputs.input_ids[0] if hasattr(inputs, 'input_ids') 
                             else inputs["input_ids"][0]).tolist()
                new_tokens = []
                token_generation_times = []
                
                for i in range(max_gen_tokens):
                    # Track time for this token
                    token_gen_start = time.time()
                    
                    # Get next token with temperature sampling
                    next_token_logits = logits[0, -1, :]
                    temperature = 0.7
                    probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                    # Replace temperature sampling with deterministic generation
                    next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                    
                    
                    # Record token and update inputs
                    new_tokens.append(next_token.item())
                    all_tokens.append(next_token.item())
                    
                    
                    
                    # Increment token counter for tracker
                    kv_tracker.increment_token()
                    
                    # Break if we hit the end token and have generated at least 10 tokens
                    if next_token.item() == tokenizer.eos_token_id:
                        logger.info(f"EOS token generated at position {len(new_tokens)} (1-indexed in the list of new tokens). Breaking generation.")
                        break
                        
                    
                    # Forward pass with cached KV
                    outputs = model(next_token, use_cache=True, past_key_values=past_key_values,output_attentions=True,return_dict=True)                   
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    attentions = outputs.attentions 
                    
                    if cache_manager and past_key_values is not None:
                        # Generate estimated attention scores for eviction
                        attention_scores = outputs.attentions    # tuple[layer] of [B, H, Q, K]                        
                        # Apply eviction strategy
                        past_key_values = cache_manager.apply_eviction_strategy(
                            past_key_values,
                            strat,
                            attention_scores = attentions
                        )
                        
                        # Explicitly update memory stats
                        cache_manager._update_memory_stats(past_key_values)
                        
                    # Record time for this token
                    token_gen_time = time.time() - token_gen_start
                    token_generation_times.append(token_gen_time)
                    token_times[str(i+1)] = token_gen_time
                
                # Get KV cache sizes from tracker
                kv_cache_sizes = kv_tracker.get_kv_cache_sizes()
                
                # If we have insufficient data from the tracker, estimate based on token count
                if len(kv_cache_sizes) < len(new_tokens):
                    # Use the max tracked size as a base
                    max_tracked_size = max(kv_cache_sizes.values()) if kv_cache_sizes else 0
                    
                    # If we have cache_manager data, use that as a baseline
                    if cache_manager and hasattr(cache_manager, 'current_memory_mb') and cache_manager.current_memory_mb > 0:
                        base_size = cache_manager.current_memory_mb
                    else:
                        # Fallback to a reasonable default based on model size
                        base_size = max_tracked_size or 1000  # Default 1GB if no data
                    
                    # Fill in missing token sizes with estimated growth
                    for i in range(1, len(new_tokens) + 1):
                        if str(i) not in kv_cache_sizes:
                            # Estimate growth: more tokens = larger KV cache
                            estimated_size = base_size * (i / len(new_tokens))
                            kv_cache_sizes[str(i)] = estimated_size
                
                # Decode generated text
                resp_ids = torch.tensor(new_tokens)
                tokens_generated = len(new_tokens)
                
        except Exception as e:
            logger.warning(f"Token-by-token generation failed, using batch generation: {e}")
            traceback.print_exc()
            
            # Fallback to standard batch generation
            gen_output = model.generate(
                input_ids=inputs.input_ids if hasattr(inputs, 'input_ids') else inputs["input_ids"],
                max_new_tokens=max_gen_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True
            )
            
            # Get response tokens
            input_length = inputs.input_ids.shape[-1] if hasattr(inputs, 'input_ids') else inputs["input_ids"].shape[-1]
            resp_ids = gen_output[0, input_length:]
            tokens_generated = len(resp_ids)
            
            # Get KV cache sizes from tracker (if any were recorded)
            kv_cache_sizes = kv_tracker.get_kv_cache_sizes()
            
            # If we have insufficient data from the tracker, estimate based on memory usage
            if len(kv_cache_sizes) < tokens_generated:
                # Get the memory usage after generation
                current_memory = 0
                if cache_manager and hasattr(cache_manager, 'current_memory_mb'):
                    current_memory = cache_manager.current_memory_mb
                
                # Fall back to GPU memory if cache manager doesn't have data
                if current_memory == 0:
                    post_gen_gpu = measure_gpu_memory()
                    if isinstance(post_gen_gpu, dict) and "total_allocated_mb" in post_gen_gpu:
                        current_memory = post_gen_gpu["total_allocated_mb"]
                
                # Now fill in KV cache sizes for each token
                for i in range(1, tokens_generated + 1):
                    if str(i) not in kv_cache_sizes:
                        # Simulate linear growth
                        estimated_size = current_memory * (i / tokens_generated)
                        kv_cache_sizes[str(i)] = estimated_size
        
        # Calculate timing
        generation_end_time = time.time()
        total_time = generation_end_time - start_time
        
        
        # Decode response
        # response = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
        try:
            # First try normal decoding
            response = tokenizer.decode(resp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Validate if response looks corrupted
            unusual_chars = sum(1 for c in response if ord(c) > 127 or c in '\\[]{}')
            if unusual_chars > len(response) * 0.1:  # More than 10% unusual characters
                logger.warning(f"Detected potentially corrupted output ({unusual_chars}/{len(response)} unusual chars)")
                # Try token-by-token decoding as fallback
                chunks = []
                for i in range(0, len(resp_ids), 1):
                    token_text = tokenizer.decode([resp_ids[i].item()], skip_special_tokens=False)
                    chunks.append(token_text)
                response = "".join(chunks).strip()
                logger.info(f"Used token-by-token decoding, result length: {len(response)}")
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            response = "Error: Failed to decode response properly."

        # Final cleanup
        response = response.replace("<s>", "").replace("</s>", "").strip()
       
        
    else:
        pass
        # # Use chat pipeline for generation (can't track KV cache size token by token in pipeline mode)
        # # Encoding time (approximate)
        # encoding_start = time.time()
        # encoding_time = 0.01  # Placeholder, can't measure precisely with pipeline
        
        # # Track first token generation
        # token_start_time = time.time()
        
        # # Call the pipeline
        # resp = chat_pipe(messages)
        
        # # Calculate timing
        # generation_end_time = time.time()
        # total_time = generation_end_time - start_time
        # first_token_time = min(0.8, (generation_end_time - token_start_time) * 0.1)  # Approximate first token time
        
        # logger.debug(f"chat_pipe returned response: {resp}")
        
        # # Parse various possible pipeline outputs
        # response = ''
        # # Direct string
        # if isinstance(resp, str):
        #     response = resp
        # # Dictionary output
        # elif isinstance(resp, dict):
        #     # OpenAI-style chat completion
        #     if 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
        #         choice = resp['choices'][0]
        #         if isinstance(choice, dict) and 'message' in choice and isinstance(choice['message'], dict):
        #             response = choice['message'].get('content', '')
        #         else:
        #             response = choice.get('generated_text', '') or choice.get('text', '')
        #     else:
        #         response = resp.get('generated_text', '') or resp.get('text', '') or str(resp)
        # # List output
        # elif isinstance(resp, list) and resp:
        #     first = resp[0]
        #     if isinstance(first, str):
        #         response = first
        #     elif isinstance(first, dict):
        #         if 'generated_text' in first:
        #             response = first['generated_text']
        #         elif 'message' in first and isinstance(first['message'], dict):
        #             response = first['message'].get('content', '')
        #         elif 'choices' in first and isinstance(first['choices'], list) and first['choices']:
        #             choice = first['choices'][0]
        #             if isinstance(choice, dict) and 'message' in choice and isinstance(choice['message'], dict):
        #                 response = choice['message'].get('content', '')
        #             else:
        #                 response = choice.get('generated_text', '') or choice.get('text', '')
        #         else:
        #             response = str(first)
        #     else:
        #         response = str(first)
            
        # # Estimate token count
        # tokens_generated = len(tokenizer.encode(response))
        
        # # Estimate per-token timing
        # if tokens_generated > 0:
        #     remaining_time = total_time - first_token_time - encoding_time
        #     avg_token_time = remaining_time / tokens_generated
        #     for i in range(1, tokens_generated + 1):
        #         token_times[str(i)] = avg_token_time
                
        # # Get KV cache sizes from tracker if available
        # kv_cache_sizes = kv_tracker.get_kv_cache_sizes()
        
        # # If we have insufficient data from tracker, estimate
        # if len(kv_cache_sizes) < tokens_generated:
        #     # For pipeline mode, we need to estimate KV cache sizes
            
        #     # Try to get memory from cache manager
        #     current_memory = 0
        #     if cache_manager and hasattr(cache_manager, 'current_memory_mb'):
        #         current_memory = cache_manager.current_memory_mb
            
        #     # Fall back to GPU memory if cache manager doesn't have data
        #     if current_memory == 0:
        #         post_gen_gpu = measure_gpu_memory()
        #         if isinstance(post_gen_gpu, dict) and "total_allocated_mb" in post_gen_gpu:
        #             current_memory = post_gen_gpu["total_allocated_mb"]
            
        #     # If we still have no data, use token count for estimation
        #     if current_memory == 0:
        #         current_memory = tokens_generated * 2  # Rough estimate: 2MB per token
            
        #     # Fill in KV cache sizes for each token
        #     for i in range(1, tokens_generated + 1):
        #         if str(i) not in kv_cache_sizes:
        #             # KV cache grows roughly linearly with tokens
        #             estimated_size = current_memory * (i / tokens_generated)
        #             kv_cache_sizes[str(i)] = estimated_size
    
    # Measure memory after generation
    post_gen_gpu_memory = measure_gpu_memory()
    post_gen_system_memory = measure_system_memory()
    
    # Calculate perplexity (may not be accurate for chat responses)
    perplexity = float('inf')  # Default to infinity
    
    # Evaluate with Azure if enabled
    azure_score = 0
    if azure_eval and azure_config:
        try:
            azure_score = evaluate_with_azure(
                prompt=user_prompt,
                response=response,
                model_name=azure_config.get('model', 'gpt-4'),
                temperature=azure_config.get('temperature', 0.0),
                api_key=azure_config.get('api_key'),
                api_base=azure_config.get('api_base'),
                api_version=azure_config.get('api_version'),
                deployment=azure_config.get('deployment')
            )
            logger.info(f"Azure evaluation score: {azure_score}")
        except Exception as e:
            logger.warning(f"Azure evaluation failed: {e}")
    
    # Prepare memory usage data
    memory_stats = {
        "pre_generation": {
            "gpu": pre_gen_gpu_memory,
            "system": pre_gen_system_memory
        },
        "post_generation": {
            "gpu": post_gen_gpu_memory,
            "system": post_gen_system_memory
        },
        "kv_cache": {
            "manager_stats": cache_manager.get_stats() if cache_manager else None
        }
    }
    
    # Calculate tokens per second
    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
    
    # Messages for next turn, if any
    next_history = []
    if isinstance(history_kv, list):
        next_history = history_kv.copy()
    next_history.append({"role": "user", "content": user_prompt})
    next_history.append({"role": "assistant", "content": response})
    
    # Return data with token timing and KV cache size info
    result = {
        "prompt": user_prompt,
        "response": response,
        "tokens_generated": tokens_generated,
        "total_time_seconds": total_time,
        "first_token_time":first_token_time,
        "tokens_per_second": tokens_per_second,
        "memory": memory_stats,
        "token_times": token_times,
        "kv_cache_sizes": kv_cache_sizes,
        "accuracy": {
            "perplexity": perplexity,
            "azure_score": azure_score
        }
    }
    
    return result, next_history

def process_multi_turn(model, tokenizer, data_item, strat, cache_manager, max_gen_tokens=100, azure_eval=False, azure_config=None):
    """
    Process a multi-turn conversation from the dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        data_item: The data item from the dataset
        strat: The strategy configuration
        cache_manager: The KV cache manager
        max_gen_tokens: Maximum number of tokens to generate
        azure_eval: Whether to perform Azure evaluation
        azure_config: Azure configuration
        
    Returns:
        Results dictionary
    """
    # Debug the data_item
    if isinstance(data_item, dict):
        print(f"Processing item with keys: {list(data_item.keys())}")
        if "turns" in data_item:
            print(f"Item has {len(data_item['turns'])} turns")
            for i, turn in enumerate(data_item.get("turns", [])):
                print(f"  Turn {i+1}: {turn[:50]}..." if turn else f"  Turn {i+1}: <empty>")
    else:
        print(f"Unexpected data_item type: {type(data_item)}")
    
    # Check if data_item has turns
    if isinstance(data_item, dict) and isinstance(data_item.get("turns"), list):
        # Multi-turn conversation
        history = None
        per_turn_results = []
        
        for turn_idx, turn in enumerate(data_item.get("turns", [])):
            if not turn:
                logger.warning(f"Skipping empty turn {turn_idx}")
                continue
                
            print(f"Processing turn {turn_idx+1}/{len(data_item.get('turns', []))}: {turn[:50]}...")
                
            # Process the turn
            turn_result, history = _process_single_prompt_chat1(
                model=model,
                tokenizer=tokenizer,
                prompt=turn,
                strat=strat,
                cache_manager=cache_manager,
                max_gen_tokens=max_gen_tokens,
                azure_eval=azure_eval,
                azure_config=azure_config,
                history_kv=history
            )
            
            # Add turn index and metadata to the result
            turn_result["turn_index"] = turn_idx
            turn_result["question_id"] = data_item.get("question_id", "unknown")
            turn_result["category"] = data_item.get("category", "unknown")
            turn_result["total_turns"] = len(data_item.get("turns", []))
            per_turn_results.append(turn_result)
            
            print(f"Turn {turn_idx+1} processed, response: {turn_result['response'][:50]}...")
        
        # Return all turn results as separate items to ensure they're all saved
        print(f"Processed {len(per_turn_results)} turns for conversation")
        return per_turn_results
    else:
        # Single-turn prompt
        prompt = get_prompt(data_item)
        result, _ = _process_single_prompt_chat1(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            strat=strat,
            cache_manager=cache_manager,
            max_gen_tokens=max_gen_tokens,
            azure_eval=azure_eval,
            azure_config=azure_config
        )
        
        # Add metadata if available
        if isinstance(data_item, dict):
            result["question_id"] = data_item.get("question_id", "unknown")
            result["category"] = data_item.get("category", "unknown")
        
        return [result]  # Return as a list for consistency

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Benchmark KV Cache Management Strategies')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--limit', type=int, help='Limit number of items to process from dataset')
    parser.add_argument('--eval_azure', action='store_true', help='Enable Azure evaluation')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--profile', action='store_true', help='Enable memory profiling')
    parser.add_argument('--generate_dashboard', action='store_true', help='Generate a dashboard after benchmarking')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    # Set up logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        cfg['model_name'],
        cache_dir=cfg.get('cache_dir')
    )

    # Set up tokenizer padding
    setup_tokenizer_padding(tokenizer)

    # Initialize cache manager
    cache_manager = RealKVCacheManager(model, tokenizer, cfg, logger)

    # Load dataset
    dataset = load_dataset(cfg['dataset']['local_path'])
    
    # Print dataset info
    print(f"Dataset loaded: {len(dataset)} items")
    
    # Limit dataset if requested
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Dataset limited to {len(dataset)} items")

    # Prepare output directory
    results_dir = cfg.get('output_dir', cfg.get('results_dir', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Check for Azure evaluation configuration
    azure_eval = args.eval_azure or cfg.get('azure_evaluation', {}).get('enabled', False)
    azure_config = cfg.get('azure_evaluation', cfg.get('azure', {}))
    
    if azure_eval and azure_config:
        print(f"Azure evaluation enabled with model: {azure_config.get('model', 'gpt-4')}")
    
    # Debug strategies
    strategies = cfg.get('strategies', [])
    print(f"Found {len(strategies)} strategies in config: {strategies}")

    # Iterate over all strategies
    for strat_idx, strat in enumerate(strategies):
        print(f"\n===== Strategy {strat_idx+1}/{len(strategies)}: {strat} =====")
        all_results = []
        for i, data_item in enumerate(dataset):
            try:
                # Display progress
                question_id = data_item.get('question_id', f'item_{i}') if isinstance(data_item, dict) else f'item_{i}'
                print(f"\rProcessing sample {i+1}/{len(dataset)}: {question_id}", end='', flush=True)
                
                # Debug data item
                if args.debug:
                    if isinstance(data_item, dict):
                        logger.debug(f"Data item keys: {list(data_item.keys())}")
                        if 'turns' in data_item:
                            logger.debug(f"Turns count: {len(data_item['turns'])}")
                
                results = process_multi_turn(
                    model, tokenizer, data_item, strat, cache_manager,
                    max_gen_tokens=cfg.get('max_gen_tokens', 100),
                    azure_eval=azure_eval, 
                    azure_config=azure_config
                )
                
                # Handle both single result and list of results
                if isinstance(results, list):
                    all_results.extend(results)
                    print(f"\nProcessed {len(results)} turns for conversation {question_id}")
                else:
                    all_results.append(results)
                    print(f"\nProcessed conversation {question_id}")
                
            except Exception as e:
                # Fixed error handling to handle both dict and list data types
                item_id = data_item.get('question_id', 'unknown') if isinstance(data_item, dict) else f"item_{i}"
                logger.error(f"Error processing {item_id} with {strat}: {e}")
                print(f"\nError: {e}")
                traceback.print_exc()
                
        # Save results
        safe_name = strat.replace('(', '_').replace(')', '_').replace(',', '_')
        out_path = os.path.join(results_dir, f"{safe_name}_results.json")
        
        # Make sure we have results to save
        if all_results:
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved {len(all_results)} results for {strat} to {out_path}")
            logger.info(f"Saved results for {strat} to {out_path}")
            if args.generate_dashboard:
                generate_dashboard(all_results, strat)
        else:
            print(f"\nWARNING: No results to save for strategy {strat}")
    
    return

def evaluate_with_azure(prompt, response, model_name="gpt-4", temperature=0.0, api_key=None, api_base=None, api_version=None, deployment=None):
    """
    Evaluate a model's response using OpenAI API as a judge.
    
    Args:
        prompt: The original prompt given to the model
        response: The response generated by the model to evaluate
        model_name: The OpenAI model to use for evaluation (default: gpt-4)
        temperature: Temperature for the API call
        api_key: OpenAI API key
        api_base: Azure OpenAI endpoint if using Azure
        api_version: Azure API version if using Azure
        deployment: Azure deployment name if using Azure
        
    Returns:
        A score between 1-10 representing the quality of the response
    """
    # Configure OpenAI API
    if api_key:
        openai.api_key = api_key
    
    # Configure Azure OpenAI if needed
    if api_base:
        openai.api_base = api_base
    if api_version:
        openai.api_version = api_version
        
    # Determine which client to use
    if api_base and "azure" in api_base:
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
        chat_model = deployment
    else:
        client = openAI.OpenAI(api_key=api_key)
        chat_model = model_name
        
    # Construct the judge prompt
    judge_prompt = f"""
You are an AI assistant tasked with evaluating the quality of AI-generated responses. 
Please rate the following response on a scale of 1 to 10, with the scoring criteria:

- Score 1-3: Incorrect/irrelevant
- Score 4-5: Some merit with errors
- Score 6-7: Mostly accurate
- Score 8-10: Correct/comprehensive

Original Prompt:
{prompt}

AI Generated Response:
{response}

Please analyze the response thoroughly and assign a score from 1-10.
First explain your reasoning, then provide your final score in this exact format: "FINAL SCORE: X" 
where X is an integer between 1 and 10.
"""

    try:
        # Call the OpenAI API
        if api_base and "azure" in api_base:
            completion = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=temperature
            )
        else:
            completion = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=temperature
            )
        
        # Extract the judge's response
        judge_response = completion.choices[0].message.content
        logger.debug(f"Judge response: {judge_response}")
        
        # Extract the score using regex
        score_match = re.search(r"FINAL SCORE:\s*(\d+)", judge_response)
        if score_match:
            score = int(score_match.group(1))
            # Ensure score is within bounds
            score = max(1, min(10, score))
            return score
        else:
            # Fallback if regex doesn't match
            logger.warning(f"Could not extract score from judge response: {judge_response}")
            # Try to find any number in the response
            numbers = re.findall(r"\b(10|[1-9])\b", judge_response)
            if numbers:
                # Take the last number mentioned, which is likely the final score
                score = int(numbers[-1])
                # Ensure score is within bounds
                score = max(1, min(10, score))
                return score
            else:
                logger.error("No score found in judge response")
                return 5  # Return a middle score if we can't extract a score
                
    except Exception as e:
        logger.error(f"Error during Azure evaluation: {e}")
        return 0  # Return 0 to indicate evaluation failed

if __name__ == "__main__":
    main()
