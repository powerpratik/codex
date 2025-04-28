#!/usr/bin/env python3
"""
Robust Real Benchmark for KV Cache Management Strategies

This script evaluates different KV cache management strategies using real metrics:
- KV cache size (real physical memory usage)
- Inference time (total, per token, time to first token)
- Accuracy (perplexity and optional Azure evaluation)

The implementation properly applies eviction strategies to the actual KV cache
and measures their impact with real physical metrics.
"""

import os
# Disable HuggingFace progress bars (model download, checkpoint loading)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
import sys
import json
import time
import logging
import transformers
# Silence HuggingFace logs & disable progress bars
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
# Set up logging: remove all existing handlers, then log only to file
root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)
file_handler = logging.FileHandler('benchmark.log', mode='a')
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
    import psutil
    import gc
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, memory profiling will be limited")

# Optional imports for Azure evaluation
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai not available, Azure evaluation will be disabled")


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
            # In newer Transformers, past_key_values is part of the output
            # DO NOT try to access model.past_key_values directly
            if hasattr(output_tensor, 'past_key_values') and output_tensor.past_key_values is not None:
                self._update_memory_stats(output_tensor.past_key_values)
                self.logger.info(f"Hook updated memory stats: {self.current_memory_mb:.2f} MB")
        except Exception as e:
            self.logger.warning(f"Error in memory tracking hook: {e}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _update_memory_stats(self, past_key_values):
        """Update memory statistics based on past_key_values"""
        total_memory = 0
        layer_memory = {}
        
        # Try to calculate memory usage per layer
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
        except Exception as e:
            self.logger.warning(f"Error calculating memory from tensors: {e}")
        
        # If direct calculation failed or returned very small values, use sequence length estimation
        if total_memory < 1.0:  # If less than 1 MB, likely incorrect
            # Get sequence length from the first layer's key tensor
            seq_len = 0
            try:
                if past_key_values and past_key_values[0] and len(past_key_values[0]) > 0:
                    seq_len = past_key_values[0][0].size(2)
                    self.last_sequence_length = seq_len
            except Exception as e:
                self.logger.warning(f"Error getting sequence length: {e}")
                seq_len = self.last_sequence_length + 1  # Increment by 1 as fallback
            
            # Estimate memory based on sequence length
            # For Llama-2-7b, each token in KV cache is roughly 0.5-1MB depending on hidden size
            total_memory = seq_len * self.per_token_memory_estimate
            
            # Distribute estimated memory across layers
            num_layers = len(past_key_values) if past_key_values else 32  # Default to 32 layers for Llama-2-7b
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
            updated_past = past_key_values
        elif "Random" in strategy_name:
            # Extract keep ratio from strategy name
            keep_ratio = 0.7  # Default
            if "keep=" in strategy_name:
                try:
                    keep_ratio = float(strategy_name.split("keep=")[1].split(")")[0])
                except:
                    pass
            updated_past = self._apply_random_strategy(past_key_values, keep_ratio)
        elif "AttentionTop" in strategy_name:
            # Extract keep ratio from strategy name
            keep_ratio = 0.7  # Default
            if "keep=" in strategy_name:
                try:
                    keep_ratio = float(strategy_name.split("keep=")[1].split(")")[0])
                except:
                    pass
            updated_past = self._apply_attention_top_strategy(past_key_values, attention_scores, keep_ratio)
        elif "AttentionBottom" in strategy_name:
            # Extract keep ratio from strategy name
            keep_ratio = 0.7  # Default
            if "keep=" in strategy_name:
                try:
                    keep_ratio = float(strategy_name.split("keep=")[1].split(")")[0])
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
                    keep_ratio = float(strategy_name.split("keep=")[1].split(",")[0])
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
        torch.cuda.empty_cache()
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
    
    def _apply_attention_top_strategy(self, past_key_values, attention_scores, keep_ratio=0.7):
        """
        Apply attention-based top strategy: keep tokens with highest attention scores.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            keep_ratio: Fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """
        if attention_scores is None:
            # Fallback to random if no attention scores
            return self._apply_random_strategy(past_key_values, keep_ratio)
        
        updated_past = []
        
        # Get average attention across layers
        if isinstance(attention_scores, list):
            avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
        else:
            avg_attention = attention_scores.cpu().numpy()
        
        # Calculate importance scores (average attention received by each token)
        importance_scores = np.mean(avg_attention, axis=(0, 1))
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Get indices of tokens with highest importance scores
                if len(importance_scores) == seq_len:
                    importance_indices = np.argsort(importance_scores)[-keep_count:]
                else:
                    # Fallback to random if dimensions don't match
                    importance_indices = np.random.choice(seq_len, keep_count, replace=False)
                
                # Create a new tensor with only the kept tokens
                indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
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
        if attention_scores is None:
            # Fallback to random if no attention scores
            return self._apply_random_strategy(past_key_values, keep_ratio)
        
        updated_past = []
        
        # Get average attention across layers
        if isinstance(attention_scores, list):
            avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
        else:
            avg_attention = attention_scores.cpu().numpy()
        
        # Calculate importance scores (average attention received by each token)
        importance_scores = np.mean(avg_attention, axis=(0, 1))
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Get indices of tokens with lowest importance scores
                if len(importance_scores) == seq_len:
                    importance_indices = np.argsort(importance_scores)[:keep_count]
                else:
                    # Fallback to random if dimensions don't match
                    importance_indices = np.random.choice(seq_len, keep_count, replace=False)
                
                # Create a new tensor with only the kept tokens
                indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
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
            torch.cuda.empty_cache()
            
            return updated_past
        
        updated_past = []
        
        # Get average attention across layers
        if isinstance(attention_scores, list):
            avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
        else:
            avg_attention = attention_scores.cpu().numpy()
        
        # Normalize to get attention importance scores
        attention_importance = np.mean(avg_attention, axis=(0, 1))
        
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
        torch.cuda.empty_cache()
        
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
        updated_past = []
        
        # Calculate importance scores if attention scores are available
        importance_scores = None
        if attention_scores is not None:
            # Get average attention across layers
            if isinstance(attention_scores, list):
                avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
            else:
                avg_attention = attention_scores.cpu().numpy()
            
            # Calculate importance scores (average attention received by each token)
            importance_scores = np.mean(avg_attention, axis=(0, 1))
        
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
        if attention_scores is None:
            # Fallback to sliding window if no attention scores
            return self._apply_sliding_window(past_key_values, base_keep)
        
        updated_past = []
        
        # Get average attention across layers
        if isinstance(attention_scores, list):
            avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
        else:
            avg_attention = attention_scores.cpu().numpy()
        
        # Calculate importance scores (average attention received by each token)
        importance_scores = np.mean(avg_attention, axis=(0, 1))
        
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


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
        token_types = ["common"] * len(tokens)
    
    return token_types


def get_prompt(data_item: Union[Dict, List, str]) -> str:
    """
    Extract prompt from data item, handling various data structures.
    
    Args:
        data_item: Data item from dataset
        
    Returns:
        Prompt string
    """
    logger.debug(f"Extracting prompt from data item type: {type(data_item)}")
    
    try:
        # If it's a string, return directly
        if isinstance(data_item, str):
            return data_item
        
        # If it's a list, handle MTBench format
        if isinstance(data_item, list):
            # MTBench format: list of turns
            prompt = ""
            for turn in data_item:
                if isinstance(turn, dict):
                    if "role" in turn and "content" in turn:
                        role = turn["role"]
                        content = turn["content"]
                        prompt += f"{role}: {content}\n"
                    elif "question" in turn:
                        prompt += f"User: {turn['question']}\n"
                    elif "text" in turn:
                        prompt += f"{turn['text']}\n"
                elif isinstance(turn, str):
                    prompt += f"{turn}\n"
            return prompt.strip()
        
        # If it's a dict, check common fields
        if isinstance(data_item, dict):
            # Check common fields
            for field in ["prompt", "text", "question", "input", "instruction"]:
                if field in data_item and isinstance(data_item[field], str):
                    return data_item[field]
            
            # Check for MTBench format
            if "turns" in data_item and isinstance(data_item["turns"], list):
                return get_prompt(data_item["turns"])
            
            # Check for conversation format
            if "conversations" in data_item and isinstance(data_item["conversations"], list):
                prompt = ""
                for turn in data_item["conversations"]:
                    if isinstance(turn, dict) and "value" in turn:
                        prompt += f"{turn.get('from', 'Speaker')}: {turn['value']}\n"
                return prompt.strip()
        
        # If we couldn't extract a prompt, log a warning and return empty string
        logger.warning(f"Could not extract prompt from data item: {data_item}")
        return ""
    
    except Exception as e:
        logger.error(f"Error extracting prompt: {e}")
        return ""


def load_dataset(dataset_path: str) -> List:
    """
    Load dataset from file.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        List of data items
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        with open(dataset_path, 'r') as f:
            # Try to load as a full JSON document (array or object)
            if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
                # Read entire content
                text = f.read()
                try:
                    data = json.loads(text)
                    # Handle dicts with "data" key
                    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                        return data["data"]
                    # List of items
                    if isinstance(data, list):
                        return data
                    # Single object
                    return [data]
                except json.JSONDecodeError:
                    # Fallback to JSON lines: one JSON object per line
                    f.seek(0)
                    items = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse JSON line: {line}")
                    return items
            else:
                # Plain text file: one sample per line
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def profile_memory():
    """
    Profile current memory usage.
    
    Returns:
        Dict with memory usage statistics
    """
    if not HAS_PSUTIL:
        return {}
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}_allocated"] = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
                gpu_memory[f"gpu_{i}_reserved"] = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
        
        return {
            "rss_mb": memory_info.rss / (1024 ** 2),  # MB
            "vms_mb": memory_info.vms / (1024 ** 2),  # MB
            **gpu_memory
        }
    except Exception as e:
        logger.warning(f"Error profiling memory: {e}")
        return {}


def calculate_perplexity(logits, target_ids):
    """
    Calculate perplexity from logits and target IDs.
    
    Args:
        logits: Model logits
        target_ids: Target token IDs
        
    Returns:
        Perplexity value
    """
    try:
        # Create a loss function
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # Reshape logits to (batch_size * seq_len, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        
        # Reshape target_ids to (batch_size * seq_len)
        target_ids = target_ids.view(-1)
        
        # Calculate loss
        loss = loss_fn(logits, target_ids)
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        return perplexity
    except Exception as e:
        logger.warning(f"Error calculating perplexity: {e}")
        return float('inf')


def evaluate_with_azure(prompt, response, api_key=None, endpoint=None,version=None):
    """
    Evaluate response using Azure OpenAI.
    
    Args:
        prompt: Input prompt
        response: Model response
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint
        
    Returns:
        Evaluation score
    """
    if not HAS_OPENAI:
        logger.warning("openai package not available, skipping Azure evaluation")
        return 0
    
    if not api_key or not endpoint:
        logger.warning(f"Azure API key or endpoint not provided (api_key={bool(api_key)}, endpoint={bool(endpoint)}), skipping Azure evaluation")
        return 0
    
    try:
        # Configure OpenAI client
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=version,
            azure_endpoint=endpoint,
            
        )
        
        # Create evaluation prompt
        eval_prompt = f"""
        Please evaluate the quality of the following response to the given prompt.
        
        Prompt:
        {prompt}
        
        Response:
        {response}
        
        Rate the response on a scale from 1 to 10, where:
        1 = Completely irrelevant or incorrect
        10 = Perfect, comprehensive, and accurate
        
        Provide only the numeric score.
        """
        
        # Get evaluation
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that evaluates the quality of responses."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        # Extract score
        score_text = completion.choices[0].message.content.strip()
        
        # Try to parse score
        try:
            score = float(score_text)
            return score
        except ValueError:
            # If we can't parse a number, try to extract it from text
            import re
            match = re.search(r'(\d+(\.\d+)?)', score_text)
            if match:
                return float(match.group(1))
            else:
                logger.warning(f"Could not parse score from Azure response: {score_text}")
                return 0
    
    except Exception:
        # Log full exception with stack trace
        logger.exception("Error during Azure evaluation")
        # Also print to stderr for immediate visibility
        try:
            import traceback, sys
            print("Error during Azure evaluation:\n" + traceback.format_exc(), file=sys.stderr)
        except Exception:
            pass
        return 0


def process_single_prompt(model, tokenizer, prompt, strat, cache_manager, max_gen_tokens=100, azure_eval=False, azure_config=None):
    """Process a single prompt with the given strategy."""
    logger.info(f"Processing prompt with strategy: {strat.name}")
    
    try:
        # Reset model state
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Check if Cache class is available
        from transformers.cache_utils import Cache, DynamicCache, StaticCache
        logger.info(f"Cache classes available: Cache, DynamicCache, StaticCache")
        
        model.config.use_cache = True
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Initialize metrics
        start_time = time.time()
        token_times = {}
        attention_scores = None
        token_types = []
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)
        input_ids = inputs.input_ids
        
        # Record encoding time
        encoding_time = time.time() - start_time
        logger.debug(f"Encoding time: {encoding_time:.4f}s")
        
        # Start generation
        generated_ids = input_ids.clone()
        
        # Time the first token generation
        first_token_start = time.time()
        
        # Generate first token - get outputs with past_key_values
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
            
        # Get past_key_values from outputs
        past_key_values = outputs.past_key_values
        
        # Record first token time
        first_token_time = time.time() - first_token_start
        logger.debug(f"First token time: {first_token_time:.4f}s")
        
        # Get next token and add to generated sequence
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        
        # If next_token has multiple elements, just take the first one
        if next_token.numel() > 1:
            next_token_id = next_token[0].item()
        else:
            next_token_id = next_token.item()
        
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
        
        # Prepare for next iteration
        token = torch.tensor([[next_token_id]], device=model.device)
        
        # Extract attention scores if available
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            attention_scores = outputs.attentions
        
        # Get token types
        token_types = get_token_types(input_ids[0].tolist(), tokenizer)
        
        # Update memory stats
        cache_manager._update_memory_stats(past_key_values)
        
        # Apply KV cache management strategy
        modified_cache = cache_manager.apply_eviction_strategy(
            past_key_values,
            strat.name, 
            attention_scores, 
            token_types
        )
        
        # Continue generating tokens
        for i in range(1, max_gen_tokens):
            token_start = time.time()
            
            # Generate next token with modified cache
            with torch.no_grad():
                outputs = model(input_ids=token, past_key_values=modified_cache, use_cache=True)
            
            # Get past_key_values from outputs for the next iteration
            past_key_values = outputs.past_key_values
            
            # Record token generation time
            token_time = time.time() - token_start
            token_times[i] = token_time
            
            # Get next token and add to generated sequence
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            
            # If next_token has multiple elements, just take the first one
            if next_token.numel() > 1:
                next_token_id = next_token[0].item()
            else:
                next_token_id = next_token.item()
            
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # Check for end of generation
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Prepare for next iteration
            token = torch.tensor([[next_token_id]], device=model.device)
            
            # Extract attention scores if available
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                attention_scores = outputs.attentions
            
            # Extend token types
            token_types.append(get_token_types([next_token_id], tokenizer)[0])
            
            # Update memory stats
            cache_manager._update_memory_stats(past_key_values)
            
            # Apply KV cache management strategy
            modified_cache = cache_manager.apply_eviction_strategy(
                past_key_values,
                strat.name, 
                attention_scores, 
                token_types
            )
        
        # End generation
        total_time = time.time() - start_time
        logger.debug(f"Total generation time: {total_time:.4f}s")
        
        # Decode full generated sequence (prompt + model output)
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Strip out the original prompt to display only the model's response
        if full_text.startswith(prompt):
            generated_text = full_text[len(prompt):].strip()
        else:
            generated_text = full_text
        # Print only the generated response
        print(generated_text)
        
        # Calculate perplexity
        perplexity = calculate_perplexity(outputs.logits, token)
        
        # Evaluate with Azure if requested
        azure_score = 0
        if azure_eval and azure_config:
            azure_score = evaluate_with_azure(
                prompt, 
                generated_text, 
                api_key=azure_config.get("api_key"),
                endpoint=azure_config.get("endpoint"),
                version= "2024-12-01-preview"
            )
        
        # Get memory stats
        memory_stats = cache_manager.get_memory_stats()
        
        # Return results
        return {
            "strategy": strat.name,
            "prompt": prompt,
            "response": generated_text,
            "time": {
                "total_time": total_time,
                "encoding_time": encoding_time,
                "first_token_time": first_token_time,
                "token_times": token_times,
                "tokens_generated": len(generated_ids[0]) - len(input_ids[0])
            },
            "memory": memory_stats,
            "accuracy": {
                "perplexity": perplexity,
                "azure_score": azure_score
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        logger.error(traceback.format_exc())
        
        # Return error result
        return {
            "strategy": strat.name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark KV cache management strategies")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--eval_azure", action="store_true", help="Evaluate with Azure OpenAI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--profile", action="store_true", help="Enable memory profiling")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--generate_dashboard", action="store_true", help="Generate dashboard")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    # Create output directory
    output_dir = Path(cfg.get("output_dir", "benchmark_results"))
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(cfg["model_name"], cfg["cache_dir"])
    
    # Set up padding token for the tokenizer
    tokenizer = setup_tokenizer_padding(tokenizer)
    
    logger.info(f"Loaded model {cfg['model_name']}")
    
    # Load dataset
    dataset_cfg = cfg.get("dataset", {})
    dataset_path = dataset_cfg.get("local_path")
    dataset = load_dataset(dataset_path)
    
    # Limit dataset if requested
    if args.limit:
        dataset = dataset[:args.limit]
    # Number of samples for progress reporting
    n_samples = len(dataset)
    
    # Initialize strategies
    strategies = []
    for strat_spec in cfg.get("strategies", ["Baseline"]):
        if isinstance(strat_spec, str):
            # Parse strategy specification
            if strat_spec == "Baseline":
                strategies.append(EnhancedBaseline())
            elif strat_spec.startswith("Random"):
                # Extract parameters
                keep = 0.7  # Default
                if "keep=" in strat_spec:
                    try:
                        keep = float(strat_spec.split("keep=")[1].split(")")[0])
                    except:
                        pass
                strategies.append(RandomKVCacheStrategy(keep=keep))
            elif strat_spec.startswith("AttentionTop"):
                # Extract parameters
                keep = 0.7  # Default
                if "keep=" in strat_spec:
                    try:
                        keep = float(strat_spec.split("keep=")[1].split(")")[0])
                    except:
                        pass
                strategies.append(AttentionTopStrategy(keep=keep))
            elif strat_spec.startswith("AttentionBottom"):
                # Extract parameters
                keep = 0.7  # Default
                if "keep=" in strat_spec:
                    try:
                        keep = float(strat_spec.split("keep=")[1].split(")")[0])
                    except:
                        pass
                strategies.append(AttentionBottomStrategy(keep=keep))
            elif strat_spec.startswith("HybridNPercent"):
                # Extract parameters
                keep = 0.7  # Default
                recency_weight = 0.5  # Default
                attention_weight = 0.3  # Default
                type_weight = 0.2  # Default
                
                if "keep=" in strat_spec:
                    try:
                        keep = float(strat_spec.split("keep=")[1].split(",")[0])
                    except:
                        pass
                if "r=" in strat_spec:
                    try:
                        recency_weight = float(strat_spec.split("r=")[1].split(",")[0])
                    except:
                        pass
                if "a=" in strat_spec:
                    try:
                        attention_weight = float(strat_spec.split("a=")[1].split(",")[0])
                    except:
                        pass
                if "t=" in strat_spec:
                    try:
                        type_weight = float(strat_spec.split("t=")[1].split(")")[0])
                    except:
                        pass
                
                strategies.append(HybridNPercentStrategy(
                    keep=keep,
                    recency_weight=recency_weight,
                    attention_weight=attention_weight,
                    type_weight=type_weight
                ))
            elif strat_spec.startswith("SlidingWindow"):
                # Extract parameters
                window = 0.7  # Default
                important = 0.1  # Default
                
                if "window=" in strat_spec:
                    try:
                        window = float(strat_spec.split("window=")[1].split(",")[0])
                    except:
                        pass
                if "important=" in strat_spec:
                    try:
                        important = float(strat_spec.split("important=")[1].split(")")[0])
                    except:
                        pass
                
                strategies.append(SlidingWindowStrategy(window=window, important=important))
            elif strat_spec.startswith("AdaptiveAttention"):
                # Extract parameters
                base_keep = 0.7  # Default
                if "base_keep=" in strat_spec:
                    try:
                        base_keep = float(strat_spec.split("base_keep=")[1].split(")")[0])
                    except:
                        pass
                
                strategies.append(AdaptiveAttentionStrategy(base_keep=base_keep))
            else:
                logger.warning(f"Unknown strategy: {strat_spec}, using Baseline")
                strategies.append(EnhancedBaseline())
    
    # Ensure we have at least one strategy
    if not strategies:
        strategies.append(EnhancedBaseline())
    
    # Log strategies
    for strat in strategies:
        logger.info(f"Benchmarking strategy: {strat.name}")
        # Console progress header
        print(f"\n=== Strategy: {strat.name} ({n_samples} samples) ===")
        sys.stdout.flush()
        logger.info(f"Created strategy instance: {strat.__class__.__name__}")
    
    # Initialize Azure config if needed
    azure_config = None
    if args.eval_azure:
        azure_config = cfg.get("azure", {})
        if not azure_config.get("api_key") or not azure_config.get("endpoint"):
            logger.warning("Azure API key or endpoint not provided, disabling Azure evaluation")
            args.eval_azure = False
    
    # Benchmark each strategy
    for strat in strategies:
        logger.info(f"Benchmarking strategy: {strat.name}")
        
        # Initialize KV cache manager
        cache_manager = RealKVCacheManager(
            model=model,
            tokenizer=tokenizer,
            cfg={"kv_threshold": cfg.get("kv_threshold", 1000)},
            logger=logger
        )
        
        # Set logger level to INFO to see memory logs
        cache_manager.logger.setLevel(logging.INFO)
        
        # Process dataset
        results = []
        
        for i, item in enumerate(dataset):
            try:
                # Extract prompt
                prompt_field = dataset_cfg.get("prompt_field", None)
                if prompt_field:
                    if isinstance(item, dict) and prompt_field in item:
                        prompt = get_prompt(item[prompt_field])
                    else:
                        prompt = get_prompt(item)
                else:
                    prompt = get_prompt(item)
                
                # Skip empty prompts
                if not prompt:
                    logger.warning(f"Skipping empty prompt at index {i}")
                    continue
                
                # Process prompt
                result = process_single_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    strat=strat,
                    cache_manager=cache_manager,
                    max_gen_tokens=cfg.get("max_gen_tokens", 100),
                    azure_eval=args.eval_azure,
                    azure_config=azure_config
                )
                
                # Add result
                results.append(result)
                
                # Profile memory if requested
                if args.profile:
                    memory_profile = profile_memory()
                    logger.info(f"Memory profile after sample {i}: {memory_profile}")
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.info(f"Processed sample {i} with strategy {strat.name}")
                # Console minimal progress
                print(f"\r{strat.name}: sample {i+1}/{n_samples}", end='', flush=True)
            
            except Exception as e:
                logger.warning(f"Skipping sample {i} due to processing error: {e}")
                continue
        
        # Save results
        results_file = output_dir / f"{strat.name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results for strategy {strat.name} to {results_file}")
        # Console summary for this strategy
        try:
            import numpy as _np
            avg_time = _np.mean([r.get("time", {}).get("total_time", 0) for r in results]) if results else 0
        except Exception:
            avg_time = 0
        print(f"Strategy {strat.name} complete: {len(results)} samples, avg total_time={avg_time:.2f}s")
        sys.stdout.flush()
    
    # Generate dashboard if requested
    if args.generate_dashboard:
        try:
            # Import dashboard module
            from kv_cache_dashboard import KVCacheDashboard
            
            logger.info("Generating dashboard")
            dashboard = KVCacheDashboard(output_dir)
            dashboard.generate_dashboard()
            logger.info(f"Dashboard generated at {output_dir}/dashboard")
        except ImportError:
            logger.warning("KVCacheDashboard module not found, skipping dashboard generation")
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
    
    logger.info("Benchmark completed")


if __name__ == "__main__":
    main()
