#!/usr/bin/env python3
"""
Stateful Multi-Turn Benchmark for KV Cache Management Strategies (v2)

This script implements stateful multi-turn conversation processing, where the
KV cache (past_key_values) is maintained and incrementally grown across turns
of the same question. It resets only for new questions.
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
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm

# Ensure transformers and necessary dependencies are installed
try:
    import transformers
    from transformers.utils import logging as hf_logging
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.cache_utils import DynamicCache # For type checking
except ImportError:
    print("Please install/update transformers, torch, accelerate: pip install -U transformers torch accelerate")
    exit(1)

try:
    import openai # For Azure evaluation
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Silence HuggingFace logs
hf_logging.set_verbosity_error()

# --- Logging Setup ---
log_filename = f'stateful_benchmark.log'
root_logger = logging.getLogger()
for h in list(root_logger.handlers): root_logger.removeHandler(h) # Clear existing handlers
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'))
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger('stateful_benchmark')
logger.propagate = True

# --- Optional Imports & System Checks ---
HAS_GC = 'gc' in globals() or 'gc' in locals()
HAS_PSUTIL = 'psutil' in globals() or 'psutil' in locals()
if not HAS_PSUTIL: logger.warning("psutil not available, system memory profiling will be limited")
if not HAS_OPENAI: logger.warning("openai library not available, Azure evaluation will be disabled")


# --- Memory Measurement Functions ---
def measure_gpu_memory():
    if not torch.cuda.is_available(): return {"error": "CUDA not available"}
    try:
        # Reset peak stats for the current interval of measurement
        if hasattr(torch.cuda, 'reset_peak_memory_stats'): # Available in newer PyTorch
            for i in range(torch.cuda.device_count()):
                 torch.cuda.reset_peak_memory_stats(i)
        
        allocated = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024 * 1024)
        reserved = sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())) / (1024 * 1024)
        # Peak might be more reliable if reset per measurement interval, otherwise it's overall peak
        peak_allocated = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024 * 1024)
        peak_reserved = sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) / (1024 * 1024)

        return {
            "total_allocated_mb": allocated, "total_reserved_mb": reserved,
            "peak_allocated_mb_interval": peak_allocated, "peak_reserved_mb_interval": peak_reserved,
        }
    except Exception as e:
        logger.error(f"Error measuring GPU memory: {e}")
        return {"error": str(e)}

def measure_system_memory():
    if not HAS_PSUTIL: return {"error": "psutil not available"}
    try:
        mem = psutil.virtual_memory()
        return {"total_gb": mem.total / (1024**3), "available_gb": mem.available / (1024**3), "used_gb": mem.used / (1024**3), "percent_used": mem.percent}
    except Exception as e:
        logger.error(f"Error measuring system memory: {e}")
        return {"error": str(e)}

def measure_kv_cache_size(past_key_values, model_config=None):
    logger.debug(f"model_config is None: {model_config is None}. past_key_values type: {type(past_key_values)}")
    if model_config: logger.debug(f"model_config type: {type(model_config)}")

    total_size_bytes = 0
    if past_key_values is None: return 0.0

    cache_type_measured = "Unknown"
    try:
        if isinstance(past_key_values, DynamicCache):
            cache_type_measured = "DynamicCache"
            if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache') and \
               isinstance(past_key_values.key_cache, list) and isinstance(past_key_values.value_cache, list):
                cache_type_measured += " (.key_cache/.value_cache lists)"
                for k_tensor in past_key_values.key_cache:
                    if isinstance(k_tensor, torch.Tensor): total_size_bytes += k_tensor.nelement() * k_tensor.element_size()
                for v_tensor in past_key_values.value_cache:
                    if isinstance(v_tensor, torch.Tensor): total_size_bytes += v_tensor.nelement() * v_tensor.element_size()
                if total_size_bytes > 0: logger.debug(f"SUCCESS: Measured via {cache_type_measured}: {total_size_bytes / (1024*1024):.2f} MB")
                else: logger.warning(f"{cache_type_measured} access but total_size_bytes is 0.")
            else:
                logger.warning(f"DynamicCache object does not have expected .key_cache/.value_cache list attributes. Dir: {dir(past_key_values)}")


        elif isinstance(past_key_values, tuple) and \
             past_key_values and all(isinstance(layer_kv, tuple) and len(layer_kv) >= 2 for layer_kv in past_key_values):
            cache_type_measured = "TupleCache"
            logger.debug(f"Attempting measurement via {cache_type_measured}")
            for layer_past in past_key_values:
                for tensor in layer_past[:2]:
                    if isinstance(tensor, torch.Tensor): total_size_bytes += tensor.nelement() * tensor.element_size()
            if total_size_bytes > 0: logger.debug(f"SUCCESS: Measured {cache_type_measured}: {total_size_bytes / (1024*1024):.2f} MB")

        if total_size_bytes == 0: # Fallback if direct methods yield 0
            logger.warning(f"KV Cache direct measurement (last attempted: {cache_type_measured}) yielded 0 bytes. Type: {type(past_key_values)}. Attempting fallback.")
            current_seq_len = 0
            if hasattr(past_key_values, 'get_seq_length') and callable(past_key_values.get_seq_length):
                current_seq_len = past_key_values.get_seq_length()
            elif hasattr(past_key_values, 'seen_tokens') and isinstance(past_key_values.seen_tokens, int):
                 current_seq_len = past_key_values.seen_tokens

            if current_seq_len > 0 and model_config is not None:
                num_layers = getattr(model_config, 'num_hidden_layers', 32)
                hidden_size = getattr(model_config, 'hidden_size', 4096)
                num_q_heads = getattr(model_config, 'num_attention_heads', 32) # Query heads
                num_kv_heads = getattr(model_config, 'num_key_value_heads', num_q_heads) # KV heads (for GQA/MQA)
                head_dim = hidden_size // num_q_heads if num_q_heads > 0 else 0

                if head_dim > 0:
                    bytes_per_element = 2 # Assuming float16
                    bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * bytes_per_element
                    total_size_bytes = current_seq_len * num_layers * bytes_per_token_per_layer
                    logger.warning(f"KV Cache: Using fallback estimation. SeqLen={current_seq_len}, Est. Size: {total_size_bytes / (1024*1024):.2f} MB")
                else: logger.error("Head dim is 0, cannot use fallback estimation.")
            elif total_size_bytes == 0:
                 logger.error(f"KV Cache measurement failed. Fallback conditions not met (seq_len={current_seq_len}, model_config_is_None={model_config is None}).")
        
        return total_size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Exception in measure_kv_cache_size: {e}\n{traceback.format_exc()}")
        return 0.0

# --- KV Cache Manager Class ---
class RealKVCacheManager:
    def __init__(self, model, tokenizer, cfg, logger_instance):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or {}
        self.threshold_mb = self.cfg.get("kv_threshold_mb", 2048) # Default 2GB threshold
        self.logger = logger_instance
        self.eviction_count_total = 0
        self.total_eviction_time_total = 0
        self.evictions_this_turn = 0
        self.eviction_time_this_turn = 0

    def reset_turn_stats(self): self.evictions_this_turn = 0; self.eviction_time_this_turn = 0
    def get_turn_stats(self): return {"evictions_this_turn": self.evictions_this_turn, "avg_eviction_time_this_turn": self.eviction_time_this_turn / max(1, self.evictions_this_turn)}

    def apply_eviction_strategy_if_needed(self, past_key_values, strat_cfg: Dict, model_config, attention_scores=None):
        if not past_key_values: return past_key_values
        current_memory_mb = measure_kv_cache_size(past_key_values, model_config)
        
        # Ensure strategy name is present
        strategy_name = strat_cfg.get("name")
        if not strategy_name:
            self.logger.error("Strategy name missing in strat_cfg. Skipping eviction.")
            return past_key_values

        self.logger.debug(f"Eviction check: Cache size {current_memory_mb:.2f}MB, Threshold {self.threshold_mb:.2f}MB for strategy {strategy_name}")

        if current_memory_mb < self.threshold_mb:
            return past_key_values
        
        self.logger.info(f"Threshold triggered for {strategy_name}. Current: {current_memory_mb:.2f}MB, Threshold: {self.threshold_mb:.2f}MB.")
        eviction_start_time = time.time()
        updated_past = past_key_values

        if strategy_name == "Baseline":
            self.logger.debug("Baseline strategy: no eviction performed.")
        elif strategy_name == "Random":
            keep_ratio = float(strat_cfg.get("keep_ratio", 0.7))
            self.logger.info(f"Applying Random strategy with keep_ratio: {keep_ratio}")
            updated_past = self._apply_random_strategy(past_key_values, keep_ratio)
        elif strategy_name == "AttentionTop":
            keep_ratio = float(strat_cfg.get("keep_ratio", 0.7))
            if attention_scores:
                self.logger.info(f"Applying AttentionTop strategy with keep_ratio: {keep_ratio}")
                updated_past = self._apply_attention_top_strategy(past_key_values, attention_scores, keep_ratio)
            else:
                self.logger.warning("AttentionTop strategy selected, but no attention_scores provided. Skipping eviction.")
        elif strategy_name == "AttentionBottom":
                keep_ratio = float(strat_cfg.get("keep_ratio", 0.7))
                if attention_scores:
                    self.logger.info(f"Applying AttentionBottom strategy with keep_ratio: {keep_ratio}")
                    updated_past = self._apply_attention_bottom_strategy(past_key_values, attention_scores, keep_ratio)
                else:
                    self.logger.warning("AttentionBottom: No attention_scores provided. Falling back to random.")
                    updated_past = self._apply_random_strategy(past_key_values, keep_ratio)
            
        
        elif strategy_name == "SlidingWindowSimple":
            keep_ratio = float(strat_cfg.get("keep_ratio", 0.7))
            self.logger.info(f"Applying SlidingWindowSimple strategy with keep_ratio: {keep_ratio}")
            updated_past = self._apply_sliding_window_simple_strategy(past_key_values, keep_ratio)

        elif strategy_name == "SlidingWindowGist":
            gist_token_count = int(strat_cfg.get("gist_token_count", 512)) # Default to 512
            recent_token_count = int(strat_cfg.get("recent_token_count", 2048)) # Default to 2048
            self.logger.info(f"Applying SlidingWindowGist strategy with gist_tokens={gist_token_count}, recent_tokens={recent_token_count}")
            updated_past = self._apply_sliding_window_gist_strategy(past_key_values, gist_token_count, recent_token_count)
            
        elif strategy_name == "SlidingWindowGistAttentive": # New name for this complex one
                keep_ratio = float(strat_cfg.get("keep_ratio", 0.9)) # Overall keep ratio
                gist_token_count = int(strat_cfg.get("gist_token_count", 512))
                recent_token_count = int(strat_cfg.get("recent_token_count", 2048))
                self.logger.info(f"Applying SlidingWindowAttentiveGist with overall_keep_ratio={keep_ratio}, gist={gist_token_count}, recent={recent_token_count}")
                updated_past = self._apply_sliding_window_attentive_gist_strategy(
                    past_key_values, attention_scores, model_config, # Pass model_config
                    keep_ratio, gist_token_count, recent_token_count
                )
        elif strategy_name == "EvictOldest":
                discard_prefix_ratio = float(strat_cfg.get("discard_prefix_ratio", 0.25)) # Default to discarding 25% oldest
                self.logger.info(f"Applying EvictOldest strategy with discard_prefix_ratio: {discard_prefix_ratio}")
                # This strategy doesn't use attention_scores
                updated_past = self._apply_evict_oldest_strategy(past_key_values, discard_prefix_ratio)
            
        else:
            self.logger.warning(f"Unknown or unsupported eviction strategy '{strategy_name}'. Applying Baseline.")
            updated_past = past_key_values

        eviction_time = time.time() - eviction_start_time
        if updated_past is not past_key_values or current_memory_mb >= self.threshold_mb : # if something was intended to happen
            self.evictions_this_turn += 1
            self.eviction_time_this_turn += eviction_time
            self.eviction_count_total += 1
            self.total_eviction_time_total += eviction_time
            new_size = measure_kv_cache_size(updated_past, model_config)
            self.logger.info(f"Eviction strategy {strategy_name} applied in {eviction_time:.4f}s. Original size: {current_memory_mb:.2f}MB, New size: {new_size:.2f}MB.")
        return updated_past

    def _apply_random_strategy(self, past_key_values: DynamicCache, keep_ratio: float) -> DynamicCache:
        self.logger.info(f"Applying Random strategy with keep_ratio: {keep_ratio}") # Added INFO level
        if not (isinstance(past_key_values, DynamicCache) and 
                hasattr(past_key_values, 'key_cache') and past_key_values.key_cache and 
                hasattr(past_key_values, 'value_cache') and past_key_values.value_cache and
                len(past_key_values.key_cache) > 0 and past_key_values.key_cache[0] is not None): # More robust check
            self.logger.warning("Random Eviction: KV cache is empty, not DynamicCache, or key_cache[0] is None. Skipping.")
            return past_key_values
        
        seq_len = past_key_values.get_seq_length()
        if seq_len <= 1:
            self.logger.debug(f"Random Eviction: seq_len ({seq_len}) too short. Skipping.")
            return past_key_values
        
        keep_count = max(1, int(seq_len * keep_ratio))
        if keep_count >= seq_len:
            self.logger.debug(f"Random Eviction: keep_count ({keep_count}) >= seq_len ({seq_len}). No eviction needed.")
            return past_key_values

        initial_device_for_indices = past_key_values.key_cache[0].device 
        indices_to_keep_template = torch.randperm(seq_len, device=initial_device_for_indices)[:keep_count]
        indices_to_keep_template, _ = torch.sort(indices_to_keep_template)
        
        self.logger.debug(f"Random Eviction: Original seq_len={seq_len}, Keeping {keep_count} tokens.")

        new_key_cache_list, new_value_cache_list = [], []
        for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
            current_tensor_device = k_tensor.device
            indices_on_correct_device = indices_to_keep_template.to(current_tensor_device)
            
            new_key_cache_list.append(k_tensor.index_select(2, indices_on_correct_device))
            new_value_cache_list.append(v_tensor.index_select(2, indices_on_correct_device))
        
        # Update the existing cache object by replacing its internal tensor lists
        return self._create_new_cache_with_updated_tensors(past_key_values, new_key_cache_list, new_value_cache_list)


    # Inside RealKVCacheManager class

    # Inside RealKVCacheManager class

    def _process_attention_scores(self, attention_scores_tuple: Tuple[torch.Tensor, ...], seq_len: int, aggregation_device: torch.device) -> Optional[torch.Tensor]:
        """
        Helper to process attention scores from model output.
        Handles both prefill (query_len > 1) and decoding (query_len = 1) attentions.
        Returns a 1D tensor of importance scores of shape [seq_len].
        """
        avg_attn_scores_per_key = None
        num_valid_layers = 0
        
        if not attention_scores_tuple : # Check if the tuple itself is None or empty
            self.logger.warning("Attention Processing: attention_scores_tuple is None or empty. Cannot compute importance.")
            return None

        self.logger.debug(f"Attention Processing: Received {len(attention_scores_tuple)} layer attentions. Cache seq_len: {seq_len}.")

        for layer_idx, layer_attn in enumerate(attention_scores_tuple):
            if not isinstance(layer_attn, torch.Tensor): # Check if item is a tensor
                self.logger.warning(f"Attention Processing: Layer {layer_idx} data is not a tensor ({type(layer_attn)}). Skipping.")
                continue
            
            # Expected shape: [batch_size=1, num_heads, query_length, key_length]
            if layer_attn.ndim != 4 or layer_attn.shape[0] != 1: # batch_size should be 1
                self.logger.warning(f"Attention Processing: Layer {layer_idx} attention tensor has unexpected ndim or batch_size: {layer_attn.shape}. Skipping.")
                continue

            # Move to aggregation device before processing
            layer_attn_on_target = layer_attn.to(aggregation_device)
            
            # Squeeze the batch_size dimension. Shape becomes: [num_heads, query_length, key_length]
            squeezed_attn = layer_attn_on_target.squeeze(0)

            # The key_length of the attention tensor (dim 2) must match the current sequence length of the KV cache
            if squeezed_attn.shape[2] != seq_len:
                self.logger.warning(f"Attention Processing: Layer {layer_idx} attention key_length {squeezed_attn.shape[2]} != cache seq_len {seq_len}. Attn shape: {layer_attn.shape}. Skipping layer.")
                continue

            # To get importance for each key_position:
            # Sum the attention it received from all query_positions, then average over heads.
            # squeezed_attn shape: [num_heads, query_length, key_length]
            # .sum(dim=1) sums over the query_length dimension. Result: [num_heads, key_length]
            # .mean(dim=0) averages over the num_heads dimension. Result: [key_length] (which is seq_len)
            current_layer_scores = squeezed_attn.sum(dim=1).mean(dim=0) 
            
            if current_layer_scores.shape[0] == seq_len: # Final sanity check
                if avg_attn_scores_per_key is None:
                    avg_attn_scores_per_key = current_layer_scores.clone().detach()
                else:
                    avg_attn_scores_per_key += current_layer_scores.clone().detach()
                num_valid_layers += 1
            else:
                # This should ideally not be reached if the previous checks are correct
                self.logger.error(f"Attention Processing: Mismatch after processing layer {layer_idx}. Scores shape: {current_layer_scores.shape}, expected seq_len: {seq_len}")
                
        if avg_attn_scores_per_key is not None and num_valid_layers > 0:
            self.logger.info(f"Attention Processing: Successfully computed and averaged importance scores from {num_valid_layers} valid layers.") # Changed to INFO for success
            return avg_attn_scores_per_key / num_valid_layers 
        
        self.logger.warning(f"Attention Processing: No valid layers found or scores could not be computed (num_valid_layers: {num_valid_layers}). This can happen if attention shapes consistently mismatch cache seq_len.")
        return None
    def _apply_evict_oldest_strategy(self, past_key_values: DynamicCache, discard_prefix_ratio: float) -> DynamicCache:
        self.logger.info(f"Attempting EvictOldest strategy, discarding oldest {discard_prefix_ratio*100:.0f}% of tokens.")
        
        if not (isinstance(past_key_values, DynamicCache) and 
                hasattr(past_key_values, 'key_cache') and past_key_values.key_cache and 
                hasattr(past_key_values, 'value_cache') and past_key_values.value_cache and
                len(past_key_values.key_cache) > 0 and past_key_values.key_cache[0] is not None):
            self.logger.warning("EvictOldest: KV cache is empty or not suitable. Skipping.")
            return past_key_values

        seq_len = past_key_values.get_seq_length()
        if seq_len <= 1:
            self.logger.debug(f"EvictOldest: seq_len ({seq_len}) too short. Skipping.")
            return past_key_values
        
        if not (0.0 < discard_prefix_ratio < 1.0):
            self.logger.error(f"EvictOldest: discard_prefix_ratio ({discard_prefix_ratio}) must be between 0 and 1 (exclusive). Skipping eviction.")
            return past_key_values

        num_tokens_to_discard = int(seq_len * discard_prefix_ratio)
        
        if num_tokens_to_discard <= 0:
            self.logger.debug(f"EvictOldest: Calculated num_tokens_to_discard ({num_tokens_to_discard}) is not positive. No eviction needed.")
            return past_key_values
        
        if num_tokens_to_discard >= seq_len: # Safety to ensure we keep at least one token
            self.logger.warning(f"EvictOldest: discard_prefix_ratio ({discard_prefix_ratio}) would discard all tokens. Keeping only the last token.")
            num_tokens_to_discard = seq_len - 1 
            if num_tokens_to_discard <=0 : return past_key_values # if seq_len was 1

        # Indices to keep are from num_tokens_to_discard up to seq_len - 1
        start_index_to_keep = num_tokens_to_discard
        indices_to_keep = torch.arange(start_index_to_keep, seq_len, device=past_key_values.key_cache[0].device)
        
        final_keep_count = len(indices_to_keep)
        self.logger.debug(f"EvictOldest: Original seq_len={seq_len}, Discarding first {num_tokens_to_discard} tokens, Keeping last {final_keep_count} tokens.")

        new_key_cache_list, new_value_cache_list = [], []
        for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
            current_tensor_device = k_tensor.device
            indices_on_correct_device = indices_to_keep.to(current_tensor_device) 
            
            new_key_cache_list.append(k_tensor.index_select(2, indices_on_correct_device))
            new_value_cache_list.append(v_tensor.index_select(2, indices_on_correct_device))
        
        return self._create_new_cache_with_updated_tensors(past_key_values, new_key_cache_list, new_value_cache_list)


    def _apply_attention_top_strategy(self, past_key_values: DynamicCache, attention_scores_tuple: Optional[Tuple[torch.Tensor, ...]], keep_ratio: float) -> DynamicCache:
        self.logger.info(f"Attempting AttentionTop strategy with keep_ratio: {keep_ratio}")
        if not (isinstance(past_key_values, DynamicCache) and hasattr(past_key_values, 'key_cache') and past_key_values.key_cache and past_key_values.key_cache[0] is not None):
            self.logger.warning("AttentionTop: KV cache is empty or not suitable. Skipping.")
            return past_key_values
        
        # MODIFICATION: If no attention_scores are provided (e.g., during pre-computation check), fallback immediately.
        if not attention_scores_tuple:
            self.logger.warning("AttentionTop: No attention scores provided (e.g. pre-computation call). Falling back to random.")
            return self._apply_random_strategy(past_key_values, keep_ratio)

        seq_len = past_key_values.get_seq_length()
        if seq_len <= 1: return past_key_values
        keep_count = max(1, int(seq_len * keep_ratio))
        if keep_count >= seq_len: return past_key_values

        try:
            aggregation_device = past_key_values.key_cache[0].device
            # _process_attention_scores is now only called if attention_scores_tuple is not None
            importance_scores = self._process_attention_scores(attention_scores_tuple, seq_len, aggregation_device)

            if importance_scores is None: # This means _process_attention_scores failed internally
                self.logger.warning("AttentionTop: Could not compute valid importance scores from _process_attention_scores. Falling back to random.")
                return self._apply_random_strategy(past_key_values, keep_ratio)
            
            # ... (rest of the logic: topk, index_select, creating new cache) ...
            _, indices_to_keep_template = torch.topk(importance_scores, keep_count, largest=True)
            indices_to_keep_template, _ = torch.sort(indices_to_keep_template)
            self.logger.debug(f"AttentionTop Eviction: Original seq_len={seq_len}, Keeping {keep_count} tokens.")

            new_key_cache_list, new_value_cache_list = [], []
            for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
                current_tensor_device = k_tensor.device
                indices_on_correct_device = indices_to_keep_template.to(current_tensor_device)
                new_key_cache_list.append(k_tensor.index_select(2, indices_on_correct_device))
                new_value_cache_list.append(v_tensor.index_select(2, indices_on_correct_device))
            
            return self._create_new_cache_with_updated_tensors(past_key_values, new_key_cache_list, new_value_cache_list)

        except Exception as e:
            self.logger.error(f"Error in _apply_attention_top_strategy: {e}", exc_info=True)
            return self._apply_random_strategy(past_key_values, keep_ratio) # Fallback on any other error

    def _apply_attention_bottom_strategy(self, past_key_values: DynamicCache, attention_scores_tuple: Tuple[torch.Tensor, ...], keep_ratio: float) -> DynamicCache:
        self.logger.info(f"Attempting AttentionBottom strategy with keep_ratio: {keep_ratio}")
        if not (isinstance(past_key_values, DynamicCache) and hasattr(past_key_values, 'key_cache') and past_key_values.key_cache and past_key_values.key_cache[0] is not None):
            self.logger.warning("AttentionBottom: KV cache is empty or not suitable. Skipping.")
            return past_key_values
        if not attention_scores_tuple:
            self.logger.warning("AttentionBottom: No attention scores provided. Falling back to random.")
            return self._apply_random_strategy(past_key_values, keep_ratio)

        seq_len = past_key_values.get_seq_length()
        if seq_len <= 1: return past_key_values
        keep_count = max(1, int(seq_len * keep_ratio))
        if keep_count >= seq_len: return past_key_values
        
        try:
            aggregation_device = past_key_values.key_cache[0].device
            importance_scores = self._process_attention_scores(attention_scores_tuple, seq_len, aggregation_device)

            if importance_scores is None:
                self.logger.warning("AttentionBottom: Could not compute valid importance scores. Falling back to random.")
                return self._apply_random_strategy(past_key_values, keep_ratio)

            _, indices_to_keep_template = torch.topk(importance_scores, keep_count, largest=False) # Key change: largest=False
            indices_to_keep_template, _ = torch.sort(indices_to_keep_template)
            self.logger.debug(f"AttentionBottom Eviction: Original seq_len={seq_len}, Keeping {keep_count} tokens (lowest attention).")

            new_key_cache_list, new_value_cache_list = [], []
            for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
                current_tensor_device = k_tensor.device
                indices_on_correct_device = indices_to_keep_template.to(current_tensor_device)
                new_key_cache_list.append(k_tensor.index_select(2, indices_on_correct_device))
                new_value_cache_list.append(v_tensor.index_select(2, indices_on_correct_device))
            
            return self._create_new_cache_with_updated_tensors(past_key_values, new_key_cache_list, new_value_cache_list)
        except Exception as e:
            self.logger.error(f"Error in _apply_attention_bottom_strategy: {e}", exc_info=True)
            return self._apply_random_strategy(past_key_values, keep_ratio)

    def _apply_sliding_window_simple_strategy(self, past_key_values: DynamicCache, keep_ratio: float) -> DynamicCache:
        self.logger.info(f"Attempting SimpleSlidingWindow strategy with keep_ratio: {keep_ratio}")
        if not (isinstance(past_key_values, DynamicCache) and 
                hasattr(past_key_values, 'key_cache') and past_key_values.key_cache and 
                hasattr(past_key_values, 'value_cache') and past_key_values.value_cache and
                len(past_key_values.key_cache) > 0 and past_key_values.key_cache[0] is not None):
            self.logger.warning("SimpleSlidingWindow: KV cache is empty or not suitable. Skipping.")
            return past_key_values

        seq_len = past_key_values.get_seq_length()
        if seq_len <= 1:
            self.logger.debug(f"SimpleSlidingWindow: seq_len ({seq_len}) too short. Skipping.")
            return past_key_values
        
        keep_count = max(1, int(seq_len * keep_ratio))
        if keep_count >= seq_len:
            self.logger.debug(f"SimpleSlidingWindow: keep_count ({keep_count}) >= seq_len ({seq_len}). No eviction needed.")
            return past_key_values

        # Indices for the most recent 'keep_count' tokens
        start_index = seq_len - keep_count
        indices_to_keep = torch.arange(start_index, seq_len, device=past_key_values.key_cache[0].device)
        
        self.logger.debug(f"SimpleSlidingWindow Eviction: Original seq_len={seq_len}, Keeping last {keep_count} tokens.")

        new_key_cache_list, new_value_cache_list = [], []
        for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
            current_tensor_device = k_tensor.device # Should be same as indices_to_keep if derived from key_cache[0]
            # Ensure indices are on the same device (though arange should have handled it above)
            indices_on_correct_device = indices_to_keep.to(current_tensor_device) 
            
            new_key_cache_list.append(k_tensor.index_select(2, indices_on_correct_device))
            new_value_cache_list.append(v_tensor.index_select(2, indices_on_correct_device))
        
        return self._create_new_cache_with_updated_tensors(past_key_values, new_key_cache_list, new_value_cache_list)

    def _apply_sliding_window_gist_strategy(self, past_key_values: DynamicCache, gist_token_count: int, recent_token_count: int) -> DynamicCache:
        self.logger.info(f"Attempting SlidingWindowGist: gist_tokens={gist_token_count}, recent_tokens={recent_token_count}")
        if not (isinstance(past_key_values, DynamicCache) and 
                hasattr(past_key_values, 'key_cache') and past_key_values.key_cache and 
                hasattr(past_key_values, 'value_cache') and past_key_values.value_cache and
                len(past_key_values.key_cache) > 0 and past_key_values.key_cache[0] is not None):
            self.logger.warning("SlidingWindowGist: KV cache is empty or not suitable. Skipping.")
            return past_key_values

        seq_len = past_key_values.get_seq_length()
        
        # Ensure gist_token_count and recent_token_count are not negative and are capped by seq_len
        gist_to_keep = max(0, min(seq_len, gist_token_count))
        recent_to_keep = max(0, min(seq_len, recent_token_count))

        if gist_to_keep + recent_to_keep >= seq_len: # If we are keeping everything or more
            self.logger.debug(f"SlidingWindowGist: gist ({gist_to_keep}) + recent ({recent_to_keep}) >= seq_len ({seq_len}). No eviction needed.")
            return past_key_values
        if gist_to_keep == 0 and recent_to_keep == 0 and seq_len > 0 : # Avoid keeping zero tokens if seq_len > 0
             self.logger.warning("SlidingWindowGist: both gist and recent counts are zero. Defaulting to keep last token.")
             recent_to_keep = 1


        gist_indices = list(range(gist_to_keep))
        # recent_indices start from (seq_len - recent_to_keep) up to seq_len -1
        # Ensure no overlap between gist and recent by adjusting the start of recent_indices
        # if gist_to_keep is large enough to overlap with the window of recent_to_keep tokens.
        recent_start_index = max(gist_to_keep, seq_len - recent_to_keep)
        recent_indices_list = list(range(recent_start_index, seq_len))
        
        indices_to_keep_set = set(gist_indices)
        indices_to_keep_set.update(recent_indices_list)
        
        indices_to_keep = sorted(list(indices_to_keep_set))
        final_keep_count = len(indices_to_keep)

        self.logger.debug(f"SlidingWindowGist Eviction: Original seq_len={seq_len}, Keeping {final_keep_count} tokens (gist: first {gist_to_keep}, recent: last relevant {len(recent_indices_list)}).")
        
        if final_keep_count == 0 and seq_len > 0: # Safety for empty indices_to_keep
            self.logger.warning("SlidingWindowGist: Resulted in 0 tokens to keep. Keeping last token instead.")
            indices_to_keep = [seq_len - 1]
        
        indices_to_keep_tensor = torch.tensor(indices_to_keep, device=past_key_values.key_cache[0].device, dtype=torch.long)

        new_key_cache_list, new_value_cache_list = [], []
        for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
            current_tensor_device = k_tensor.device
            indices_on_correct_device = indices_to_keep_tensor.to(current_tensor_device)
            new_key_cache_list.append(k_tensor.index_select(2, indices_on_correct_device))
            new_value_cache_list.append(v_tensor.index_select(2, indices_on_correct_device))
            
        return self._create_new_cache_with_updated_tensors(past_key_values, new_key_cache_list, new_value_cache_list)


    def _apply_sliding_window_attentive_gist_strategy(
        self, 
        past_key_values: DynamicCache, 
        attention_scores_tuple: Optional[Tuple[torch.Tensor, ...]], 
        model_config, # Added for _measure_kv_cache_size consistency if needed by it
        keep_ratio: float, # Overall target keep ratio
        gist_token_count: int, 
        recent_token_count: int
    ) -> DynamicCache:
        self.logger.info(f"Attempting SlidingWindowAttentiveGist: keep_ratio={keep_ratio}, gist={gist_token_count}, recent={recent_token_count}")

        original_seq_len = past_key_values.get_seq_length()
        if original_seq_len <= 1: return past_key_values

        target_total_keep_count = max(1, int(original_seq_len * keep_ratio))
        
        # Ensure gist and recent counts are not more than seq_len
        actual_gist_to_keep = min(gist_token_count, original_seq_len)
        actual_recent_to_keep = min(recent_token_count, original_seq_len)

        if actual_gist_to_keep + actual_recent_to_keep >= original_seq_len:
            # Gist + Recent already covers the whole sequence, or more than keep_ratio allows
            # In this case, we might just keep the most recent `target_total_keep_count` tokens if
            # gist+recent is too large, or all if it's smaller/equal to target.
            # For simplicity now, if they overlap or cover all, just keep min(original_seq_len, target_total_keep_count) recent tokens.
            # This simplifies to SlidingWindowSimple if gist/recent overlap significantly and target_total_keep_count is the main driver.
            # A more nuanced handling of overlap is needed for a true Gist + Recent if their sum exceeds target_total_keep_count.

            # Let's define: if gist+recent is already a good portion, we just keep those,
            # ensuring it doesn't exceed target_total_keep_count by too much (or if it does, prioritize recent over gist if forced to choose)

            gist_indices = list(range(actual_gist_to_keep))
            # Ensure recent indices don't overlap gist if possible, but prioritize their full count from end
            recent_start_idx = max(actual_gist_to_keep, original_seq_len - actual_recent_to_keep)
            recent_indices = list(range(recent_start_idx, original_seq_len))
            
            combined_indices = sorted(list(set(gist_indices + recent_indices)))
            
            if len(combined_indices) > target_total_keep_count:
                # If keeping both gist and full recent exceeds target, we need to trim more.
                # Prioritize keeping the full recent block, then fill with gist.
                self.logger.warning(f"SlidingWindowAttentiveGist: Gist ({actual_gist_to_keep}) + Recent ({actual_recent_to_keep}) exceeds target keep count ({target_total_keep_count}). Prioritizing recency and then gist within limit.")
                
                indices_to_keep_set = set(range(original_seq_len - actual_recent_to_keep, original_seq_len))
                remaining_to_keep = target_total_keep_count - len(indices_to_keep_set)
                if remaining_to_keep > 0:
                    indices_to_keep_set.update(range(min(actual_gist_to_keep, remaining_to_keep)))
                indices_to_keep = sorted(list(indices_to_keep_set))
            else:
                indices_to_keep = combined_indices
            
            if not indices_to_keep: # Safety if all logic above results in empty
                 indices_to_keep = list(range(max(0, original_seq_len -1 ), original_seq_len)) # keep last token

        else: # There's a middle section to consider for attention-based eviction
            tokens_to_keep_from_middle = target_total_keep_count - (actual_gist_to_keep + actual_recent_to_keep)
            
            middle_start_idx = actual_gist_to_keep
            middle_end_idx = original_seq_len - actual_recent_to_keep 
            middle_seq_len = middle_end_idx - middle_start_idx

            indices_to_keep = list(range(actual_gist_to_keep)) # Start with gist

            if tokens_to_keep_from_middle > 0 and middle_seq_len > 0:
                if attention_scores_tuple:
                    aggregation_device = past_key_values.key_cache[0].device
                    # Process attentions for the *entire original sequence* to get global importance
                    full_importance_scores = self._process_attention_scores(attention_scores_tuple, original_seq_len, aggregation_device)
                    
                    if full_importance_scores is not None:
                        # Select scores only for the middle segment
                        middle_importance_scores = full_importance_scores[middle_start_idx:middle_end_idx]
                        
                        num_to_pick_from_middle = min(tokens_to_keep_from_middle, middle_seq_len)
                        
                        if num_to_pick_from_middle > 0:
                            _, top_relative_indices_in_middle = torch.topk(middle_importance_scores, num_to_pick_from_middle, largest=True)
                            # Convert relative middle indices to original sequence indices
                            original_indices_from_middle = top_relative_indices_in_middle + middle_start_idx
                            indices_to_keep.extend(original_indices_from_middle.tolist())
                        else:
                            self.logger.debug("SlidingWindowAttentiveGist: No tokens to pick from middle based on budget.")
                    else:
                        self.logger.warning("SlidingWindowAttentiveGist: Could not compute attention scores for middle. Keeping random from middle.")
                        # Fallback: random from middle
                        num_to_pick_from_middle = min(tokens_to_keep_from_middle, middle_seq_len)
                        if num_to_pick_from_middle > 0:
                             middle_indices_pool = torch.arange(middle_start_idx, middle_end_idx, device=past_key_values.key_cache[0].device)
                             random_middle_indices = middle_indices_pool[torch.randperm(len(middle_indices_pool))[:num_to_pick_from_middle]]
                             indices_to_keep.extend(random_middle_indices.tolist())
                else: # No attention scores, pick random from middle or just keep gist+recent
                    self.logger.warning("SlidingWindowAttentiveGist: No attention scores for middle. Keeping random from middle if budget allows.")
                    num_to_pick_from_middle = min(tokens_to_keep_from_middle, middle_seq_len)
                    if num_to_pick_from_middle > 0:
                        middle_indices_pool = torch.arange(middle_start_idx, middle_end_idx, device=past_key_values.key_cache[0].device)
                        random_middle_indices = middle_indices_pool[torch.randperm(len(middle_indices_pool))[:num_to_pick_from_middle]]
                        indices_to_keep.extend(random_middle_indices.tolist())
            
            indices_to_keep.extend(list(range(original_seq_len - actual_recent_to_keep, original_seq_len))) # Add recent
            indices_to_keep = sorted(list(set(indices_to_keep))) # Deduplicate and sort

        final_keep_count = len(indices_to_keep)
        self.logger.debug(f"SlidingWindowAttentiveGist: Original seq_len={original_seq_len}, Target keep={target_total_keep_count}, Actual keeping {final_keep_count} tokens.")

        if final_keep_count == 0 and original_seq_len > 0:
            indices_to_keep = [original_seq_len - 1] # Keep at least the last token
            self.logger.warning("SlidingWindowAttentiveGist: After all logic, 0 tokens to keep. Defaulting to last token.")
        
        indices_to_keep_tensor = torch.tensor(indices_to_keep, device=past_key_values.key_cache[0].device, dtype=torch.long)

        new_key_cache_list, new_value_cache_list = [], []
        for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
            current_tensor_device = k_tensor.device
            indices_on_correct_device = indices_to_keep_tensor.to(current_tensor_device)
            new_key_cache_list.append(k_tensor.index_select(2, indices_on_correct_device))
            new_value_cache_list.append(v_tensor.index_select(2, indices_on_correct_device))
            
        return self._create_new_cache_with_updated_tensors(past_key_values, new_key_cache_list, new_value_cache_list)
    
    def _create_new_cache_with_updated_tensors(self, original_past_key_values: DynamicCache, 
                                             new_key_list: List[torch.Tensor], 
                                             new_value_list: List[torch.Tensor]) -> DynamicCache:
        """
        Updates the provided DynamicCache object's tensors by replacing its key_cache and value_cache.
        The 'seen_tokens' property will then re-evaluate based on these new tensors.
        """
        original_past_key_values.key_cache = new_key_list
        original_past_key_values.value_cache = new_value_list
        if new_key_list and new_key_list[0] is not None: # Check if cache is not empty
            # Log the new sequence length derived from the actual cache object
            self.logger.debug(f"Updated DynamicCache. New seq_len via get_seq_length(): {original_past_key_values.get_seq_length()}")
        else:
            self.logger.debug("Updated DynamicCache, but it's now empty or key_cache[0] is None.")
        return original_past_key_values



# --- Model Loading (ensure output_attentions is True for AttentionTop) ---
def load_model_and_tokenizer(model_name: str, cache_dir: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    logger.info(f"Loading model {model_name} from {cache_dir or 'default cache'}")
    try:
        # Load config first to modify it before loading model
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        config.output_attentions = True # Ensure attentions are output for AttentionTop strategy
        logger.info("Set model.config.output_attentions=True")

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config, # Pass modified config
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto", # Handles multi-GPU if available
            trust_remote_code=True
        )
        logger.info(f"Model {model_name} loaded on device(s): {model.hf_device_map if hasattr(model, 'hf_device_map') else model.device}")
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
            else: # Should not happen for most models
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                logger.warning("Added [PAD] as pad_token and resized model embeddings.")
        
        # Fallback chat template for Llama-3 if not present (verify against official template)
        if not getattr(tokenizer, 'chat_template', None) and "llama-3" in model_name.lower():
            logger.warning("Attempting to set fallback Llama-3 chat template.")
            llama3_instruct_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 and bos_token %}{% set content = bos_token + content %}{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{% endif %}"
            )
            tokenizer.chat_template = llama3_instruct_template
            logger.info("Fallback Llama-3 chat template applied.")


        return tokenizer, model
    except Exception as e:
        logger.critical(f"Fatal error loading model/tokenizer: {e}", exc_info=True)
        raise
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
        client = openai.OpenAI(api_key=api_key)
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
    
def _process_single_turn_stateful(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, current_turn_prompt_text: str,
    strat_cfg: Dict, cache_manager: RealKVCacheManager,
    max_gen_tokens: int, azure_eval: bool, azure_config: Optional[Dict],
    incoming_past_key_values: Optional[DynamicCache],
    is_first_turn: bool, max_input_length: int
) -> Tuple[Dict, Optional[DynamicCache]]:

    turn_start_time = time.time()
    logger.info(f"Processing Turn ({'FIRST' if is_first_turn else 'SUBSEQUENT'}). incoming_past_kv is None: {incoming_past_key_values is None}")
    logger.info(f"  Turn Prompt Text (first 100 chars): '{current_turn_prompt_text[:100]}...'")

    # Initialize turn-specific results and state variables
    kv_cache_sizes_during_generation = {}
    token_times = {}
    response_text = "[Error: Processing failed before generation]"
    tokens_generated_count = 0
    time_to_first_token = -1.0
    model_config = model.config
    azure_score = 0  # Initialize azure_score
    new_token_ids = [] # Initialize new_token_ids here
    
    # This will be the cache state passed to the model for the current turn's input processing
    past_kv_for_model_call = incoming_past_key_values 
    # This will be the cache state *after* the current turn's processing and generation
    output_past_key_values = incoming_past_key_values # Default to incoming if errors occur early

    pre_turn_gpu_mem = measure_gpu_memory()
    cache_manager.reset_turn_stats()
    
    try:
        # --- 1. Prepare input_ids for the current turn ---
        encoding_start_time = time.time()
        if is_first_turn:
            logger.debug("First turn: Tokenizing using apply_chat_template for single user message.")
            messages = [{"role": "user", "content": current_turn_prompt_text}]
            # No past_kv for the very first model call of a new question
            past_kv_for_model_call = None 
            try:
                inputs_tensor = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
                current_length = inputs_tensor.shape[-1]
                if current_length > max_input_length:
                    logger.warning(f"First turn input ({current_length} tokens) exceeded max_input_length ({max_input_length}). Truncating.")
                    bos_token_id = tokenizer.bos_token_id
                    if bos_token_id is not None and inputs_tensor.shape[1] > 0 and inputs_tensor[0, 0] == bos_token_id: # Check if BOS is actually first
                        inputs_tensor = torch.cat((inputs_tensor[:, :1], inputs_tensor[:, -(max_input_length - 1):]), dim=1)
                    else:
                        inputs_tensor = inputs_tensor[:, -max_input_length:]
                    logger.info(f"First turn input truncated to length: {inputs_tensor.shape[-1]}")
                input_ids = inputs_tensor.to(model.device)
            except Exception as e_template:
                logger.error(f"Failed to apply chat template for first turn: {e_template}. Falling back to basic tokenization.", exc_info=True)
                input_ids = tokenizer(current_turn_prompt_text, return_tensors="pt", max_length=max_input_length, truncation=True).input_ids.to(model.device)
        else: # Subsequent turn (Turn 2+)
            logger.debug("Subsequent turn: Tokenizing only current prompt text and using incoming KV cache.")
            # For Llama-3 style, a user turn is <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>
            # For subsequent turns, we should not add BOS if it's already in the cache.
            # `apply_chat_template` with a single message and `add_generation_prompt=False` (then manually add assistant prompt)
            # OR manually format the segment.
            # Let's try apply_chat_template for the segment, then remove BOS if it was added.
            messages = [{"role": "user", "content": current_turn_prompt_text}]
            try:
                # We want the tokens for the user turn *segment* only, and the assistant prompt
                # This is tricky with apply_chat_template for just one segment for stateful.
                # A simpler approach for Llama-3 might be:
                user_turn_segment = f"<|start_header_id|>user<|end_header_id|>\n\n{current_turn_prompt_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                input_ids = tokenizer(user_turn_segment, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
                
                # If incoming_past_key_values is None (meaning previous turn failed, starting fresh mid-conversation)
                # and no BOS in segment, prepend BOS. This case should ideally be handled by resetting the conversation.
                if past_kv_for_model_call is None and input_ids[0,0] != tokenizer.bos_token_id:
                    logger.warning("Subsequent turn but no past_kv. Prepending BOS to new segment.")
                    input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=model.device), input_ids], dim=1)

            except Exception as e_template:
                logger.error(f"Failed to tokenize subsequent turn prompt: {e_template}. Using basic tokenization.", exc_info=True)
                input_ids = tokenizer(current_turn_prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device) # Fallback, likely suboptimal

        encoding_time = time.time() - encoding_start_time
        logger.info(f"Input encoding time: {encoding_time:.4f}s. Current turn input_ids shape: {input_ids.shape}")

        if input_ids.numel() == 0:
            logger.warning("Input IDs are empty. Skipping model generation.")
            response_text = "[Error: Empty Input after tokenization]"
            output_past_key_values = past_kv_for_model_call # Return original cache
            # Construct a minimal error result
            # (Error 2 fix: Ensure all fields are present even on early exit)
            result_on_error = {
                "prompt_this_turn": current_turn_prompt_text, "response": response_text, "tokens_generated_this_turn": 0,
                "time_total_turn_seconds": time.time() - turn_start_time, "time_to_first_token_of_turn": -1.0,
                "tokens_per_second_this_turn": 0.0,
                "memory": {
                    "gpu_pre_turn_mb": pre_turn_gpu_mem.get("total_allocated_mb"), "gpu_post_turn_mb": measure_gpu_memory().get("total_allocated_mb"),
                    "kv_cache_end_of_turn_mb": measure_kv_cache_size(output_past_key_values, model_config),
                    "eviction_stats_this_turn": cache_manager.get_turn_stats()
                },
                "token_gen_times_this_turn": {}, "kv_cache_sizes_during_gen": {},
                "accuracy": {"azure_score": 0} # Ensure azure_score is initialized
            }
            return result_on_error, output_past_key_values

        # --- 2. Pre-computation Eviction on incoming_past_key_values ---
        if cache_manager and past_kv_for_model_call:
            size_of_incoming_cache = measure_kv_cache_size(past_kv_for_model_call, model_config)
            if size_of_incoming_cache >= cache_manager.threshold_mb:
                 logger.info(f"Applying pre-computation eviction. Incoming cache size {size_of_incoming_cache:.2f} MB >= threshold {cache_manager.threshold_mb:.2f} MB.")
                 past_kv_for_model_call = cache_manager.apply_eviction_strategy_if_needed(
                     past_kv_for_model_call, strat_cfg, model_config, attention_scores=None
                 )
        
        # --- 3. Process current input and generate response ---
        generation_loop_start_time = time.time()
        current_attentions = None
        # new_token_ids = [] # Moved to the top of the function

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_kv_for_model_call,
                use_cache=True,
                output_attentions=model.config.output_attentions,
                return_dict=True
            )
            logits = outputs.logits
            updated_kv_cache = outputs.past_key_values
            if model.config.output_attentions: current_attentions = outputs.attentions

            size_after_input_processing = measure_kv_cache_size(updated_kv_cache, model_config)
            logger.info(f"KV cache size after processing current turn's input tokens: {size_after_input_processing:.2f} MB")
            kv_cache_sizes_during_generation["input_processed"] = size_after_input_processing
            
            output_past_key_values = updated_kv_cache # Initialize with cache after prefill

            for i in range(max_gen_tokens):
                loop_iter_start_time = time.time()
                if cache_manager and output_past_key_values: # Use the latest cache state
                    output_past_key_values = cache_manager.apply_eviction_strategy_if_needed(
                        output_past_key_values, strat_cfg, model_config, attention_scores=current_attentions
                    )
                
                next_token_logits = logits[:, -1, :]
                next_token_id_tensor = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                next_token_id = next_token_id_tensor.item()

                if i == 0:
                    time_to_first_token = time.time() - generation_loop_start_time
                    # For token_times, key is 1-indexed count of generated tokens *this turn*
                    token_times[str(tokens_generated_count + 1)] = time_to_first_token
                    logger.info(f"Time to first token of this turn: {time_to_first_token:.4f}s")

                new_token_ids.append(next_token_id)
                tokens_generated_count += 1
                
                kv_cache_sizes_during_generation[str(tokens_generated_count)] = measure_kv_cache_size(output_past_key_values, model_config)

                is_eos = (tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id) or \
                           (hasattr(tokenizer, 'eot_id') and next_token_id == tokenizer.eot_id)
                if is_eos:
                    logger.info(f"EOS token {next_token_id} ('{tokenizer.decode([next_token_id], skip_special_tokens=False)}') detected at gen token {tokens_generated_count}. Stopping.")
                    break
                
                next_input_ids_for_loop = next_token_id_tensor.to(model.device)
                
                outputs = model(
                    input_ids=next_input_ids_for_loop,
                    past_key_values=output_past_key_values, # Pass the potentially evicted cache
                    use_cache=True,
                    output_attentions=model.config.output_attentions,
                    return_dict=True
                )
                logits = outputs.logits
                output_past_key_values = outputs.past_key_values # Update cache for next iteration
                if model.config.output_attentions: current_attentions = outputs.attentions
                
                if i > 0: token_times[str(tokens_generated_count)] = time.time() - loop_iter_start_time
            
        if new_token_ids:
            response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            response_text = response_text.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "").strip()
            if not response_text and new_token_ids: response_text = "[Empty Response, only special tokens?]"
        else:
            response_text = "[No tokens generated]"
        logger.info(f"Generated {tokens_generated_count} tokens. Response snippet: '{response_text[:100]}...'")

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM during turn (generated {tokens_generated_count} tokens for current prompt '{current_turn_prompt_text[:50]}...'): {e}", exc_info=True)
        response_text = "[OOM Error]"
        output_past_key_values = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Exception during turn (generated {tokens_generated_count} tokens for current prompt '{current_turn_prompt_text[:50]}...'): {e}", exc_info=True)
        response_text = "[Runtime Error]"
        output_past_key_values = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    turn_total_time = time.time() - turn_start_time
    tokens_per_second = tokens_generated_count / turn_total_time if turn_total_time > 0 and tokens_generated_count > 0 else 0.0
    post_turn_gpu_mem = measure_gpu_memory()
    
    final_kv_cache_size_mb = measure_kv_cache_size(output_past_key_values, model_config) if output_past_key_values else 0.0
    logger.info(f"End of turn. Final KV cache size: {final_kv_cache_size_mb:.2f} MB. Tokens generated this turn: {tokens_generated_count}.")

    if azure_eval and "Error" not in response_text and response_text != "[No tokens generated]": # Avoid sending errors for eval
        try:
            azure_eval_params = {
                'api_key': azure_config.get('api_key'),
                'api_version': azure_config.get('api_version'),
                'api_base': azure_config.get('api_base'),  # evaluate_with_azure expects 'api_base' for the endpoint
                'deployment': azure_config.get('deployment'), # This is used as the model/deployment name for Azure
                'model_name': azure_config.get('model', 'gpt-4') # For non-Azure or as a fallback
                # 'temperature' can also be passed from azure_config if it exists there
            }
            if 'temperature' in azure_config:
                azure_eval_params['temperature'] = azure_config.get('temperature')
            # Filter out None params for OpenAI client
            azure_eval_params_clean = {k:v for k,v in azure_eval_params.items() if v is not None}

            azure_score = evaluate_with_azure(current_turn_prompt_text, response_text, **azure_eval_params_clean)
            logger.info(f"Azure eval score: {azure_score}")
        except Exception as az_e:
            logger.error(f"Azure evaluation call failed: {az_e}", exc_info=True)
            azure_score = -1 # Indicate error
    
    result = {
        "prompt_this_turn": current_turn_prompt_text, "response": response_text,
        "tokens_generated_this_turn": tokens_generated_count,
        "time_total_turn_seconds": turn_total_time, "time_to_first_token_of_turn": time_to_first_token,
        "tokens_per_second_this_turn": tokens_per_second,
        "memory": {
            "gpu_pre_turn_mb": pre_turn_gpu_mem.get("total_allocated_mb"),
            "gpu_post_turn_mb": post_turn_gpu_mem.get("total_allocated_mb"),
            "kv_cache_end_of_turn_mb": final_kv_cache_size_mb,
            "eviction_stats_this_turn": cache_manager.get_turn_stats()
        },
        "token_gen_times_this_turn": token_times,
        "kv_cache_sizes_during_gen": kv_cache_sizes_during_generation,
        "accuracy": {"azure_score": azure_score}
    }
    return result, output_past_key_values
# --- Dataset Loading ---
def load_dataset(dataset_path: str) -> List[Dict]:
    """Load JSONL dataset from file."""
    items: List[Dict] = []
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f: # Added encoding
            content = f.read().strip()
            # Handle JSON array format
            if content.startswith('[') and content.endswith(']'):
                try:
                    items = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON array from dataset file: {e}")
                    return [] # Return empty if malformed
            else: # Assume JSONL
                for line_number, line in enumerate(content.splitlines(), 1):
                    if line.strip():
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON line {line_number} in dataset file: {e} - Line content: '{line[:100]}...'")
                            continue # Skip malformed lines
        logger.info(f"Loaded {len(items)} items from dataset")
        if not items: logger.warning("Dataset is empty or all lines were malformed!")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        logger.error(traceback.format_exc()) # Log full traceback for unexpected errors
    return items

# --- Stateful Multi-Turn Processing ---
def process_multi_turn(model, tokenizer, data_item, strat_cfg, cache_manager,
                       max_gen_tokens=100, max_input_length=4096, # Max length for first turn
                       azure_eval=False, azure_config=None):
    """
    Processes a multi-turn conversation statefully, managing KV cache across turns.
    """
    if not isinstance(data_item, dict) or "turns" not in data_item or not isinstance(data_item["turns"], list):
        logger.error(f"Invalid data item format for multi-turn processing: {data_item}")
        return [] # Skip invalid items

    question_id = data_item.get("question_id", "unknown")
    category = data_item.get("category", "unknown")
    turns_text_list = data_item.get("turns", []) # List of strings, each is a turn's text
    
    # Ensure turns_text_list contains only strings
    # (The dataset format shows it's a list of strings, but good to be robust)
    if not all(isinstance(turn_text, str) for turn_text in turns_text_list):
        logger.error(f"Invalid turn format in data_item for question {question_id}. Expected list of strings.")
        return []

    logger.info(f"--- Starting Question {question_id} (Total turns: {len(turns_text_list)}) ---")

    # State maintained across turns of the *same question*
    current_question_accumulated_kv_cache = None
    # Text history for logging/reference and for constructing prompts if needed by _process_single_turn_stateful
    # (though for stateful, _process_single_turn_stateful mainly uses current_turn_text)
    text_message_history_for_logging = []
    all_turn_results_for_this_question = []

    for turn_idx, current_turn_text in enumerate(turns_text_list):
        if not current_turn_text: # Skip empty turn strings
            logger.warning(f"Question {question_id}, Turn {turn_idx+1}: Skipping empty turn text.")
            # Add a placeholder result or simply skip adding to all_turn_results
            error_result = {
                "prompt_this_turn": "[Skipped Empty Turn Text]", "response": "[Skipped Empty Turn Text]",
                "tokens_generated_this_turn": 0, "turn_index": turn_idx, "question_id": question_id,
                "category": category, "total_turns_in_question": len(turns_text_list), "error": "Empty turn text"
            }
            all_turn_results_for_this_question.append(error_result)
            continue

        is_first_turn_of_question = (turn_idx == 0)
        logger.info(f"Processing Question {question_id}, Turn {turn_idx+1}/{len(turns_text_list)} ({'FIRST' if is_first_turn_of_question else 'SUBSEQUENT'}).")

        text_message_history_for_logging.append({"role": "user", "content": current_turn_text})

        # Process this single turn, passing the current KV cache state
        turn_result_dict, returned_past_key_values = _process_single_turn_stateful(
            model=model,
            tokenizer=tokenizer,
            current_turn_prompt_text=current_turn_text,
            strat_cfg=strat_cfg,
            cache_manager=cache_manager,
            max_gen_tokens=max_gen_tokens,
            azure_eval=azure_eval,
            azure_config=azure_config,
            incoming_past_key_values=current_question_accumulated_kv_cache, # Pass cache from previous turn
            is_first_turn=is_first_turn_of_question,
            max_input_length=max_input_length
        )

        # Update the KV cache state for the next turn *of this question*
        current_question_accumulated_kv_cache = returned_past_key_values

        # Add metadata to the turn result
        turn_result_dict["turn_index"] = turn_idx
        turn_result_dict["question_id"] = question_id
        turn_result_dict["category"] = category
        turn_result_dict["total_turns_in_question"] = len(turns_text_list)
        # Add accumulated context length (from KV cache) to results for this turn
        if current_question_accumulated_kv_cache and hasattr(current_question_accumulated_kv_cache, 'get_seq_length'):
            turn_result_dict["kv_cache_seq_len_end_of_turn"] = current_question_accumulated_kv_cache.get_seq_length()
        
        all_turn_results_for_this_question.append(turn_result_dict)

        assistant_response = turn_result_dict.get("response", "")
        if "Error" not in assistant_response and assistant_response != "[No tokens generated]":
             text_message_history_for_logging.append({"role": "assistant", "content": assistant_response})
        else:
             logger.warning(f"Turn {turn_idx+1} for question {question_id} resulted in response: '{assistant_response}'.")
             # If a turn fails and loses cache (e.g., OOM), stop processing subsequent turns for this question.
             if current_question_accumulated_kv_cache is None:
                  logger.error(f"KV Cache lost. Stopping further processing for question {question_id} after error in turn {turn_idx+1}.")
                  break
        if turn_idx == 10: # Turn 10 is when turn_idx (0-indexed) is 9
            logger.info(f"DEBUG: Manually stopping processing for question {question_id} after Turn {turn_idx+1} as requested for debugging.")
            break # Exit the loop for the current question's turns
    
    logger.info(f"--- Finished Question {question_id}. Processed {len(all_turn_results_for_this_question)} turns. ---")
    return all_turn_results_for_this_question
def main():
    parser = argparse.ArgumentParser(description='Stateful Benchmark for KV Cache Management')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--limit', type=int, help='Limit number of questions to process from dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {args.config}")
        logger.critical(f"Configuration file not found at {args.config}")
        return
    except json.JSONDecodeError as e:
         print(f"ERROR: Invalid JSON in configuration file {args.config}: {e}")
         logger.critical(f"Invalid JSON in configuration file {args.config}: {e}")
         return

    # Set up logging level based on debug flag
    # Ensure log_filename is defined globally or passed appropriately if not already
    global log_filename 
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Reconfigure root logger if not already configured, or specific logger
    # This ensures file handler from global scope gets the right level too.
    root_logger.setLevel(log_level) 
    for handler in root_logger.handlers: # Adjust existing file handler's level
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(log_level)
            
    logger.setLevel(log_level) # Set our specific logger's level
    logger.info(f"Logging to file: {log_filename}") # log_filename should be defined at the top of the script
    logger.info(f"Log level set to: {logging.getLevelName(logger.level)}")


    # Load model and tokenizer
    try:
        tokenizer, model = load_model_and_tokenizer(
            cfg['model_name'],
            cache_dir=cfg.get('cache_dir')
        )
    except Exception as e:
        logger.critical(f"Failed to load model/tokenizer. Exiting. Error: {e}")
        return # Cannot proceed without model

    # Initialize cache manager
    cache_manager = RealKVCacheManager(model, tokenizer, cfg, logger)

    # Load dataset
    dataset = load_dataset(cfg['dataset']['local_path'])
    if not dataset:
         logger.critical("Dataset failed to load or is empty. Exiting.")
         return

    # Limit dataset if requested
    if args.limit and args.limit > 0:
        dataset = dataset[:args.limit]
        logger.info(f"Dataset limited to first {len(dataset)} questions.")

    # Prepare output directory
    results_dir = cfg.get('output_dir', 'stateful_results') # Default output directory
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    # Get Azure config if enabled
    azure_eval = cfg.get('azure_evaluation', {}).get('enabled', False) and HAS_OPENAI
    azure_config = cfg.get('azure_evaluation', {}) if azure_eval else None # Pass the whole dict
    if azure_eval: logger.info("Azure evaluation enabled.")
    else: logger.info("Azure evaluation disabled (or openai library missing/config issue).")

    strategies = cfg.get('strategies', [])
    if not strategies:
         logger.warning("No strategies found in configuration file. Exiting.")
         return

    logger.info(f"Starting benchmark with {len(strategies)} strategies.")

    # Iterate over strategies
    for strat_cfg in strategies: # strat_cfg is now expected to be a dictionary
        strategy_name = strat_cfg.get("name", "UnknownStrategy") # Get name from dict
        if strategy_name == "UnknownStrategy":
            logger.error(f"Strategy configuration missing 'name': {strat_cfg}. Skipping.")
            continue
            
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', strategy_name) # Sanitize name for filename
        # Append parameters to filename for strategies like Random(keep_ratio=0.7) for uniqueness
        if strategy_name == "Random" and "keep_ratio" in strat_cfg:
            safe_name += f"_keep{int(strat_cfg['keep_ratio']*100)}"
        elif strategy_name == "AttentionTop" and "keep_ratio" in strat_cfg:
            safe_name += f"_keep{int(strat_cfg['keep_ratio']*100)}"
            
        logger.info(f"\n===== Running Strategy: {strategy_name} (config: {strat_cfg}) =====")

        all_results_for_strategy = []
        pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Strategy {strategy_name}")
        for i, data_item in pbar:
            question_id = data_item.get('question_id', f'item_{i}')
            pbar.set_description(f"Strategy {strategy_name} (Q: {question_id})")
            try:
                # Process all turns for this question statefully
                turn_results = process_multi_turn(
                    model, tokenizer, data_item, strat_cfg, cache_manager, # Pass the dict strat_cfg
                    max_gen_tokens=cfg.get('max_gen_tokens', 256),
                    max_input_length=cfg.get('max_input_length', 4096),
                    azure_eval=azure_eval,
                    azure_config=azure_config
                )
                all_results_for_strategy.extend(turn_results)

            except Exception as e:
                logger.error(f"FATAL ERROR processing question {question_id} with strategy {strategy_name}: {e}")
                logger.error(traceback.format_exc())
                all_results_for_strategy.append({
                    "question_id": question_id, "error": str(e),
                    "strategy_cfg": strat_cfg, # Log strategy that failed
                    "traceback": traceback.format_exc()
                })
            
            # Optional: Clear CUDA cache between questions
            if HAS_GC: gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Save results for this strategy
        if all_results_for_strategy:
            out_path = os.path.join(results_dir, f"{safe_name}_results.json")
            try:
                with open(out_path, 'w') as f:
                    json.dump(all_results_for_strategy, f, indent=2)
                logger.info(f"Saved {len(all_results_for_strategy)} turn results for {strategy_name} to {out_path}")
            except Exception as e:
                 logger.error(f"Failed to save results for {strategy_name} to {out_path}: {e}")
        else:
            logger.warning(f"No results generated for strategy {strategy_name}.")

    logger.info("===== Stateful Benchmark Finished =====")

if __name__ == "__main__":
    # This ensures main() is called only when the script is executed directly
    main()
