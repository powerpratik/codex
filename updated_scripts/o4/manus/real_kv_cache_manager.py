import torch
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

class RealKVCacheManager:
    """
    KV Cache Manager that properly applies eviction strategies and measures real metrics
    for Llama-2-7b-chat-hf model.
    """
    def __init__(self, model, tokenizer, config, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger or logging.getLogger("RealKVCacheManager")
        
        # Memory tracking
        self.peak_memory = 0
        self.current_memory = 0
        self.eviction_count = 0
        self.total_eviction_time = 0
        
        # Detailed metrics
        self.step_wise_cache_sizes = []
        self.layer_wise_cache_sizes = []
        self.eviction_stats = []
        
        # Layer-specific configurations
        self.num_layers = self._detect_num_layers()
        self.layer_thresholds = self._init_layer_thresholds()
        
        # Register hooks for tracking KV cache
        self._register_hooks()
    
    def _detect_num_layers(self):
        """Detect number of layers in the model"""
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'num_hidden_layers'):
                return self.model.config.num_hidden_layers
        # Fallback to manual inspection
        return len([n for n, _ in self.model.named_modules() if 'layers' in n and 'attention' in n])
    
    def _init_layer_thresholds(self):
        """Initialize layer-specific thresholds"""
        base_threshold = self.config.get("kv_threshold", 1000)
        
        # Allow more aggressive pruning in earlier layers
        layer_factors = []
        for i in range(self.num_layers):
            layer_factor = 0.7 + 0.3 * (i / max(1, self.num_layers - 1))
            layer_factors.append(layer_factor)
            
        return [base_threshold * factor for factor in layer_factors]
    
    def _register_hooks(self):
        """Register forward hooks to intercept and modify KV cache"""
        self.hooks = []
        
        # Find all attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                # Register pre-forward hook to capture input KV cache
                pre_hook = module.register_forward_pre_hook(self._pre_attention_hook)
                self.hooks.append(pre_hook)
                
                # Register forward hook to capture and modify output KV cache
                post_hook = module.register_forward_hook(self._post_attention_hook)
                self.hooks.append(post_hook)
    
    def _pre_attention_hook(self, module, args):
        """Pre-attention hook to capture input KV cache"""
        # This hook runs before the attention module's forward pass
        # We can use it to measure the size of the incoming KV cache
        if len(args) > 2 and args[2] is not None:  # past_key_value is typically the 3rd argument
            past_key_value = args[2]
            # Measure but don't modify yet
            size = self.compute_cache_size(past_key_value)
            if size > 0:
                self.current_memory = size
                self.track_memory(size)
        return None
    
    def _post_attention_hook(self, module, args, output):
        """Post-attention hook to capture and modify output KV cache"""
        # This hook runs after the attention module's forward pass
        # We can use it to apply eviction strategies to the KV cache
        
        # For Llama models, output is typically a tuple where the last element is past_key_value
        if isinstance(output, tuple) and len(output) > 1:
            # The last element is typically the updated KV cache
            if hasattr(output[-1], 'key_cache') and hasattr(output[-1], 'value_cache'):
                # This is a LlamaCache object
                # Apply eviction strategy here
                start_time = time.time()
                
                # Record pre-eviction size
                pre_size = self.compute_cache_size(output[-1])
                
                # Apply eviction strategy
                # Note: We need to implement strategy-specific eviction logic here
                # For now, we'll just measure without modifying
                
                # Record post-eviction size and time
                post_size = self.compute_cache_size(output[-1])
                eviction_time = time.time() - start_time
                
                if pre_size != post_size:
                    self.eviction_count += 1
                    self.total_eviction_time += eviction_time
                    self.eviction_stats.append({
                        "pre_size": pre_size,
                        "post_size": post_size,
                        "time": eviction_time
                    })
                
                self.current_memory = post_size
                self.track_memory(post_size)
                self.step_wise_cache_sizes.append(post_size)
                
                # Compute layer-wise sizes
                layer_sizes = self.compute_layer_sizes(output[-1])
                self.layer_wise_cache_sizes.append(layer_sizes)
        
        return output
    
    def apply_eviction_strategy(self, cache, strategy, attention_scores=None):
        """Apply the specified eviction strategy to the KV cache"""
        if cache is None:
            return cache
        
        start_time = time.time()
        
        # Record pre-eviction size
        pre_size = self.compute_cache_size(cache)
        
        # Apply strategy-specific eviction
        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            # LlamaCache object
            # We need to modify the key_cache and value_cache in-place
            # This is model-specific and requires careful implementation
            
            # For demonstration, we'll implement a simple sliding window strategy
            if strategy == "SlidingWindow":
                window_size = 0.7  # Keep the most recent 70% of tokens
                
                for layer_idx in range(len(cache.key_cache)):
                    if layer_idx < len(cache.key_cache) and layer_idx < len(cache.value_cache):
                        k = cache.key_cache[layer_idx]
                        v = cache.value_cache[layer_idx]
                        
                        if k is not None and v is not None:
                            seq_len = k.size(0)
                            keep_len = max(1, int(seq_len * window_size))
                            
                            # Keep only the most recent tokens
                            cache.key_cache[layer_idx] = k[-keep_len:].clone()
                            cache.value_cache[layer_idx] = v[-keep_len:].clone()
            
            elif strategy == "AdaptiveAttention" and attention_scores is not None:
                # Implement adaptive attention strategy
                base_keep = 0.7
                
                for layer_idx in range(len(cache.key_cache)):
                    if layer_idx < len(cache.key_cache) and layer_idx < len(cache.value_cache):
                        k = cache.key_cache[layer_idx]
                        v = cache.value_cache[layer_idx]
                        
                        if k is not None and v is not None:
                            seq_len = k.size(0)
                            
                            # Adjust keep ratio based on layer position
                            layer_position = layer_idx / max(1, self.num_layers - 1)
                            layer_keep = base_keep - (0.2 * layer_position)
                            
                            if attention_scores is not None and len(attention_scores) == seq_len:
                                # Use attention scores to determine which tokens to keep
                                keep_count = max(1, int(seq_len * layer_keep))
                                
                                # Get indices of top-k tokens by importance
                                _, indices = torch.topk(attention_scores, keep_count)
                                indices, _ = torch.sort(indices)  # Sort indices to maintain sequence order
                                
                                # Keep only the selected tokens
                                cache.key_cache[layer_idx] = k[indices].clone()
                                cache.value_cache[layer_idx] = v[indices].clone()
                            else:
                                # Fallback to keeping most recent tokens
                                keep_len = max(1, int(seq_len * layer_keep))
                                cache.key_cache[layer_idx] = k[-keep_len:].clone()
                                cache.value_cache[layer_idx] = v[-keep_len:].clone()
        
        # Record post-eviction size and time
        post_size = self.compute_cache_size(cache)
        eviction_time = time.time() - start_time
        
        if pre_size != post_size:
            self.eviction_count += 1
            self.total_eviction_time += eviction_time
            self.eviction_stats.append({
                "strategy": strategy,
                "pre_size": pre_size,
                "post_size": post_size,
                "time": eviction_time
            })
        
        return cache
    
    def compute_cache_size(self, cache):
        """Compute total KV cache size in MB - real physical measurement"""
        if cache is None:
            return 0
            
        # Handle LlamaCache object
        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            total_bytes = 0
            
            # For LlamaCache, access the internal key and value tensors
            for layer_idx in range(len(cache.key_cache)):
                if layer_idx < len(cache.key_cache) and layer_idx < len(cache.value_cache):
                    k = cache.key_cache[layer_idx]
                    v = cache.value_cache[layer_idx]
                    
                    if k is not None and v is not None:
                        # Real physical measurement using PyTorch's memory tracking
                        total_bytes += k.numel() * k.element_size()
                        total_bytes += v.numel() * v.element_size()
            
            # Convert to megabytes
            return total_bytes / (1024 ** 2)
        
        # Handle traditional tuple of (key, value) pairs
        elif isinstance(cache, tuple) and all(isinstance(layer, tuple) and len(layer) == 2 for layer in cache):
            total_bytes = 0
            for k, v in cache:
                # Real physical measurement using PyTorch's memory tracking
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()
            return total_bytes / (1024 ** 2)
            
        # Unsupported cache format
        else:
            self.logger.warning(f"Unknown cache format: {type(cache)}. Cannot compute size.")
            return 0
    
    def compute_layer_sizes(self, cache):
        """Compute KV cache size per layer in MB - real physical measurement"""
        if cache is None:
            return []
            
        layer_sizes = []
        
        # Handle LlamaCache object
        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            for layer_idx in range(len(cache.key_cache)):
                if layer_idx < len(cache.key_cache) and layer_idx < len(cache.value_cache):
                    k = cache.key_cache[layer_idx]
                    v = cache.value_cache[layer_idx]
                    
                    if k is not None and v is not None:
                        # Real physical measurement using PyTorch's memory tracking
                        layer_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()
                        layer_sizes.append(layer_bytes / (1024 ** 2))
                    else:
                        layer_sizes.append(0)
        
        # Handle traditional tuple of (key, value) pairs
        elif isinstance(cache, tuple) and all(isinstance(layer, tuple) and len(layer) == 2 for layer in cache):
            for k, v in cache:
                # Real physical measurement using PyTorch's memory tracking
                layer_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()
                layer_sizes.append(layer_bytes / (1024 ** 2))
        
        return layer_sizes
    
    def track_memory(self, current_size):
        """Track peak memory usage"""
        self.peak_memory = max(self.peak_memory, current_size)
    
    def identify_token_types(self, token_ids):
        """Identify token types for importance calculation"""
        token_types = []
        
        # Get special tokens
        special_tokens = set()
        for name in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'sep_token_id', 'cls_token_id']:
            if hasattr(self.tokenizer, name):
                token_id = getattr(self.tokenizer, name)
                if token_id is not None:
                    special_tokens.add(token_id)
        
        # Simple entity detection heuristic
        entity_pattern = []
        try:
            for token_id in token_ids:
                token_str = self.tokenizer.decode([token_id])
                if token_str and token_str[0].isupper() and len(token_str) > 1:
                    entity_pattern.append(True)
                else:
                    entity_pattern.append(False)
        except:
            entity_pattern = [False] * len(token_ids)
        
        # Assign token types
        for i, token_id in enumerate(token_ids):
            if token_id in special_tokens:
                token_types.append('special')
            elif i < len(entity_pattern) and entity_pattern[i]:
                token_types.append('entity')
            else:
                token_types.append('regular')
        
        return token_types
    
    def get_memory_stats(self):
        """Get detailed memory usage statistics"""
        return {
            "peak_memory_mb": self.peak_memory,
            "current_memory_mb": self.current_memory,
            "eviction_count": self.eviction_count,
            "avg_eviction_time": self.total_eviction_time / max(1, self.eviction_count),
            "total_eviction_time": self.total_eviction_time,
            "step_wise_cache_sizes": self.step_wise_cache_sizes,
            "layer_wise_cache_sizes": self.layer_wise_cache_sizes,
            "eviction_stats": self.eviction_stats
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.peak_memory = 0
        self.current_memory = 0
        self.eviction_count = 0
        self.total_eviction_time = 0
        self.step_wise_cache_sizes = []
        self.layer_wise_cache_sizes = []
        self.eviction_stats = []
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
