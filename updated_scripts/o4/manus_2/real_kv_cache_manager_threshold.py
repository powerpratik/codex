import torch
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

class RealKVCacheManager:
    """
    KV Cache Manager that properly applies eviction strategies and measures real metrics
    for Llama-2-7b-chat-hf model.
    
    This updated version implements threshold-based eviction that only kicks in
    when the cache size is about to exceed the threshold.
    """
    def __init__(self, model, tokenizer, config, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger or logging.getLogger("RealKVCacheManager")
        
        # KV cache threshold in MB
        self.threshold = config.get("kv_threshold", 1000)
        
        # Memory tracking
        self.peak_memory = 0
        self.current_memory = 0
        self.eviction_count = 0
        self.total_eviction_time = 0
        
        # Detailed metrics
        self.step_wise_cache_sizes = []
        self.layer_wise_cache_sizes = []
        self.eviction_stats = []
        self.threshold_triggers = []
        
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
        base_threshold = self.threshold
        
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
                # Extract layer index if possible
                layer_idx = -1
                try:
                    if 'layers.' in name:
                        parts = name.split('layers.')
                        if len(parts) > 1:
                            layer_idx = int(parts[1].split('.')[0])
                except:
                    pass
                
                # Store layer index in module for later use
                module._layer_idx = layer_idx
                
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
                # Record pre-eviction size
                pre_size = self.compute_cache_size(output[-1])
                
                # Record post-eviction size and time
                post_size = pre_size  # Will be updated if eviction occurs
                eviction_time = 0
                
                self.current_memory = post_size
                self.track_memory(post_size)
                self.step_wise_cache_sizes.append(post_size)
                
                # Compute layer-wise sizes
                layer_sizes = self.compute_layer_sizes(output[-1])
                self.layer_wise_cache_sizes.append(layer_sizes)
        
        return output
    
    def apply_eviction_strategy(self, cache, strategy_name, attention_scores=None):
        """
        Apply the specified eviction strategy to the KV cache,
        but only if the cache size is about to exceed the threshold.
        """
        if cache is None:
            return cache
        
        # Measure current cache size
        current_size = self.compute_cache_size(cache)
        
        # Check if we're approaching the threshold (within 5%)
        threshold_margin = 0.95  # Apply eviction when we reach 95% of threshold
        approaching_threshold = current_size >= (self.threshold * threshold_margin)
        
        # For baseline strategy, never evict
        if strategy_name == "Baseline":
            return cache
        
        # Only apply eviction if approaching threshold
        if not approaching_threshold:
            return cache
            
        # Record threshold trigger
        self.threshold_triggers.append({
            "step": len(self.step_wise_cache_sizes),
            "pre_size": current_size,
            "threshold": self.threshold,
            "strategy": strategy_name
        })
        
        start_time = time.time()
        
        # Apply strategy-specific eviction
        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            # LlamaCache object - we need to modify the key_cache and value_cache in-place
            
            if strategy_name.startswith("SlidingWindow"):
                # Extract parameters if available
                window_size = 0.7  # Default
                important_ratio = 0.1  # Default
                
                if "(" in strategy_name:
                    param_str = strategy_name.split("(")[1].rstrip(")")
                    for p in param_str.split(","):
                        if "=" in p:
                            k, v = p.split("=")
                            if k.strip() == "window":
                                window_size = float(v.strip())
                            elif k.strip() == "important":
                                important_ratio = float(v.strip())
                
                # Apply sliding window strategy to each layer
                for layer_idx in range(len(cache.key_cache)):
                    if layer_idx < len(cache.key_cache) and layer_idx < len(cache.value_cache):
                        k = cache.key_cache[layer_idx]
                        v = cache.value_cache[layer_idx]
                        
                        if k is not None and v is not None:
                            seq_len = k.size(0)
                            
                            # Number of recent tokens to keep
                            recent_count = max(1, int(seq_len * window_size))
                            
                            # Always keep the most recent tokens
                            keep_indices = list(range(seq_len - recent_count, seq_len))
                            
                            # If we have attention scores, also keep some important historical tokens
                            if attention_scores is not None and len(attention_scores) == seq_len:
                                historical_indices = list(range(0, seq_len - recent_count))
                                
                                # Only keep important_ratio of historical tokens
                                important_count = max(1, int(len(historical_indices) * important_ratio))
                                
                                # Get scores for historical tokens only
                                if len(historical_indices) > 0:
                                    historical_scores = attention_scores[historical_indices]
                                    
                                    # Find the most important historical tokens
                                    if len(historical_scores) > 0:
                                        top_indices = torch.topk(historical_scores, 
                                                               min(important_count, len(historical_scores))).indices
                                        important_historical = [historical_indices[i] for i in top_indices.tolist()]
                                        keep_indices.extend(important_historical)
                            
                            # Sort indices
                            keep_indices = sorted(set(keep_indices))
                            
                            # Create new tensors with only the kept tokens
                            # This is crucial for actually reducing memory usage
                            new_k = k[keep_indices].clone()
                            new_v = v[keep_indices].clone()
                            
                            # Replace the original tensors
                            cache.key_cache[layer_idx] = new_k
                            cache.value_cache[layer_idx] = new_v
                            
                            # Force CUDA to actually free the memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
            
            elif strategy_name.startswith("AdaptiveAttention"):
                # Extract parameters if available
                base_keep = 0.7  # Default
                
                if "(" in strategy_name:
                    param_str = strategy_name.split("(")[1].rstrip(")")
                    for p in param_str.split(","):
                        if "=" in p:
                            k, v = p.split("=")
                            if k.strip() == "base_keep":
                                base_keep = float(v.strip())
                
                # Apply adaptive attention strategy to each layer
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
                                
                                # Create new tensors with only the kept tokens
                                new_k = k[indices].clone()
                                new_v = v[indices].clone()
                                
                                # Replace the original tensors
                                cache.key_cache[layer_idx] = new_k
                                cache.value_cache[layer_idx] = new_v
                            else:
                                # Fallback to keeping most recent tokens
                                keep_len = max(1, int(seq_len * layer_keep))
                                
                                # Create new tensors with only the kept tokens
                                new_k = k[-keep_len:].clone()
                                new_v = v[-keep_len:].clone()
                                
                                # Replace the original tensors
                                cache.key_cache[layer_idx] = new_k
                                cache.value_cache[layer_idx] = new_v
                            
                            # Force CUDA to actually free the memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
        
        # Record post-eviction size and time
        post_size = self.compute_cache_size(cache)
        eviction_time = time.time() - start_time
        
        if post_size < current_size:
            self.eviction_count += 1
            self.total_eviction_time += eviction_time
            self.eviction_stats.append({
                "strategy": strategy_name,
                "pre_size": current_size,
                "post_size": post_size,
                "time": eviction_time,
                "tokens_removed": current_size - post_size,
                "reduction_percent": (1 - post_size / current_size) * 100
            })
            
            self.logger.info(f"Applied {strategy_name} eviction: {current_size:.2f}MB → {post_size:.2f}MB " +
                           f"({(1 - post_size / current_size) * 100:.1f}% reduction)")
        
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
            "eviction_stats": self.eviction_stats,
            "threshold_triggers": self.threshold_triggers,
            "threshold_mb": self.threshold
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
        self.threshold_triggers = []
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
