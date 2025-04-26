# kv_cache_manager.py
import torch
import time
import logging
from attention_utils import extract_layerwise_attention, dynamic_token_importance

class KVCacheManager:
    """
    Advanced KV Cache Manager for newer LLaMA models
    """
    def __init__(self, model, tokenizer, config, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger or logging.getLogger("KVCacheManager")
        
        # Memory tracking
        self.peak_memory = 0
        self.eviction_count = 0
        self.total_eviction_time = 0
        
        # Layer-specific configurations
        self.num_layers = self._detect_num_layers()
        self.layer_thresholds = self._init_layer_thresholds()
    
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
    
    def compute_cache_size(self, cache):
        """Compute total KV cache size in MB"""
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
                        total_bytes += k.numel() * k.element_size()
                        total_bytes += v.numel() * v.element_size()
            
            # Convert to megabytes
            return total_bytes / (1024 ** 2)
        
        # Handle traditional tuple of (key, value) pairs
        elif isinstance(cache, tuple) and all(isinstance(layer, tuple) and len(layer) == 2 for layer in cache):
            total_bytes = 0
            for k, v in cache:
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()
            return total_bytes / (1024 ** 2)
            
        # Unsupported cache format
        else:
            self.logger.warning(f"Unknown cache format: {type(cache)}. Cannot compute size.")
            return 0
    
    def compute_layer_sizes(self, cache):
        """Compute KV cache size per layer in MB"""
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
                        layer_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()
                        layer_sizes.append(layer_bytes / (1024 ** 2))
                    else:
                        layer_sizes.append(0)
        
        # Handle traditional tuple of (key, value) pairs
        elif isinstance(cache, tuple) and all(isinstance(layer, tuple) and len(layer) == 2 for layer in cache):
            for k, v in cache:
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
        """Get memory usage statistics"""
        return {
            "peak_memory_mb": self.peak_memory,
            "eviction_count": self.eviction_count,
            "avg_eviction_time": self.total_eviction_time / max(1, self.eviction_count),
            "total_eviction_time": self.total_eviction_time
        }