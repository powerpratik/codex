import torch
import random
import numpy as np
from enhanced_strategies import EnhancedBaseStrategy

class RandomKVCacheStrategy(EnhancedBaseStrategy):
    """
    Random KV cache management strategy that randomly selects tokens to keep
    when the cache size exceeds the threshold.
    """
    def __init__(self, threshold, keep_ratio=0.7):
        super().__init__(threshold)
        self.keep_ratio = keep_ratio
        self.name = f"Random(keep={keep_ratio})"
    
    def evict(self, past_key_values, attention_scores=None):
        """Global eviction strategy"""
        all_k, all_v = zip(*past_key_values)
        
        T = all_k[0].shape[0]  # Sequence length
        
        # Number of tokens to keep
        keep_count = max(1, int(T * self.keep_ratio))
        
        # Randomly select indices to keep
        keep_indices = sorted(random.sample(range(T), keep_count))
        
        # Apply eviction
        def prune(t):
            return t[keep_indices, ...]
        
        return tuple((prune(k), prune(v)) for k, v in past_key_values)
    
    def evict_layer(self, layer_kv, attention_scores=None, layer_idx=0, total_layers=12):
        """Layer-specific eviction strategy"""
        k, v = layer_kv
        T = k.shape[0]  # Sequence length
        
        # Number of tokens to keep
        keep_count = max(1, int(T * self.keep_ratio))
        
        # Randomly select indices to keep
        keep_indices = sorted(random.sample(range(T), keep_count))
        
        # Apply eviction
        return k[keep_indices, ...], v[keep_indices, ...]

class AttentionTopStrategy(EnhancedBaseStrategy):
    """
    Attention-based strategy that keeps tokens with the highest attention scores
    when the cache size exceeds the threshold.
    """
    def __init__(self, threshold, keep_ratio=0.7):
        super().__init__(threshold)
        self.keep_ratio = keep_ratio
        self.name = f"AttentionTop(keep={keep_ratio})"
    
    def evict(self, past_key_values, attention_scores=None):
        """Global eviction strategy"""
        all_k, all_v = zip(*past_key_values)
        
        T = all_k[0].shape[0]  # Sequence length
        
        # Number of tokens to keep
        keep_count = max(1, int(T * self.keep_ratio))
        
        # If we have attention scores, keep tokens with highest scores
        if attention_scores is not None and len(attention_scores) == T:
            # Get indices of top-k tokens by importance
            _, indices = torch.topk(attention_scores, keep_count)
            keep_indices = sorted(indices.tolist())
        else:
            # Fallback to keeping most recent tokens
            keep_indices = list(range(T - keep_count, T))
        
        # Apply eviction
        def prune(t):
            return t[keep_indices, ...]
        
        return tuple((prune(k), prune(v)) for k, v in past_key_values)
    
    def evict_layer(self, layer_kv, attention_scores=None, layer_idx=0, total_layers=12):
        """Layer-specific eviction strategy"""
        k, v = layer_kv
        T = k.shape[0]  # Sequence length
        
        # Number of tokens to keep
        keep_count = max(1, int(T * self.keep_ratio))
        
        # If we have attention scores, keep tokens with highest scores
        if attention_scores is not None and len(attention_scores) == T:
            # Get indices of top-k tokens by importance
            _, indices = torch.topk(attention_scores, keep_count)
            keep_indices = sorted(indices.tolist())
        else:
            # Fallback to keeping most recent tokens
            keep_indices = list(range(T - keep_count, T))
        
        # Apply eviction
        return k[keep_indices, ...], v[keep_indices, ...]

class AttentionBottomStrategy(EnhancedBaseStrategy):
    """
    Attention-based strategy that keeps tokens with the lowest attention scores
    when the cache size exceeds the threshold.
    """
    def __init__(self, threshold, keep_ratio=0.7):
        super().__init__(threshold)
        self.keep_ratio = keep_ratio
        self.name = f"AttentionBottom(keep={keep_ratio})"
    
    def evict(self, past_key_values, attention_scores=None):
        """Global eviction strategy"""
        all_k, all_v = zip(*past_key_values)
        
        T = all_k[0].shape[0]  # Sequence length
        
        # Number of tokens to keep
        keep_count = max(1, int(T * self.keep_ratio))
        
        # If we have attention scores, keep tokens with lowest scores
        if attention_scores is not None and len(attention_scores) == T:
            # Get indices of bottom-k tokens by importance
            _, indices = torch.topk(-attention_scores, keep_count)  # Negative to get lowest
            keep_indices = sorted(indices.tolist())
        else:
            # Fallback to keeping most recent tokens
            keep_indices = list(range(T - keep_count, T))
        
        # Apply eviction
        def prune(t):
            return t[keep_indices, ...]
        
        return tuple((prune(k), prune(v)) for k, v in past_key_values)
    
    def evict_layer(self, layer_kv, attention_scores=None, layer_idx=0, total_layers=12):
        """Layer-specific eviction strategy"""
        k, v = layer_kv
        T = k.shape[0]  # Sequence length
        
        # Number of tokens to keep
        keep_count = max(1, int(T * self.keep_ratio))
        
        # If we have attention scores, keep tokens with lowest scores
        if attention_scores is not None and len(attention_scores) == T:
            # Get indices of bottom-k tokens by importance
            _, indices = torch.topk(-attention_scores, keep_count)  # Negative to get lowest
            keep_indices = sorted(indices.tolist())
        else:
            # Fallback to keeping most recent tokens
            keep_indices = list(range(T - keep_count, T))
        
        # Apply eviction
        return k[keep_indices, ...], v[keep_indices, ...]

class HybridNPercentStrategy(EnhancedBaseStrategy):
    """
    Hybrid strategy that selects a percentage of tokens using a combination of
    recency, attention scores, and token type importance.
    """
    def __init__(self, threshold, keep_ratio=0.7, recency_weight=0.5, attention_weight=0.3, type_weight=0.2):
        super().__init__(threshold)
        self.keep_ratio = keep_ratio
        self.recency_weight = recency_weight
        self.attention_weight = attention_weight
        self.type_weight = type_weight
        self.name = f"HybridNPercent(keep={keep_ratio},r={recency_weight},a={attention_weight},t={type_weight})"
    
    def evict(self, past_key_values, attention_scores=None, token_types=None):
        """Global eviction strategy"""
        all_k, all_v = zip(*past_key_values)
        
        T = all_k[0].shape[0]  # Sequence length
        
        # Number of tokens to keep
        keep_count = max(1, int(T * self.keep_ratio))
        
        # Create a combined importance score
        importance = torch.zeros(T, device=all_k[0].device)
        
        # 1. Recency component - more recent tokens get higher scores
        recency_scores = torch.linspace(0, 1, T, device=all_k[0].device)
        importance += self.recency_weight * recency_scores
        
        # 2. Attention component - if available
        if attention_scores is not None and len(attention_scores) == T:
            # Normalize attention scores to [0, 1]
            norm_attention = attention_scores / attention_scores.max()
            importance += self.attention_weight * norm_attention
        
        # 3. Token type component - if available
        if token_types is not None and len(token_types) == T:
            type_scores = torch.zeros(T, device=all_k[0].device)
            for i, token_type in enumerate(token_types):
                if token_type == 'special':
                    type_scores[i] = 1.0  # Highest importance
                elif token_type == 'entity':
                    type_scores[i] = 0.8  # High importance
                else:
                    type_scores[i] = 0.5  # Medium importance
            importance += self.type_weight * type_scores
        
        # Get indices of top-k tokens by combined importance
        _, indices = torch.topk(importance, keep_count)
        keep_indices = sorted(indices.tolist())
        
        # Apply eviction
        def prune(t):
            return t[keep_indices, ...]
        
        return tuple((prune(k), prune(v)) for k, v in past_key_values)
    
    def evict_layer(self, layer_kv, attention_scores=None, token_types=None, layer_idx=0, total_layers=12):
        """Layer-specific eviction strategy"""
        k, v = layer_kv
        T = k.shape[0]  # Sequence length
        
        # Number of tokens to keep
        keep_count = max(1, int(T * self.keep_ratio))
        
        # Create a combined importance score
        importance = torch.zeros(T, device=k.device)
        
        # 1. Recency component - more recent tokens get higher scores
        recency_scores = torch.linspace(0, 1, T, device=k.device)
        importance += self.recency_weight * recency_scores
        
        # 2. Attention component - if available
        if attention_scores is not None and len(attention_scores) == T:
            # Normalize attention scores to [0, 1]
            norm_attention = attention_scores / attention_scores.max()
            importance += self.attention_weight * norm_attention
        
        # 3. Token type component - if available
        if token_types is not None and len(token_types) == T:
            type_scores = torch.zeros(T, device=k.device)
            for i, token_type in enumerate(token_types):
                if token_type == 'special':
                    type_scores[i] = 1.0  # Highest importance
                elif token_type == 'entity':
                    type_scores[i] = 0.8  # High importance
                else:
                    type_scores[i] = 0.5  # Medium importance
            importance += self.type_weight * type_scores
        
        # Get indices of top-k tokens by combined importance
        _, indices = torch.topk(importance, keep_count)
        keep_indices = sorted(indices.tolist())
        
        # Apply eviction
        return k[keep_indices, ...], v[keep_indices, ...]
