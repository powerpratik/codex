# enhanced_strategies.py
import torch
import random
import numpy as np
from abc import ABC, abstractmethod

class EnhancedBaseStrategy(ABC):
    def __init__(self, threshold):
        self.threshold = threshold
        self.name = self.__class__.__name__
    
    @abstractmethod
    def evict(self, past_key_values, attention_scores=None):
        """Global eviction strategy"""
        pass
    
    @abstractmethod
    def evict_layer(self, layer_kv, attention_scores=None, layer_idx=0, total_layers=12):
        """Layer-specific eviction strategy"""
        pass
    
    def get_stats(self):
        """Return strategy-specific stats"""
        return {"name": self.name, "threshold": self.threshold}

class EnhancedBaseline(EnhancedBaseStrategy):
    """Baseline strategy that does no eviction"""
    def evict(self, past_key_values, attention_scores=None):
        return past_key_values
    
    def evict_layer(self, layer_kv, attention_scores=None, layer_idx=0, total_layers=12):
        return layer_kv

class SlidingWindowStrategy(EnhancedBaseStrategy):
    """
    Sliding window strategy that keeps only the most recent tokens 
    plus a few important ones from history
    """
    def __init__(self, threshold, window_size=0.7, important_ratio=0.1):
        super().__init__(threshold)
        self.window_size = window_size  # Percentage of recent tokens to keep
        self.important_ratio = important_ratio  # Percentage of important historical tokens to keep
        
    def evict(self, past_key_values, attention_scores=None):
        # Global eviction using sliding window
        all_k, all_v = zip(*past_key_values)
        
        T = all_k[0].shape[0]  # Sequence length
        
        # Number of recent tokens to keep
        recent_count = max(1, int(T * self.window_size))
        
        # Always keep the most recent tokens
        keep_indices = list(range(T - recent_count, T))
        
        # If we have attention scores, also keep some important historical tokens
        if attention_scores is not None and len(attention_scores) == T:
            historical_indices = list(range(0, T - recent_count))
            
            # Only keep important_ratio of historical tokens
            important_count = max(1, int(len(historical_indices) * self.important_ratio))
            
            # Get scores for historical tokens only
            historical_scores = attention_scores[historical_indices]
            
            # Find the most important historical tokens
            if len(historical_scores) > 0:
                top_indices = torch.topk(historical_scores, min(important_count, len(historical_scores))).indices
                important_historical = [historical_indices[i] for i in top_indices.tolist()]
                keep_indices.extend(important_historical)
        
        # Sort indices
        keep_indices = sorted(set(keep_indices))
        
        # Apply eviction
        def prune(t):
            return t[keep_indices, ...]
        
        return tuple((prune(k), prune(v)) for k, v in past_key_values)
    
    def evict_layer(self, layer_kv, attention_scores=None, layer_idx=0, total_layers=12):
        """Layer-specific sliding window with adaptive ratios"""
        k, v = layer_kv
        T = k.shape[0]  # Sequence length
        
        # Adjust window size based on layer position
        # Later layers focus more on recent tokens, earlier layers preserve more context
        layer_progress = layer_idx / max(1, total_layers - 1)
        adjusted_window = self.window_size * (1 + 0.3 * layer_progress)  # Gradually increase for later layers
        adjusted_important = self.important_ratio * (1 - 0.5 * layer_progress)  # Gradually decrease for later layers
        
        # Number of recent tokens to keep
        recent_count = max(1, int(T * adjusted_window))
        
        # Always keep the most recent tokens
        keep_indices = list(range(T - recent_count, T))
        
        # If we have attention scores, also keep some important historical tokens
        if attention_scores is not None and len(attention_scores) == T:
            historical_indices = list(range(0, T - recent_count))
            
            # Only keep important_ratio of historical tokens
            important_count = max(1, int(len(historical_indices) * adjusted_important))
            
            # Get scores for historical tokens only
            historical_scores = attention_scores[historical_indices]
            
            # Find the most important historical tokens
            if len(historical_scores) > 0:
                top_indices = torch.topk(historical_scores, min(important_count, len(historical_scores))).indices
                important_historical = [historical_indices[i] for i in top_indices.tolist()]
                keep_indices.extend(important_historical)
        
        # Sort indices
        keep_indices = sorted(set(keep_indices))
        
        # Apply eviction
        return k[keep_indices, ...], v[keep_indices, ...]

class AdaptiveAttentionStrategy(EnhancedBaseStrategy):
    """
    Adaptive strategy that adjusts eviction based on token patterns
    """
    def __init__(self, threshold, base_keep=0.7):
        super().__init__(threshold)
        self.base_keep = base_keep
        self.token_history = []  # Track patterns of important tokens
        
    def evict(self, past_key_values, attention_scores=None):
        all_k, all_v = zip(*past_key_values)
        T = all_k[0].shape[0]  # Sequence length
        
        if attention_scores is None or len(attention_scores) != T:
            # Fallback to standard approach if no attention scores
            keep_count = max(1, int(T * self.base_keep))
            keep_indices = sorted(random.sample(range(T), keep_count))
        else:
            # Normalize scores
            norm_scores = attention_scores / attention_scores.sum()
            
            # Detect different token patterns
            # Check variance in attention - high variance means some tokens are much more important
            score_variance = attention_scores.var().item()
            
            # Adjust eviction based on variance
            if score_variance > 0.01:  # High variance - be more selective
                keep_ratio = max(0.4, min(0.9, self.base_keep - 0.2))
            else:  # Low variance - more uniform importance
                keep_ratio = min(0.95, self.base_keep + 0.1)
            
            keep_count = max(1, int(T * keep_ratio))
            
            # Get top-k tokens by importance
            keep_indices = torch.topk(attention_scores, keep_count).indices.tolist()
            
            # Always ensure we keep some recent tokens for coherence
            recent_count = min(50, T // 4)
            recent_indices = list(range(T - recent_count, T))
            keep_indices = sorted(set(keep_indices + recent_indices))
            
            # Update token history for future pattern recognition
            self.token_history.append((keep_indices, norm_scores.tolist()))
            if len(self.token_history) > 10:
                self.token_history.pop(0)
        
        # Apply eviction
        def prune(t):
            return t[keep_indices, ...]
        
        return tuple((prune(k), prune(v)) for k, v in past_key_values)
    
    def evict_layer(self, layer_kv, attention_scores=None, layer_idx=0, total_layers=12):
        """Layer-specific adaptive strategy"""
        k, v = layer_kv
        T = k.shape[0]  # Sequence length
        
        # Early layers need more context, later layers more focused
        layer_position = layer_idx / max(1, total_layers - 1)
        
        # Adjust base keep ratio based on layer position
        layer_keep = self.base_keep - (0.2 * layer_position)
        
        if attention_scores is None or len(attention_scores) != T:
            # Fallback to standard approach if no attention scores
            keep_count = max(1, int(T * layer_keep))
            keep_indices = sorted(random.sample(range(T), keep_count))
        else:
            # Normalize scores
            norm_scores = attention_scores / attention_scores.sum()
            
            # Detect different token patterns
            score_variance = attention_scores.var().item()
            
            # Adjust eviction based on variance
            if score_variance > 0.01:  # High variance - be more selective
                keep_ratio = max(0.4, min(0.9, layer_keep - 0.2))
            else:  # Low variance - more uniform importance
                keep_ratio = min(0.95, layer_keep + 0.1)
            
            keep_count = max(1, int(T * keep_ratio))
            
            # Get top-k tokens by importance
            keep_indices = torch.topk(attention_scores, keep_count).indices.tolist()
            
            # Always ensure we keep some recent tokens for coherence
            recent_count = min(50, T // 4)
            recent_indices = list(range(T - recent_count, T))
            keep_indices = sorted(set(keep_indices + recent_indices))
        
        # Apply eviction
        return k[keep_indices, ...], v[keep_indices, ...]