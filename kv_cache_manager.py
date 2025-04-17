"""
kv_cache_manager.py

Provides a physical implementation for managing key-value (KV) cache
during Transformer-based language model inference. This allows users
to plug in different eviction strategies (e.g., window, LRU) and
interact with the KV cache in-memory for testing or debugging.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

# Define a type alias for a single layer's key/value pair
KeyValue = Tuple[torch.Tensor, torch.Tensor]

def unpack_kv(past_kvs) -> List[KeyValue]:  # noqa
    """
    Convert model past_key_values to a list of (key, value) tuples.
    Supports HF Cache API (DynamicCache) and legacy tuple/lists.
    """
    # HF new Cache API: subclass of Cache with to_legacy_cache()
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_kvs, Cache):
            legacy = past_kvs.to_legacy_cache()
            return [(k, v) for (k, v) in legacy]
    except ImportError:
        pass
    # legacy API: tuple or list of (key, value)
    return [(k, v) for (k, v) in past_kvs]

def pack_kv(kv_list: List[KeyValue]):  # noqa
    """
    Convert a list of (key, value) tuples back to model past_key_values API.
    Uses HF DynamicCache.from_legacy_cache if available, else falls back to tuple of tuples.
    """
    try:
        from transformers.cache_utils import DynamicCache
        legacy = tuple((k, v) for (k, v) in kv_list)
        return DynamicCache.from_legacy_cache(legacy)
    except ImportError:
        # fallback: legacy format
        return tuple((k, v) for (k, v) in kv_list)

class EvictionStrategy(ABC):
    """
    Abstract base class for cache eviction strategies.
    """
    @abstractmethod
    def evict(self, past_kvs: List[KeyValue]) -> List[KeyValue]:  # pragma: no cover
        """
        Given a list of past (key, value) tensors for each layer,
        return a new list after applying the eviction policy.
        """
        pass

class NoOpStrategy(EvictionStrategy):
    """
    Eviction strategy that does not remove any cached entries.
    """
    def evict(self, past_kvs: List[KeyValue]) -> List[KeyValue]:
        return past_kvs

class WindowStrategy(EvictionStrategy):
    """
    Eviction strategy that retains only the most recent `max_length` tokens
    in the cache for each layer.
    """
    def __init__(self, max_length: int):
        self.max_length = max_length

    def evict(self, past_kvs: List[KeyValue]) -> List[KeyValue]:
        new_past: List[KeyValue] = []
        for key, value in past_kvs:
            # key, value shape: (batch_size, num_heads, seq_len, head_dim)
            seq_len = key.size(-2)
            if seq_len > self.max_length:
                # keep only the last `max_length` positions
                key = key[..., -self.max_length :, :]
                value = value[..., -self.max_length :, :]
            new_past.append((key, value))
        return new_past
 
class RandomSamplingStrategy(EvictionStrategy):
    """
    Eviction strategy that randomly samples up to `max_length` tokens
    from the cache for each layer.
    """
    def __init__(self, max_length: int, seed: Optional[int] = None):
        self.max_length = max_length
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None

    def evict(self, past_kvs: List[KeyValue]) -> List[KeyValue]:
        new_past: List[KeyValue] = []
        for key, value in past_kvs:
            # key/value shape: (batch_size, num_heads, seq_len, head_dim)
            seq_len = key.size(-2)
            if seq_len > self.max_length:
                # sample random indices and sort to preserve order
                idx = torch.randperm(seq_len, generator=self.generator)[: self.max_length]
                idx, _ = torch.sort(idx)
                key = key[..., idx, :]
                value = value[..., idx, :]
            new_past.append((key, value))
        return new_past

class StridedStrategy(EvictionStrategy):
    """
    Eviction strategy that keeps every N-th token up to `max_length` tokens,
    sampling uniformly across the sequence.
    """
    def __init__(self, max_length: int):
        self.max_length = max_length

    def evict(self, past_kvs: List[KeyValue]) -> List[KeyValue]:
        new_past: List[KeyValue] = []
        for key, value in past_kvs:
            seq_len = key.size(-2)
            if seq_len > self.max_length:
                # generate uniformly spaced indices
                idx = torch.linspace(0, seq_len - 1, steps=self.max_length).long()
                key = key[..., idx, :]
                value = value[..., idx, :]
            new_past.append((key, value))
        return new_past

class BlockAverageStrategy(EvictionStrategy):
    """
    Eviction strategy that divides the sequence into blocks of size `block_size`
    and averages keys/values within each block, reducing sequence length.
    """
    def __init__(self, block_size: int):
        self.block_size = block_size

    def evict(self, past_kvs: List[KeyValue]) -> List[KeyValue]:
        new_past: List[KeyValue] = []
        for key, value in past_kvs:
            # key/value shape: (batch_size, num_heads, seq_len, head_dim)
            seq_len = key.size(-2)
            if seq_len > self.block_size:
                num_blocks = (seq_len + self.block_size - 1) // self.block_size
                k_blocks = []
                v_blocks = []
                for b in range(num_blocks):
                    start = b * self.block_size
                    end = min((b + 1) * self.block_size, seq_len)
                    # average over the token dimension
                    k_blk = key[..., start:end, :].mean(dim=-2, keepdim=True)
                    v_blk = value[..., start:end, :].mean(dim=-2, keepdim=True)
                    k_blocks.append(k_blk)
                    v_blocks.append(v_blk)
                key = torch.cat(k_blocks, dim=-2)
                value = torch.cat(v_blocks, dim=-2)
            new_past.append((key, value))
        return new_past
 
class AttentionScoreStrategy(EvictionStrategy):
    """
    Eviction strategy that selects tokens with highest attention-like scores
    with respect to the most recent token’s key vector. Always retains
    the newest token and picks the top (max_length-1) prior tokens by score.
    """
    def __init__(self, max_length: int):
        self.max_length = max_length

    def evict(self, past_kvs: List[KeyValue]) -> List[KeyValue]:
        import torch
        new_past: List[KeyValue] = []
        for key, value in past_kvs:
            # key/value shape: (batch_size, num_heads, seq_len, head_dim)
            batch, heads, seq_len, head_dim = key.shape
            if seq_len > self.max_length:
                # query vector: last token
                q = key[..., -1:, :]
                # past keys: all except last
                k_past = key[..., :-1, :]
                # compute dot-product scores: (batch, heads, seq_len-1)
                scores = (k_past * q).sum(dim=-1)
                # aggregate over heads: (batch, seq_len-1)
                scores = scores.mean(dim=1)
                # only supports batch_size=1 for now
                if batch != 1:
                    raise ValueError("AttentionScoreStrategy supports batch_size=1 only")
                # select top (max_length-1) indices from past tokens
                sc = scores[0]  # (seq_len-1,)
                k_keep = self.max_length - 1
                topk = torch.topk(sc, k=k_keep, largest=True).indices
                idx = torch.sort(topk).values
                # include the last index (newest token)
                last_idx = torch.tensor([seq_len - 1], device=key.device, dtype=idx.dtype)
                idx_full = torch.cat([idx, last_idx])
                idx_full, _ = torch.sort(idx_full)
                key = key[..., idx_full, :]
                value = value[..., idx_full, :]
            new_past.append((key, value))
        return new_past

class KVCacheManager:
    """
    Manager for Transformer KV cache. Users can update the cache with
    new `past_key_values` from the model, and eviction strategies will
    be applied automatically.
    """
    def __init__(self, strategy: EvictionStrategy):
        self.strategy = strategy
        self.cache: Optional[List[KeyValue]] = None

    def update(self, past_kvs: List[KeyValue]) -> List[KeyValue]:
        """
        Update the internal cache with new past_key_values from the model,
        then apply the eviction strategy.
        Returns the pruned cache list.
        """
        self.cache = self.strategy.evict(past_kvs)
        return self.cache

    def get_cache(self) -> Optional[List[KeyValue]]:
        """
        Get the current cached key/value pairs (one tuple per layer).
        """
        return self.cache

    def clear(self) -> None:
        """
        Clear the cache completely.
        """
        self.cache = None