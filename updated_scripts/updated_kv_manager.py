# -*- coding: utf-8 -*-
"""
kv_cache_manager.py

Provides a physical implementation for managing key-value (KV) cache
during Transformer-based language model inference. This allows users
to plug in different eviction strategies (e.g., window, ratio-based) and
interact with the KV cache in-memory for testing or debugging.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
import math # Added for ceiling calculation

# Define a type alias for a single layer's key/value pair
KeyValue = Tuple[torch.Tensor, torch.Tensor]

def unpack_kv(past_kvs) -> List[KeyValue]: # noqa
    """
    Convert model past_key_values to a list of (key, value) tuples.
    Supports HF Cache API (DynamicCache) and legacy tuple/lists.
    """
    # HF new Cache API: subclass of Cache with to_legacy_cache()
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_kvs, Cache):
            legacy = past_kvs.to_legacy_cache()
            # Ensure output is list of tuples
            if legacy is not None:
                return [(k, v) for k, v in legacy]
            else:
                # Handle cases where the cache might be empty or uninitialized
                return []
    except ImportError:
        pass # Fallback to legacy below

    # legacy API: tuple or list of (key, value)
    if past_kvs is None:
        return []
    return [(k, v) for k, v in past_kvs]


def pack_kv(kv_list: List[KeyValue]): # noqa
    """
    Convert a list of (key, value) tuples back to model past_key_values API.
    Uses HF DynamicCache.from_legacy_cache if available, else falls back to tuple of tuples.
    """
    if not kv_list: # Handle empty list case
        return None

    try:
        from transformers.cache_utils import DynamicCache
        # Check if the first element is a tuple of tensors before creating legacy tuple
        if kv_list and isinstance(kv_list[0], tuple) and len(kv_list[0]) == 2:
             legacy = tuple((k, v) for k, v in kv_list)
             # Create dummy tensors if needed by DynamicCache for empty cache init
             # Note: This might need adjustment based on specific HF versions or models
             # For now, assuming from_legacy_cache handles empty/None correctly if kv_list is empty.
             if legacy:
                 return DynamicCache.from_legacy_cache(legacy)
             else:
                 return DynamicCache() # Return an empty DynamicCache if list was empty
        else:
             # Fallback or handle unexpected structure
             return tuple((k, v) for k, v in kv_list)


    except ImportError:
        # fallback: legacy format
        return tuple((k, v) for k, v in kv_list)


class EvictionStrategy(ABC):
    """
    Abstract base class for cache eviction strategies.
    """

    @abstractmethod
    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]: # pragma: no cover
        """
        Given a list of past (key, value) tensors for each layer,
        return a new list after applying the eviction policy.
        Also returns the original sequence length and the kept sequence length.
        """
        pass

class NoOpStrategy(EvictionStrategy):
    """
    Eviction strategy that does not remove any cached entries.
    """
    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
             return [], 0, 0
        seq_len = past_kvs[0][0].size(-2)
        return past_kvs, seq_len, seq_len

class WindowStrategy(EvictionStrategy):
    """
    Eviction strategy that retains only the most recent `max_length` tokens
    in the cache for each layer.
    """
    def __init__(self, max_length: int):
        if max_length <= 0:
             raise ValueError("max_length must be positive")
        self.max_length = max_length

    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
             return [], 0, 0

        new_past: List[KeyValue] = []
        # Get original sequence length from the first layer (assume all layers have same length)
        seq_len = past_kvs[0][0].size(-2)
        kept_len = seq_len # Default if no pruning happens

        for key, value in past_kvs:
            # key, value shape: (batch_size, num_heads, seq_len, head_dim)
            current_seq_len = key.size(-2)
            if current_seq_len > self.max_length:
                # keep only the last `max_length` positions
                keep_indices = torch.arange(current_seq_len - self.max_length, current_seq_len, device=key.device)
                key = torch.index_select(key, -2, keep_indices)
                value = torch.index_select(value, -2, keep_indices)
                kept_len = self.max_length # Update kept_len
            new_past.append((key, value))

        return new_past, seq_len, kept_len


# --- New Ratio-Based Strategies ---

class TopRatioStrategy(EvictionStrategy):
    """
    Keeps the first N tokens based on a ratio, plus the most recent token.
    """
    def __init__(self, pruning_ratio: float):
        if not 0.0 <= pruning_ratio <= 1.0:
            raise ValueError("pruning_ratio must be between 0.0 and 1.0")
        self.pruning_ratio = pruning_ratio

    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
            return [], 0, 0

        new_past: List[KeyValue] = []
        seq_len = past_kvs[0][0].size(-2)

        if seq_len <= 1: # Cannot prune if only one token
             return past_kvs, seq_len, seq_len

        # Calculate how many tokens to keep from the beginning (excluding the last one)
        # Ensure at least one token is kept if ratio > 0, otherwise keep only the last.
        num_to_keep_from_past = math.ceil((seq_len - 1) * self.pruning_ratio)
        num_to_keep_total = num_to_keep_from_past + 1 # Add the last token

        if num_to_keep_total >= seq_len: # No pruning needed if keeping all or more
             return past_kvs, seq_len, seq_len

        kept_len = num_to_keep_total
        first_indices = torch.arange(num_to_keep_from_past, device=past_kvs[0][0].device)
        last_index = torch.tensor([seq_len - 1], device=past_kvs[0][0].device)
        indices_to_keep = torch.cat([first_indices, last_index])

        for key, value in past_kvs:
            pruned_key = torch.index_select(key, -2, indices_to_keep)
            pruned_value = torch.index_select(value, -2, indices_to_keep)
            new_past.append((pruned_key, pruned_value))

        return new_past, seq_len, kept_len


class BottomRatioStrategy(EvictionStrategy):
    """
    Keeps the last N tokens based on a ratio, plus the most recent token
    (which is inherently included in the 'last N'). Equivalent to WindowStrategy
    if window size is calculated from ratio.
    """
    def __init__(self, pruning_ratio: float):
        if not 0.0 <= pruning_ratio <= 1.0:
            raise ValueError("pruning_ratio must be between 0.0 and 1.0")
        self.pruning_ratio = pruning_ratio

    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
            return [], 0, 0

        new_past: List[KeyValue] = []
        seq_len = past_kvs[0][0].size(-2)

        if seq_len <= 1: # Cannot prune if only one token
             return past_kvs, seq_len, seq_len

        # Calculate how many tokens to keep in total (including the last one)
        num_to_keep_total = math.ceil(seq_len * self.pruning_ratio)
        num_to_keep_total = max(1, num_to_keep_total) # Ensure at least the last token is kept

        if num_to_keep_total >= seq_len: # No pruning needed
            return past_kvs, seq_len, seq_len

        kept_len = num_to_keep_total
        start_index = seq_len - num_to_keep_total
        indices_to_keep = torch.arange(start_index, seq_len, device=past_kvs[0][0].device)

        for key, value in past_kvs:
            pruned_key = torch.index_select(key, -2, indices_to_keep)
            pruned_value = torch.index_select(value, -2, indices_to_keep)
            new_past.append((pruned_key, pruned_value))

        return new_past, seq_len, kept_len


class BothRatioStrategy(EvictionStrategy):
    """
    Keeps N/2 tokens from the start and N/2 tokens from the end, based on a ratio N,
    plus the most recent token (ensuring it's included).
    """
    def __init__(self, pruning_ratio: float):
        if not 0.0 <= pruning_ratio <= 1.0:
            raise ValueError("pruning_ratio must be between 0.0 and 1.0")
        self.pruning_ratio = pruning_ratio

    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
            return [], 0, 0

        new_past: List[KeyValue] = []
        seq_len = past_kvs[0][0].size(-2)

        if seq_len <= 1: # Cannot prune if only one token
            return past_kvs, seq_len, seq_len

        # Calculate total tokens to keep (excluding the last one initially)
        num_to_keep_from_past = math.ceil((seq_len - 1) * self.pruning_ratio)

        if num_to_keep_from_past >= seq_len - 1: # Keeping all past tokens + the last one
            return past_kvs, seq_len, seq_len

        # Split keepers between top and bottom
        num_top = num_to_keep_from_past // 2
        num_bottom = num_to_keep_from_past - num_top

        top_indices = torch.arange(num_top, device=past_kvs[0][0].device)

        # Bottom indices need to exclude the very last token (index seq_len - 1) for now
        bottom_start_index = (seq_len - 1) - num_bottom
        bottom_indices = torch.arange(bottom_start_index, seq_len - 1, device=past_kvs[0][0].device)

        # Always include the last token
        last_index = torch.tensor([seq_len - 1], device=past_kvs[0][0].device)

        # Combine, remove duplicates (in case bottom includes last_index range), and sort
        indices_to_keep = torch.cat([top_indices, bottom_indices, last_index])
        indices_to_keep = torch.unique(indices_to_keep) # Handles overlap if num_bottom is large
        indices_to_keep = torch.sort(indices_to_keep).values
        kept_len = len(indices_to_keep)

        for key, value in past_kvs:
            pruned_key = torch.index_select(key, -2, indices_to_keep)
            pruned_value = torch.index_select(value, -2, indices_to_keep)
            new_past.append((pruned_key, pruned_value))

        return new_past, seq_len, kept_len


# --- End New Strategies ---

class RandomSamplingStrategy(EvictionStrategy):
    """
    Eviction strategy that randomly samples up to `max_length` tokens
    from the cache for each layer, always keeping the most recent token.
    """
    def __init__(self, max_length: int, seed: Optional[int] = None):
        if max_length <= 0:
             raise ValueError("max_length must be positive")
        self.max_length = max_length
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
             return [], 0, 0

        new_past: List[KeyValue] = []
        seq_len = past_kvs[0][0].size(-2)
        kept_len = seq_len

        if seq_len > self.max_length:
             kept_len = self.max_length
             num_to_sample = self.max_length - 1 # Need to sample N-1 from the past
             if num_to_sample > 0:
                 # Sample from indices 0 to seq_len-2
                 perm = torch.randperm(seq_len - 1, generator=self.generator, device=past_kvs[0][0].device)
                 past_indices = perm[:num_to_sample]
                 # Include the last index
                 last_index = torch.tensor([seq_len - 1], device=past_kvs[0][0].device)
                 idx = torch.cat([past_indices, last_index])
                 idx, _ = torch.sort(idx) # Sort to approximately preserve order
             else: # Only keep the last token
                 idx = torch.tensor([seq_len - 1], device=past_kvs[0][0].device)

             for key, value in past_kvs:
                 key = key[..., idx, :]
                 value = value[..., idx, :]
                 new_past.append((key, value))
        else:
            # No pruning needed
            new_past = past_kvs

        return new_past, seq_len, kept_len


class StridedStrategy(EvictionStrategy):
    """
    Eviction strategy that keeps every N-th token up to `max_length` tokens,
    sampling uniformly across the sequence, always including the last token.
    """
    def __init__(self, max_length: int):
        if max_length <= 0:
             raise ValueError("max_length must be positive")
        self.max_length = max_length

    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
             return [], 0, 0

        new_past: List[KeyValue] = []
        seq_len = past_kvs[0][0].size(-2)
        kept_len = seq_len

        if seq_len > self.max_length:
             kept_len = self.max_length
             # generate uniformly spaced indices from 0 to seq_len-2
             # We want max_length-1 indices from the past
             indices = torch.linspace(0, seq_len - 2, steps=self.max_length - 1, device=past_kvs[0][0].device).long()
             last_index = torch.tensor([seq_len - 1], device=past_kvs[0][0].device)
             idx = torch.cat([indices, last_index])
             idx = torch.unique(idx) # Ensure last index isn't duplicated
             idx, _ = torch.sort(idx) # Maintain order

             # Adjust kept_len if unique operation reduced size
             kept_len = len(idx)

             for key, value in past_kvs:
                 key = key[..., idx, :]
                 value = value[..., idx, :]
                 new_past.append((key, value))
        else:
            new_past = past_kvs

        return new_past, seq_len, kept_len

class BlockAverageStrategy(EvictionStrategy):
    """
    Eviction strategy that divides the sequence into blocks of size `block_size`
    and averages keys/values within each block, reducing sequence length.
    WARNING: This modifies the actual key/value content, not just selects tokens.
    """
    def __init__(self, block_size: int):
        if block_size <= 0:
             raise ValueError("block_size must be positive")
        self.block_size = block_size

    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
             return [], 0, 0

        new_past: List[KeyValue] = []
        seq_len = past_kvs[0][0].size(-2)
        kept_len = seq_len # Initial assumption

        for key, value in past_kvs:
            # key/value shape: (batch_size, num_heads, seq_len, head_dim)
            current_seq_len = key.size(-2)
            if current_seq_len > self.block_size:
                num_blocks = math.ceil(current_seq_len / self.block_size)
                k_blocks = []
                v_blocks = []

                for b in range(num_blocks):
                    start = b * self.block_size
                    end = min((b + 1) * self.block_size, current_seq_len)
                    # average over the token dimension
                    k_blk = key[..., start:end, :].mean(dim=-2, keepdim=True)
                    v_blk = value[..., start:end, :].mean(dim=-2, keepdim=True)
                    k_blocks.append(k_blk)
                    v_blocks.append(v_blk)

                key = torch.cat(k_blocks, dim=-2)
                value = torch.cat(v_blocks, dim=-2)
                kept_len = key.size(-2) # Update kept_len after averaging

            new_past.append((key, value))

        return new_past, seq_len, kept_len


class AttentionScoreStrategy(EvictionStrategy):
    """
    Eviction strategy that selects tokens with highest attention-like scores
    with respect to the most recent token’s key vector. Always retains
    the newest token and picks the top (max_length-1) prior tokens by score.
    """
    def __init__(self, max_length: int):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length = max_length

    def evict(self, past_kvs: List[KeyValue]) -> Tuple[List[KeyValue], int, int]:
        if not past_kvs:
            return [], 0, 0

        new_past: List[KeyValue] = []
        seq_len = past_kvs[0][0].size(-2)
        kept_len = seq_len

        if seq_len > self.max_length:
            kept_len = self.max_length
            num_to_keep_from_past = self.max_length - 1

            # Use scores from the first layer only for simplicity, assuming similar patterns
            # In a real scenario, might average scores across layers or use other heuristics
            key_l0, value_l0 = past_kvs[0]
            batch, heads, _, head_dim = key_l0.shape

            if batch != 1:
                # Keep original behavior for simplicity, but could average over batch
                print("Warning: AttentionScoreStrategy performance degrades for batch_size > 1, using first batch item.")
                # Or raise ValueError("AttentionScoreStrategy currently supports batch_size=1 only")

            # Query vector: last token from the first batch item
            q = key_l0[0, :, -1:, :] # Shape: (heads, 1, head_dim)

            # Past keys: all except last from the first batch item
            k_past = key_l0[0, :, :-1, :] # Shape: (heads, seq_len-1, head_dim)

            # Compute dot-product scores: (heads, seq_len-1)
            # Permute q to (heads, head_dim, 1) for batch matmul
            scores = torch.matmul(k_past, q.permute(0, 2, 1)).squeeze(-1) # Shape: (heads, seq_len-1)

            # Aggregate over heads: (seq_len-1,)
            scores_agg = scores.mean(dim=0)

            idx = []
            if num_to_keep_from_past > 0:
                # Select top indices from past tokens
                topk_indices = torch.topk(scores_agg, k=min(num_to_keep_from_past, scores_agg.shape[0]), largest=True).indices
                idx.append(topk_indices)

            # Include the last index (newest token)
            last_idx = torch.tensor([seq_len - 1], device=key_l0.device, dtype=torch.long) # Ensure long dtype
            idx.append(last_idx)

            # Combine, sort, and apply to all layers
            idx_full = torch.cat(idx)
            idx_full = torch.unique(idx_full) # Should not be needed if num_to_keep_from_past < seq_len-1
            idx_full, _ = torch.sort(idx_full)

            # Adjust kept_len based on actual unique indices kept
            kept_len = len(idx_full)

            for key, value in past_kvs:
                 # Apply indices to all items in the batch dim if batch > 1
                 pruned_key = torch.index_select(key, -2, idx_full)
                 pruned_value = torch.index_select(value, -2, idx_full)
                 new_past.append((pruned_key, pruned_value))

        else: # No pruning needed
             new_past = past_kvs

        return new_past, seq_len, kept_len


class KVCacheManager:
    """
    Manager for Transformer KV cache. Users can update the cache with
    new `past_key_values` from the model, and eviction strategies will
    be applied automatically.
    """
    def __init__(self, strategy: EvictionStrategy):
        self.strategy = strategy
        # self.cache: Optional[List[KeyValue]] = None # Internal cache state not needed if update returns directly

    def update(self, past_kvs_from_model) -> Tuple[List[KeyValue], int, int]:
        """
        Update the internal cache with new past_key_values from the model,
        then apply the eviction strategy.

        Handles unpacking the model output format.

        Returns:
            - The pruned cache list (suitable for next model input).
            - Original sequence length before eviction.
            - Kept sequence length after eviction.
        """
        unpacked_kv_list = unpack_kv(past_kvs_from_model)
        pruned_kv_list, orig_len, kept_len = self.strategy.evict(unpacked_kv_list)
        # self.cache = pruned_kv_list # Update internal state if needed elsewhere
        return pruned_kv_list, orig_len, kept_len

    # get_cache and clear are less relevant if update doesn't maintain internal state
    # but can be kept if direct access is sometimes needed.
    # def get_cache(self) -> Optional[List[KeyValue]]:
    #     """
    #     Get the current cached key/value pairs (one tuple per layer).
    #     """
    #     return self.cache

    # def clear(self) -> None:
    #     """
    #     Clear the cache completely.
    #     """
    #     self.cache = None


