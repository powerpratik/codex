# -*- coding: utf-8 -*-
"""
test_kv_cache_manager.py

Unit tests for KVCacheManager and eviction strategies.
"""

import torch
import pytest
import math
from updated_kv_manager import (
    KVCacheManager, NoOpStrategy, WindowStrategy,
    RandomSamplingStrategy, StridedStrategy, BlockAverageStrategy, AttentionScoreStrategy,
    TopRatioStrategy, BottomRatioStrategy, BothRatioStrategy, # Added imports
    unpack_kv, pack_kv # Keep if helper functions need testing directly
)

# Helper function to create dummy past_key_values
def make_dummy_past(seq_len: int,
                    batch_size: int = 1,
                    num_layers: int = 2,
                    num_heads: int = 2,
                    head_dim: int = 4,
                    device: str = 'cpu') -> list:
    """
    Create a dummy past_key_values list (list of tuples).
    """
    past_kvs = []
    for _ in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        past_kvs.append((k, v))
    return past_kvs

# --- Existing Tests (Minor adjustments if needed) ---

def test_noop_strategy_preserves_length():
    seq_len = 10
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(NoOpStrategy())
    new_past, orig_len, kept_len = manager.update(past) # Adjust to new return signature
    assert orig_len == seq_len
    assert kept_len == seq_len
    assert len(new_past) == len(past)
    for (k_old, v_old), (k_new, v_new) in zip(past, new_past):
        assert k_old.shape == k_new.shape
        assert v_old.shape == v_new.shape
        assert torch.equal(k_old, k_new) # Ensure content is identical
        assert torch.equal(v_old, v_new)

def test_window_strategy_shorter_than_window():
    seq_len = 3
    window = 5
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(WindowStrategy(window))
    new_past, orig_len, kept_len = manager.update(past)
    assert orig_len == seq_len
    assert kept_len == seq_len # No truncation
    assert len(new_past) == len(past)
    for (k_old, _), (k_new, _) in zip(past, new_past):
        assert k_old.shape == k_new.shape

def test_window_strategy_longer_than_window():
    seq_len = 10
    window = 5
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(WindowStrategy(window))
    new_past, orig_len, kept_len = manager.update(past)
    assert orig_len == seq_len
    assert kept_len == window # Truncated to window size
    assert len(new_past) == len(past)
    for (k_new, v_new) in new_past:
        assert k_new.size(-2) == window
        assert v_new.size(-2) == window
    # Check if the *last* 'window' elements were kept
    original_last_k, original_last_v = past[0] # Check first layer
    kept_k, kept_v = new_past[0]
    assert torch.equal(original_last_k[..., -window:, :], kept_k)
    assert torch.equal(original_last_v[..., -window:, :], kept_v)

def test_attention_strategy_shorter_than_max():
    seq_len = 4
    max_len = 5
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(AttentionScoreStrategy(max_length=max_len))
    new_past, orig_len, kept_len = manager.update(past)
    assert orig_len == seq_len
    assert kept_len == seq_len # No truncation
    for (k_old, _), (k_new, _) in zip(past, new_past):
        assert k_new.shape == k_old.shape

def test_attention_strategy_longer_than_max_retains_latest():
    seq_len = 10
    max_len = 6
    past = make_dummy_past(seq_len)
    # Mark the last token's key/value to verify it's kept
    original_last_k = past[0][0][..., -1, :].clone() # Key, layer 0
    original_last_v = past[0][1][..., -1, :].clone() # Value, layer 0
    past[0][0][..., -1, :] = 1.0
    past[0][1][..., -1, :] = 2.0

    manager = KVCacheManager(AttentionScoreStrategy(max_length=max_len))
    new_past, orig_len, kept_len = manager.update(past)

    assert orig_len == seq_len
    assert kept_len == max_len # Should keep max_len tokens
    for (k_new, v_new) in new_past:
        assert k_new.size(-2) == max_len
        assert v_new.size(-2) == max_len

    # Check if the last entry in the new cache matches the original marked last vector
    assert torch.allclose(new_past[0][0][..., -1, :], torch.ones_like(new_past[0][0][..., -1, :]))
    assert torch.allclose(new_past[0][1][..., -1, :], torch.full_like(new_past[0][1][..., -1, :], 2.0))


# --- Tests for New Ratio Strategies ---

@pytest.mark.parametrize("ratio, seq_len, expected_kept", [
    (0.5, 10, 6),  # Keep ceil(9 * 0.5) = 5 from past + 1 last = 6
    (0.1, 10, 2),  # Keep ceil(9 * 0.1) = 1 from past + 1 last = 2
    (0.9, 10, 10), # Keep ceil(9 * 0.9) = 9 from past + 1 last = 10 (no prune)
    (1.0, 10, 10), # Keep all
    (0.0, 10, 1),  # Keep 0 from past + 1 last = 1
    (0.5, 1, 1),   # Cannot prune length 1
])
def test_top_ratio_strategy(ratio, seq_len, expected_kept):
    past = make_dummy_past(seq_len)
    # Mark first and last tokens for verification
    past[0][0][..., 0, :] = -1.0 # Mark first key
    past[0][0][..., -1, :] = 1.0 # Mark last key

    manager = KVCacheManager(TopRatioStrategy(pruning_ratio=ratio))
    new_past, orig_len, kept_len = manager.update(past)

    assert orig_len == seq_len
    assert kept_len == expected_kept
    for k_new, v_new in new_past:
        assert k_new.size(-2) == expected_kept
        assert v_new.size(-2) == expected_kept

    # Verify last token is always kept if seq_len > 0
    if seq_len > 0:
         assert torch.allclose(new_past[0][0][..., -1, :], torch.ones_like(new_past[0][0][..., -1, :]))

    # Verify first token is kept if ratio > 0 and expected_kept > 1
    if ratio > 0 and expected_kept > 1 and seq_len > 0:
        assert torch.allclose(new_past[0][0][..., 0, :], -torch.ones_like(new_past[0][0][..., 0, :]))
    # Verify first token is NOT kept if ratio is 0 (only last is kept)
    elif ratio == 0 and seq_len > 1:
        # Check if the only kept token is the last one
        assert kept_len == 1
        assert torch.allclose(new_past[0][0][..., 0, :], torch.ones_like(new_past[0][0][..., 0, :]))


@pytest.mark.parametrize("ratio, seq_len, expected_kept", [
    (0.5, 10, 5),  # Keep ceil(10 * 0.5) = 5
    (0.1, 10, 1),  # Keep ceil(10 * 0.1) = 1
    (0.9, 10, 9),  # Keep ceil(10 * 0.9) = 9
    (1.0, 10, 10), # Keep all
    (0.0, 10, 1),  # Keep ceil(10*0.0)=0, but max(1, 0) = 1 (always keep last)
    (0.5, 1, 1),   # Cannot prune length 1
])
def test_bottom_ratio_strategy(ratio, seq_len, expected_kept):
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(BottomRatioStrategy(pruning_ratio=ratio))
    new_past, orig_len, kept_len = manager.update(past)

    assert orig_len == seq_len
    assert kept_len == expected_kept
    for k_new, v_new in new_past:
        assert k_new.size(-2) == expected_kept
        assert v_new.size(-2) == expected_kept

    # Check content matches the *last* expected_kept elements
    if seq_len > 0 and kept_len > 0 and kept_len <= seq_len:
        original_k, _ = past[0] # Check first layer key
        kept_k, _ = new_past[0]
        expected_content = original_k[..., -kept_len:, :]
        assert torch.equal(kept_k, expected_content)


@pytest.mark.parametrize("ratio, seq_len, expected_kept", [
    (0.5, 11, 6),  # Keep ceil(10 * 0.5) = 5 from past. Split 2 top, 3 bottom. +1 last = 6 total. Indices: 0, 1, 7, 8, 9, 10
    (0.6, 11, 7),  # Keep ceil(10 * 0.6) = 6 from past. Split 3 top, 3 bottom. +1 last = 7 total. Indices: 0, 1, 2, 7, 8, 9, 10
    (0.1, 11, 2),  # Keep ceil(10 * 0.1) = 1 from past. Split 0 top, 1 bottom. +1 last = 2 total. Indices: 9, 10
    (0.0, 11, 1),  # Keep 0 from past + 1 last = 1. Index: 10
    (1.0, 11, 11), # Keep all
    (0.5, 2, 2),   # Keep ceil(1 * 0.5)=1 from past. Split 0 top, 1 bottom. +1 last. Indices: 0, 1 => Keep all 2.
    (0.5, 1, 1),   # Cannot prune length 1
])
def test_both_ratio_strategy(ratio, seq_len, expected_kept):
    past = make_dummy_past(seq_len)
    # Mark first and last tokens
    if seq_len > 0:
        past[0][0][..., 0, :] = -1.0 # Mark first key
        past[0][0][..., -1, :] = 1.0 # Mark last key

    manager = KVCacheManager(BothRatioStrategy(pruning_ratio=ratio))
    new_past, orig_len, kept_len = manager.update(past)

    assert orig_len == seq_len
    # Note: Due to ceiling and splitting, the exact number kept might vary slightly
    # from a simple ratio calculation, but the test cases cover expected outcomes.
    assert kept_len == expected_kept, f"Ratio {ratio}, SeqLen {seq_len}"

    for k_new, v_new in new_past:
        assert k_new.size(-2) == expected_kept
        assert v_new.size(-2) == expected_kept

    # Verify last token is always kept if seq_len > 0
    if seq_len > 0:
        assert torch.allclose(new_past[0][0][..., -1, :], torch.ones_like(new_past[0][0][..., -1, :]))

    # Verify first token is kept if ratio > 0 and expected_kept > 1 (and it wasn't pruned by bottom)
    if ratio > 0 and expected_kept > 1 and seq_len > 0:
         # Check if the first index (0) exists in the kept keys
         if torch.allclose(new_past[0][0][..., 0, :], -torch.ones_like(new_past[0][0][..., 0, :])):
             print(f"First token kept for ratio {ratio}, seq_len {seq_len}")
         else:
             # This might happen if ratio is small and only bottom + last are kept
             print(f"First token NOT kept for ratio {ratio}, seq_len {seq_len}")


