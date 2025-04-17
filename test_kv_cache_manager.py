"""
test_kv_cache_manager.py

Unit tests for KVCacheManager and eviction strategies.
"""
import torch

import pytest

from kv_cache_manager import KVCacheManager, NoOpStrategy, WindowStrategy
from kv_cache_manager import RandomSamplingStrategy, StridedStrategy, BlockAverageStrategy, AttentionScoreStrategy

def make_dummy_past(seq_len: int,
                    num_layers: int = 2,
                    num_heads: int = 2,
                    head_dim: int = 4) -> list:
    """
    Create a dummy past_key_values list with given sequence length,
    layers, heads, and head dimension.
    """
    past_kvs = []
    for _ in range(num_layers):
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        past_kvs.append((k, v))
    return past_kvs

def test_noop_strategy_preserves_length():
    seq_len = 10
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(NoOpStrategy())
    new_past = manager.update(past)
    for (k_old, v_old), (k_new, v_new) in zip(past, new_past):
        assert k_old.shape == k_new.shape
        assert v_old.shape == v_new.shape

def test_window_strategy_shorter_than_window():
    seq_len = 3
    window = 5
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(WindowStrategy(window))
    new_past = manager.update(past)
    # If sequence length < window, shapes remain unchanged
    for (k_old, _), (k_new, _) in zip(past, new_past):
        assert k_old.shape == k_new.shape

def test_window_strategy_longer_than_window():
    seq_len = 10
    window = 5
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(WindowStrategy(window))
    new_past = manager.update(past)
    # If sequence length > window, sequence length should be truncated to window
    for (k_new, _) in new_past:
        assert k_new.size(-2) == window

def test_cache_clear_and_get():
    past = make_dummy_past(4)
    manager = KVCacheManager(NoOpStrategy())
    assert manager.get_cache() is None
    manager.update(past)
    assert manager.get_cache() is not None
    manager.clear()
    assert manager.get_cache() is None

def test_attention_strategy_shorter_than_max():
    seq_len = 4
    max_len = 5
    past = make_dummy_past(seq_len)
    manager = KVCacheManager(AttentionScoreStrategy(max_length=max_len))
    new_past = manager.update(past)
    # Sequence shorter than max: unchanged
    for (k_old, _), (k_new, _) in zip(past, new_past):
        assert k_new.shape == k_old.shape

def test_attention_strategy_longer_than_max_retains_latest():
    seq_len = 10
    max_len = 6
    past = make_dummy_past(seq_len)
    # fill past with known data so last vector is identifiable
    for i, (k, v) in enumerate(past):
        # set last key vector to all ones
        k[..., -1, :] = 1.0
        v[..., -1, :] = 2.0
    manager = KVCacheManager(AttentionScoreStrategy(max_length=max_len))
    new_past = manager.update(past)
    for (k_new, v_new) in new_past:
        # new sequence length equals max_len
        assert k_new.size(-2) == max_len
        # last entry in new cache must match original last vector
        assert torch.allclose(k_new[..., -1, :], torch.ones_like(k_new[..., -1, :]))
        assert torch.allclose(v_new[..., -1, :], torch.full_like(v_new[..., -1, :], 2.0))