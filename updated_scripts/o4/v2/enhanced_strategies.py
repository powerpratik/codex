import random
import torch


class NoOpStrategy:
    def evict(self, past, **kwargs):
        return past


class WindowStrategy:
    def __init__(self, window_size_tokens: int):
        self.window_size = window_size_tokens

    def evict(self, past, **kwargs):
        seq_len = past[0][0].size(2)
        if seq_len <= self.window_size:
            return past
        start = seq_len - self.window_size
        pruned = [
            (k[:, :, start:, :].contiguous(), v[:, :, start:, :].contiguous())
            for k, v in past
        ]
        return tuple(pruned)


class RandomSamplingStrategy:
    def __init__(self, sample_size_tokens: int):
        self.sample_size = sample_size_tokens

    def evict(self, past, **kwargs):
        seq_len = past[0][0].size(2)
        if seq_len <= self.sample_size:
            return past
        indices = sorted(random.sample(range(seq_len), self.sample_size))
        pruned = [
            (k[:, :, indices, :].contiguous(), v[:, :, indices, :].contiguous())
            for k, v in past
        ]
        return tuple(pruned)


class StridedStrategy:
    def __init__(self, num_tokens: int):
        self.num_tokens = num_tokens

    def evict(self, past, **kwargs):
        seq_len = past[0][0].size(2)
        if seq_len <= self.num_tokens:
            return past
        stride = seq_len / self.num_tokens
        indices = [int(i * stride) for i in range(self.num_tokens)]
        pruned = [
            (k[:, :, indices, :].contiguous(), v[:, :, indices, :].contiguous())
            for k, v in past
        ]
        return tuple(pruned)


class BlockAverageStrategy:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks

    def evict(self, past, **kwargs):
        seq_len = past[0][0].size(2)
        if seq_len <= self.num_blocks:
            return past
        block_size = seq_len // self.num_blocks
        pruned = []
        for k, v in past:
            # unfold dims: (batch, heads, num_blocks, block_size, head_dim)
            k_unf = k.unfold(2, block_size, block_size)
            v_unf = v.unfold(2, block_size, block_size)
            # average over the block_size dimension (dim=3)
            k_avg = k_unf.mean(dim=3).contiguous()
            v_avg = v_unf.mean(dim=3).contiguous()
            pruned.append((k_avg, v_avg))
        return tuple(pruned)


class AttentionScoreStrategy:
    def __init__(self, top_k: int):
        self.top_k = top_k

    def evict(self, past, attention_scores=None):
        """
        attention_scores: tuple of length num_layers, each a Tensor
            [batch, heads, seq_len, key_len]
        """
        assert attention_scores is not None, "Must pass attentions to AttentionScoreStrategy"
        pruned = []
        for layer_idx, (k, v) in enumerate(past):
            attn = attention_scores[layer_idx]     # [B, H, L_q, L_k]
            # sum over batch, heads, and queries → importance per key position
            scores = attn.sum(dim=(0, 1, 2))        # [L_k]
            topk = torch.topk(scores, min(self.top_k, scores.size(0)), sorted=True).indices
            indices = topk.sort().values.tolist()   # back to ascending order
            k_p = k[:, :, indices, :].contiguous()
            v_p = v[:, :, indices, :].contiguous()
            pruned.append((k_p, v_p))
        return tuple(pruned)
