# strategies.py
from abc import ABC, abstractmethod
import random
import torch

class BaseStrategy(ABC):
    def __init__(self, threshold: int):
        self.threshold = threshold

    @abstractmethod
    def evict(self, past_key_values, attention_scores=None):
        """
        past_key_values: tuple of (k, v) per layer
        attention_scores: optional list of token-level scores
        Return pruned past_key_values under threshold
        """
        pass

class Baseline(BaseStrategy):
    def evict(self, past_key_values, attention_scores=None):
        return past_key_values  # no eviction

class TopAttention(BaseStrategy):
    def __init__(self, threshold, pct):
        super().__init__(threshold)
        self.pct = pct / 100.0

    def evict(self, past, attention_scores):
        return _attention_evict(past, attention_scores, keep_top=self.pct)

class BottomAttention(TopAttention):
    def evict(self, past, attention_scores):
        return _attention_evict(past, attention_scores, keep_bottom=self.pct)

class HybridAttention(BaseStrategy):
    def __init__(self, threshold, top_pct, bot_pct):
        super().__init__(threshold)
        self.top_pct = top_pct / 100.0
        self.bot_pct = bot_pct / 100.0

    def evict(self, past, attention_scores):
        top = _attention_evict(past, attention_scores, keep_top=self.top_pct)
        return _attention_evict(top, attention_scores, keep_bottom=self.bot_pct)

class RandomEvict(BaseStrategy):
    def __init__(self, threshold, pct):
        super().__init__(threshold)
        self.pct = pct / 100.0

    def evict(self, past, attention_scores=None):
        # flatten tokens across layers, randomly pick keep_count
        all_k, all_v = zip(*past)  # list of tensors [layer x (T,head,...)]

        total_tokens = all_k[0].shape[0]
        keep = int(total_tokens * (1 - self.pct))
        idxs = list(range(total_tokens))
        keep_idxs = set(random.sample(idxs, keep))

        def prune_tensor(tensor):
            return tensor[list(sorted(keep_idxs)), ...]

        return tuple(
          (prune_tensor(k), prune_tensor(v))
          for k, v in past
        )

# helper for attention
def _attention_evict(past, scores, keep_top=None, keep_bottom=None):
    # scores: [T] float tensor
    T = scores.shape[0]
    keep = []
    if keep_top:
        top_k = int(T * keep_top)
        top_idxs = torch.topk(scores, top_k).indices.tolist()
        keep += top_idxs
    if keep_bottom:
        bot_k = int(T * keep_bottom)
        bot_idxs = torch.topk(-scores, bot_k).indices.tolist()
        keep += bot_idxs
    keep = sorted(set(keep))
    def prune(t):
        return t[keep, ...]
    return tuple((prune(k), prune(v)) for k,v in past)
