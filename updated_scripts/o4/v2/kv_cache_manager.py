import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def measure_cache_mb(past_key_values):
    """
    Sum up all bytes in past_key_values and return megabytes.
    """
    total_bytes = 0
    for layer in past_key_values:
        for tensor in layer:  # tensor is key or value
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


class ThresholdEvictionStrategy:
    """
    Evicts oldest tokens once measured cache size (MB) exceeds memory_threshold_mb.
    """

    def __init__(self, memory_threshold_mb: float, model_config):
        # Convert threshold to raw bytes
        self.memory_threshold_bytes = memory_threshold_mb * 1024 ** 2

        # Estimate bytes per token per layer: key+value each contain hidden_size floats
        hidden_size = model_config.hidden_size
        dtype_size = torch.finfo(torch.float32).bits // 8  # usually 4 bytes
        # each token in one layer => key + value => 2 * hidden_size * dtype_size
        self.bytes_per_token_per_layer = 2 * hidden_size * dtype_size

        # Total layers
        num_layers = model_config.num_hidden_layers
        # bytes per token across the **entire** cache:
        self.bytes_per_token_total = self.bytes_per_token_per_layer * num_layers

        # Compute how many tokens correspond to the byte threshold
        self.token_threshold = int(self.memory_threshold_bytes / self.bytes_per_token_total)
        logger.info(
            f"[ThresholdEvict] memory threshold: {memory_threshold_mb:.1f} MB "
            f"→ token threshold: {self.token_threshold} tokens"
        )

    def evict(self, past_key_values):
        """
        If cache size exceeds bytes threshold, drop the oldest tokens
        (i.e. slice off the front of the sequence dimension).
        """
        current_mb = measure_cache_mb(past_key_values)
        if current_mb <= (self.memory_threshold_bytes / 1024**2):
            return past_key_values

        # all layers have same seq_len
        seq_len = past_key_values[0][0].size(2)
        to_drop = seq_len - self.token_threshold
        logger.debug(
            f"[ThresholdEvict] Dropping {to_drop}/{seq_len} tokens "
            f"({current_mb:.1f}MB > threshold)"
        )

        pruned = []
        for key, value in past_key_values:
            # key, value shapes: [batch, heads, seq_len, head_dim]
            k_pruned = key[:, :, to_drop:, :].contiguous()
            v_pruned = value[:, :, to_drop:, :].contiguous()
            pruned.append((k_pruned, v_pruned))

        return tuple(pruned)
