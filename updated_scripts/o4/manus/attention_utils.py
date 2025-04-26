# attention_utils.py
import torch
import numpy as np

def extract_layerwise_attention(attentions, layer_weights=None):
    """
    Extract attention scores with layer-specific weighting.
    
    Args:
        attentions: List of attention tensors from model output
        layer_weights: Optional list of weights for each layer (later layers typically more important)
        
    Returns:
        torch.Tensor: Weighted attention scores per token
    """
    if not attentions:
        return None
        
    # Default to exponential weighting if not provided (favors later layers)
    if layer_weights is None:
        num_layers = len(attentions)
        # Exponential weighting: later layers get exponentially more weight
        layer_weights = [np.exp(i/5) for i in range(num_layers)]
        # Normalize weights
        sum_weights = sum(layer_weights)
        layer_weights = [w / sum_weights for w in layer_weights]
    
    # Convert to tensor if not already
    if not isinstance(layer_weights, torch.Tensor):
        layer_weights = torch.tensor(
            layer_weights, 
            device=attentions[0].device, 
            dtype=attentions[0].dtype
        )
    
    # Stack attention matrices and apply weights along layer dimension
    all_layers = torch.stack(attentions)  # [num_layers, batch, num_heads, seq_len, seq_len]
    
    # Apply layer weights - reshape for broadcasting
    layer_weights = layer_weights.view(-1, 1, 1, 1, 1)
    weighted_layers = all_layers * layer_weights
    
    # Sum across layers to get layer-weighted attention
    layer_weighted = weighted_layers.sum(dim=0)  # [batch, num_heads, seq_len, seq_len]
    
    # Now handle attention heads - we'll identify and use the most "selective" heads
    # These are heads that have high variance in their attention distributions
    head_variances = layer_weighted.var(dim=-1).mean(dim=-1)  # [batch, num_heads]
    
    # Get the top half most selective heads
    num_heads = head_variances.shape[1]
    top_heads = torch.topk(head_variances, num_heads // 2 + 1, dim=1).indices[0]
    
    # Only keep attention from selective heads
    selective_attention = layer_weighted[:, top_heads]
    
    # Average across selective heads
    head_avg = selective_attention.mean(dim=1)  # [batch, seq_len, seq_len]
    
    # For each token position, compute how much attention it receives (importance)
    # This is the column-wise sum of the attention matrix
    token_importance = head_avg.sum(dim=1)[0]  # [seq_len]
    
    return token_importance

def dynamic_token_importance(current_scores, token_types, position_indices, decay_factor=0.85):
    """
    Adjust token importance based on token type and position
    
    Args:
        current_scores: Raw attention-based scores
        token_types: List of token types (e.g., 'special', 'entity', 'regular')
        position_indices: Original position indices of tokens
        decay_factor: How much to decay older tokens (0.85 means 15% decay per step)
        
    Returns:
        Adjusted importance scores
    """
    # Copy scores to avoid modifying the original
    adjusted_scores = current_scores.clone()
    
    # Position-based time decay: more recent tokens are more important
    # We apply exponential decay based on position
    seq_len = adjusted_scores.shape[0]
    
    # Calculate position-based decay factors
    # Newer tokens (higher position_indices) get less decay
    max_pos = max(position_indices)
    position_decay = torch.tensor(
        [decay_factor ** (max_pos - pos) for pos in position_indices],
        device=adjusted_scores.device,
        dtype=adjusted_scores.dtype
    )
    
    # Apply position-based decay
    adjusted_scores = adjusted_scores * position_decay
    
    # Apply token-type based adjustments
    for i, token_type in enumerate(token_types):
        if token_type == 'special':
            # Special tokens (like BOS, EOS) get higher importance
            adjusted_scores[i] *= 1.5
        elif token_type == 'entity':
            # Named entities get higher importance
            adjusted_scores[i] *= 1.2
    
    return adjusted_scores