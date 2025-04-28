import time
import logging
import numpy as np
import torch
import gc
from typing import List, Dict, Tuple, Optional, Union, Any
from transformers.cache_utils import Cache, DynamicCache, StaticCache



class RealKVCacheManager:
    """
    Modified KV Cache Manager that physically manipulates the modern Cache object.
    Directly handles the newer Transformers Cache format.
    """
    
    def __init__(self, model, tokenizer=None, cfg=None, logger=None):
        # Original initialization
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or {}
        self.threshold = self.cfg.get("kv_threshold", 1000)
        self.threshold_trigger_ratio = 0.95
        
        # Set up logger
        self.logger = logger or logging.getLogger("kv_cache_manager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize memory tracking
        self.current_memory_mb = 0
        self.peak_memory_mb = 0
        self.layer_memory = {}
        self.step_memory = []
        
        # Initialize eviction stats
        self.eviction_count = 0
        self.total_eviction_time = 0
        
        # Add direct memory tracking
        self.last_sequence_length = 0
        self.per_token_memory_estimate = 0.5  # MB per token for Llama-2-7b (adjust if needed)
        
        # Register hooks for memory tracking
        self.hooks = []
        self._register_hooks()
        self.logger.info("Memory tracking hooks registered")
    
    def _register_hooks(self):
        """Register hooks to track memory usage"""
        # Remove existing hooks
        self.remove_hooks()
        
        # Add new hooks for attention modules
        hook_count = 0
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                hook = module.register_forward_hook(self._memory_tracking_hook)
                self.hooks.append(hook)
                hook_count += 1
                self.logger.info(f"Registered memory tracking hook for {name}")
        
        self.logger.info(f"Total hooks registered: {hook_count}")
    
    def _memory_tracking_hook(self, module, input_tensor, output_tensor):
        """Hook to track memory usage during forward pass"""
        try:
            # Force memory stats update on each forward pass
            if hasattr(self.model, "past_key_values") and self.model.past_key_values is not None:
                self._update_memory_stats(self.model.past_key_values)
                self.logger.info(f"Hook updated memory stats: {self.current_memory_mb:.2f} MB")
        except Exception as e:
            self.logger.warning(f"Error in memory tracking hook: {e}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _get_legacy_cache(self, past_key_values):
        """Get the legacy cache format from a Cache object or return the original if already in legacy format"""
        if self._is_cache_object(past_key_values):
            if hasattr(past_key_values, 'to_legacy_cache'):
                return past_key_values.to_legacy_cache()
            # If no direct conversion method, we'll need to implement our own
            return self._manual_cache_extraction(past_key_values)
        return past_key_values

    def _manual_cache_extraction(self, cache_obj):
        """Extract tensors from a Cache object manually"""
        # This is a fallback for when to_legacy_cache is not available
        # Implementation depends on the specific Cache implementation
        # For DynamicCache, we can extract the key_values directly
        if hasattr(cache_obj, 'key_values'):
            return tuple(cache_obj.key_values)
        # Fallback to an empty cache if we can't extract
        self.logger.warning("Unable to extract tensors from Cache object, using empty cache")
        return ()
    
    def _update_memory_stats(self, past_key_values):
        """Update memory statistics based on Cache object or legacy past_key_values"""
        total_memory = 0
        layer_memory = {}
        
        try:
            # Check if we have a Cache object
            if isinstance(past_key_values, (Cache, DynamicCache, StaticCache)):
                # Try to get sequence length from Cache
                seq_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 0
                
                # Try to access key_value tensors directly
                key_value_dict = past_key_values.key_value_dict if hasattr(past_key_values, 'key_value_dict') else {}
                
                # Process the internal tensors if we can access them
                if key_value_dict:
                    for layer_idx, tensors in key_value_dict.items():
                        layer_total = 0
                        for tensor in tensors.values():
                            tensor_memory = tensor.numel() * tensor.element_size() / (1024 * 1024)  # MB
                            layer_total += tensor_memory
                        
                        layer_memory[f"layer_{layer_idx}"] = layer_total
                        total_memory += layer_total
                else:
                    # Fallback to estimation based on sequence length
                    model_config = self.model.config
                    hidden_size = getattr(model_config, 'hidden_size', 4096)
                    num_layers = getattr(model_config, 'num_hidden_layers', 32)
                    num_heads = getattr(model_config, 'num_attention_heads', 32)
                    head_dim = hidden_size // num_heads
                    
                    # Calculate actual memory: 2 (key+value) * seq_len * head_dim * num_heads * num_layers * bytes_per_param
                    bytes_per_param = 2  # fp16 uses 2 bytes per parameter
                    total_memory = 2 * seq_len * head_dim * num_heads * num_layers * bytes_per_param / (1024 * 1024)  # MB
                    
                    # Distribute across layers
                    for i in range(num_layers):
                        layer_memory[f"layer_{i}"] = total_memory / num_layers
                    
                    # Update sequence length
                    self.last_sequence_length = seq_len
            
            # Handle legacy tuple format
            elif isinstance(past_key_values, tuple) and past_key_values:
                for i, layer_cache in enumerate(past_key_values):
                    layer_total = 0
                    for tensor in layer_cache:
                        if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                            tensor_memory = tensor.numel() * tensor.element_size() / (1024 * 1024)  # MB
                            layer_total += tensor_memory
                    
                    layer_memory[f"layer_{i}"] = layer_total
                    total_memory += layer_total
                
                # Try to get sequence length
                if past_key_values[0] and past_key_values[0][0] is not None:
                    self.last_sequence_length = past_key_values[0][0].size(2)
            
            # If both methods failed or returned very small values, use sequence length estimation
            if total_memory < 1.0:  # If less than 1 MB, likely incorrect
                seq_len = self.last_sequence_length + 1  # Increment as fallback
                total_memory = seq_len * self.per_token_memory_estimate
                
                # Distribute estimated memory across layers
                num_layers = 32  # Default for Llama-2-7b
                for i in range(num_layers):
                    layer_memory[f"layer_{i}"] = total_memory / num_layers
        
        except Exception as e:
            self.logger.warning(f"Error calculating memory: {e}")
            # Fallback to sequence length estimation
            seq_len = self.last_sequence_length + 1  # Increment by 1 as fallback
            total_memory = seq_len * self.per_token_memory_estimate
            
            # Distribute estimated memory across layers
            num_layers = 32  # Default to 32 layers for Llama-2-7b
            for i in range(num_layers):
                layer_memory[f"layer_{i}"] = total_memory / num_layers
        
        # Update stats
        self.current_memory_mb = total_memory
        self.peak_memory_mb = max(self.peak_memory_mb, total_memory)
        self.layer_memory = layer_memory
        self.step_memory.append(total_memory)
        
        # Log memory usage
        self.logger.info(f"Current KV cache size: {total_memory:.2f} MB, Peak: {self.peak_memory_mb:.2f} MB, Seq len: {self.last_sequence_length}")
    def get_memory_stats(self):
        """Get current memory statistics"""
        return {
            "current_memory_mb": self.current_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "layer_memory": self.layer_memory,
            "step_memory": self.step_memory,
            "eviction_count": self.eviction_count,
            "avg_eviction_time": self.total_eviction_time / max(1, self.eviction_count)
        }
    
    def _is_cache_object(self, past_key_values):
    """Check if past_key_values is a Cache object or a tuple of tensors"""
        return hasattr(past_key_values, "get_usable_length") or hasattr(past_key_values, "to_legacy_cache")

    def _convert_to_legacy_cache(self, past_key_values):
        """Convert Cache object to legacy tuple of tensors format if needed"""
        if self._is_cache_object(past_key_values) and hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()
        return past_key_values

    def _convert_back_to_cache(self, legacy_cache, original_cache):
        """Convert modified legacy cache back to Cache object if needed"""
        if self._is_cache_object(original_cache):
            # For newer models with Cache objects, we need to create a new Cache
            # This is a simplified approach and might need customization for specific models
            if hasattr(original_cache, "get_seq_length"):
                seq_length = original_cache.get_seq_length()
            else:
                # Estimate from the first layer's key tensor
                seq_length = legacy_cache[0][0].size(2) if legacy_cache else 0
                
            # Create a new Cache with the modified tensors
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            new_cache.update(legacy_cache)
            return new_cache
        return legacy_cache

    def apply_eviction_strategy(self, past_key_values, strategy_name, attention_scores=None, token_types=None):
        """
        Apply the specified eviction strategy to the KV cache.
        Handles modern Cache objects for newer Transformers versions.
        
        Args:
            past_key_values: The current KV cache (Cache object or tuple)
            strategy_name: Name of the strategy to apply
            attention_scores: Attention scores for each token (optional)
            token_types: Types of each token (optional)
            
        Returns:
            Updated past_key_values (same type as input)
        """
        # Check if we need to apply eviction
        if self.current_memory_mb < self.threshold * 0.1:
            self.logger.info(f"Skipping eviction, current memory {self.current_memory_mb:.2f} MB < threshold {self.threshold * 0.1:.2f} MB")
            return past_key_values
        else:
            self.logger.info(f"Applying eviction, current memory {self.current_memory_mb:.2f} MB >= threshold {self.threshold * 0.1:.2f} MB")
        
        # Start timing eviction
        eviction_start = time.time()
        
        try:
            # Check if we have a Cache object from newer Transformers
            if isinstance(past_key_values, (Cache, DynamicCache, StaticCache)):
                # For Cache objects, we have to create a new Cache with the evicted tokens removed
                
                # Extract sequence length
                seq_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 0
                
                # Determine which indices to keep based on strategy
                indices_to_keep = self._get_indices_to_keep(
                    seq_len, 
                    strategy_name, 
                    attention_scores=attention_scores,
                    token_types=token_types
                )
                
                # For modern Cache objects, we need to create a new one since they're often immutable
                new_cache = DynamicCache()
                
                # Get the underlying key_value tensors
                if hasattr(past_key_values, 'key_value_dict'):
                    for layer_idx, tensor_dict in past_key_values.key_value_dict.items():
                        # Create new tensors with only the kept indices
                        new_tensors = {}
                        for tensor_name, tensor in tensor_dict.items():
                            # Extract tensor dimensions
                            batch_size, num_heads = tensor.shape[:2]
                            head_dim = tensor.shape[3] if tensor.dim() > 3 else tensor.shape[2]
                            
                            # Convert indices to tensor
                            tensor_indices = torch.tensor(indices_to_keep, device=tensor.device, dtype=torch.long)
                            
                            # Select the indices to keep
                            if tensor.dim() > 3:  # 4D tensor (batch, num_heads, seq_len, head_dim)
                                new_tensor = tensor.index_select(2, tensor_indices)
                            else:  # 3D tensor (batch, seq_len, hidden_size)
                                new_tensor = tensor.index_select(1, tensor_indices)
                            
                            new_tensors[tensor_name] = new_tensor
                        
                        # Update the new cache with the modified tensors
                        for tensor_name, tensor in new_tensors.items():
                            new_cache.update_tensor(layer_idx, tensor_name, tensor)
                
                updated_past = new_cache
            
            # Handle legacy tuple format
            elif isinstance(past_key_values, tuple):
                updated_past = []
                
                for layer_idx, layer_cache in enumerate(past_key_values):
                    updated_layer = []
                    
                    for tensor in layer_cache:
                        # Get sequence length
                        seq_len = tensor.size(2)
                        
                        # Determine which indices to keep based on strategy
                        indices_to_keep = self._get_indices_to_keep(
                            seq_len, 
                            strategy_name, 
                            attention_scores=attention_scores,
                            token_types=token_types
                        )
                        
                        # Convert indices to tensor
                        tensor_indices = torch.tensor(indices_to_keep, device=tensor.device, dtype=torch.long)
                        
                        # Create a new tensor with only the kept tokens
                        new_tensor = tensor.index_select(2, tensor_indices)
                        updated_layer.append(new_tensor)
                    
                    updated_past.append(tuple(updated_layer))
                
                updated_past = tuple(updated_past)
            else:
                # Unknown format, return unchanged
                self.logger.warning(f"Unknown past_key_values format: {type(past_key_values)}, skipping eviction")
                updated_past = past_key_values
        
        except Exception as e:
            self.logger.error(f"Error applying eviction strategy: {e}")
            updated_past = past_key_values
        
        # End timing and update stats
        eviction_time = time.time() - eviction_start
        self.eviction_count += 1
        self.total_eviction_time += eviction_time
        
        # Force CUDA to free memory
        torch.cuda.empty_cache()
        
        return updated_past

        

    
    def _apply_random_strategy(self, past_key_values, keep_ratio=0.7):
        """
        Apply random eviction strategy: randomly select tokens to keep.
        
        Args:
            past_key_values: The current KV cache
            keep_ratio: Fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """
        updated_past = []
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Randomly select indices to keep
                indices_to_keep = torch.randperm(seq_len, device=tensor.device)[:keep_count]
                
                # Create a new tensor with only the kept tokens
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def _apply_attention_top_strategy(self, past_key_values, attention_scores, keep_ratio=0.7):
        """
        Apply attention-based top strategy: keep tokens with highest attention scores.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            keep_ratio: Fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """
        if attention_scores is None:
            # Fallback to random if no attention scores
            return self._apply_random_strategy(past_key_values, keep_ratio)
        
        updated_past = []
        
        # Get average attention across layers
        if isinstance(attention_scores, list):
            avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
        else:
            avg_attention = attention_scores.cpu().numpy()
        
        # Calculate importance scores (average attention received by each token)
        importance_scores = np.mean(avg_attention, axis=(0, 1))
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Get indices of tokens with highest importance scores
                if len(importance_scores) == seq_len:
                    importance_indices = np.argsort(importance_scores)[-keep_count:]
                else:
                    # Fallback to random if dimensions don't match
                    importance_indices = np.random.choice(seq_len, keep_count, replace=False)
                
                # Create a new tensor with only the kept tokens
                indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def _apply_attention_bottom_strategy(self, past_key_values, attention_scores, keep_ratio=0.7):
        """
        Apply attention-based bottom strategy: keep tokens with lowest attention scores.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            keep_ratio: Fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """
        if attention_scores is None:
            # Fallback to random if no attention scores
            return self._apply_random_strategy(past_key_values, keep_ratio)
        
        updated_past = []
        
        # Get average attention across layers
        if isinstance(attention_scores, list):
            avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
        else:
            avg_attention = attention_scores.cpu().numpy()
        
        # Calculate importance scores (average attention received by each token)
        importance_scores = np.mean(avg_attention, axis=(0, 1))
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Get indices of tokens with lowest importance scores
                if len(importance_scores) == seq_len:
                    importance_indices = np.argsort(importance_scores)[:keep_count]
                else:
                    # Fallback to random if dimensions don't match
                    importance_indices = np.random.choice(seq_len, keep_count, replace=False)
                
                # Create a new tensor with only the kept tokens
                indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def _apply_hybrid_strategy(self, past_key_values, attention_scores, token_types, 
                              keep_ratio=0.7, recency_weight=0.5, attention_weight=0.3, type_weight=0.2):
        """
        Apply hybrid strategy: combine recency, attention, and token type importance.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            token_types: Types of each token
            keep_ratio: Fraction of tokens to keep
            recency_weight: Weight for recency factor
            attention_weight: Weight for attention factor
            type_weight: Weight for token type factor
            
        Returns:
            Updated past_key_values
        """
        # Check if we need to apply eviction
        if self.current_memory_mb < self.threshold * 0.1:  # Lower to 10% for testing
            return past_key_values
        
        # Start timing eviction
        eviction_start = time.time()
        
        if attention_scores is None or token_types is None:
            # Fallback to sliding window if missing data
            updated_past = self._apply_sliding_window(past_key_values, keep_ratio)
            
            # End timing and update stats
            eviction_time = time.time() - eviction_start
            self.eviction_count += 1
            self.total_eviction_time += eviction_time
            
            # Force CUDA to free memory
            torch.cuda.empty_cache()
            
            return updated_past
        
        updated_past = []
        
        # Get average attention across layers
        if isinstance(attention_scores, list):
            avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
        else:
            avg_attention = attention_scores.cpu().numpy()
        
        # Normalize to get attention importance scores
        attention_importance = np.mean(avg_attention, axis=(0, 1))
        
        # Convert token types to importance scores
        type_importance = np.zeros(len(token_types))
        for i, token_type in enumerate(token_types):
            if token_type == "special":
                type_importance[i] = 1.0
            elif token_type == "rare":
                type_importance[i] = 0.8
            elif token_type == "punctuation":
                type_importance[i] = 0.3
            else:  # common
                type_importance[i] = 0.5
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * keep_ratio))
                
                # Calculate hybrid importance scores
                hybrid_scores = np.zeros(seq_len)
                
                # Add recency component
                recency_scores = np.linspace(0, 1, seq_len)
                hybrid_scores += recency_weight * recency_scores
                
                # Add attention component if available
                if len(attention_importance) == seq_len:
                    # Normalize attention scores to [0, 1]
                    norm_attention = (attention_importance - attention_importance.min()) 
                    if attention_importance.max() > attention_importance.min():
                        norm_attention /= (attention_importance.max() - attention_importance.min())
                    hybrid_scores += attention_weight * norm_attention
                
                # Add token type component if available
                if len(type_importance) == seq_len:
                    hybrid_scores += type_weight * type_importance
                
                # Get indices of tokens with highest hybrid scores
                importance_indices = np.argsort(hybrid_scores)[-keep_count:]
                
                # Create a new tensor with only the kept tokens
                indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        # End timing and update stats
        eviction_time = time.time() - eviction_start
        self.eviction_count += 1
        self.total_eviction_time += eviction_time
        
        # Force CUDA to free memory
        torch.cuda.empty_cache()
        
        return tuple(updated_past)
    
    def _apply_sliding_window(self, past_key_values, window_size=0.7, important_ratio=0.1, attention_scores=None):
        """
        Apply sliding window strategy: keep recent tokens and important tokens.
        
        Args:
            past_key_values: The current KV cache
            window_size: Fraction of recent tokens to keep
            important_ratio: Fraction of important tokens to keep
            attention_scores: Attention scores for each token (optional)
            
        Returns:
            Updated past_key_values
        """
        updated_past = []
        
        # Calculate importance scores if attention scores are available
        importance_scores = None
        if attention_scores is not None:
            # Get average attention across layers
            if isinstance(attention_scores, list):
                avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
            else:
                avg_attention = attention_scores.cpu().numpy()
            
            # Calculate importance scores (average attention received by each token)
            importance_scores = np.mean(avg_attention, axis=(0, 1))
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate how many recent tokens to keep
                recent_count = max(1, int(seq_len * window_size))
                
                # Calculate how many important tokens to keep
                important_count = max(0, int(seq_len * important_ratio))
                
                # Get indices of recent tokens
                recent_indices = list(range(seq_len - recent_count, seq_len))
                
                # Get indices of important tokens if available
                important_indices = []
                if importance_scores is not None and len(importance_scores) == seq_len and important_count > 0:
                    # Get indices of tokens with highest importance scores, excluding recent tokens
                    candidate_indices = list(range(0, seq_len - recent_count))
                    candidate_scores = importance_scores[:seq_len - recent_count]
                    if len(candidate_indices) > 0:
                        important_indices = np.argsort(candidate_scores)[-important_count:]
                
                # Combine indices
                indices_to_keep = list(set(recent_indices + list(important_indices)))
                indices_to_keep.sort()  # Sort to maintain order
                
                # Create a new tensor with only the kept tokens
                indices_tensor = torch.tensor(indices_to_keep, device=tensor.device)
                new_tensor = tensor.index_select(2, indices_tensor)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def _apply_adaptive_attention(self, past_key_values, attention_scores, base_keep=0.7):
        """
        Apply adaptive attention strategy: keep tokens based on attention distribution.
        
        Args:
            past_key_values: The current KV cache
            attention_scores: Attention scores for each token
            base_keep: Base fraction of tokens to keep
            
        Returns:
            Updated past_key_values
        """
        if attention_scores is None:
            # Fallback to sliding window if no attention scores
            return self._apply_sliding_window(past_key_values, base_keep)
        
        updated_past = []
        
        # Get average attention across layers
        if isinstance(attention_scores, list):
            avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
        else:
            avg_attention = attention_scores.cpu().numpy()
        
        # Calculate importance scores (average attention received by each token)
        importance_scores = np.mean(avg_attention, axis=(0, 1))
        
        for layer_idx, layer_cache in enumerate(past_key_values):
            updated_layer = []
            
            for tensor in layer_cache:
                # Get sequence length
                seq_len = tensor.size(2)
                
                # Calculate adaptive keep ratio based on attention distribution
                if len(importance_scores) == seq_len:
                    # Calculate entropy of attention distribution
                    norm_scores = importance_scores / np.sum(importance_scores)
                    entropy = -np.sum(norm_scores * np.log2(norm_scores + 1e-10))
                    max_entropy = np.log2(seq_len)
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    # Adjust keep ratio based on entropy
                    # Higher entropy (more uniform attention) -> keep fewer tokens
                    # Lower entropy (more focused attention) -> keep more tokens
                    adaptive_keep = base_keep * (1.0 - 0.5 * normalized_entropy)
                else:
                    adaptive_keep = base_keep
                
                # Calculate how many tokens to keep
                keep_count = max(1, int(seq_len * adaptive_keep))
                
                # Get indices of tokens with highest importance scores
                if len(importance_scores) == seq_len:
                    importance_indices = np.argsort(importance_scores)[-keep_count:]
                else:
                    # Fallback to recent tokens if dimensions don't match
                    importance_indices = list(range(seq_len - keep_count, seq_len))
                
                # Create a new tensor with only the kept tokens
                indices_to_keep = torch.tensor(importance_indices, device=tensor.device)
                new_tensor = tensor.index_select(2, indices_to_keep)
                updated_layer.append(new_tensor)
            
            updated_past.append(tuple(updated_layer))
        
        return tuple(updated_past)
    
    def __del__(self):
        """Clean up when the object is deleted"""
        self.remove_hooks()


def _get_indices_to_keep(self, seq_len, strategy_name, attention_scores=None, token_types=None):
    """
    Determine which token indices to keep based on the strategy.
    
    Args:
        seq_len: Current sequence length
        strategy_name: Name of the strategy to apply
        attention_scores: Attention scores for each token (optional)
        token_types: Types of each token (optional)
        
    Returns:
        List of indices to keep
    """
    # Extract keep ratio from strategy name
    keep_ratio = 0.7  # Default
    if "keep=" in strategy_name:
        try:
            keep_ratio = float(strategy_name.split("keep=")[1].split(")")[0])
        except:
            pass
    
    # Calculate how many tokens to keep
    keep_count = max(1, int(seq_len * keep_ratio))
    
    # Determine which indices to keep based on strategy
    if "Baseline" in strategy_name:
        # Baseline keeps all tokens
        return list(range(seq_len))

    # Bind helper function to the RealKVCacheManager class
    RealKVCacheManager._get_indices_to_keep = _get_indices_to_keep
        
        elif "Random" in strategy_name:
            # Random strategy: randomly select tokens to keep
            # Use numpy because torch.randperm can be slow for large seq_len
            indices = np.random.permutation(seq_len)[:keep_count].tolist()
            indices.sort()  # Sort to maintain order
            return indices
        
        elif "AttentionTop" in strategy_name:
            # Keep tokens with highest attention scores
            if attention_scores is None:
                # Fallback to random if no attention scores
                indices = np.random.permutation(seq_len)[:keep_count].tolist()
                indices.sort()
                return indices
            
            # Calculate importance scores from attention
            if isinstance(attention_scores, list):
                avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
            else:
                avg_attention = attention_scores.cpu().numpy()
            
            importance_scores = np.mean(avg_attention, axis=(0, 1))
            
            # Handle case where importance_scores length doesn't match seq_len
            if len(importance_scores) != seq_len:
                # Fallback to random
                indices = np.random.permutation(seq_len)[:keep_count].tolist()
                indices.sort()
                return indices
            
            # Get indices of tokens with highest importance scores
            importance_indices = np.argsort(importance_scores)[-keep_count:].tolist()
            importance_indices.sort()  # Sort to maintain order
            return importance_indices
        
        elif "AttentionBottom" in strategy_name:
            # Keep tokens with lowest attention scores
            if attention_scores is None:
                # Fallback to random if no attention scores
                indices = np.random.permutation(seq_len)[:keep_count].tolist()
                indices.sort()
                return indices
            
            # Calculate importance scores from attention
            if isinstance(attention_scores, list):
                avg_attention = np.mean([scores.cpu().numpy() for scores in attention_scores], axis=0)
            else:
                avg_attention = attention_scores.cpu().numpy()
            
            importance_scores = np.mean(avg_attention, axis=(0, 1))
            
            # Handle case where importance_scores length doesn't match seq_len
            if len(importance_scores) != seq_len:
                # Fallback to random
                indices = np.random.permutation(seq_len)[:keep_count].tolist()
                indices.sort()
                return indices
            
            # Get indices of tokens with lowest importance scores
            importance_indices = np.argsort(importance_scores)[:keep_count].tolist()
            importance_indices.sort()  # Sort to maintain order
            return importance_indices
        
        elif "SlidingWindow" in strategy_name:
            # Extract window size and important token ratio
            window_size = 0.7  # Default
            important_ratio = 0.1  # Default
            
            if "window=" in strategy_name:
                try:
                    window_size = float(strategy_name.split("window=")[1].split(",")[0])
                except:
                    pass
            if "important=" in strategy_name:
                try:
                    important_ratio = float(strategy_name.split("important=")[1].split(")")[0])
                except:
                    pass
            
            # Calculate how many recent tokens to keep
            recent_count = max(1, int(seq_len * window_size))
            
            # Calculate how many important tokens to keep
            important_count = max(0, int(seq_len * important_ratio))
            
            # Get indices of recent tokens
            recent_indices = list(range(seq_len - recent_count, seq_len))
            
            # Get indices of important tokens if available
            important_indices = []
            if attention_scores is not None and len(np.mean(attention_scores.cpu().numpy(), axis=(0, 1))) == seq_len and important_count > 0:
                # Calculate importance scores from attention
                avg_attention = np.mean(attention_scores.cpu().numpy(), axis=(0, 1))
                
                # Get indices of tokens with highest importance scores, excluding recent tokens
                candidate_indices = list(range(0, seq_len - recent_count))
                if candidate_indices:
                    candidate_scores = avg_attention[:seq_len - recent_count]
                    important_indices = np.argsort(candidate_scores)[-important_count:].tolist()
            
            # Combine indices
            indices_to_keep = list(set(recent_indices + important_indices))
            indices_to_keep.sort()  # Sort to maintain order
            return indices_to_keep
        
        elif "HybridNPercent" in strategy_name:
            # Extract parameters
            recency_weight = 0.5  # Default
            attention_weight = 0.3  # Default
            type_weight = 0.2  # Default
            
            if "r=" in strategy_name:
                try:
                    recency_weight = float(strategy_name.split("r=")[1].split(",")[0])
                except:
                    pass
            if "a=" in strategy_name:
                try:
                    attention_weight = float(strategy_name.split("a=")[1].split(",")[0])
                except:
                    pass
            if "t=" in strategy_name:
                try:
                    type_weight = float(strategy_name.split("t=")[1].split(")")[0])
                except:
                    pass
            
            # Calculate hybrid importance scores
            hybrid_scores = np.zeros(seq_len)
            
            # Add recency component
            recency_scores = np.linspace(0, 1, seq_len)
            hybrid_scores += recency_weight * recency_scores
            
            # Add attention component if available
            if attention_scores is not None:
                avg_attention = np.mean(attention_scores.cpu().numpy(), axis=(0, 1))
                if len(avg_attention) == seq_len:
                    # Normalize attention scores to [0, 1]
                    norm_attention = (avg_attention - avg_attention.min())
                    if avg_attention.max() > avg_attention.min():
                        norm_attention /= (avg_attention.max() - avg_attention.min())
                    hybrid_scores += attention_weight * norm_attention
            
            # Add token type component if available
            if token_types and len(token_types) == seq_len:
                type_importance = np.zeros(seq_len)
                for i, token_type in enumerate(token_types):
                    if token_type == "special":
                        type_importance[i] = 1.0
                    elif token_type == "rare":
                        type_importance[i] = 0.8
                    elif token_type == "punctuation":
                        type_importance[i] = 0.3
                    else:  # common
                        type_importance[i] = 0.5
                hybrid_scores += type_weight * type_importance
            
            # Get indices of tokens with highest hybrid scores
            importance_indices = np.argsort(hybrid_scores)[-keep_count:].tolist()
            importance_indices.sort()  # Sort to maintain order
            return importance_indices
        
        elif "AdaptiveAttention" in strategy_name:
            # Extract base keep ratio
            base_keep = 0.7  # Default
            if "base_keep=" in strategy_name:
                try:
                    base_keep = float(strategy_name.split("base_keep=")[1].split(")")[0])
                except:
                    pass
            
            # Calculate adaptive keep ratio based on attention distribution
            adaptive_keep = base_keep
            
            if attention_scores is not None:
                avg_attention = np.mean(attention_scores.cpu().numpy(), axis=(0, 1))
                if len(avg_attention) == seq_len:
                    # Calculate entropy of attention distribution
                    norm_scores = avg_attention / np.sum(avg_attention)
                    entropy = -np.sum(norm_scores * np.log2(norm_scores + 1e-10))
                    max_entropy = np.log2(seq_len)
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    # Adjust keep ratio based on entropy
                    # Higher entropy (more uniform attention) -> keep fewer tokens
                    # Lower entropy (more focused attention) -> keep more tokens
                    adaptive_keep = base_keep * (1.0 - 0.5 * normalized_entropy)
            
            # Calculate how many tokens to keep with adaptive ratio
            adaptive_keep_count = max(1, int(seq_len * adaptive_keep))
            
            # Get indices of tokens with highest importance scores
            if attention_scores is not None:
                avg_attention = np.mean(attention_scores.cpu().numpy(), axis=(0, 1))
                if len(avg_attention) == seq_len:
                    importance_indices = np.argsort(avg_attention)[-adaptive_keep_count:].tolist()
                    importance_indices.sort()  # Sort to maintain order
                    return importance_indices
            
            # Fallback to recent tokens
            return list(range(seq_len - adaptive_keep_count, seq_len))
        
        else:
            # Unknown strategy, return all indices
            self.logger.warning(f"Unknown strategy: {strategy_name}, keeping all tokens")
            return list(range(seq_len))