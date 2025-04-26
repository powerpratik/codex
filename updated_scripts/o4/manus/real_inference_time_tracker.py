import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

class InferenceTimeTracker:
    """
    Tracks detailed inference time metrics for LLM generation.
    """
    def __init__(self, model, tokenizer, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        
        # Time tracking
        self.encoding_time = 0
        self.total_generation_time = 0
        self.first_token_time = 0
        self.token_times = []
        self.token_timestamps = []
        self.eviction_times = []
        
        # Token tracking
        self.input_token_count = 0
        self.generated_token_count = 0
        
        # Detailed metrics
        self.per_token_metrics = []
        self.per_layer_metrics = []
        
        # Register hooks for tracking inference time
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to track inference time"""
        self.hooks = []
        
        # Find all modules that contribute to inference time
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ['attention', 'mlp', 'ffn', 'head']):
                # Register pre-forward hook to capture start time
                pre_hook = module.register_forward_pre_hook(self._pre_module_hook)
                self.hooks.append(pre_hook)
                
                # Register forward hook to capture end time
                post_hook = module.register_forward_hook(self._post_module_hook)
                self.hooks.append(post_hook)
    
    def _pre_module_hook(self, module, args):
        """Pre-module hook to capture start time"""
        # Store start time in module for later retrieval
        module._start_time = time.time()
        return None
    
    def _post_module_hook(self, module, args, output):
        """Post-module hook to capture end time and compute duration"""
        if hasattr(module, '_start_time'):
            end_time = time.time()
            duration = end_time - module._start_time
            
            # Store in module metrics
            if not hasattr(module, '_time_metrics'):
                module._time_metrics = []
            module._time_metrics.append(duration)
            
            # Add to per-layer metrics if this is a major component
            module_name = module.__class__.__name__
            if any(x in module_name.lower() for x in ['attention', 'mlp', 'ffn']):
                self.per_layer_metrics.append({
                    'module': module_name,
                    'time': duration
                })
        
        return output
    
    def start_encoding(self):
        """Mark the start of prompt encoding"""
        self.encoding_start_time = time.time()
    
    def end_encoding(self, input_tokens):
        """Mark the end of prompt encoding and record metrics"""
        self.encoding_time = time.time() - self.encoding_start_time
        self.input_token_count = len(input_tokens)
    
    def start_generation(self):
        """Mark the start of token generation"""
        self.generation_start_time = time.time()
        self.token_generation_start_time = time.time()
    
    def token_generated(self, token_id, eviction_time=0):
        """Record metrics for a generated token"""
        now = time.time()
        token_time = now - self.token_generation_start_time
        
        # For first token, record special metric
        if len(self.token_times) == 0:
            self.first_token_time = token_time
        
        # Record token time
        self.token_times.append(token_time)
        self.token_timestamps.append(now)
        self.eviction_times.append(eviction_time)
        self.generated_token_count += 1
        
        # Record per-token metrics
        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        self.per_token_metrics.append({
            'token_id': token_id,
            'token_text': token_text,
            'generation_time': token_time,
            'eviction_time': eviction_time,
            'timestamp': now
        })
        
        # Reset for next token
        self.token_generation_start_time = now
    
    def end_generation(self):
        """Mark the end of generation and compute final metrics"""
        self.total_generation_time = time.time() - self.generation_start_time
    
    def get_time_stats(self):
        """Get comprehensive time statistics"""
        if not self.token_times:
            return {
                "encoding_time": self.encoding_time,
                "total_generation_time": 0,
                "tokens_per_second": 0,
                "avg_token_time": 0,
                "first_token_time": 0
            }
        
        avg_token_time = sum(self.token_times) / len(self.token_times)
        tokens_per_second = self.generated_token_count / max(0.001, self.total_generation_time)
        
        return {
            "encoding_time": self.encoding_time,
            "total_generation_time": self.total_generation_time,
            "tokens_per_second": tokens_per_second,
            "avg_token_time": avg_token_time,
            "first_token_time": self.first_token_time,
            "token_times": self.token_times,
            "token_timestamps": self.token_timestamps,
            "eviction_times": self.eviction_times,
            "eviction_overhead_percent": sum(self.eviction_times) / max(0.001, self.total_generation_time) * 100,
            "input_tokens": self.input_token_count,
            "generated_tokens": self.generated_token_count,
            "per_token_metrics": self.per_token_metrics,
            "per_layer_metrics": self.per_layer_metrics
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.encoding_time = 0
        self.total_generation_time = 0
        self.first_token_time = 0
        self.token_times = []
        self.token_timestamps = []
        self.eviction_times = []
        self.input_token_count = 0
        self.generated_token_count = 0
        self.per_token_metrics = []
        self.per_layer_metrics = []
        
        # Reset module metrics
        for name, module in self.model.named_modules():
            if hasattr(module, '_time_metrics'):
                delattr(module, '_time_metrics')
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
