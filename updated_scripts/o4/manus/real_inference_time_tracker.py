import torch
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

class InferenceTimeTracker:
    """
    Tracks detailed inference time metrics for LLM generation.
    
    This class measures:
    - Encoding time (time to process the initial prompt)
    - Generation time (total time for token generation)
    - Per-token generation time
    - Time to first token
    - Eviction overhead time
    """
    def __init__(self, model, tokenizer, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger or logging.getLogger("InferenceTimeTracker")
        
        # Time tracking
        self.encoding_start_time = 0
        self.encoding_end_time = 0
        self.generation_start_time = 0
        self.generation_end_time = 0
        self.first_token_time = 0
        self.token_generation_times = []
        self.eviction_overhead_times = []
        
        # Token tracking
        self.input_token_count = 0
        self.generated_token_count = 0
        
        # Module time tracking
        self.module_times = {}
        
        # Register hooks for detailed timing
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to track time spent in different modules"""
        # Find key modules to track
        for name, module in self.model.named_modules():
            # Track attention modules
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(self._module_time_hook(name))
                self.hooks.append(hook)
            
            # Track MLP modules
            elif 'mlp' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(self._module_time_hook(name))
                self.hooks.append(hook)
    
    def _module_time_hook(self, name):
        """Create a hook function for the specified module"""
        def hook(module, input, output):
            if not hasattr(module, '_start_time'):
                return
            
            # Calculate time spent in this module
            module_time = time.time() - module._start_time
            
            # Record time
            if name not in self.module_times:
                self.module_times[name] = []
            self.module_times[name].append(module_time)
            
            # Clean up
            delattr(module, '_start_time')
        
        return hook
    
    def _pre_forward_hook(self, module, input):
        """Record start time before module execution"""
        module._start_time = time.time()
        return None
    
    def start_encoding(self):
        """Start timing the encoding phase"""
        self.encoding_start_time = time.time()
    
    def end_encoding(self, input_tokens):
        """End timing the encoding phase"""
        self.encoding_end_time = time.time()
        self.input_token_count = len(input_tokens)
        
        encoding_time = self.encoding_end_time - self.encoding_start_time
        self.logger.info(f"Encoding time: {encoding_time:.4f}s for {self.input_token_count} tokens")
    
    def start_generation(self):
        """Start timing the generation phase"""
        self.generation_start_time = time.time()
    
    def token_generated(self, token_id, eviction_time=0):
        """Record time for a generated token"""
        current_time = time.time()
        
        # For the first token, record special timing
        if self.generated_token_count == 0:
            self.first_token_time = current_time - self.generation_start_time
            self.logger.info(f"Time to first token: {self.first_token_time:.4f}s")
        
        # Record generation time for this token
        if self.generated_token_count > 0:
            # Time since last token, excluding eviction time
            token_time = current_time - self.token_generation_times[-1]["end_time"] - eviction_time
        else:
            # First token time, excluding eviction time
            token_time = self.first_token_time - eviction_time
        
        # Record eviction overhead
        self.eviction_overhead_times.append(eviction_time)
        
        # Record token info
        self.token_generation_times.append({
            "token_id": token_id,
            "token_text": self.tokenizer.decode([token_id], skip_special_tokens=True),
            "time": token_time,
            "eviction_time": eviction_time,
            "total_time": token_time + eviction_time,
            "end_time": current_time
        })
        
        # Increment counter
        self.generated_token_count += 1
    
    def end_generation(self):
        """End timing the generation phase"""
        self.generation_end_time = time.time()
        
        total_generation_time = self.generation_end_time - self.generation_start_time
        tokens_per_second = self.generated_token_count / max(0.001, total_generation_time)
        
        self.logger.info(f"Generation time: {total_generation_time:.4f}s for {self.generated_token_count} tokens")
        self.logger.info(f"Tokens per second: {tokens_per_second:.2f}")
    
    def get_time_stats(self):
        """Get detailed timing statistics"""
        # Calculate encoding time
        encoding_time = self.encoding_end_time - self.encoding_start_time
        
        # Calculate generation time
        if self.generation_end_time > 0:
            total_generation_time = self.generation_end_time - self.generation_start_time
        else:
            total_generation_time = 0
        
        # Calculate per-token stats
        token_times = [t["time"] for t in self.token_generation_times]
        eviction_times = [t["eviction_time"] for t in self.token_generation_times]
        
        # Avoid division by zero
        if self.generated_token_count > 0:
            avg_token_time = sum(token_times) / self.generated_token_count
            avg_eviction_time = sum(eviction_times) / self.generated_token_count
            tokens_per_second = self.generated_token_count / max(0.001, total_generation_time)
        else:
            avg_token_time = 0
            avg_eviction_time = 0
            tokens_per_second = 0
        
        # Calculate module time stats
        module_time_stats = {}
        for name, times in self.module_times.items():
            if times:
                module_time_stats[name] = {
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
        
        return {
            "encoding_time": encoding_time,
            "total_generation_time": total_generation_time,
            "first_token_time": self.first_token_time if self.generated_token_count > 0 else 0,
            "avg_token_time": avg_token_time,
            "avg_eviction_time": avg_eviction_time,
            "tokens_per_second": tokens_per_second,
            "token_times": self.token_generation_times,
            "eviction_overhead_times": self.eviction_overhead_times,
            "module_times": module_time_stats,
            "input_token_count": self.input_token_count,
            "generated_token_count": self.generated_token_count
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.encoding_start_time = 0
        self.encoding_end_time = 0
        self.generation_start_time = 0
        self.generation_end_time = 0
        self.first_token_time = 0
        self.token_generation_times = []
        self.eviction_overhead_times = []
        self.input_token_count = 0
        self.generated_token_count = 0
        self.module_times = {}
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
