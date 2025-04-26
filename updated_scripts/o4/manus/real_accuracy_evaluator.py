import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import time

class AccuracyEvaluator:
    """
    Evaluates the accuracy of LLM responses using various metrics.
    """
    def __init__(self, model, tokenizer, config, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger or logging.getLogger("AccuracyEvaluator")
        
        # Metrics tracking
        self.perplexity_scores = []
        self.azure_scores = []
        self.per_sample_metrics = []
        
        # Azure evaluation setup
        self.use_azure = config.get("use_azure", False)
        if self.use_azure:
            try:
                from azure_eval import score_with_azure
                self.score_with_azure = score_with_azure
            except ImportError:
                self.logger.warning("Azure evaluation module not found. Azure scoring will be disabled.")
                self.use_azure = False
    
    def calculate_perplexity(self, input_ids, generated_ids):
        """
        Calculate perplexity of the generated sequence as a proxy for quality.
        Lower perplexity generally indicates better quality.
        """
        # Combine input and generated IDs for context
        full_sequence = torch.tensor(input_ids + generated_ids[1:], device=self.model.device).unsqueeze(0)
        
        # We'll evaluate perplexity on just the generated part
        target_ids = full_sequence[:, len(input_ids):]
        
        with torch.no_grad():
            # Get logits from the model
            outputs = self.model(full_sequence[:, :-1])
            logits = outputs.logits[:, len(input_ids)-1:-1, :]  # Only consider logits for generated tokens
            
            # Calculate log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Gather log probs for the actual next tokens
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            
            # Calculate perplexity: exp(-mean(log_probs))
            mean_log_prob = token_log_probs.mean().item()
            perplexity = np.exp(-mean_log_prob)
            
            return perplexity
    
    def evaluate_with_azure(self, prompt, response):
        """
        Evaluate response using Azure OpenAI as a judge.
        Returns a score from 1-10.
        """
        if not self.use_azure:
            self.logger.warning("Azure evaluation is disabled. Returning None.")
            return None
        
        try:
            score = self.score_with_azure(prompt, response, self.config)
            return score
        except Exception as e:
            self.logger.error(f"Error in Azure evaluation: {e}")
            return None
    
    def evaluate_response(self, prompt, response, input_ids, generated_ids):
        """
        Evaluate a single response using multiple metrics.
        """
        start_time = time.time()
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(input_ids, generated_ids)
        self.perplexity_scores.append(perplexity)
        
        # Azure evaluation if enabled
        azure_score = None
        if self.use_azure:
            azure_score = self.evaluate_with_azure(prompt, response)
            if azure_score is not None:
                self.azure_scores.append(azure_score)
        
        # Record metrics for this sample
        metrics = {
            "perplexity": perplexity,
            "azure_score": azure_score,
            "evaluation_time": time.time() - start_time
        }
        
        self.per_sample_metrics.append(metrics)
        return metrics
    
    def get_accuracy_stats(self):
        """
        Get comprehensive accuracy statistics.
        """
        stats = {
            "perplexity": {
                "mean": np.mean(self.perplexity_scores) if self.perplexity_scores else None,
                "min": min(self.perplexity_scores) if self.perplexity_scores else None,
                "max": max(self.perplexity_scores) if self.perplexity_scores else None,
                "std": np.std(self.perplexity_scores) if self.perplexity_scores else None,
                "values": self.perplexity_scores
            }
        }
        
        if self.azure_scores:
            stats["azure"] = {
                "mean": np.mean(self.azure_scores),
                "min": min(self.azure_scores),
                "max": max(self.azure_scores),
                "std": np.std(self.azure_scores),
                "values": self.azure_scores
            }
        
        stats["per_sample"] = self.per_sample_metrics
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self.perplexity_scores = []
        self.azure_scores = []
        self.per_sample_metrics = []
