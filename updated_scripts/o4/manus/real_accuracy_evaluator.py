import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

class AccuracyEvaluator:
    """
    Evaluates the accuracy of LLM responses using multiple metrics.
    
    This class measures:
    - Perplexity as a proxy for quality
    - Azure OpenAI evaluation when available
    - Token-level comparison with reference outputs when available
    """
    def __init__(self, model, tokenizer, config, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger or logging.getLogger("AccuracyEvaluator")
        
        # Metrics tracking
        self.perplexity_values = []
        self.azure_scores = []
        self.sample_metrics = []
        
        # Azure evaluation setup
        self.use_azure = config.get("use_azure", False)
        if self.use_azure:
            try:
                from openai import AzureOpenAI
                self.azure_client = AzureOpenAI(
                    api_key=config["azure"]["api_key"],
                    azure_endpoint=config["azure"]["endpoint"],
                    api_version=config["azure"]["api_version"]
                )
                self.azure_deployment = config["azure"]["deployment_name"]
                self.logger.info("Azure OpenAI client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                self.use_azure = False
    
    def evaluate_response(self, prompt, response, input_tokens, output_tokens):
        """
        Evaluate the quality of a generated response using multiple metrics.
        
        Args:
            prompt: The input prompt
            response: The generated response
            input_tokens: List of input token IDs
            output_tokens: List of all token IDs (input + output)
        
        Returns:
            Dictionary of accuracy metrics
        """
        metrics = {}
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(input_tokens, output_tokens)
        metrics["perplexity"] = perplexity
        self.perplexity_values.append(perplexity)
        
        # Azure evaluation if enabled
        if self.use_azure:
            azure_score = self.evaluate_with_azure(prompt, response)
            metrics["azure_score"] = azure_score
            if azure_score is not None:
                self.azure_scores.append(azure_score)
        
        # Record sample metrics
        self.sample_metrics.append(metrics)
        
        return metrics
    
    def calculate_perplexity(self, input_tokens, output_tokens):
        """
        Calculate perplexity of the generated sequence.
        
        Perplexity is a measure of how well a probability model predicts a sample.
        Lower perplexity indicates better prediction quality.
        """
        # We need at least input + 1 output token
        if len(output_tokens) <= len(input_tokens):
            return None
        
        # Get only the generated part
        generated_tokens = output_tokens[len(input_tokens):]
        
        # Create input context (we need this to get the logits for the generated tokens)
        context_length = min(len(input_tokens), 32)  # Limit context to avoid OOM
        context = input_tokens[-context_length:]
        
        # Convert to tensor
        input_ids = torch.tensor([context], device=self.model.device)
        
        # Get logits for each generated token
        log_probs = []
        
        with torch.no_grad():
            # Initial forward pass
            outputs = self.model(input_ids=input_ids)
            past_key_values = outputs.past_key_values
            
            # For each generated token
            for i, token in enumerate(generated_tokens):
                # Get logits for next token
                if i > 0:
                    # Use past_key_values for efficiency
                    token_input = torch.tensor([[token]], device=self.model.device)
                    outputs = self.model(input_ids=token_input, past_key_values=past_key_values)
                    past_key_values = outputs.past_key_values
                
                # Get probability of the actual next token
                next_token_idx = generated_tokens[i] if i < len(generated_tokens) - 1 else generated_tokens[-1]
                logits = outputs.logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                prob = probs[0, next_token_idx].item()
                
                # Avoid log(0)
                prob = max(prob, 1e-10)
                log_prob = np.log(prob)
                log_probs.append(log_prob)
        
        # Calculate perplexity
        if not log_probs:
            return None
        
        avg_log_prob = sum(log_probs) / len(log_probs)
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity
    
    def evaluate_with_azure(self, prompt, response):
        """
        Evaluate the response using Azure OpenAI as a judge.
        
        Returns a score from 1-10 based on the quality of the response.
        """
        if not self.use_azure:
            return None
        
        try:
            # Construct evaluation prompt
            eval_prompt = f"""
            You are an expert evaluator for large language model outputs.
            Please evaluate the following response to the given prompt on a scale of 1-10,
            where 1 is extremely poor and 10 is excellent.
            
            Consider factors such as:
            - Accuracy and factual correctness
            - Relevance to the prompt
            - Coherence and clarity
            - Depth and thoroughness
            
            Prompt: {prompt}
            
            Response: {response}
            
            Provide your evaluation as a single number between 1 and 10.
            """
            
            # Call Azure OpenAI
            completion = self.azure_client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for large language model outputs."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            # Extract score
            score_text = completion.choices[0].message.content.strip()
            
            # Try to parse the score
            try:
                # Extract just the number
                score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
                
                # Ensure it's in range 1-10
                score = max(1, min(10, score))
                
                return score
            except:
                self.logger.warning(f"Failed to parse Azure score: {score_text}")
                return None
            
        except Exception as e:
            self.logger.error(f"Azure evaluation failed: {e}")
            return None
    
    def get_accuracy_stats(self):
        """Get detailed accuracy statistics"""
        stats = {}
        
        # Perplexity stats
        if self.perplexity_values:
            valid_values = [p for p in self.perplexity_values if p is not None]
            if valid_values:
                stats["perplexity"] = {
                    "mean": sum(valid_values) / len(valid_values),
                    "min": min(valid_values),
                    "max": max(valid_values),
                    "values": valid_values
                }
        
        # Azure stats
        if self.azure_scores:
            stats["azure"] = {
                "mean": sum(self.azure_scores) / len(self.azure_scores),
                "min": min(self.azure_scores),
                "max": max(self.azure_scores),
                "values": self.azure_scores
            }
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self.perplexity_values = []
        self.azure_scores = []
        self.sample_metrics = []
