#!/usr/bin/env python3
"""
Robust Real Benchmark for KV Cache Management Strategies

This script evaluates different KV cache management strategies using real metrics:
- KV cache size (real physical memory usage)
- Inference time (total, per token, time to first token)
- Accuracy (perplexity and optional Azure evaluation)

The implementation properly applies eviction strategies to the actual KV cache
and measures their impact with real physical metrics.
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from complete_real_kv_cache_manager import RealKVCacheManager
from fixed_kv_cache_dashboard_final import KVCacheDashboard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger('real_benchmark')

# Optional imports for memory profiling
try:
    import psutil
    import gc
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, memory profiling will be limited")

# Optional imports for Azure evaluation
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai not available, Azure evaluation will be disabled")


class EnhancedBaseline:
    """Baseline strategy that does no eviction"""
    def __init__(self, **kwargs):
        self.name = "Baseline"


class RandomKVCacheStrategy:
    """Random KV cache management strategy"""
    def __init__(self, keep=0.7, **kwargs):
        self.keep = keep
        self.name = f"Random(keep={self.keep})"


class AttentionTopStrategy:
    """Attention-based top strategy: keep tokens with highest attention scores"""
    def __init__(self, keep=0.7, **kwargs):
        self.keep = keep
        self.name = f"AttentionTop(keep={self.keep})"


class AttentionBottomStrategy:
    """Attention-based bottom strategy: keep tokens with lowest attention scores"""
    def __init__(self, keep=0.7, **kwargs):
        self.keep = keep
        self.name = f"AttentionBottom(keep={self.keep})"


class HybridNPercentStrategy:
    """Hybrid strategy: combine recency, attention, and token type importance"""
    def __init__(self, keep=0.7, recency_weight=0.5, attention_weight=0.3, type_weight=0.2, **kwargs):
        self.keep = keep
        self.recency_weight = recency_weight
        self.attention_weight = attention_weight
        self.type_weight = type_weight
        self.name = f"HybridNPercent(keep={self.keep},r={self.recency_weight},a={self.attention_weight},t={self.type_weight})"


class SlidingWindowStrategy:
    """Sliding window strategy: keep recent tokens and important tokens"""
    def __init__(self, window=0.7, important=0.1, **kwargs):
        self.window = window
        self.important = important
        self.name = f"SlidingWindow(window={self.window},important={self.important})"


class AdaptiveAttentionStrategy:
    """Adaptive attention strategy: keep tokens based on attention distribution"""
    def __init__(self, base_keep=0.7, **kwargs):
        self.base_keep = base_keep
        self.name = f"AdaptiveAttention(base_keep={self.base_keep})"


def load_model_and_tokenizer(model_name: str, cache_dir: Optional[str] = None) -> Tuple:
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_name: Name of the model to load
        cache_dir: Directory to cache model files
        
    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading model {model_name} from {cache_dir if cache_dir else 'default cache'}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def setup_tokenizer_padding(tokenizer):
    """
    Set up padding token for the tokenizer if it doesn't have one.
    This is necessary for Llama tokenizers which don't have a padding token by default.
    """
    # Check if the tokenizer already has a padding token
    if tokenizer.pad_token is None:
        # Use the EOS token as the padding token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        else:
            # If there's no EOS token either, add a new special token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added [PAD] as pad_token")
    
    return tokenizer


def get_token_types(tokens: List[int], tokenizer) -> List[str]:
    """
    Classify tokens into types: special, rare, common, punctuation.
    
    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer to decode tokens
        
    Returns:
        List of token types
    """
    token_types = []
    
    try:
        # Get vocabulary size
        vocab_size = len(tokenizer.get_vocab())
        
        # Define thresholds
        special_tokens = set(tokenizer.all_special_ids)
        rare_threshold = int(vocab_size * 0.9)  # Top 10% are rare
        
        # Punctuation characters
        punctuation = set(".,;:!?()[]{}-_\"'`/\\<>@#$%^&*+=|~")
        
        for token in tokens:
            if token in special_tokens:
                token_types.append("special")
            elif token > rare_threshold:
                token_types.append("rare")
            else:
                # Check if it's punctuation
                token_str = tokenizer.decode([token])
                if any(c in punctuation for c in token_str):
                    token_types.append("punctuation")
                else:
                    token_types.append("common")
    except Exception as e:
        logger.warning(f"Error classifying tokens: {e}")
        # Fallback: mark all as common
        token_types = ["common"] * len(tokens)
    
    return token_types


def get_prompt(data_item: Union[Dict, List, str]) -> str:
    """
    Extract prompt from data item, handling various data structures.
    
    Args:
        data_item: Data item from dataset
        
    Returns:
        Prompt string
    """
    logger.debug(f"Extracting prompt from data item type: {type(data_item)}")
    
    try:
        # If it's a string, return directly
        if isinstance(data_item, str):
            return data_item
        
        # If it's a list, handle MTBench format
        if isinstance(data_item, list):
            # MTBench format: list of turns
            prompt = ""
            for turn in data_item:
                if isinstance(turn, dict):
                    if "role" in turn and "content" in turn:
                        role = turn["role"]
                        content = turn["content"]
                        prompt += f"{role}: {content}\n"
                    elif "question" in turn:
                        prompt += f"User: {turn['question']}\n"
                    elif "text" in turn:
                        prompt += f"{turn['text']}\n"
                elif isinstance(turn, str):
                    prompt += f"{turn}\n"
            return prompt.strip()
        
        # If it's a dict, check common fields
        if isinstance(data_item, dict):
            # Check common fields
            for field in ["prompt", "text", "question", "input", "instruction"]:
                if field in data_item and isinstance(data_item[field], str):
                    return data_item[field]
            
            # Check for MTBench format
            if "turns" in data_item and isinstance(data_item["turns"], list):
                return get_prompt(data_item["turns"])
            
            # Check for conversation format
            if "conversations" in data_item and isinstance(data_item["conversations"], list):
                prompt = ""
                for turn in data_item["conversations"]:
                    if isinstance(turn, dict) and "value" in turn:
                        prompt += f"{turn.get('from', 'Speaker')}: {turn['value']}\n"
                return prompt.strip()
        
        # If we couldn't extract a prompt, log a warning and return empty string
        logger.warning(f"Could not extract prompt from data item: {data_item}")
        return ""
    
    except Exception as e:
        logger.error(f"Error extracting prompt: {e}")
        return ""


def load_dataset(dataset_path: str) -> List:
    """
    Load dataset from file.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        List of data items
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        with open(dataset_path, 'r') as f:
            if dataset_path.endswith('.json'):
                data = json.load(f)
                
                # Handle different dataset formats
                if isinstance(data, dict) and "data" in data:
                    # Format: {"data": [...]}
                    return data["data"]
                elif isinstance(data, list):
                    # Format: [...]
                    return data
                else:
                    # Format: {...}
                    return [data]
            else:
                # Assume text file with one sample per line
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def profile_memory():
    """
    Profile current memory usage.
    
    Returns:
        Dict with memory usage statistics
    """
    if not HAS_PSUTIL:
        return {}
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}_allocated"] = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
                gpu_memory[f"gpu_{i}_reserved"] = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
        
        return {
            "rss_mb": memory_info.rss / (1024 ** 2),  # MB
            "vms_mb": memory_info.vms / (1024 ** 2),  # MB
            **gpu_memory
        }
    except Exception as e:
        logger.warning(f"Error profiling memory: {e}")
        return {}


def calculate_perplexity(logits, target_ids):
    """
    Calculate perplexity from logits and target IDs.
    
    Args:
        logits: Model logits
        target_ids: Target token IDs
        
    Returns:
        Perplexity value
    """
    try:
        # Create a loss function
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # Reshape logits to (batch_size * seq_len, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        
        # Reshape target_ids to (batch_size * seq_len)
        target_ids = target_ids.view(-1)
        
        # Calculate loss
        loss = loss_fn(logits, target_ids)
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        return perplexity
    except Exception as e:
        logger.warning(f"Error calculating perplexity: {e}")
        return float('inf')


def evaluate_with_azure(prompt, response, api_key=None, endpoint=None):
    """
    Evaluate response using Azure OpenAI.
    
    Args:
        prompt: Input prompt
        response: Model response
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint
        
    Returns:
        Evaluation score
    """
    if not HAS_OPENAI:
        logger.warning("openai package not available, skipping Azure evaluation")
        return 0
    
    if not api_key or not endpoint:
        logger.warning("Azure API key or endpoint not provided, skipping Azure evaluation")
        return 0
    
    try:
        # Configure OpenAI client
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )
        
        # Create evaluation prompt
        eval_prompt = f"""
        Please evaluate the quality of the following response to the given prompt.
        
        Prompt:
        {prompt}
        
        Response:
        {response}
        
        Rate the response on a scale from 1 to 10, where:
        1 = Completely irrelevant or incorrect
        10 = Perfect, comprehensive, and accurate
        
        Provide only the numeric score.
        """
        
        # Get evaluation
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that evaluates the quality of responses."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        # Extract score
        score_text = completion.choices[0].message.content.strip()
        
        # Try to parse score
        try:
            score = float(score_text)
            return score
        except ValueError:
            # If we can't parse a number, try to extract it from text
            import re
            match = re.search(r'(\d+(\.\d+)?)', score_text)
            if match:
                return float(match.group(1))
            else:
                logger.warning(f"Could not parse score from Azure response: {score_text}")
                return 0
    
    except Exception as e:
        logger.warning(f"Error evaluating with Azure: {e}")
        return 0


def process_single_prompt(model, tokenizer, prompt, strat, cache_manager, max_gen_tokens=100, azure_eval=False, azure_config=None):
    """
    Process a single prompt with the given strategy.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        strat: KV cache management strategy
        cache_manager: KV cache manager
        max_gen_tokens: Maximum number of tokens to generate
        azure_eval: Whether to evaluate with Azure
        azure_config: Azure configuration
        
    Returns:
        Dict with results
    """
    logger.info(f"Processing prompt with strategy: {strat.name}")
    
    try:
        # Reset model state
        model.config.use_cache = True
        if hasattr(model, "past_key_values") and model.past_key_values is not None:
            model.past_key_values = None
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Initialize metrics
        start_time = time.time()
        token_times = {}
        attention_scores = None
        token_types = []
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)
        input_ids = inputs.input_ids
        
        # Record encoding time
        encoding_time = time.time() - start_time
        logger.debug(f"Encoding time: {encoding_time:.4f}s")
        
        # Start generation
        generated_ids = input_ids.clone()
        first_token_time = None
        
        # Time the first token generation
        first_token_start = time.time()
        
        # Generate first token
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        
        # Record first token time
        first_token_time = time.time() - first_token_start
        logger.debug(f"First token time: {first_token_time:.4f}s")
        
        # Get next token and add to generated sequence
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        
        # If next_token has multiple elements, just take the first one
        if next_token.numel() > 1:
            next_token_id = next_token[0].item()
            logger.warning(f"Multiple tokens generated, using first one: {next_token}")
        else:
            next_token_id = next_token.item()
        
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
        
        # Prepare for next iteration - reshape to ensure it's a single token
        token = torch.tensor([[next_token_id]], device=model.device)
        
        # Extract attention scores if available
        if hasattr(out, "attentions") and out.attentions is not None:
            attention_scores = out.attentions
        
        # Get token types
        token_types = get_token_types(input_ids[0].tolist(), tokenizer)
        
        # Apply KV cache management strategy
        past = model.past_key_values
        
        # Update memory stats
        cache_manager._update_memory_stats(past)
        
        # Apply strategy
        if "HybridNPercent" in strat.name:
            # For HybridNPercent, we need to use the hybrid strategy
            past = cache_manager._apply_hybrid_strategy(
                past, 
                attention_scores, 
                token_types,
                keep_ratio=getattr(strat, 'keep', 0.7),
                recency_weight=getattr(strat, 'recency_weight', 0.5),
                attention_weight=getattr(strat, 'attention_weight', 0.3),
                type_weight=getattr(strat, 'type_weight', 0.2)
            )
        else:
            # For other strategies, use the normal apply_eviction_strategy method
            past = cache_manager.apply_eviction_strategy(past, strat.name, attention_scores)
        
        # Update model's past_key_values
        model.past_key_values = past
        
        # Continue generating tokens
        for i in range(1, max_gen_tokens):
            token_start = time.time()
            
            # Generate next token
            with torch.no_grad():
                out = model(input_ids=token, use_cache=True)
            
            # Record token generation time
            token_time = time.time() - token_start
            token_times[i] = token_time
            
            # Get next token and add to generated sequence
            next_token = out.logits[:, -1, :].argmax(dim=-1)
            
            # If next_token has multiple elements, just take the first one
            if next_token.numel() > 1:
                next_token_id = next_token[0].item()
            else:
                next_token_id = next_token.item()
            
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # Check for end of generation
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Prepare for next iteration
            token = torch.tensor([[next_token_id]], device=model.device)
            
            # Extract attention scores if available
            if hasattr(out, "attentions") and out.attentions is not None:
                attention_scores = out.attentions
            
            # Extend token types
            token_types.append(get_token_types([next_token_id], tokenizer)[0])
            
            # Apply KV cache management strategy
            past = model.past_key_values
            
            # Update memory stats
            cache_manager._update_memory_stats(past)
            
            # Apply strategy
            if "HybridNPercent" in strat.name:
                # For HybridNPercent, we need to use the hybrid strategy
                past = cache_manager._apply_hybrid_strategy(
                    past, 
                    attention_scores, 
                    token_types,
                    keep_ratio=getattr(strat, 'keep', 0.7),
                    recency_weight=getattr(strat, 'recency_weight', 0.5),
                    attention_weight=getattr(strat, 'attention_weight', 0.3),
                    type_weight=getattr(strat, 'type_weight', 0.2)
                )
            else:
                # For other strategies, use the normal apply_eviction_strategy method
                past = cache_manager.apply_eviction_strategy(past, strat.name, attention_scores)
            
            # Update model's past_key_values
            model.past_key_values = past
        
        # End generation
        total_time = time.time() - start_time
        logger.debug(f"Total generation time: {total_time:.4f}s")
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Calculate perplexity
        perplexity = calculate_perplexity(out.logits, token)
        
        # Evaluate with Azure if requested
        azure_score = 0
        if azure_eval and azure_config:
            azure_score = evaluate_with_azure(
                prompt, 
                generated_text, 
                api_key=azure_config.get("api_key"),
                endpoint=azure_config.get("endpoint")
            )
        
        # Get memory stats
        memory_stats = cache_manager.get_memory_stats()
        
        # Return results
        return {
            "strategy": strat.name,
            "prompt": prompt,
            "response": generated_text,
            "time": {
                "total_time": total_time,
                "encoding_time": encoding_time,
                "first_token_time": first_token_time,
                "token_times": token_times,
                "tokens_generated": len(generated_ids[0]) - len(input_ids[0])
            },
            "memory": memory_stats,
            "accuracy": {
                "perplexity": perplexity,
                "azure_score": azure_score
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        logger.error(traceback.format_exc())
        
        # Return error result
        return {
            "strategy": strat.name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark KV cache management strategies")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--eval_azure", action="store_true", help="Evaluate with Azure OpenAI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--profile", action="store_true", help="Enable memory profiling")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--generate_dashboard", action="store_true", help="Generate dashboard")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    # Create output directory
    output_dir = Path(cfg.get("output_dir", "benchmark_results"))
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(cfg["model_name"], cfg["cache_dir"])
    
    # Set up padding token for the tokenizer
    tokenizer = setup_tokenizer_padding(tokenizer)
    
    logger.info(f"Loaded model {cfg['model_name']}")
    
    # Load dataset
    dataset_cfg = cfg.get("dataset", {})
    dataset_path = dataset_cfg.get("local_path")
    dataset = load_dataset(dataset_path)
    
    # Limit dataset if requested
    if args.limit:
        dataset = dataset[:args.limit]
    
    # Initialize strategies
    strategies = []
    for strat_spec in cfg.get("strategies", ["Baseline"]):
        if isinstance(strat_spec, str):
            # Parse strategy specification
            if strat_spec == "Baseline":
                strategies.append(EnhancedBaseline())
            elif strat_spec.startswith("Random"):
                # Extract parameters
                keep = 0.7  # Default
                if "keep=" in strat_spec:
                    try:
                        keep = float(strat_spec.split("keep=")[1].split(")")[0])
                    except:
                        pass
                strategies.append(RandomKVCacheStrategy(keep=keep))
            elif strat_spec.startswith("AttentionTop"):
                # Extract parameters
                keep = 0.7  # Default
                if "keep=" in strat_spec:
                    try:
                        keep = float(strat_spec.split("keep=")[1].split(")")[0])
                    except:
                        pass
                strategies.append(AttentionTopStrategy(keep=keep))
            elif strat_spec.startswith("AttentionBottom"):
                # Extract parameters
                keep = 0.7  # Default
                if "keep=" in strat_spec:
                    try:
                        keep = float(strat_spec.split("keep=")[1].split(")")[0])
                    except:
                        pass
                strategies.append(AttentionBottomStrategy(keep=keep))
            elif strat_spec.startswith("HybridNPercent"):
                # Extract parameters
                keep = 0.7  # Default
                recency_weight = 0.5  # Default
                attention_weight = 0.3  # Default
                type_weight = 0.2  # Default
                
                if "keep=" in strat_spec:
                    try:
                        keep = float(strat_spec.split("keep=")[1].split(",")[0])
                    except:
                        pass
                if "r=" in strat_spec:
                    try:
                        recency_weight = float(strat_spec.split("r=")[1].split(",")[0])
                    except:
                        pass
                if "a=" in strat_spec:
                    try:
                        attention_weight = float(strat_spec.split("a=")[1].split(",")[0])
                    except:
                        pass
                if "t=" in strat_spec:
                    try:
                        type_weight = float(strat_spec.split("t=")[1].split(")")[0])
                    except:
                        pass
                
                strategies.append(HybridNPercentStrategy(
                    keep=keep,
                    recency_weight=recency_weight,
                    attention_weight=attention_weight,
                    type_weight=type_weight
                ))
            elif strat_spec.startswith("SlidingWindow"):
                # Extract parameters
                window = 0.7  # Default
                important = 0.1  # Default
                
                if "window=" in strat_spec:
                    try:
                        window = float(strat_spec.split("window=")[1].split(",")[0])
                    except:
                        pass
                if "important=" in strat_spec:
                    try:
                        important = float(strat_spec.split("important=")[1].split(")")[0])
                    except:
                        pass
                
                strategies.append(SlidingWindowStrategy(window=window, important=important))
            elif strat_spec.startswith("AdaptiveAttention"):
                # Extract parameters
                base_keep = 0.7  # Default
                if "base_keep=" in strat_spec:
                    try:
                        base_keep = float(strat_spec.split("base_keep=")[1].split(")")[0])
                    except:
                        pass
                
                strategies.append(AdaptiveAttentionStrategy(base_keep=base_keep))
            else:
                logger.warning(f"Unknown strategy: {strat_spec}, using Baseline")
                strategies.append(EnhancedBaseline())
    
    # Ensure we have at least one strategy
    if not strategies:
        strategies.append(EnhancedBaseline())
    
    # Log strategies
    for strat in strategies:
        logger.info(f"Benchmarking strategy: {strat.name}")
        logger.info(f"Created strategy instance: {strat.__class__.__name__}")
    
    # Initialize Azure config if needed
    azure_config = None
    if args.eval_azure:
        azure_config = cfg.get("azure", {})
        if not azure_config.get("api_key") or not azure_config.get("endpoint"):
            logger.warning("Azure API key or endpoint not provided, disabling Azure evaluation")
            args.eval_azure = False
    
    # Benchmark each strategy
    for strat in strategies:
        logger.info(f"Benchmarking strategy: {strat.name}")
        
        # Initialize KV cache manager
        cache_manager = RealKVCacheManager(
            model=model,
            tokenizer=tokenizer,
            cfg={"kv_threshold": cfg.get("kv_threshold", 1000)},
            logger=logger
        )
        
        # Set logger level to INFO to see memory logs
        cache_manager.logger.setLevel(logging.INFO)
        
        # Process dataset
        results = []
        
        for i, item in enumerate(dataset):
            try:
                # Extract prompt
                prompt_field = dataset_cfg.get("prompt_field", None)
                if prompt_field:
                    if isinstance(item, dict) and prompt_field in item:
                        prompt = get_prompt(item[prompt_field])
                    else:
                        prompt = get_prompt(item)
                else:
                    prompt = get_prompt(item)
                
                # Skip empty prompts
                if not prompt:
                    logger.warning(f"Skipping empty prompt at index {i}")
                    continue
                
                # Process prompt
                result = process_single_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    strat=strat,
                    cache_manager=cache_manager,
                    max_gen_tokens=cfg.get("max_gen_tokens", 100),
                    azure_eval=args.eval_azure,
                    azure_config=azure_config
                )
                
                # Add result
                results.append(result)
                
                # Profile memory if requested
                if args.profile:
                    memory_profile = profile_memory()
                    logger.info(f"Memory profile after sample {i}: {memory_profile}")
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.info(f"Processed sample {i} with strategy {strat.name}")
            
            except Exception as e:
                logger.warning(f"Skipping sample {i} due to processing error: {e}")
                continue
        
        # Save results
        results_file = output_dir / f"{strat.name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results for strategy {strat.name} to {results_file}")
    
    # Generate dashboard if requested
    if args.generate_dashboard:
        logger.info("Generating dashboard")
        dashboard = KVCacheDashboard(output_dir)
        dashboard.generate_dashboard()
        logger.info(f"Dashboard generated at {output_dir}/dashboard")
    
    logger.info("Benchmark completed")


if __name__ == "__main__":
    main()
