# enhanced_benchmark.py
import argparse
import time
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.cuda
from tqdm import tqdm

from utils import (
    load_config,
    ensure_dataset,
    load_model_and_tokenizer,
    save_json,
)
from enhanced_strategies import EnhancedBaseline, SlidingWindowStrategy, AdaptiveAttentionStrategy
from attention_utils import extract_layerwise_attention, dynamic_token_importance
from kv_cache_manager import KVCacheManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_benchmark")

# Common prompt keys for different dataset formats
COMMON_PROMPT_KEYS = ["prompt", "instruction", "input", "text"]

def get_prompt(sample, cfg):
    # Extract prompt from sample using various format conventions
    # 1) If sample is just a string
    if isinstance(sample, str):
        return sample

    # 2) If user specified a prompt_field in config
    pf = cfg["dataset"].get("prompt_field")
    if pf and pf in sample:
        return sample[pf]

    # 3) MT-Bench style with "turns"
    turns = sample.get("turns")
    if isinstance(turns, list) and len(turns) > 0:
        return turns[0]

    # 4) Fallback to common keys
    for k in COMMON_PROMPT_KEYS:
        if k in sample:
            return sample[k]

    # 5) Give up
    raise KeyError(f"No prompt found in sample keys {list(sample.keys())}")

def get_enhanced_strategy(name, threshold, config):
    """Get strategy implementation based on name and configuration"""
    if name == "Baseline":
        return EnhancedBaseline(threshold)
    
    if name.startswith("SlidingWindow"):
        # Extract parameters from name: SlidingWindow(window=0.7,important=0.1)
        params = {}
        if "(" in name:
            param_str = name.split("(")[1].rstrip(")")
            for p in param_str.split(","):
                if "=" in p:
                    k, v = p.split("=")
                    params[k.strip()] = float(v.strip())
        
        window_size = params.get("window", 0.7)
        important_ratio = params.get("important", 0.1)
        return SlidingWindowStrategy(threshold, window_size, important_ratio)
    
    if name.startswith("AdaptiveAttention"):
        # Extract parameters from name: AdaptiveAttention(base_keep=0.7)
        params = {}
        if "(" in name:
            param_str = name.split("(")[1].rstrip(")")
            for p in param_str.split(","):
                if "=" in p:
                    k, v = p.split("=")
                    params[k.strip()] = float(v.strip())
        
        base_keep = params.get("base_keep", 0.7)
        return AdaptiveAttentionStrategy(threshold, base_keep)
    
    # Fallback to baseline
    logger.warning(f"Unknown strategy {name}, falling back to Baseline")
    return EnhancedBaseline(threshold)

def profile_memory():
    """Profile current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved": torch.cuda.memory_reserved() / (1024 ** 2),
            "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 2)
        }
    return {"allocated": 0, "reserved": 0, "max_allocated": 0}

# Updated benchmark code for newer LLaMA models
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_template.json")
    parser.add_argument("--eval_azure", action="store_true", help="Evaluate with Azure")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--profile", action="store_true", help="Enable memory profiling")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    cfg = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Load dataset
    dataset = ensure_dataset(cfg)
    if args.limit:
        dataset = dataset[:args.limit]
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(cfg["model_name"], cfg["cache_dir"])
    logger.info(f"Loaded model {cfg['model_name']}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Record global stats for reporting
    global_stats = {
        "model_name": cfg["model_name"],
        "dataset_size": len(dataset),
        "max_gen_tokens": cfg["max_gen_tokens"],
        "kv_threshold": cfg["kv_threshold"],
        "strategies": [],
        "memory_profile": profile_memory() if args.profile else {}
    }
    
    # Benchmark each strategy
    for strat_name in cfg["strategies"]:
        logger.info(f"Benchmarking strategy: {strat_name}")
        
        # Create strategy instance
        strat = get_enhanced_strategy(strat_name, cfg["kv_threshold"], cfg)
        logger.info(f"Created strategy instance: {strat.name}")
        
        # Create KV cache manager
        cache_manager = KVCacheManager(model, tokenizer, cfg, logger)
        
        # Per-strategy metrics
        logs = []
        strategy_metrics = {
            "name": strat_name,
            "total_time": 0,
            "total_tokens": 0,
            "memory_usage": [],
            "token_times": []
        }
        
        # Process each sample
        for sample_idx, sample in enumerate(tqdm(dataset[:11], desc=strat_name)):
            # Extract metadata
            question_id = sample.get("question_id", f"q{sample_idx}")
            category = sample.get("category", "uncategorized")
            
            # Get the prompt
            prompt = get_prompt(sample, cfg)
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_tokens = inputs.input_ids[0].tolist()
            
            # Record memory before processing
            if args.profile:
                pre_memory = profile_memory()
                strategy_metrics["memory_usage"].append(pre_memory)
            
            # === Generate response with timing ===
            generation_start = time.time()
            times = []
            sizes = []
            
            # === Encode-only prefix ===
            start = time.time()
            with torch.no_grad():
                # Important: use_cache=True but don't pass past_key_values
                out = model(**inputs, use_cache=True, output_attentions=True)
            times.append(time.time() - start)
            
            # Get the cache object from the output
            past = out.past_key_values  # This is now a LlamaCache object
            sizes.append(cache_manager.compute_cache_size(past))
            
            # Extract attention scores and token types for the prompt
            attention_scores = extract_layerwise_attention(out.attentions)
            token_types = cache_manager.identify_token_types(input_tokens)
            
            # Prepare for generation
            token = inputs.input_ids[:, -1:].to(model.device)
            gen_tokens = input_tokens
            generated_text = ""
            
            # For new models, we can't directly manipulate the cache
            # We'll need to use the model's built-in functions
            # or just benchmark without eviction for now
            
            # === Generation loop ===
            for i in range(cfg["max_gen_tokens"]):
                # Generate next token
                start = time.time()
                with torch.no_grad():
                    out = model(
                        input_ids=token, 
                        past_key_values=past,  # Pass the cache object
                        use_cache=True, 
                        output_attentions=True
                    )
                elapsed = time.time() - start
                times.append(elapsed)
                strategy_metrics["token_times"].append(elapsed)
                
                # Update KV cache metrics
                past = out.past_key_values
                current_size = cache_manager.compute_cache_size(past)
                sizes.append(current_size)
                cache_manager.track_memory(current_size)
                
                # Update attention scores with new information
                new_attention = extract_layerwise_attention(out.attentions)
                
                # Get next token and add to generated sequence
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                next_token_id = next_token.item()
                gen_tokens.append(next_token_id)
                
                # Update token types with the new token
                new_token_type = cache_manager.identify_token_types([next_token_id])[0]
                token_types.append(new_token_type)
                
                # Decode the newest token and add to generated text
                token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                generated_text += token_text
                
                # Stop if we hit an end-of-sequence token
                if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                    break
                
                # Prepare for next iteration
                token = next_token
            
            # === End of generation ===
            generation_time = time.time() - generation_start
            
            # Record final memory usage
            if args.profile:
                post_memory = profile_memory()
                
            # Decode the full sequence for better handling of word boundaries
            generated = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Accuracy evaluation (if requested)
            accuracy = None
            if args.eval_azure:
                from azure_eval import score_with_azure
                accuracy = score_with_azure(prompt, generated, cfg)
            
            # Calculate metrics
            total_tokens = len(gen_tokens) - len(input_tokens)
            strategy_metrics["total_tokens"] += total_tokens
            strategy_metrics["total_time"] += generation_time
            
            # Build detailed log entry
            log = {
                "question_id": question_id,
                "category": category,
                "prompt": prompt,
                "response": generated,
                "strategy": strat_name,
                "sizes": {
                    "avg": sum(sizes) / len(sizes),
                    "peak": max(sizes),
                    "history": sizes,
                    "layer_breakdown": cache_manager.compute_layer_sizes(past)
                },
                "times": {
                    "total": generation_time,
                    "per_token": generation_time / max(1, total_tokens),
                    "first_token": times[0],
                    "breakdown": times
                },
                "tokens": {
                    "input_length": len(input_tokens),
                    "output_length": total_tokens,
                    "total_length": len(gen_tokens)
                },
                "cache_stats": cache_manager.get_memory_stats()
            }
            
            if accuracy is not None:
                log["accuracy"] = accuracy
            
            logs.append(log)
            
            # Log progress
            if (sample_idx + 1) % 5 == 0 or sample_idx == len(dataset) - 1:
                logger.info(f"Processed {sample_idx + 1}/{len(dataset)} samples with {strat_name}")
        
        # Save per-strategy logs
        save_json(logs, cfg, "per_strategy", strat_name)
        logger.info(f"Saved logs for {strat_name} to {output_dir/'per_strategy'/f'{strat_name}.json'}")
        
        # Add strategy summary to global stats
        tokens_per_second = strategy_metrics["total_tokens"] / max(0.001, strategy_metrics["total_time"])
        avg_per_token = strategy_metrics["total_time"] / max(1, strategy_metrics["total_tokens"])
        
        strategy_summary = {
            "name": strat_name,
            "tokens_per_second": tokens_per_second,
            "avg_token_time": avg_per_token,
            "total_generation_time": strategy_metrics["total_time"],
            "total_tokens_generated": strategy_metrics["total_tokens"],
            "peak_kv_cache_mb": cache_manager.peak_memory,
            "eviction_count": cache_manager.eviction_count,
            "avg_eviction_time": cache_manager.total_eviction_time / max(1, cache_manager.eviction_count)
        }
        
        global_stats["strategies"].append(strategy_summary)
    
    # Save global stats summary
    save_json(global_stats, cfg, ".", "benchmark_summary")
    logger.info(f"Saved benchmark summary to {output_dir/'benchmark_summary.json'}")
    
    # Generate comparative report
    generate_comparative_report(global_stats, output_dir)
    logger.info(f"Generated comparative report at {output_dir/'comparative_report.md'}")

def generate_comparative_report(stats, output_dir):
    """Generate a markdown report comparing strategy performance"""
    report = [
        "# KV Cache Strategy Benchmark Report\n",
        f"## Model: {stats['model_name']}\n",
        f"Dataset size: {stats['dataset_size']} samples  ",
        f"Max generation tokens: {stats['max_gen_tokens']}  ",
        f"KV threshold: {stats['kv_threshold']} MB\n"
    ]
    
    # Strategy comparison table
    report.append("## Strategy Performance Comparison\n")
    report.append("| Strategy | Tokens/sec | Avg token time (ms) | Peak KV Cache (MB) | Evictions | Avg eviction time (ms) |")
    report.append("|----------|------------|---------------------|-------------------|-----------|------------------------|")
    
    for strat in stats["strategies"]:
        report.append(
            f"| {strat['name']} | {strat['tokens_per_second']:.2f} | "
            f"{strat['avg_token_time'] * 1000:.2f} | {strat['peak_kv_cache_mb']:.2f} | "
            f"{strat['eviction_count']} | {strat['avg_eviction_time'] * 1000:.2f} |"
        )
    
    # Write report to file
    with open(output_dir / "comparative_report.md", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    main()