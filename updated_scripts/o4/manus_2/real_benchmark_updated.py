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
from real_kv_cache_manager_threshold import RealKVCacheManager
from real_inference_time_tracker import InferenceTimeTracker
from real_accuracy_evaluator import AccuracyEvaluator
from kv_cache_dashboard import KVCacheDashboard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("real_benchmark")

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

# Updated benchmark code with real metric collection and threshold-based eviction
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_template.json")
    parser.add_argument("--eval_azure", action="store_true", help="Evaluate with Azure")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--profile", action="store_true", help="Enable memory profiling")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--generate_dashboard", action="store_true", help="Generate visualization dashboard")
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
    
    # Update config with Azure flag
    cfg["use_azure"] = args.eval_azure
    
    # Benchmark each strategy
    for strat_name in cfg["strategies"]:
        logger.info(f"Benchmarking strategy: {strat_name}")
        
        # Create strategy instance
        strat = get_enhanced_strategy(strat_name, cfg["kv_threshold"], cfg)
        logger.info(f"Created strategy instance: {strat.name}")
        
        # Create real metric trackers
        cache_manager = RealKVCacheManager(model, tokenizer, cfg, logger)
        time_tracker = InferenceTimeTracker(model, tokenizer, logger)
        accuracy_evaluator = AccuracyEvaluator(model, tokenizer, cfg, logger)
        
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
        for sample_idx, sample in enumerate(tqdm(dataset, desc=strat_name)):
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
            
            # Reset metrics for this sample
            cache_manager.reset_stats()
            time_tracker.reset_stats()
            
            # === Start encoding with timing ===
            time_tracker.start_encoding()
            with torch.no_grad():
                # Important: use_cache=True but don't pass past_key_values
                out = model(**inputs, use_cache=True, output_attentions=True)
            time_tracker.end_encoding(input_tokens)
            
            # Get the cache object from the output
            past = out.past_key_values
            
            # Extract attention scores and token types for the prompt
            attention_scores = extract_layerwise_attention(out.attentions)
            token_types = cache_manager.identify_token_types(input_tokens)
            
            # Prepare for generation
            token = inputs.input_ids[:, -1:].to(model.device)
            gen_tokens = input_tokens.copy()
            generated_text = ""
            
            # === Start generation with timing ===
            time_tracker.start_generation()
            
            # === Generation loop ===
            for i in range(cfg["max_gen_tokens"]):
                # Apply eviction strategy to the KV cache if needed
                # This will only trigger when approaching the threshold
                eviction_start = time.time()
                past = cache_manager.apply_eviction_strategy(past, strat.name, attention_scores)
                eviction_time = time.time() - eviction_start
                
                # Generate next token
                with torch.no_grad():
                    out = model(
                        input_ids=token, 
                        past_key_values=past,
                        use_cache=True, 
                        output_attentions=True
                    )
                
                # Get next token and add to generated sequence
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                next_token_id = next_token.item()
                gen_tokens.append(next_token_id)
                
                # Record token generation time
                time_tracker.token_generated(next_token_id, eviction_time)
                
                # Update KV cache
                past = out.past_key_values
                
                # Update attention scores with new information
                new_attention = extract_layerwise_attention(out.attentions)
                attention_scores = new_attention
                
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
            time_tracker.end_generation()
            
            # Record final memory usage
            if args.profile:
                post_memory = profile_memory()
                
            # Decode the full sequence for better handling of word boundaries
            generated = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Evaluate accuracy
            accuracy_metrics = accuracy_evaluator.evaluate_response(
                prompt, generated, input_tokens, gen_tokens
            )
            
            # Calculate metrics
            total_tokens = len(gen_tokens) - len(input_tokens)
            strategy_metrics["total_tokens"] += total_tokens
            strategy_metrics["total_time"] += time_tracker.get_time_stats()["total_generation_time"]
            
            # Build detailed log entry
            log = {
                "question_id": question_id,
                "category": category,
                "prompt": prompt,
                "response": generated,
                "strategy": strat_name,
                "kv_cache": cache_manager.get_memory_stats(),
                "time": time_tracker.get_time_stats(),
                "accuracy": accuracy_metrics,
                "tokens": {
                    "input_length": len(input_tokens),
                    "output_length": total_tokens,
                    "total_length": len(gen_tokens)
                }
            }
            
            logs.append(log)
            
            # Log progress
            if (sample_idx + 1) % 5 == 0 or sample_idx == len(dataset) - 1:
                logger.info(f"Processed {sample_idx + 1}/{len(dataset)} samples with {strat_name}")
        
        # Clean up hooks
        cache_manager.remove_hooks()
        time_tracker.remove_hooks()
        
        # Save per-strategy logs
        save_json(logs, cfg, "per_strategy", f"real_{strat_name}")
        logger.info(f"Saved logs for {strat_name} to {output_dir/'per_strategy'/f'real_{strat_name}.json'}")
        
        # Add strategy summary to global stats
        time_stats = time_tracker.get_time_stats()
        cache_stats = cache_manager.get_memory_stats()
        accuracy_stats = accuracy_evaluator.get_accuracy_stats()
        
        strategy_summary = {
            "name": strat_name,
            "tokens_per_second": time_stats["tokens_per_second"],
            "avg_token_time": time_stats["avg_token_time"],
            "first_token_time": time_stats["first_token_time"],
            "total_generation_time": time_stats["total_generation_time"],
            "total_tokens_generated": time_tracker.generated_token_count,
            "peak_kv_cache_mb": cache_stats["peak_memory_mb"],
            "eviction_count": cache_stats["eviction_count"],
            "avg_eviction_time": cache_stats["avg_eviction_time"],
            "perplexity": accuracy_stats["perplexity"]["mean"] if "perplexity" in accuracy_stats else None
        }
        
        if "azure" in accuracy_stats:
            strategy_summary["azure_score"] = accuracy_stats["azure"]["mean"]
        
        global_stats["strategies"].append(strategy_summary)
    
    # Save global stats summary
    save_json(global_stats, cfg, ".", "real_benchmark_summary")
    logger.info(f"Saved benchmark summary to {output_dir/'real_benchmark_summary.json'}")
    
    # Generate comparative report
    generate_comparative_report(global_stats, output_dir)
    logger.info(f"Generated comparative report at {output_dir/'real_comparative_report.md'}")
    
    # Generate dashboard if requested
    if args.generate_dashboard:
        logger.info("Generating visualization dashboard...")
        dashboard = KVCacheDashboard(output_dir)
        dashboard.generate_dashboard()
        logger.info(f"Dashboard generated at {output_dir/'dashboard'}")

def generate_comparative_report(stats, output_dir):
    """Generate a markdown report comparing strategy performance"""
    report = [
        "# Real KV Cache Strategy Benchmark Report\n",
        f"## Model: {stats['model_name']}\n",
        f"Dataset size: {stats['dataset_size']} samples  ",
        f"Max generation tokens: {stats['max_gen_tokens']}  ",
        f"KV threshold: {stats['kv_threshold']} MB\n"
    ]
    
    # Strategy comparison table
    report.append("## Strategy Performance Comparison\n")
    report.append("| Strategy | Tokens/sec | Avg token time (ms) | First token time (ms) | Peak KV Cache (MB) | Evictions | Avg eviction time (ms) |")
    report.append("|----------|------------|---------------------|----------------------|-------------------|-----------|------------------------|")
    
    for strat in stats["strategies"]:
        report.append(
            f"| {strat['name']} | {strat['tokens_per_second']:.2f} | "
            f"{strat['avg_token_time'] * 1000:.2f} | {strat['first_token_time'] * 1000:.2f} | "
            f"{strat['peak_kv_cache_mb']:.2f} | {strat['eviction_count']} | "
            f"{strat['avg_eviction_time'] * 1000:.2f} |"
        )
    
    # Accuracy comparison if available
    has_perplexity = any('perplexity' in strat and strat['perplexity'] is not None for strat in stats["strategies"])
    has_azure = any('azure_score' in strat for strat in stats["strategies"])
    
    if has_perplexity or has_azure:
        report.append("\n## Accuracy Metrics\n")
        
        headers = ["Strategy"]
        if has_perplexity:
            headers.append("Perplexity (lower is better)")
        if has_azure:
            headers.append("Azure Score (1-10)")
        
        report.append("| " + " | ".join(headers) + " |")
        report.append("|" + "|".join(["-" * len(h) for h in headers]) + "|")
        
        for strat in stats["strategies"]:
            row = [strat['name']]
            if has_perplexity:
                perplexity = strat.get('perplexity', "N/A")
                row.append(f"{perplexity:.2f}" if isinstance(perplexity, float) else "N/A")
            if has_azure:
                azure = strat.get('azure_score', "N/A")
                row.append(f"{azure:.2f}" if isinstance(azure, float) else "N/A")
            
            report.append("| " + " | ".join(row) + " |")
    
    # Write report to file
    with open(output_dir / "real_comparative_report.md", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    main()
