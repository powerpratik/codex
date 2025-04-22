# -*- coding: utf-8 -*-
"""
benchmark_kv_cache.py

Benchmarking different KV cache eviction strategies on Llama 2 7B
using a variety of Hugging Face Datasets. Measures:
- Average KV cache size (MB)
- Average input sequence length before pruning
- Average kept sequence length after pruning (and ratio)
- Average inference time per token (seconds)
- Perplexity (generation accuracy via next-token NLL)

Strategies included:
* Baseline Full Cache (NoOpStrategy)
* WindowStrategy
* TopRatioStrategy
* BottomRatioStrategy
* BothRatioStrategy
* RandomSamplingStrategy
* StridedStrategy
* BlockAverageStrategy
* AttentionScoreStrategy

By default uses WikiText-2 (test split). To choose another dataset,
pass --dataset_name, --dataset_config (or leave empty), and --dataset_split.

Usage:
python benchmark_kv_cache.py \
--dataset_name wikitext --dataset_config wikitext-2-raw-v1 --dataset_split test \
--seq_prefix 64 --gen_length 128 --num_samples 10 \
--pruning_ratios 0.5,0.25
"""

import argparse
import time
import math
import torch
import numpy as np # For averaging

try:
    from datasets import load_dataset
except ImportError:
    print("Please install the 'datasets' library: pip install datasets")
    load_dataset = None

try:
    from transformers import LlamaForCausalLM, LlamaTokenizer
except ImportError:
    print("Please install the 'transformers' library: pip install transformers")
    LlamaForCausalLM = None
    LlamaTokenizer = None

# Import necessary classes from kv_cache_manager
from updated_kv_manager import (
    KVCacheManager,
    NoOpStrategy,
    WindowStrategy,
    RandomSamplingStrategy,
    StridedStrategy,
    BlockAverageStrategy,
    AttentionScoreStrategy,
    TopRatioStrategy, # Added
    BottomRatioStrategy, # Added
    BothRatioStrategy, # Added
    unpack_kv,
    pack_kv,
)


def evaluate_strategy(
    name: str,
    strategy,
    model,
    tokenizer,
    dataset,
    device,
    seq_prefix: int,
    gen_length: int,
    num_samples: int,
):
    """Evaluates a given strategy, collecting detailed metrics."""

    cache_mgr = KVCacheManager(strategy)

    # --- Metric Accumulators ---
    total_tokens_generated = 0
    total_inference_time = 0.0
    total_nll = 0.0
    # For KV cache size
    all_step_kv_sizes_bytes = []
    # For input/kept lengths
    all_step_input_lengths = []
    all_step_kept_lengths = []
    # --- End Metric Accumulators ---

    # Filter dataset for sufficiently long examples and select samples
    try:
        # Attempt filtering - might fail if dataset doesn't have 'text' or is streaming
        samples = dataset.filter(lambda ex: 'text' in ex and len(tokenizer(ex['text'], return_tensors='pt').input_ids[0]) >= seq_prefix + gen_length)
        samples = samples.select(range(min(num_samples, len(samples))))
        print(f"Filtered dataset to {len(samples)} samples.")
    except Exception as e:
        print(f"Warning: Could not filter dataset ({e}). Using first {num_samples} samples directly.")
        samples = dataset.select(range(num_samples))


    start_strategy_eval_time = time.time()

    # Process each sample
    for i, ex in enumerate(samples):
        print(f" Processing sample {i+1}/{len(samples)}...", end='\r')
        text = ex['text']
        toks = tokenizer(text, return_tensors='pt')
        ids = toks.input_ids[0]

        # prefix and target split
        prefix_ids = ids[:seq_prefix].unsqueeze(0).to(device) # Add batch dim
        target_ids = ids[seq_prefix : seq_prefix + gen_length]

        # Initialize KV cache with the prefix
        with torch.no_grad():
            outputs = model(prefix_ids, use_cache=True)
            # Update KV cache - returns pruned list, original len, kept len
            pruned_past_list, orig_len, kept_len = cache_mgr.update(outputs.past_key_values)

            # Log initial state metrics (optional, uncomment if needed)
            # current_kv_size_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in pruned_past_list) if pruned_past_list else 0
            # all_step_kv_sizes_bytes.append(current_kv_size_bytes)
            # all_step_input_lengths.append(orig_len)
            # all_step_kept_lengths.append(kept_len)

            # Pack the cache for the next step
            past_key_values = pack_kv(pruned_past_list)

        # Generate token by token for the target sequence
        for token_idx, target_token_id in enumerate(target_ids):
            input_token = target_token_id.view(1, 1).to(device) # Batch dim = 1, Seq len = 1

            step_start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids=input_token, past_key_values=past_key_values, use_cache=True)
            step_inference_time = time.time() - step_start_time

            # --- Collect Metrics for this step ---
            total_tokens_generated += 1
            total_inference_time += step_inference_time

            # Calculate NLL for perplexity
            logits = outputs.logits[:, -1, :] # Get logits for the last generated token
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs[0, target_token_id].item() # NLL of the *correct* next token
            total_nll += nll

            # Update KV cache and get lengths
            pruned_past_list, orig_len, kept_len = cache_mgr.update(outputs.past_key_values)
            all_step_input_lengths.append(orig_len)
            all_step_kept_lengths.append(kept_len)

            # Calculate KV cache size *after* pruning for this step
            current_kv_size_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in pruned_past_list) if pruned_past_list else 0
            all_step_kv_sizes_bytes.append(current_kv_size_bytes)

            # Pack the pruned cache for the next iteration
            past_key_values = pack_kv(pruned_past_list)
            # --- End Metric Collection ---

            # Clean up CUDA cache periodically if needed
            if token_idx % 50 == 0:
                 torch.cuda.empty_cache()

    # --- Calculate Averages ---
    end_strategy_eval_time = time.time()
    total_strategy_time_s = end_strategy_eval_time - start_strategy_eval_time

    avg_inference_time_per_token = total_inference_time / total_tokens_generated if total_tokens_generated else 0
    avg_perplexity = math.exp(total_nll / total_tokens_generated) if total_tokens_generated else float('inf')
    avg_kv_size_mb = (np.mean(all_step_kv_sizes_bytes) / (1024 * 1024)) if all_step_kv_sizes_bytes else 0
    avg_input_length = np.mean(all_step_input_lengths) if all_step_input_lengths else 0
    avg_kept_length = np.mean(all_step_kept_lengths) if all_step_kept_lengths else 0
    avg_kept_ratio = avg_kept_length / avg_input_length if avg_input_length > 0 else 0

    print(f"\nFinished processing {len(samples)} samples for strategy '{name}'.") # Newline after progress indicator

    return {
        'strategy': name,
        'avg_time_per_token_s': avg_inference_time_per_token,
        'perplexity': avg_perplexity,
        'avg_kv_size_mb': avg_kv_size_mb,
        'avg_input_length': avg_input_length,
        'avg_kept_length': avg_kept_length,
        'avg_kept_ratio': avg_kept_ratio,
        'total_tokens_generated': total_tokens_generated,
        'total_strategy_time_s': total_strategy_time_s,
        # Optionally return raw lists for more detailed analysis:
         'raw_kv_sizes_bytes': all_step_kv_sizes_bytes,
         'raw_input_lengths': all_step_input_lengths,
         'raw_kept_lengths': all_step_kept_lengths,
    }


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Benchmark KV cache eviction strategies on Llama 2 7B")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Hugging Face model identifier")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Hugging Face dataset name (e.g., wikitext, ptb_text_only)")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset config name (leave empty for none)")
    parser.add_argument("--dataset_split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--seq_prefix", type=int, default=64,
                        help="Initial context length (tokens)")
    parser.add_argument("--gen_length", type=int, default=1024, # Increased default
                        help="Number of tokens to predict/generate per sample")
    parser.add_argument("--num_samples", type=int, default=10, # Reduced default for faster testing
                        help="Number of sequences to test")
    parser.add_argument("--cache_dir", type=str, default=None, help="Where to store HF model files and tokenizer")
    parser.add_argument("--pruning_ratios", type=str, default="0.5,0.25",
                        help="Comma-separated list of pruning ratios (e.g., 0.5,0.25) for ratio-based strategies")
    parser.add_argument("--max_length_ratio", type=float, default=0.5,
                       help="Ratio of max sequence length (prefix+gen) to use for max_length in non-ratio strategies")


    args = parser.parse_args()

    # Check libraries
    if load_dataset is None or LlamaForCausalLM is None or LlamaTokenizer is None:
        print("Missing required libraries (datasets, transformers). Please install them.")
        return

    model_name = args.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seq_prefix = args.seq_prefix
    gen_length = args.gen_length
    num_samples = args.num_samples
    ratios_to_test = [float(r) for r in args.pruning_ratios.split(',')]
    max_len_abs = int((seq_prefix + gen_length) * args.max_length_ratio)
    max_len_abs = max(1, max_len_abs) # Ensure it's at least 1
    print(f"Max length for Window/Random/etc.: {max_len_abs} (derived from ratio {args.max_length_ratio})")


    # load dataset
    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config or 'default'}) split '{args.dataset_split}'")
    try:
        if args.dataset_config:
            ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, cache_dir=args.cache_dir)
        else:
            ds = load_dataset(args.dataset_name, split=args.dataset_split, cache_dir=args.cache_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # load model & tokenizer
    print(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
        if tokenizer.pad_token_id is None:
            print("Setting pad_token_id to eos_token_id")
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16, # Use float16 for efficiency
            # device_map='auto' # Consider if model is too large for single GPU
        ).to(device)
        model.eval()
        print("Model and tokenizer loaded.")
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return


    # --- Define Strategies ---
    strategies_to_run = []
    strategies_to_run.append(("baseline_full", NoOpStrategy()))
    strategies_to_run.append((f"window_{max_len_abs}t", WindowStrategy(max_length=max_len_abs)))

    # Add ratio-based strategies
    for ratio in ratios_to_test:
        ratio_pct = int(ratio * 100)
        strategies_to_run.append((f"top_{ratio_pct}pct", TopRatioStrategy(pruning_ratio=ratio)))
        strategies_to_run.append((f"bottom_{ratio_pct}pct", BottomRatioStrategy(pruning_ratio=ratio))) # Similar to Window
        strategies_to_run.append((f"both_{ratio_pct}pct", BothRatioStrategy(pruning_ratio=ratio)))

    # Add other fixed-size strategies
    strategies_to_run.append((f"random_{max_len_abs}t", RandomSamplingStrategy(max_length=max_len_abs, seed=42)))
    strategies_to_run.append((f"strided_{max_len_abs}t", StridedStrategy(max_length=max_len_abs)))
    strategies_to_run.append((f"attn_top{max_len_abs}t", AttentionScoreStrategy(max_length=max_len_abs)))
    # BlockAverage might change semantics, include with caution or specific block sizes
    # strategies_to_run.append(("block_avg_size_4", BlockAverageStrategy(block_size=4)))
    # --- End Define Strategies ---


    # --- Run Benchmarks ---
    results = []
    print(f"\nStarting benchmark: {num_samples} samples, prefix={seq_prefix}, gen_length={gen_length}")
    for name, strat in strategies_to_run:
        print(f"\n--- Evaluating strategy: {name} ---")
        try:
            res = evaluate_strategy(name, strat, model, tokenizer, ds, device, seq_prefix, gen_length, num_samples)
            results.append(res)
            # Print intermediate result summary
            print(f"  Avg Time/Token: {res['avg_time_per_token_s']:.5f}s")
            print(f"  Perplexity:     {res['perplexity']:.3f}")
            print(f"  Avg KV Size:    {res['avg_kv_size_mb']:.2f} MB")
            print(f"  Avg Kept Ratio: {res['avg_kept_ratio']:.1%}")
            print(f"  Total Eval Time:{res['total_strategy_time_s']:.2f}s")

        except Exception as e:
             print(f"ERROR evaluating strategy {name}: {e}")
             import traceback
             traceback.print_exc()
        finally:
            # Clear CUDA cache between strategies
            torch.cuda.empty_cache()
    # --- End Run Benchmarks ---


    # --- Summary ---
    print("\n--- Benchmark Results Summary ---")
    # Header - adjusted for new metrics
    print(f"{'Strategy':<20} | {'Avg Time/Tok (s)':<18} | {'Perplexity':<12} | {'Avg KV (MB)':<12} | {'Avg Kept Ratio':<15} | {'Total Eval Time(s)':<18}")
    print("-" * 105) # Separator line

    # Sort results maybe by perplexity or KV size? Optional. For now, insertion order.
    for r in results:
        # Row formatting - adjusted for new metrics
        print(f"{r['strategy']:<20} | {r['avg_time_per_token_s']:<18.5f} | {r['perplexity']:<12.3f} | {r['avg_kv_size_mb']:<12.2f} | {r['avg_kept_ratio']:<15.1%} | {r['total_strategy_time_s']:<18.2f}")

    print("-" * 105)
    print(f"Settings: model={model_name}, prefix={seq_prefix}, gen_len={gen_length}, samples={num_samples}")
    # --- End Summary ---

if __name__ == "__main__":
    main()


