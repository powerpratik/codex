"""
benchmark_kv_cache.py

Benchmarking different KV cache eviction strategies on Llama 2 7B
using a variety of Hugging Face Datasets. Measures:
  - Average KV cache size (# elements in keys + values)
  - Average inference time per token (seconds)
  - Perplexity (generation accuracy via next-token NLL)

Strategies included:
  * Baseline Full Cache (NoOpStrategy)
  * WindowStrategy
  * RandomSamplingStrategy
  * StridedStrategy
  * BlockAverageStrategy
  * AttentionScoreStrategy

By default uses WikiText-2 (test split). To choose another dataset,
pass --dataset_name, --dataset_config (or leave empty), and --dataset_split.

Usage:
  python benchmark_kv_cache.py \
    --dataset_name wikitext --dataset_config wikitext-2-raw-v1 --dataset_split test \
    --seq_prefix 64 --gen_length 32 --num_samples 5
"""
import argparse
import time
import math
import torch
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
try:
    from transformers import LlamaForCausalLM, LlamaTokenizer
except ImportError:
    LlamaForCausalLM = None
    LlamaTokenizer = None

from kv_cache_manager import (
    KVCacheManager,
    NoOpStrategy,
    WindowStrategy,
    RandomSamplingStrategy,
    StridedStrategy,
    BlockAverageStrategy,
    AttentionScoreStrategy,
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
    total_tokens = 0
    total_time = 0.0
    total_nll = 0.0
    total_kv_elems = 0
    samples = dataset.filter(lambda ex: len(tokenizer(ex['text'], return_tensors='pt').input_ids[0]) >= seq_prefix + gen_length)
    samples = samples.select(range(min(num_samples, len(samples))))
    for ex in samples:
        text = ex['text']
        toks = tokenizer(text, return_tensors='pt')
        ids = toks.input_ids[0]
        # prefix and target split
        prefix_ids = ids[:seq_prefix]
        target_ids = ids[seq_prefix : seq_prefix + gen_length]

        cache_mgr = KVCacheManager(strategy)
        # initial build cache
        with torch.no_grad():
            out = model(prefix_ids.unsqueeze(0).to(device), use_cache=True)
        # unpack to list of (key, value), evict, then repack for model
        past_list = unpack_kv(out.past_key_values)
        past_list = cache_mgr.update(past_list)
        past = pack_kv(past_list)

        # token-by-token generation/prediction
        for t in target_ids:
            inp = t.unsqueeze(0).unsqueeze(0).to(device)
            start = time.time()
            with torch.no_grad():
                out = model(inp, past_key_values=past, use_cache=True)
                logits = out.logits.squeeze(0).squeeze(0)
            elapsed = time.time() - start
            # nll of true next token
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs[t].item()
            # unpack, evict, repack cache
            past_list = unpack_kv(out.past_key_values)
            past_list = cache_mgr.update(past_list)
            past = pack_kv(past_list)
            # count KV elements
            kv_count = sum(k.numel() + v.numel() for k, v in past_list)

            total_tokens += 1
            total_time += elapsed
            total_nll += nll
            total_kv_elems += kv_count

    # metrics
    avg_time = total_time / total_tokens
    avg_perplexity = math.exp(total_nll / total_tokens)
    avg_kv = total_kv_elems / total_tokens
    return {
        'strategy': name,
        'avg_time_per_token_s': avg_time,
        'perplexity': avg_perplexity,
        'avg_kv_elements': avg_kv,
        'total_kv_elements': total_kv_elems
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
    parser.add_argument("--gen_length", type=int, default=32,
                        help="Number of tokens to predict")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of sequences to test")
    parser.add_argument("--cache_dir", type=str, default=None, help="where to store HF model files and tokenizer")
    args = parser.parse_args()

    model_name = args.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_prefix = args.seq_prefix
    gen_length = args.gen_length
    num_samples = args.num_samples
    print(num_samples)

    # load dataset
    if load_dataset is None:
        raise ImportError("Please install the 'datasets' library: pip install datasets")
    if args.dataset_config:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    else:
        ds = load_dataset(args.dataset_name, split=args.dataset_split)

    # load model & tokenizer
    if LlamaForCausalLM is None or LlamaTokenizer is None:
        raise ImportError("Please install the 'transformers' library: pip install transformers")
    tokenizer = LlamaTokenizer.from_pretrained(model_name,cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaForCausalLM.from_pretrained(model_name,cache_dir= args.cache_dir).to(device)
    model.eval()

    # define strategies
    max_len = seq_prefix + gen_length  # worst-case cache length
    strategies = [
        ("baseline_full", NoOpStrategy()),
        ("window_{}_tokens".format(max_len // 2), WindowStrategy(max_length=max_len // 2)),
        ("random_{}_tokens".format(max_len // 2), RandomSamplingStrategy(max_length=max_len // 2, seed=42)),
        ("strided_{}_tokens".format(max_len // 2), StridedStrategy(max_length=max_len // 2)),
        ("block_avg_size_4", BlockAverageStrategy(block_size=4)),
        ("attention_top{}_tokens".format(max_len // 2), AttentionScoreStrategy(max_length=max_len // 2)),
    ]

    # run benchmarks
    results = []
    for name, strat in strategies:
        print(f"Evaluating strategy: {name}")
        strategy_start_time = time.time()
        res = evaluate_strategy(name, strat, model, tokenizer, ds, device, seq_prefix, gen_length, num_samples)
        strategy_end_time = time.time()
        res['total_strategy_time_s'] = strategy_end_time - strategy_start_time
        print(res)
        results.append(res)

    # summary
    print("\nBenchmark results:")
    # Add 'Total Time(s)' and 'Total KV elems' to the header
    print(f"{'Strategy':<20} {'Avg Token Time(s)':<18} {'Total Time(s)':<15} {'Perplexity':<12} {'Avg KV elems':<12} {'Total KV elems':<15}")

    for r in results:
    # Add r['total_strategy_time_s'] and r['total_kv_elements'] to the output row
        print(f"{r['strategy']:<20} {r['avg_time_per_token_s']:<18.4f} {r['total_strategy_time_s']:<15.4f} {r['perplexity']:<12.2f} {r['avg_kv_elements']:<12.0f} {r['total_kv_elements']:<15}") # <-- Note the added metric at the end
if __name__ == "__main__":
    main()
