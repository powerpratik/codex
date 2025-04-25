import argparse
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache_manager import ThresholdEvictionStrategy, measure_cache_mb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--memory_threshold_mb", type=float, default=250.0,
                   help="Max KB cache in MB before eviction kicks in")
    p.add_argument("--gen_length", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=5)
    return p.parse_args()


def run_one_sample(model, tokenizer, cache_manager, input_ids, gen_length):
    past = None
    cache_sizes = []
    times = []

    for _ in range(gen_length):
        # 1) Forward + timing
        t0 = time.time()
        outputs = model(input_ids, past_key_values=past)
        t1 = time.time()

        # 2) Raw past, then evict
        raw_past = outputs.past_key_values
        past = cache_manager.evict(raw_past)

        # 3) Measure AFTER eviction
        cache_sizes.append(measure_cache_mb(past))
        times.append(t1 - t0)

        # 4) Next token (greedy for simplicity)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = next_token

    return cache_sizes, times


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map="auto", 
        torch_dtype=torch.float32
    ).to(device)

    # prepare eviction
    cache_manager = ThresholdEvictionStrategy(
        memory_threshold_mb=args.memory_threshold_mb,
        model_config=model.config
    )

    # load a few prefixes
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    texts = ds["text"][: args.num_samples]

    all_cache = []
    all_times = []
    for txt in texts:
        # encode a short prompt
        input_ids = tokenizer(txt, return_tensors="pt").input_ids.to(device)
        cache_sizes, times = run_one_sample(
            model, tokenizer, cache_manager, input_ids, args.gen_length
        )
        all_cache.append(cache_sizes)
        all_times.append(times)

    # dump results
    import json
    out = {
        "cache_mb": all_cache,
        "time_s": all_times
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
