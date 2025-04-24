import argparse
import time
import json
from pathlib import Path

from tqdm import tqdm
import torch

from utils import (
    load_config,
    ensure_dataset,
    load_model_and_tokenizer,
    compute_cache_size,
    save_json,
)
from strategies import Baseline, TopAttention, BottomAttention, HybridAttention, RandomEvict

COMMON_PROMPT_KEYS = ["prompt", "instruction", "input", "text"]


def get_prompt(sample, cfg):
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


def get_strategy(name, threshold):
    if name == "Baseline":
        return Baseline(threshold)
    if name.startswith("TopAttention"):
        pct = int(name.split("(")[1].rstrip("%)"))
        return TopAttention(threshold, pct)
    if name.startswith("BottomAttention"):
        pct = int(name.split("(")[1].rstrip("%)"))
        return BottomAttention(threshold, pct)
    if name.startswith("HybridAttention"):
        args = name.split("(")[1].rstrip(")").split(",")
        return HybridAttention(threshold, int(args[0]), int(args[1]))
    if name.startswith("Random"):
        pct = int(name.split("(")[1].rstrip("%)"))
        return RandomEvict(threshold, pct)
    raise ValueError(f"Unknown strategy {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_template.json")
    parser.add_argument("--eval_azure", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = ensure_dataset(cfg)
    tokenizer, model = load_model_and_tokenizer(cfg["model_name"],cfg["cache_dir"])
    model.eval()

    for strat_name in cfg["strategies"]:
        strat = get_strategy(strat_name, cfg["kv_threshold"])
        logs = []

        for sample in tqdm(dataset[:5], desc=strat_name):
            # Extract metadata
            question_id = sample.get("question_id")
            category = sample.get("category")

            prompt = get_prompt(sample, cfg)

            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # === Encode-only prefix ===
            times = []
            sizes = []
            start = time.time()
            out = model(**inputs, use_cache=True)
            times.append(time.time() - start)

            past = out.past_key_values
            sizes.append(compute_cache_size(past))

            # Prepare the first token for generation loop
            token = inputs.input_ids[:, -1:].to(model.device)
            generated = prompt

            # === Generation loop ===
            for _ in range(cfg["max_gen_tokens"]):
                if compute_cache_size(past) >= cfg["kv_threshold"]:
                    scores = torch.rand(compute_cache_size(past), device=model.device)
                    past = strat.evict(past, scores)

                start = time.time()
                out = model(input_ids=token, past_key_values=past, use_cache=True)
                elapsed = time.time() - start
                times.append(elapsed)

                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated += tokenizer.decode(next_token[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)
                past = out.past_key_values
                sizes.append(compute_cache_size(past))
                token = next_token

            # Build log entry with category and question_id
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
                },
                "times": {
                    "total": sum(times),
                    "per_token": sum(times) / len(times),
                    "first_token": times[0],
                }
            }

            if args.eval_azure:
                from azure_eval import score_with_azure
                log["accuracy"] = score_with_azure(prompt, generated, cfg)

            logs.append(log)

        save_json(logs, cfg, "per_strategy", strat_name)

if __name__ == "__main__":
    main()
