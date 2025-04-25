# utils.py
import os, json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def ensure_dataset(cfg):
    path = Path(cfg["dataset"]["local_path"])
    if not path.exists():
        # implement your download logic here—for example via HF Datasets
        raise FileNotFoundError(f"{path} not found. Please download and place it.")
    return json.load(open(path, "r"))

def save_json(log, cfg, subdir, name):
    out_dir = Path(cfg["output_dir"]) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(log, f, indent=2)

def load_model_and_tokenizer(model_name,cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
        device_map="auto", torch_dtype="auto",cache_dir=cache_dir)
    return tokenizer, model

# utils.py

def compute_cache_size(past_key_values):
    """
    Estimate total KV‐cache memory footprint in megabytes.
    past_key_values: tuple of (key, value) pairs per layer.
    """
    total_bytes = 0
    for k, v in past_key_values:
        # number of elements × bytes per element
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    # convert to megabytes
    return total_bytes / (1024 ** 2)


def load_config(path="config_template.json"):
    return json.load(open(path, "r"))
