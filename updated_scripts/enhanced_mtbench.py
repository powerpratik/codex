#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fixed_mtbench_eval.py

Fixed version of the MT-Bench KV cache evaluation script with additional
debugging, error handling, and fixes for multi-GPU setups.
"""

import os
import json
import torch
import argparse
import numpy as np
import time
import requests
import re
import gc
import signal
import traceback
from tqdm import tqdm
from contextlib import contextmanager

# Ensure transformers and necessary dependencies are installed
try:
    from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
except ImportError:
    print("Please install transformers, bitsandbytes, accelerate: pip install transformers bitsandbytes accelerate")
    exit()

# Import from our kv_cache_manager
from updated_kv_manager import (
    KVCacheManager, NoOpStrategy, WindowStrategy,
    TopRatioStrategy, BottomRatioStrategy, BothRatioStrategy,
    RandomSamplingStrategy, StridedStrategy, AttentionScoreStrategy, BlockAverageStrategy,
    unpack_kv, pack_kv
)

# --- Timeout handler ---
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time of a block of code."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm

# --- MTBenchDataset Class ---
class MTBenchDataset:
    """
    Dataset class for MT-Bench evaluation
    """
    def __init__(self, data_path="mt_bench_data.jsonl"):
        self.data_path = data_path
        if not os.path.exists(data_path):
            self._download_dataset()
        self.questions = self._load_questions()
        print(f"Loaded {len(self.questions)} questions from MT-Bench")

    def _download_dataset(self):
        """Download MT-Bench dataset from Hugging Face"""
        print("Downloading MT-Bench dataset...")
        url = "https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/raw/main/question.jsonl"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            questions = []
            for line in response.text.strip().split('\n'):
                questions.append(json.loads(line))
            with open(self.data_path, 'w') as f:
                json.dump(questions, f, indent=2)
            print(f"MT-Bench dataset downloaded to {self.data_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading MT-Bench dataset: {e}")
            print("Please download manually from https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts and save as mt_bench_data.jsonl")
            raise

    def _load_questions(self):
        """Load questions from JSON file"""
        try:
            with open(self.data_path, 'r') as f:
                questions = json.load(f)
            # Ensure 'turns' is a list
            for q in questions:
                if 'turns' not in q or not isinstance(q['turns'], list):
                    print(f"Warning: Question {q.get('question_id')} has missing or invalid 'turns'. Skipping.")
            return [q for q in questions if 'turns' in q and isinstance(q['turns'], list) and len(q['turns']) > 0]
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {self.data_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.data_path}. The file might be corrupted or incomplete.")
            raise

    def get_question_by_id(self, question_id):
        """Get a specific question by ID"""
        for q in self.questions:
            if q["question_id"] == question_id:
                return q
        return None

    def get_questions_by_category(self, category):
        """Get questions for a specific category"""
        return [q for q in self.questions if q["category"] == category]

    def format_prompt(self, question, turn_idx=0, model_name=""):
        """Format the prompt for the model (first turn only for this script)"""
        if turn_idx >= len(question["turns"]):
            print(f"Warning: turn_idx {turn_idx} out of bounds for question {question.get('question_id')}")
            return ""
        prompt = question["turns"][turn_idx]

        # Format based on model type (Llama-2 chat specific formatting)
        if "Llama-2" in model_name and "chat" in model_name.lower():
            # Basic Llama2 chat format for single turn
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            # Generic prompt format
            formatted_prompt = prompt

        return formatted_prompt

# --- Fixed Generation Function with Enhanced Debugging ---
def generate_response(model, tokenizer, prompt, strategy=None, max_new_tokens=None, timeout_seconds=60):
    """Generate a response with KV cache pruning and detailed metrics collection"""
    if not prompt:
        return "Error: Empty prompt received", {}

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]
    is_cuda = model.device.type == 'cuda'

    print(f"[DEBUG] Starting generation with {input_token_count} input tokens")
    print(f"[DEBUG] Strategy: {strategy.__class__.__name__}")
    
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(model.device)

    if strategy is None:
        strategy = NoOpStrategy()
    cache_manager = KVCacheManager(strategy)

    # Initialize step metrics
    step_metrics = {
        "kv_cache_sizes_bytes": [],
        "input_lengths": [],
        "kept_lengths": [],
        "token_inference_times": []
    }
    
    try:
        with time_limit(timeout_seconds):
            generated_ids = inputs["input_ids"]
            past_key_values = None
            prompt_start_time = time.time()
            max_steps = 1000 if max_new_tokens is None else max_new_tokens
            
            # --- Generation Loop ---
            for step in range(max_steps):
                token_gen_start = time.time()
                current_input_ids = generated_ids if past_key_values is None else generated_ids[:, -1:]

                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True
                    )

                if is_cuda: torch.cuda.synchronize()
                token_time = time.time() - token_gen_start
                step_metrics["token_inference_times"].append(token_time)

                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                # Debug the KV update step which might be causing issues
                try:
                    print(f"[DEBUG] Step {step}: Processing KV update")
                    
                    # Handle potential device mismatch issues in multi-GPU setups
                    # Unpack and get device information
                    unpacked_past_kvs = unpack_kv(outputs.past_key_values)
                    if unpacked_past_kvs:
                        first_k, first_v = unpacked_past_kvs[0]
                        kv_device = first_k.device
                        print(f"[DEBUG] KV cache device: {kv_device}")
                    
                    # Update KV cache using the manager
                    pruned_past_list, orig_len, kept_len = cache_manager.update(outputs.past_key_values)
                    past_key_values = pack_kv(pruned_past_list)
                    
                    print(f"[DEBUG] Step {step}: KV update successful - orig_len={orig_len}, kept_len={kept_len}")
                    
                    # Collect metrics for this step
                    kv_size_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                                      for k, v in pruned_past_list) if pruned_past_list else 0
                    step_metrics["kv_cache_sizes_bytes"].append(kv_size_bytes)
                    step_metrics["input_lengths"].append(orig_len)
                    step_metrics["kept_lengths"].append(kept_len)
                    
                except Exception as e:
                    print(f"[ERROR] KV update failed: {e}")
                    traceback.print_exc()
                    raise

                # Check stopping conditions
                if next_token_id.item() == tokenizer.eos_token_id:
                    print(f"[DEBUG] EOS token generated at step {step}")
                    break
                
                # Check token limit only if specified
                if max_new_tokens is not None and (step + 1) >= max_new_tokens:
                    print(f"[DEBUG] Reached max_new_tokens limit ({max_new_tokens})")
                    break
                
                # Print periodic status
                if step > 0 and step % 20 == 0:
                    print(f"[DEBUG] Generated {step} tokens so far")

            # --- End Generation Loop ---

            if is_cuda: torch.cuda.synchronize()
            prompt_total_time = time.time() - prompt_start_time
            generated_text = tokenizer.decode(generated_ids[0, input_token_count:], skip_special_tokens=True)
            
            print(f"[DEBUG] Generation complete: {len(generated_text)} chars, {generated_ids.shape[1] - input_token_count} tokens")
            
            # Compute current and peak memory
            current_mem_bytes = torch.cuda.memory_allocated(model.device) if is_cuda else 0
            prompt_peak_mem_bytes = torch.cuda.max_memory_allocated(model.device) if is_cuda else 0

            # Calculate average metrics
            avg_token_time = np.mean(step_metrics["token_inference_times"]) if step_metrics["token_inference_times"] else 0
            avg_kv_size_mb = (np.mean(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024)) if step_metrics["kv_cache_sizes_bytes"] else 0
            avg_input_len = np.mean(step_metrics["input_lengths"]) if step_metrics["input_lengths"] else 0
            avg_kept_len = np.mean(step_metrics["kept_lengths"]) if step_metrics["kept_lengths"] else 0
            avg_kept_ratio = avg_kept_len / avg_input_len if avg_input_len > 0 else 0
            
            output_tokens = generated_ids.shape[1] - input_token_count
            tokens_per_second = output_tokens / prompt_total_time if prompt_total_time > 0 else 0

            # Compile detailed metrics
            final_metrics = {
                "input_tokens": input_token_count,
                "output_tokens": output_tokens,
                "total_tokens_processed": generated_ids.shape[1],
                "total_time_s": prompt_total_time,
                "avg_token_time_s": avg_token_time,
                "tokens_per_second": tokens_per_second,
                "avg_kv_cache_size_mb": avg_kv_size_mb,
                "avg_kept_ratio": avg_kept_ratio,
                "current_memory_mb": current_mem_bytes / (1024 * 1024),
                "peak_memory_mb": prompt_peak_mem_bytes / (1024 * 1024),
                # Only include summary of raw step metrics to avoid huge JSON files
                "raw_metrics_summary": {
                    "num_steps": len(step_metrics["token_inference_times"]),
                    "min_kv_cache_mb": min(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024) if step_metrics["kv_cache_sizes_bytes"] else 0,
                    "max_kv_cache_mb": max(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024) if step_metrics["kv_cache_sizes_bytes"] else 0,
                    "min_kept_ratio": min(step_metrics["kept_lengths"]) / max(step_metrics["input_lengths"]) if step_metrics["input_lengths"] else 0,
                    "max_kept_ratio": max(step_metrics["kept_lengths"]) / max(step_metrics["input_lengths"]) if step_metrics["input_lengths"] else 0,
                }
            }

    except TimeoutException as e:
        print(f"[ERROR] Generation timeout: {e}")
        return f"Error: Generation timed out after {timeout_seconds} seconds", {}
    except Exception as e:
        print(f"[ERROR] Generation error: {e}")
        traceback.print_exc()
        return f"Error during generation: {str(e)}", {}

    return generated_text, final_metrics

# --- Evaluation Function for a Strategy ---
def evaluate_mtbench(model, tokenizer, dataset, strategy_name="baseline", strategy=None, category=None, 
                   num_samples=None, max_tokens_per_response=1024, timeout_seconds=120):
    """Evaluate a strategy on MT-Bench dataset, collecting detailed metrics per prompt."""
    results = {
        "strategy": strategy_name,
        "responses": [],
        "aggregated_metrics": {}
    }
    prompt_metrics_list = []

    if category:
        questions = dataset.get_questions_by_category(category)
    else:
        questions = dataset.questions
    print(f"Starting evaluation for '{strategy_name}' on {len(questions)} questions ({'Category: ' + category if category else 'All Categories'}).")

    if num_samples and num_samples < len(questions):
        import random
        questions = random.sample(questions, min(num_samples, len(questions)))
        print(f"Limited to {num_samples} randomly selected questions.")

    model_name_for_prompt = model.config._name_or_path if hasattr(model.config, "_name_or_path") else ""

    success_count = 0
    error_count = 0
    
    for i, question in enumerate(tqdm(questions, desc=f"Evaluating {strategy_name}")):
        prompt = dataset.format_prompt(question, turn_idx=0, model_name=model_name_for_prompt)
        if not prompt: 
            print(f"[WARNING] Empty prompt for question {question.get('question_id')}. Skipping.")
            continue

        print(f"\n[INFO] Processing question {i+1}/{len(questions)}: {question.get('question_id')} (Category: {question.get('category')})")
        
        try:
            # We'll set a timeout to prevent hanging
            response_text, prompt_metrics = generate_response(
                model, tokenizer, prompt, 
                strategy=strategy,
                max_new_tokens=max_tokens_per_response if max_tokens_per_response > 0 else None,
                timeout_seconds=timeout_seconds
            )
            
            if response_text.startswith("Error:"):
                print(f"[ERROR] Generation failed: {response_text}")
                error_count += 1
            else:
                success_count += 1
                
            results["responses"].append({
                "question_id": question["question_id"],
                "category": question["category"],
                "prompt": prompt,
                "response": response_text,
                "metrics": prompt_metrics,
            })

            if prompt_metrics:
                prompt_metrics_list.append(prompt_metrics)

            # Print per-question summary
            if prompt_metrics:
                print(f"[INFO] Results for question {question.get('question_id')}:")
                print(f"  Tokens: {prompt_metrics.get('input_tokens', 0)} in, {prompt_metrics.get('output_tokens', 0)} out")
                print(f"  Time: {prompt_metrics.get('total_time_s', 0):.2f}s ({prompt_metrics.get('tokens_per_second', 0):.2f} tokens/s)")
                print(f"  KV Cache: {prompt_metrics.get('avg_kv_cache_size_mb', 0):.2f} MB (kept ratio: {prompt_metrics.get('avg_kept_ratio', 0):.1%})")
                print(f"  Memory: {prompt_metrics.get('peak_memory_mb', 0):.2f} MB peak")
            
            # Force cleanup after each question
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"[ERROR] Failed to process question {question.get('question_id')}: {e}")
            traceback.print_exc()
            error_count += 1
            results["responses"].append({
                "question_id": question["question_id"],
                "category": question["category"],
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "metrics": {},
            })
            
            # Try to recover from errors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"[INFO] Strategy '{strategy_name}' completed: {success_count} successful, {error_count} errors")

    # --- Aggregate metrics across all successful prompts ---
    if prompt_metrics_list:
        agg_metrics = results["aggregated_metrics"]
        agg_metrics["avg_total_time_per_prompt_s"] = np.mean([m.get("total_time_s", 0) for m in prompt_metrics_list])
        agg_metrics["avg_peak_memory_per_prompt_mb"] = np.mean([m.get("peak_memory_mb", 0) for m in prompt_metrics_list])
        agg_metrics["avg_token_time_s"] = np.mean([m.get("avg_token_time_s", 0) for m in prompt_metrics_list])
        agg_metrics["avg_kv_cache_size_mb"] = np.mean([m.get("avg_kv_cache_size_mb", 0) for m in prompt_metrics_list])
        agg_metrics["avg_kept_ratio"] = np.mean([m.get("avg_kept_ratio", 0) for m in prompt_metrics_list])

        total_output_toks = sum(m.get("output_tokens", 0) for m in prompt_metrics_list)
        total_gen_time = sum(m.get("total_time_s", 0) for m in prompt_metrics_list)
        agg_metrics["overall_tokens_per_second"] = total_output_toks / total_gen_time if total_gen_time > 0 else 0
        
        # Additional helpful metrics
        agg_metrics["success_rate"] = success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
        agg_metrics["avg_output_tokens"] = np.mean([m.get("output_tokens", 0) for m in prompt_metrics_list])
    else:
        print(f"[WARNING] No successful prompt evaluations for strategy {strategy_name}.")
        results["aggregated_metrics"] = {k: 0 for k in [
            "avg_total_time_per_prompt_s", "avg_peak_memory_per_prompt_mb", "avg_token_time_s",
            "avg_kv_cache_size_mb", "avg_kept_ratio", "overall_tokens_per_second", 
            "success_rate", "avg_output_tokens"
        ]}

    return results

# --- Utility Functions ---
def save_results(results_list, output_file="mtbench_results.json"):
    """Save a list of results (one per strategy) to JSON file."""
    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (dict, list)): return convert_numpy_recursive(obj)
        return obj
    
    def convert_numpy_recursive(data):
        if isinstance(data, list): return [convert_numpy(item) for item in data]
        if isinstance(data, dict): return {key: convert_numpy(value) for key, value in data.items()}
        return convert_numpy(data)

    processed_results_list = convert_numpy_recursive(results_list)
    try:
        # First save to a temporary file in case of write errors
        temp_output_file = output_file + ".tmp"
        with open(temp_output_file, 'w') as f:
            json.dump(processed_results_list, f, indent=2)
        
        # If successful, rename to the actual output file
        os.replace(temp_output_file, output_file)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        traceback.print_exc()

def print_memory_stats(device):
    """Print current GPU memory stats for debugging."""
    if device.type != 'cuda':
        return
    
    print("\n--- GPU Memory Statistics ---")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
        print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate KV cache pruning strategies on MT-Bench")
    
    # Model and Data
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model name or path (e.g., meta-llama/Llama-2-7b-chat-hf)")
    parser.add_argument("--mtbench_data_path", default="mt_bench_data.jsonl",
                        help="Path to MT-Bench questions JSON file")
    
    # Evaluation Scope
    parser.add_argument("--category", help="Only test specific MT-Bench category (e.g., 'writing', 'reasoning')")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples (questions) to test per strategy")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for sample selection")
    
    # Pruning Strategies
    parser.add_argument("--pruning_ratios", type=str, default="0.5,0.25",
                       help="Comma-separated list of pruning ratios (e.g., 0.5,0.25) for ratio-based strategies")
    parser.add_argument("--max_length_ratio", type=float, default=0.5,
                       help="Ratio for fixed-size strategies - applied to estimated max length")
    parser.add_argument("--block_sizes", type=str, default="4,8",
                       help="Comma-separated list of block sizes for BlockAverageStrategy")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Where to store HF model files and tokenizer")
    
    # Strategy Selection
    parser.add_argument("--strategies", type=str, default="baseline,window,top,bottom,both,random,strided,attention,block",
                       help="Comma-separated list of strategies to run (e.g., 'baseline,window,top')")
    parser.add_argument("--only_strategy", type=str, default=None,
                       help="Run only this specific strategy (overrides --strategies)")
    
    # Generation Controls
    parser.add_argument("--max_tokens_per_response", type=int, default=0,
                       help="Max tokens per response (0 for unlimited)")
    parser.add_argument("--timeout_seconds", type=int, default=120,
                       help="Timeout per prompt generation in seconds")
    
    # Output
    parser.add_argument("--output_file", type=str, default="mtbench_results_fixed.json",
                       help="Output file for detailed JSON results")
    parser.add_argument("--save_intermediate", action="store_true",
                       help="Save results after each strategy evaluation")
    parser.add_argument("--debug", action="store_true",
                       help="Enable verbose debug output")
    
    args = parser.parse_args()

    # Set random seed
    import random
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
            print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")

    pruning_ratios_to_test = [float(r) for r in args.pruning_ratios.split(',')]
    block_sizes_to_test = [int(s) for s in args.block_sizes.split(',')]
    strategy_types_to_run = args.strategies.split(',') if not args.only_strategy else [args.only_strategy]
    
    print(f"Strategy types to run: {strategy_types_to_run}")
    print(f"Pruning ratios: {pruning_ratios_to_test}")
    print(f"Block sizes: {block_sizes_to_test}")
    
    try:
        dataset = MTBenchDataset(data_path=args.mtbench_data_path)
    except Exception as e:
        print(f"Failed to load MT-Bench dataset: {e}")
        traceback.print_exc()
        return

    # --- Load Model & Tokenizer ---
    print(f"Loading model: {args.model}")
    model_load_start = time.time()
    try:
        tokenizer = LlamaTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # --- Quantization for large models ---
        quantization_config = None
        model_name_lower = args.model.lower()
        if "13b" in model_name_lower or "70b" in model_name_lower:
            print("Applying 4-bit quantization for 13B/70B model...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print(f"Quantization config: {quantization_config.to_dict()}")

        model = LlamaForCausalLM.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )
        model.eval()
        print(f"Model loaded successfully in {time.time() - model_load_start:.2f}s.")
        if hasattr(model, 'hf_device_map'): 
            print(f"Model device map: {model.hf_device_map}")
        
        # Print memory usage after model loading
        print_memory_stats(device)

    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        traceback.print_exc()
        return

    # --- Estimate max tokens for fixed-size strategies ---
    # Use a higher estimate to avoid unnecessary constraints
    estimated_max_total_len = 2048
    max_len_abs = max(1, int(estimated_max_total_len * args.max_length_ratio))
    print(f"Using max_len={max_len_abs} for fixed-size strategies")

    # --- Define All Strategies ---
    all_strategies_to_run = []
    
    # Baseline (NoOp strategy)
    if "baseline" in strategy_types_to_run:
        all_strategies_to_run.append(("baseline_full", NoOpStrategy()))
    
    # Window strategy
    if "window" in strategy_types_to_run:
        all_strategies_to_run.append((f"window_{max_len_abs}t", WindowStrategy(max_length=max_len_abs)))
    
    # Ratio-based strategies
    for ratio in pruning_ratios_to_test:
        ratio_pct = int(ratio * 100)
        if "top" in strategy_types_to_run:
            all_strategies_to_run.append((f"top_{ratio_pct}pct", TopRatioStrategy(pruning_ratio=ratio)))
        if "bottom" in strategy_types_to_run:
            all_strategies_to_run.append((f"bottom_{ratio_pct}pct", BottomRatioStrategy(pruning_ratio=ratio)))
        if "both" in strategy_types_to_run:
            all_strategies_to_run.append((f"both_{ratio_pct}pct", BothRatioStrategy(pruning_ratio=ratio)))
    
    # Random sampling strategy
    if "random" in strategy_types_to_run:
        all_strategies_to_run.append((f"random_{max_len_abs}t", RandomSamplingStrategy(max_length=max_len_abs, seed=args.sample_seed)))
    
    # Strided strategy
    if "strided" in strategy_types_to_run:
        all_strategies_to_run.append((f"strided_{max_len_abs}t", StridedStrategy(max_length=max_len_abs)))
    
    # Attention score strategy
    if "attention" in strategy_types_to_run:
        all_strategies_to_run.append((f"attn_top{max_len_abs}t", AttentionScoreStrategy(max_length=max_len_abs)))
    
    # Block average strategy
    if "block" in strategy_types_to_run:
        for block_size in block_sizes_to_test:
            all_strategies_to_run.append((f"block_avg_size_{block_size}", BlockAverageStrategy(block_size=block_size)))
    
    print(f"Total strategies to evaluate: {len(all_strategies_to_run)}")
    for name, _ in all_strategies_to_run:
        print(f"  - {name}")

    # --- Run Evaluations ---
    all_results_list = []
    evaluation_start_time = time.time()

    for i, (name, strat_instance) in enumerate(all_strategies_to_run):
        print(f"\n===== EVALUATING STRATEGY: {name} ({i+1}/{len(all_strategies_to_run)}) =====")
        strategy_start_time = time.time()
        
        # Clear cache before starting strategy eval for cleaner memory stats
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            print_memory_stats(device)

        try:
            results = evaluate_mtbench(
                model, tokenizer, dataset,
                strategy_name=name,
                strategy=strat_instance,
                category=args.category,
                num_samples=args.num_samples,
                max_tokens_per_response=args.max_tokens_per_response,
                timeout_seconds=args.timeout_seconds
            )
            all_results_list.append(results)

            # Print intermediate summary
            agg_metrics = results.get("aggregated_metrics", {})
            print(f"--- Strategy '{name}' Summary ---")
            print(f"  Avg Total Time/Prompt: {agg_metrics.get('avg_total_time_per_prompt_s', 0):.3f} s")
            print(f"  Avg Peak Mem/Prompt:   {agg_metrics.get('avg_peak_memory_per_prompt_mb', 0):.2f} MB")
            print(f"  Avg KV Size/Step:      {agg_metrics.get('avg_kv_cache_size_mb', 0):.2f} MB")
            print(f"  Avg Kept Ratio/Step:   {agg_metrics.get('avg_kept_ratio', 0):.1%}")
            print(f"  Overall Throughput:    {agg_metrics.get('overall_tokens_per_second', 0):.2f} tokens/s")
            print(f"  Success Rate:          {agg_metrics.get('success_rate', 0):.1%}")
            print(f"  Strategy Eval Time:    {time.time() - strategy_start_time:.2f} s")
            
            # Save intermediate results if enabled
            if args.save_intermediate:
                interim_file = f"{os.path.splitext(args.output_file)[0]}_{name}.json"
                save_results([results], interim_file)
                print(f"  Intermediate results saved to: {interim_file}")

        except Exception as e:
            print(f"ERROR evaluating strategy {name}: {e}")
            traceback.print_exc()
            # Add an empty result to maintain order
            all_results_list.append({
                "strategy": name,
                "responses": [],
                "aggregated_metrics": {},
                "error": str(e)
            })

    evaluation_end_time = time.time()
    print(f"\nTotal evaluation duration (excluding model load): {evaluation_end_time - evaluation_start_time:.2f} s")

    # --- Save Final Results ---
    save_results(all_results_list, args.output_file)

    # --- Print Final Summary Table ---
    print("\n===== FINAL SUMMARY (Aggregated Metrics) =====")
    
    # Header
    print(f"{'Strategy':<20} | {'Avg Time/Prompt(s)':<20} | {'Avg Peak Mem(MB)':<18} | {'Avg KV Size(MB)':<16} | {'Avg Kept Ratio':<15} | {'Success Rate':<15} | {'Throughput(tok/s)':<20}")
    print("-" * 130)

    # Rows
    for result in all_results_list:
        strategy = result["strategy"]
        metrics = result.get("aggregated_metrics", {})
        print(f"{strategy:<20} | "
              f"{metrics.get('avg_total_time_per_prompt_s', 0):<20.3f} | "
              f"{metrics.get('avg_peak_memory_per_prompt_mb', 0):<18.2f} | "
              f"{metrics.get('avg_kv_cache_size_mb', 0):<16.2f} | "
              f"{metrics.get('avg_kept_ratio', 0):<15.1%} | "
              f"{metrics.get('success_rate', 0):<15.1%} | "
              f"{metrics.get('overall_tokens_per_second', 0):<20.2f}")

    print("-" * 130)
    print(f"Settings: model={args.model}, samples={args.num_samples}, category={args.category or 'all'}, device={device}, max_tokens={args.max_tokens_per_response}")

if __name__ == "__main__":
    main()
