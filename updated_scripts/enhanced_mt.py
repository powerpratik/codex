# NEW ONE WITH THRESHOLD BASED MT
# Ensure 'unpack_kv' is imported from updated_kv_manager
import torch # Make sure torch is imported
import argparse
import numpy as np
import os
import json
import time
import requests
import re
import gc
import signal
import traceback
from tqdm import tqdm
from contextlib import contextmanager

from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from updated_kv_manager import (
    KVCacheManager, NoOpStrategy, WindowStrategy,
    TopRatioStrategy, BottomRatioStrategy, BothRatioStrategy,
    RandomSamplingStrategy, StridedStrategy, AttentionScoreStrategy, BlockAverageStrategy,
    unpack_kv, pack_kv
)

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

def get_kv_cache_size(past_key_values) -> int:
    """Calculate the total size in bytes of the KV cache."""
    if past_key_values is None:
        return 0
    size_bytes = 0
    try:
        # Use the unpack_kv function to handle both HF Cache and legacy formats
        unpacked_kvs = unpack_kv(past_key_values)
        if not unpacked_kvs: return 0

        for k, v in unpacked_kvs:
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                size_bytes += k.numel() * k.element_size()
                size_bytes += v.numel() * v.element_size()
            else:
                # Handle potential inconsistencies if unpacking fails or returns non-tensors
                # Turn off debug print by default
                # print(f"[Warning] Non-tensor found in KV cache structure during size calculation.")
                pass
        return size_bytes
    except Exception as e:
        print(f"[Error] Failed to calculate KV cache size: {e}")
        return 0 # Return 0 or raise to indicate failure

# --- Updated Generation Function with Conditional Pruning ---
def generate_response(model, tokenizer, prompt, strategy=None, max_new_tokens=None, timeout_seconds=60, pruning_threshold_bytes=float('inf')):
    """Generate a response with CONDITIONAL KV cache pruning and detailed metrics collection"""
    if not prompt:
        return "Error: Empty prompt received", {}

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]
    is_cuda = model.device.type == 'cuda'
    # Debug print removed
    # print(f"[DEBUG] Starting generation with {input_token_count} input tokens")
    # print(f"[DEBUG] Strategy: {strategy.__class__.__name__}")
    # print(f"[DEBUG] Pruning Threshold: {pruning_threshold_bytes / (1024*1024):.2f} MB")

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(model.device)

    if strategy is None:
        strategy = NoOpStrategy()
    # Instantiate the manager but we'll call update conditionally
    cache_manager = KVCacheManager(strategy)

    step_metrics = {
        "kv_cache_sizes_bytes": [], "input_lengths": [],
        "kept_lengths": [], "token_inference_times": [],
        "pruning_applied_flags": [] # NEW: Track when pruning occurs
    }

    generated_text = "" # Initialize in case of early exit or error

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

                # Get the latest KV cache state from the model output
                new_past_key_values = outputs.past_key_values

                # --- CONDITIONAL PRUNING LOGIC ---
                current_cache_size_bytes = get_kv_cache_size(new_past_key_values)
                pruning_applied_this_step = False
                orig_len, kept_len = 0, 0 # Initialize lengths for metric recording

                if current_cache_size_bytes > pruning_threshold_bytes:
                    # Debug print removed
                    # print(f"[DEBUG] Step {step}: Cache size {current_cache_size_bytes / (1024*1024):.2f} MB > threshold {pruning_threshold_bytes / (1024*1024):.2f} MB. Applying pruning.")
                    try:
                        # Apply the user-defined strategy via the manager
                        pruned_past_list, orig_len, kept_len = cache_manager.update(new_past_key_values)
                        past_key_values = pack_kv(pruned_past_list) # Use pruned cache for next step
                        # Recalculate size after pruning for metrics
                        kv_size_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in pruned_past_list) if pruned_past_list else 0
                        pruning_applied_this_step = True
                        # Debug print removed
                        # print(f"[DEBUG] Step {step}: KV update successful - orig_len={orig_len}, kept_len={kept_len}, new_size={kv_size_bytes / (1024*1024):.2f} MB")
                    except Exception as e:
                        print(f"[ERROR] KV update (pruning) failed at step {step}: {e}")
                        traceback.print_exc()
                        raise # Propagate error
                else:
                    # Cache size is within threshold, do not prune. Use the output cache directly.
                    past_key_values = new_past_key_values
                    kv_size_bytes = current_cache_size_bytes
                    # Get lengths for metrics (no pruning means kept_len == orig_len)
                    unpacked_kvs = unpack_kv(past_key_values)
                    if unpacked_kvs:
                        # Safely access tensor size if unpacked_kvs is not empty and contains tensors
                        if unpacked_kvs[0] and len(unpacked_kvs[0]) > 0 and isinstance(unpacked_kvs[0][0], torch.Tensor):
                           orig_len = unpacked_kvs[0][0].size(-2)
                           kept_len = orig_len
                        else:
                           orig_len, kept_len = 0, 0 # Or handle appropriately
                    else:
                        orig_len, kept_len = 0, 0
                    pruning_applied_this_step = False

                # --- Record metrics for this step ---
                step_metrics["kv_cache_sizes_bytes"].append(kv_size_bytes)
                step_metrics["input_lengths"].append(orig_len) # Length *before* potential pruning
                step_metrics["kept_lengths"].append(kept_len) # Length *after* potential pruning
                step_metrics["pruning_applied_flags"].append(pruning_applied_this_step)

                # --- Continue generation ---
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                # Check stopping conditions
                if next_token_id.item() == tokenizer.eos_token_id:
                    # Debug print removed
                    # print(f"[DEBUG] EOS token generated at step {step}")
                    break
                if max_new_tokens is not None and (step + 1) >= max_new_tokens:
                    # Debug print removed
                    # print(f"[DEBUG] Reached max_new_tokens limit ({max_new_tokens})")
                    break
                # Print periodic status (optional, keeping it as INFO level)
                # if step > 0 and step % 100 == 0: # Changed frequency
                #     print(f"[INFO] Generated {step} tokens so far...")


            # --- End Generation Loop ---

            if is_cuda: torch.cuda.synchronize()
            prompt_total_time = time.time() - prompt_start_time

            generated_text = tokenizer.decode(generated_ids[0, input_token_count:], skip_special_tokens=True)
            # Debug print removed
            # print(f"[DEBUG] Generation complete: {len(generated_text)} chars, {generated_ids.shape[1] - input_token_count} tokens")

            current_mem_bytes = torch.cuda.memory_allocated(model.device) if is_cuda else 0
            prompt_peak_mem_bytes = torch.cuda.max_memory_allocated(model.device) if is_cuda else 0

            # Calculate average metrics
            avg_token_time = np.mean(step_metrics["token_inference_times"]) if step_metrics["token_inference_times"] else 0
            avg_kv_size_mb = (np.mean(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024)) if step_metrics["kv_cache_sizes_bytes"] else 0
            output_tokens = generated_ids.shape[1] - input_token_count
            tokens_per_second = output_tokens / prompt_total_time if prompt_total_time > 0 else 0

            # Calculate avg_kept_ratio ONLY for steps where pruning was applied
            pruned_steps_indices = [i for i, applied in enumerate(step_metrics["pruning_applied_flags"]) if applied]
            if pruned_steps_indices:
                pruned_input_lengths = [step_metrics["input_lengths"][i] for i in pruned_steps_indices]
                pruned_kept_lengths = [step_metrics["kept_lengths"][i] for i in pruned_steps_indices]
                # Guard against division by zero if input length was 0 on a pruned step (shouldn't happen)
                step_ratios = [(k / i) for k, i in zip(pruned_kept_lengths, pruned_input_lengths) if i > 0]
                avg_kept_ratio_on_prune = np.mean(step_ratios) if step_ratios else 0.0
            else:
                avg_kept_ratio_on_prune = 0.0 # Indicate N/A or 0 if no pruning happened

            pruning_trigger_rate = np.mean(step_metrics["pruning_applied_flags"]) if step_metrics["pruning_applied_flags"] else 0.0

            final_metrics = {
                "input_tokens": input_token_count,
                "output_tokens": output_tokens,
                "total_tokens_processed": generated_ids.shape[1],
                "total_time_s": prompt_total_time,
                "avg_token_time_s": avg_token_time,
                "tokens_per_second": tokens_per_second,
                "avg_kv_cache_size_mb": avg_kv_size_mb, # Average size across all steps
                "avg_kept_ratio_when_pruned": avg_kept_ratio_on_prune, # Avg ratio when pruning triggered
                "pruning_trigger_rate": pruning_trigger_rate, # % of steps where pruning happened
                "current_memory_mb": current_mem_bytes / (1024 * 1024),
                "peak_memory_mb": prompt_peak_mem_bytes / (1024 * 1024),
                # Summary of raw metrics
                "raw_metrics_summary": {
                    "num_steps": len(step_metrics["token_inference_times"]),
                    "min_kv_cache_mb": min(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024) if step_metrics.get("kv_cache_sizes_bytes") else 0,
                    "max_kv_cache_mb": max(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024) if step_metrics.get("kv_cache_sizes_bytes") else 0,
                }
            }

    except TimeoutException as e:
        print(f"[ERROR] Generation timeout: {e}")
        # Return minimal metrics on timeout
        return f"Error: Generation timed out after {timeout_seconds} seconds", {
            "input_tokens": input_token_count, "output_tokens": 0, "total_time_s": timeout_seconds, "error": "timeout"
        }
    except Exception as e:
        print(f"[ERROR] Generation error: {e}")
        traceback.print_exc()
         # Return minimal metrics on other errors
        return f"Error during generation: {str(e)}", {
            "input_tokens": input_token_count, "output_tokens": 0, "total_time_s": time.time() - prompt_start_time if 'prompt_start_time' in locals() else 0, "error": str(e)
        }

    return generated_text, final_metrics


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

# --- Main Function (incorporating conditional pruning threshold) ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate KV cache pruning strategies on MT-Bench")

    # Model and Data
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model name or path (e.g., meta-llama/Llama-2-7b-chat-hf)")
    parser.add_argument("--mtbench_data_path", default="mt_bench_data.jsonl",
                        help="Path to MT-Bench questions JSON file")

    # Evaluation Scope
    parser.add_argument("--category", help="Only test specific MT-Bench category (e.g., 'writing', 'reasoning')")
    parser.add_argument("--num_samples", type=int, default=1,
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
    parser.add_argument("--cache_dir", type=str, default='/home/cc/llm_backup_chameleon_recent/codex/updated_scripts',
                        help="Where to store HF model files and tokenizer")

    # *** NEW: Conditional Pruning Threshold ***
    parser.add_argument("--pruning_threshold_mb", type=float, default=float('inf'),
                        help="KV cache size threshold in MB to trigger pruning. "
                             "'inf' (default) means apply strategy always (if not NoOp). "
                             "Set 0 to disable pruning beyond NoOp.")

    # Strategy Selection
    parser.add_argument("--strategies", type=str, default="baseline,window,top,bottom,both,random,strided,attention,block",
                        help="Comma-separated list of strategies to run (e.g., 'baseline,window,top')")
    parser.add_argument("--only_strategy", type=str, default=None,
                        help="Run only this specific strategy (overrides --strategies)")

    # Generation Controls
    parser.add_argument("--max_tokens_per_response", type=int, default=0,
                        help="Max tokens per response (0 for model default/unlimited within reason)")
    parser.add_argument("--timeout_seconds", type=int, default=120,
                        help="Timeout per prompt generation in seconds")

    # Output
    parser.add_argument("--output_file", type=str, default="mtbench_results_conditional.json", # Updated default name
                        help="Output file for detailed JSON results")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save results after each strategy evaluation")
    # Debug flag kept for potential future use, but internal prints removed
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output (currently minimal effect)")

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
            print(f" - Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")

    # --- Process Threshold ---
    pruning_threshold_bytes = args.pruning_threshold_mb * 1024 * 1024 if args.pruning_threshold_mb != float('inf') else float('inf')
    if args.pruning_threshold_mb == 0: # Handle 0MB case explicitly to mean never prune
        pruning_threshold_bytes = float('inf')
        print("[INFO] Pruning threshold set to 0 MB, effectively disabling conditional pruning.")
    elif pruning_threshold_bytes != float('inf'):
        print(f"[INFO] Pruning will be triggered when KV cache exceeds {args.pruning_threshold_mb:.2f} MB ({pruning_threshold_bytes} bytes).")
    else:
         print("[INFO] Pruning threshold is infinity. Pruning strategy (if not NoOp) will be applied based on its own logic (likely every step).")


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
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print(f"Quantization config: {quantization_config.to_dict()}")
            except Exception as q_e:
                print(f"Warning: Failed to create BitsAndBytesConfig, likely missing dependencies. Proceeding without quantization. Error: {q_e}")
                quantization_config = None


        model = LlamaForCausalLM.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            device_map="auto", # Handles multi-GPU placement
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )

        model.eval()
        print(f"Model loaded successfully in {time.time() - model_load_start:.2f}s.")
        if hasattr(model, 'hf_device_map'):
            print(f"Model device map: {model.hf_device_map}")
        print_memory_stats(device)

    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        traceback.print_exc()
        return

    # --- Estimate max tokens for fixed-size strategies ---
    estimated_max_total_len = 4096 # Using a higher estimate for general cases
    max_len_abs = max(1, int(estimated_max_total_len * args.max_length_ratio))
    print(f"Using max_len={max_len_abs} for fixed-size strategies (Window, Random, etc.)")

    # --- Define All Strategies ---
    all_strategies_to_run = []
    # Baseline (NoOp strategy)
    if "baseline" in strategy_types_to_run or "baseline_full" in strategy_types_to_run:
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
             # Ensure block size is reasonable, e.g., not larger than max_len_abs
             if block_size > 0 and block_size < max_len_abs:
                all_strategies_to_run.append((f"block_avg_size_{block_size}", BlockAverageStrategy(block_size=block_size)))

    print(f"Total strategies to evaluate: {len(all_strategies_to_run)}")
    for name, _ in all_strategies_to_run:
        print(f" - {name}")

    # --- Run Evaluations ---
    all_results_list = []
    evaluation_start_time = time.time()

    for i, (name, strat_instance) in enumerate(all_strategies_to_run):
        print(f"\n===== EVALUATING STRATEGY: {name} ({i+1}/{len(all_strategies_to_run)}) =====")
        strategy_start_time = time.time()

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        # print_memory_stats(device) # Optional: print stats before each run

        try:
            # *** Pass the threshold to evaluate_mtbench ***
            results = evaluate_mtbench(
                model, tokenizer, dataset,
                strategy_name=name,
                strategy=strat_instance,
                category=args.category,
                num_samples=args.num_samples,
                max_tokens_per_response=args.max_tokens_per_response,
                timeout_seconds=args.timeout_seconds,
                # *** ADD THIS LINE ***
                pruning_threshold_bytes=pruning_threshold_bytes
            )

            all_results_list.append(results)

            # Print intermediate summary including new metrics
            agg_metrics = results.get("aggregated_metrics", {})
            print(f"--- Strategy '{name}' Summary ---")
            print(f" Avg Total Time/Prompt: {agg_metrics.get('avg_total_time_per_prompt_s', 0):.3f} s")
            print(f" Avg Peak Mem/Prompt: {agg_metrics.get('avg_peak_memory_per_prompt_mb', 0):.2f} MB")
            print(f" Avg KV Size/Step: {agg_metrics.get('avg_kv_cache_size_mb', 0):.2f} MB")
            # Use the new metrics from generate_response's final_metrics
            print(f" Pruning Trigger Rate: {agg_metrics.get('pruning_trigger_rate', 0):.1%}")
            print(f" Avg Kept Ratio (when pruned): {agg_metrics.get('avg_kept_ratio_when_pruned', 0):.1%}")
            print(f" Overall Throughput: {agg_metrics.get('overall_tokens_per_second', 0):.2f} tokens/s")
            print(f" Success Rate: {agg_metrics.get('success_rate', 0):.1%}")
            print(f" Strategy Eval Time: {time.time() - strategy_start_time:.2f} s")

            if args.save_intermediate:
                interim_file = f"{os.path.splitext(args.output_file)[0]}_{name}.json"
                save_results([results], interim_file)
                print(f" Intermediate results saved to: {interim_file}")

        except Exception as e:
            print(f"ERROR evaluating strategy {name}: {e}")
            traceback.print_exc()
            all_results_list.append({
                "strategy": name, "responses": [], "aggregated_metrics": {}, "error": str(e)
            })

    evaluation_end_time = time.time()
    print(f"\nTotal evaluation duration (excluding model load): {evaluation_end_time - evaluation_start_time:.2f} s")

    # --- Save Final Results ---
    save_results(all_results_list, args.output_file)

    # --- Print Final Summary Table ---
    print("\n===== FINAL SUMMARY (Aggregated Metrics) =====")
    # Updated Header
    print(f"{'Strategy':<20} | {'Avg Time/Prompt(s)':<20} | {'Avg Peak Mem(MB)':<18} | {'Avg KV Size(MB)':<16} | {'Prune Rate':<12} | {'Avg Kept (P)':<13} | {'Success Rate':<15} | {'Throughput(tok/s)':<20}")
    print("-" * 150) # Adjust width

    # Updated Rows
    for result in all_results_list:
        strategy = result["strategy"]
        metrics = result.get("aggregated_metrics", {})

        # Format new metrics
        prune_rate_str = f"{metrics.get('pruning_trigger_rate', 0):.1%}"
        # Show N/A if pruning never happened
        kept_ratio_str = f"{metrics.get('avg_kept_ratio_when_pruned', 0):.1%}" if metrics.get('pruning_trigger_rate', 0) > 0 else "N/A"

        print(f"{strategy:<20} | "
              f"{metrics.get('avg_total_time_per_prompt_s', 0):<20.3f} | "
              f"{metrics.get('avg_peak_memory_per_prompt_mb', 0):<18.2f} | "
              f"{metrics.get('avg_kv_cache_size_mb', 0):<16.2f} | "
              f"{prune_rate_str:<12} | " # Pruning Trigger Rate
              f"{kept_ratio_str:<13} | " # Avg Kept Ratio (When Pruned)
              f"{metrics.get('success_rate', 0):<15.1%} | "
              f"{metrics.get('overall_tokens_per_second', 0):<20.2f}")

    print("-" * 150) # Adjust width
    print(f"Settings: model={args.model}, samples={args.num_samples}, category={args.category or 'all'}, device={device}, max_tokens={args.max_tokens_per_response}, threshold={args.pruning_threshold_mb}MB")
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

if __name__ == "__main__":
    # Ensure necessary functions like evaluate_mtbench, MTBenchDataset, save_results, print_memory_stats
    # and strategy classes are defined or imported correctly above this point.
    # Define or import evaluate_mtbench that accepts pruning_threshold_bytes
    def evaluate_mtbench(model, tokenizer, dataset, strategy_name="baseline", strategy=None, category=None,
                     num_samples=None, max_tokens_per_response=1024, timeout_seconds=120,
                     pruning_threshold_bytes=float('inf')): # Ensure signature matches
        results = { "strategy": strategy_name, "responses": [], "aggregated_metrics": {} }
        prompt_metrics_list = []
        if category:
            questions = dataset.get_questions_by_category(category)
        else:
            questions = dataset.questions
        print(f"Starting evaluation for '{strategy_name}' on {len(questions)} questions ({'Category: ' + category if category else 'All Categories'}).")

        if num_samples and num_samples > 0 and num_samples < len(questions):
           # Use seeded sampling if samples are limited
           import random
           # Ensure random state is consistent if desired across runs with same seed
           # random.seed(args.sample_seed) # Seed set globally in main
           questions = random.sample(questions, min(num_samples, len(questions)))
           print(f"Limited to {num_samples} randomly selected questions.")
        elif num_samples is None or num_samples <= 0:
             # Use all questions if num_samples is not specified or invalid
             num_samples = len(questions) # Use all if not specified or <= 0
             print(f"Using all {num_samples} questions.")

        model_name_for_prompt = model.config._name_or_path if hasattr(model.config, "_name_or_path") else ""
        success_count = 0
        error_count = 0

        for i, question in enumerate(tqdm(questions, desc=f"Evaluating {strategy_name}", total=len(questions))):
            prompt = dataset.format_prompt(question, turn_idx=0, model_name=model_name_for_prompt)
            if not prompt:
                print(f"[WARNING] Empty prompt for question {question.get('question_id')}. Skipping.")
                continue

            print(f"\n[INFO] Processing question {i+1}/{len(questions)}: {question.get('question_id')} (Category: {question.get('category')})")
            try:
                # Pass the threshold to generate_response
                response_text, prompt_metrics = generate_response(
                    model, tokenizer, prompt,
                    strategy=strategy,
                    max_new_tokens=max_tokens_per_response if max_tokens_per_response > 0 else None,
                    timeout_seconds=timeout_seconds,
                    pruning_threshold_bytes=pruning_threshold_bytes # Pass threshold here
                )

                if response_text.startswith("Error:") or "error" in prompt_metrics:
                    print(f"[ERROR] Generation failed: {response_text} | Metrics: {prompt_metrics}")
                    error_count += 1
                    # Store error response
                    results["responses"].append({
                        "question_id": question["question_id"], "category": question["category"],
                        "prompt": prompt, "response": response_text, "metrics": prompt_metrics, "status": "error"
                    })
                else:
                    success_count += 1
                    results["responses"].append({
                        "question_id": question["question_id"], "category": question["category"],
                        "prompt": prompt, "response": response_text, "metrics": prompt_metrics, "status": "success"
                    })
                    if prompt_metrics: # Should always have metrics on success
                        prompt_metrics_list.append(prompt_metrics)

                    # Print per-question summary
                    if prompt_metrics:
                        print(f"[INFO] Results for question {question.get('question_id')}:")
                        print(f" Tokens: {prompt_metrics.get('input_tokens', 0)} in, {prompt_metrics.get('output_tokens', 0)} out")
                        print(f" Time: {prompt_metrics.get('total_time_s', 0):.2f}s ({prompt_metrics.get('tokens_per_second', 0):.2f} tokens/s)")
                        print(f" KV Cache: {prompt_metrics.get('avg_kv_cache_size_mb', 0):.2f} MB (prune rate: {prompt_metrics.get('pruning_trigger_rate', 0):.1%})")
                        print(f" Memory: {prompt_metrics.get('peak_memory_mb', 0):.2f} MB peak")

                # Force cleanup after each question
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"[ERROR] Unexpected error processing question {question.get('question_id')}: {e}")
                traceback.print_exc()
                error_count += 1
                results["responses"].append({
                    "question_id": question["question_id"], "category": question["category"],
                    "prompt": prompt, "response": f"Error: {str(e)}", "metrics": {}, "status": "error"
                })
                # Try to recover
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()

        print(f"[INFO] Strategy '{strategy_name}' completed: {success_count} successful, {error_count} errors")

        # --- Aggregate metrics ---
        if prompt_metrics_list:
            agg_metrics = results["aggregated_metrics"]
            # Average standard metrics
            agg_metrics["avg_total_time_per_prompt_s"] = np.mean([m.get("total_time_s", 0) for m in prompt_metrics_list if m])
            agg_metrics["avg_peak_memory_per_prompt_mb"] = np.mean([m.get("peak_memory_mb", 0) for m in prompt_metrics_list if m])
            agg_metrics["avg_token_time_s"] = np.mean([m.get("avg_token_time_s", 0) for m in prompt_metrics_list if m])
            agg_metrics["avg_kv_cache_size_mb"] = np.mean([m.get("avg_kv_cache_size_mb", 0) for m in prompt_metrics_list if m])

            # Average the new conditional pruning metrics
            agg_metrics["pruning_trigger_rate"] = np.mean([m.get("pruning_trigger_rate", 0) for m in prompt_metrics_list if m])
            # For avg_kept_ratio_when_pruned, average only over prompts where pruning occurred at least once
            ratios_when_pruned = [m.get("avg_kept_ratio_when_pruned", 0) for m in prompt_metrics_list if m and m.get("pruning_trigger_rate", 0) > 0]
            agg_metrics["avg_kept_ratio_when_pruned"] = np.mean(ratios_when_pruned) if ratios_when_pruned else 0.0

            # Calculate overall throughput based on successful prompts
            total_output_toks = sum(m.get("output_tokens", 0) for m in prompt_metrics_list if m)
            total_gen_time = sum(m.get("total_time_s", 0) for m in prompt_metrics_list if m)
            agg_metrics["overall_tokens_per_second"] = total_output_toks / total_gen_time if total_gen_time > 0 else 0
            agg_metrics["success_rate"] = success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
            agg_metrics["avg_output_tokens"] = np.mean([m.get("output_tokens", 0) for m in prompt_metrics_list if m])
        else:
            print(f"[WARNING] No successful prompt evaluations with metrics for strategy {strategy_name}.")
            # Fill with zeros or defaults
            results["aggregated_metrics"] = {
                "avg_total_time_per_prompt_s": 0, "avg_peak_memory_per_prompt_mb": 0, "avg_token_time_s": 0,
                "avg_kv_cache_size_mb": 0, "pruning_trigger_rate": 0, "avg_kept_ratio_when_pruned": 0,
                "overall_tokens_per_second": 0, "success_rate": 0, "avg_output_tokens": 0
            }
            if error_count > 0: # If there were only errors
                 results["aggregated_metrics"]["success_rate"] = 0.0


        return results

    # Need these utility functions defined or imported
    # Assuming MTBenchDataset, save_results, print_memory_stats, TimeoutException, time_limit are defined above
    # Assuming strategy classes (NoOpStrategy, etc.) and KVCacheManager are imported from updated_kv_manager
    main()

