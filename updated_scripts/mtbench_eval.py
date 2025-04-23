"""
mtbench_eval.py

Evaluates KV cache pruning strategies on the MT-Bench dataset using LLM-as-a-judge.
Supports various Llama 2 models (7B, 13B, 70B) with quantization.

Collects detailed metrics including per-token time, total time per prompt,
peak GPU memory per prompt, KV cache size per step, retention ratio per step,
and LLM-as-a-judge scores.
"""

import os
import json
import torch
import argparse
import numpy as np
import time
import requests
import re # For judge score extraction
import gc # For garbage collection
from tqdm import tqdm
# Ensure transformers and necessary dependencies are installed
try:
    from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
except ImportError:
     print("Please install transformers, bitsandbytes, accelerate: pip install transformers bitsandbytes accelerate")
     exit()
# Ensure openai package is installed for judging
try:
    from openai import OpenAI
    from openai import AzureOpenAI
except ImportError:
     print("OpenAI package not installed for LLM-as-a-judge. Run: pip install openai")
     OpenAI = None # Set to None if not available

# Import from our kv_cache_manager
from updated_kv_manager import (
    KVCacheManager, NoOpStrategy, WindowStrategy,
    TopRatioStrategy, BottomRatioStrategy, BothRatioStrategy,
    # Assuming other strategies might be tested too (uncomment if needed)
    # RandomSamplingStrategy, StridedStrategy, AttentionScoreStrategy, BlockAverageStrategy,
    unpack_kv, pack_kv
)

# --- MTBenchDataset Class (from previous version, minor improvements) ---
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
        url = "https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/raw/main/question.jsonl" # HF link
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes
            questions = []
            for line in response.text.strip().split('\n'):
                questions.append(json.loads(line))
            with open(self.data_path, 'w') as f:
                 json.dump(questions, f, indent=2) # Added indent for readability
            print(f"MT-Bench dataset downloaded to {self.data_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading MT-Bench dataset: {e}")
            print("Please download manually from https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts and save as mt_bench_data.json")
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
                    # Optionally remove or mark the question
            return [q for q in questions if 'turns' in q and isinstance(q['turns'], list) and len(q['turns']) > 0] # Filter out invalid ones
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
# Add near the top of the file if not already present:
# import matplotlib.pyplot as plt # Needed if visualize_comparison calls plt.show()
import torch.nn.functional as F
import numpy as np
import time
from updated_kv_manager import KVCacheManager, NoOpStrategy, pack_kv # Ensure necessary imports

# --- Updated Generation Function with Visualization Capture ---
def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1024,
    strategy=None,
    # --- NEW: Visualization Args ---
    capture_step=-1, # Step at which to capture data (-1 means no capture)
    output_attentions=False # Whether model forward should output attentions
):
    """Generate a response with KV cache pruning and metrics collection, optionally capturing data."""
    if not prompt:
         return "Error: Empty prompt received", {}

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]
    is_cuda = model.device.type == 'cuda'

    # --- Setup before generation ---
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(model.device)

    # Ensure cache manager is set up
    if strategy is None: strategy = NoOpStrategy()
    cache_manager = KVCacheManager(strategy)

    # --- NEW: Storage for captured data ---
    captured_data = {
        "logits": None,
        "attentions": None, # Store processed attention for the target step
        "prev_kept_len": None # Length of KV cache *before* predicting the token at capture_step
    }

    step_metrics = { # Keep existing metrics
        "kv_cache_sizes_bytes": [], "input_lengths": [],
        "kept_lengths": [], "token_inference_times": []
    }
    generated_ids = inputs["input_ids"]
    past_key_values = None
    prompt_start_time = time.time()

    try:
        # --- Generation Loop ---
        for step in range(max_new_tokens):
            token_gen_start = time.time()
            current_input_ids = generated_ids if past_key_values is None else generated_ids[:, -1:]

            # Decide if we need attentions for this step
            should_output_attentions = output_attentions and (step == capture_step)

            with torch.no_grad():
                outputs = model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=should_output_attentions # Pass flag to model
                )

            if is_cuda: torch.cuda.synchronize()
            token_time = time.time() - token_gen_start
            step_metrics["token_inference_times"].append(token_time)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # --- NEW: Capture Data at target step BEFORE pruning ---
            if step == capture_step:
                print(f"Capturing data at step {step}...")
                captured_data["logits"] = next_token_logits.clone().cpu() # Capture logits for next token prediction
                if should_output_attentions and outputs.attentions:
                    # Process attentions: Focus on last layer, average over heads
                    # Shape: (batch, n_heads, query_len, key_len)
                    last_layer_attention = outputs.attentions[-1].clone().cpu() # (batch, n_heads, q_len, k_len)
                    # Get attention from last query token to all keys
                    # Note: q_len is 1 for step > 0, prompt_len for step 0
                    attention_last_token = last_layer_attention[0, :, -1, :] # (n_heads, k_len)
                    # Average across heads for simplicity
                    captured_data["attentions"] = attention_last_token.mean(dim=0).numpy() # (k_len,)
                # Capture the length of the KV cache *before* this prediction was made
                # This length determined the context available for this step's logits/attention
                if past_key_values:
                    # Get length from the first layer's key tensor in the packed format
                    # Handle both tuple and Cache formats if necessary
                    if isinstance(past_key_values, tuple): # Legacy format
                         captured_data["prev_kept_len"] = past_key_values[0][0].size(-2)
                    elif hasattr(past_key_values, 'get_seq_length'): # HF Cache object
                         captured_data["prev_kept_len"] = past_key_values.get_seq_length(layer_idx=0)
                    else: # Fallback if format is unknown
                         captured_data["prev_kept_len"] = step_metrics["kept_lengths"][-1] if step_metrics["kept_lengths"] else input_token_count
                else: # First step, context length is input length
                    captured_data["prev_kept_len"] = input_token_count

            # --- Update KV cache using the manager ---
            pruned_past_list, orig_len, kept_len = cache_manager.update(outputs.past_key_values)
            past_key_values = pack_kv(pruned_past_list) # Pack for the next iteration

            # --- Record metrics AFTER pruning ---
            kv_size_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                               for k, v in pruned_past_list) if pruned_past_list else 0
            step_metrics["kv_cache_sizes_bytes"].append(kv_size_bytes)
            # orig_len is length before pruning this step
            step_metrics["input_lengths"].append(orig_len)
             # kept_len is length after pruning this step
            step_metrics["kept_lengths"].append(kept_len)

            # Append generated token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id: break
            if step == capture_step and capture_step != -1: # Stop early if just visualizing
                print(f"Stopping generation after capturing step {capture_step}.")
                break
        # --- End Generation Loop ---

        if is_cuda: torch.cuda.synchronize() # Ensure all ops are done
        prompt_total_time = time.time() - prompt_start_time
        generated_text = tokenizer.decode(generated_ids[0, input_token_count:], skip_special_tokens=True)
        prompt_peak_mem_bytes = torch.cuda.max_memory_allocated(model.device) if is_cuda else 0

        # --- Calculate final aggregated metrics for this prompt ---
        avg_token_time = np.mean(step_metrics["token_inference_times"]) if step_metrics["token_inference_times"] else 0
        avg_kv_size_mb = (np.mean(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024)) if step_metrics["kv_cache_sizes_bytes"] else 0
        # Use the lengths recorded *before* pruning each step for ratio calculation
        avg_input_len = np.mean(step_metrics["input_lengths"]) if step_metrics["input_lengths"] else 0
        avg_kept_len = np.mean(step_metrics["kept_lengths"]) if step_metrics["kept_lengths"] else 0
        # Calculate avg_kept_ratio based on per-step input/kept lengths for accuracy
        step_ratios = [(k / i) for k, i in zip(step_metrics["kept_lengths"], step_metrics["input_lengths"]) if i > 0]
        avg_kept_ratio = np.mean(step_ratios) if step_ratios else 0.0


        final_metrics = {
            "input_tokens": input_token_count,
            "output_tokens": generated_ids.shape[1] - input_token_count,
            "total_tokens_processed": generated_ids.shape[1], # Includes prompt
            "total_time_s": prompt_total_time, # Total time for this prompt
            "avg_token_time_s": avg_token_time, # Average per generated token
            "avg_kv_cache_size_mb": avg_kv_size_mb, # Average size per step after pruning
            "avg_kept_ratio": avg_kept_ratio, # Average retention per step
            "peak_memory_mb": prompt_peak_mem_bytes / (1024 * 1024), # Peak memory for this prompt
            # --- NEW: Add captured data ---
            "captured_data": captured_data if capture_step != -1 else None
        }

    except Exception as e:
        print(f"Generation error: {e}"); import traceback; traceback.print_exc()
        # Return metrics collected so far and empty captured data on error
        final_metrics = {
            "input_tokens": input_token_count,
            "output_tokens": generated_ids.shape[1] - input_token_count if 'generated_ids' in locals() else 0,
            "total_tokens_processed": generated_ids.shape[1] if 'generated_ids' in locals() else input_token_count,
            "total_time_s": time.time() - prompt_start_time,
            "avg_token_time_s": np.mean(step_metrics["token_inference_times"]) if step_metrics.get("token_inference_times") else 0,
            "avg_kv_cache_size_mb": (np.mean(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024)) if step_metrics.get("kv_cache_sizes_bytes") else 0,
            "avg_kept_ratio": np.mean([(k / i) for k, i in zip(step_metrics.get("kept_lengths",[]), step_metrics.get("input_lengths",[])) if i > 0]) if step_metrics.get("input_lengths") else 0,
            "peak_memory_mb": torch.cuda.max_memory_allocated(model.device) / (1024 * 1024) if is_cuda else 0,
            "captured_data": None # No valid capture on error
        }
        return "Error during generation", final_metrics


    return generated_text, final_metrics

# --- NEW: Visualization Function ---
def visualize_comparison(
    baseline_response_data,
    pruned_response_data,
    tokenizer,
    capture_step,
    top_k=5
    ):
    """Visualizes attention and top-K probability differences."""
    print("\n--- Visualization Comparison ---")
    print(f"Comparing Baseline vs. '{pruned_response_data.get('strategy_name', 'Pruned')}' at Generation Step {capture_step}")

    baseline_capture = baseline_response_data.get("metrics", {}).get("captured_data")
    pruned_capture = pruned_response_data.get("metrics", {}).get("captured_data")

    if not baseline_capture or not pruned_capture:
        print("ERROR: Captured data missing for baseline or pruned run. Cannot visualize.")
        return

    # --- 1. Visualize Top-K Probabilities ---
    print(f"\n--- Top-{top_k} Next Token Probabilities ---")
    baseline_logits = baseline_capture.get("logits")
    pruned_logits = pruned_capture.get("logits")

    if baseline_logits is not None and pruned_logits is not None:
        # Ensure logits are tensors before applying softmax
        if not isinstance(baseline_logits, torch.Tensor): baseline_logits = torch.tensor(baseline_logits)
        if not isinstance(pruned_logits, torch.Tensor): pruned_logits = torch.tensor(pruned_logits)

        baseline_probs = F.softmax(baseline_logits.float(), dim=-1).squeeze() # Use float for stability
        pruned_probs = F.softmax(pruned_logits.float(), dim=-1).squeeze()

        baseline_top_p, baseline_top_i = torch.topk(baseline_probs, top_k)
        pruned_top_p, pruned_top_i = torch.topk(pruned_probs, top_k)

        print(f"{'Rank':<5} | {'Baseline Token':<20} {'Prob':<7} | {'Pruned Token':<20} {'Prob':<7}")
        print("-" * 70)
        for i in range(top_k):
            # Handle potential decoding errors gracefully
            try:
                b_tok = tokenizer.decode(baseline_top_i[i].item())
            except Exception:
                b_tok = f"ID:{baseline_top_i[i].item()}"
            b_prob = baseline_top_p[i].item()

            try:
                p_tok = tokenizer.decode(pruned_top_i[i].item())
            except Exception:
                p_tok = f"ID:{pruned_top_i[i].item()}"
            p_prob = pruned_top_p[i].item()

            print(f"{i+1:<5} | {b_tok:<20} {b_prob:<7.2%} | {p_tok:<20} {p_prob:<7.2%}")
    else:
        print("Logits not captured for comparison.")

    # --- 2. Visualize Attention (if captured) ---
    print(f"\n--- Attention Weights (Last Layer Avg Head) ---")
    baseline_attn = baseline_capture.get("attentions")
    pruned_attn = pruned_capture.get("attentions")
    baseline_len = baseline_capture.get("prev_kept_len", 0)
    pruned_len = pruned_capture.get("prev_kept_len", 0)


    if baseline_attn is not None and pruned_attn is not None and baseline_len > 0 and pruned_len > 0:
        # Ensure attention arrays are numpy
        if not isinstance(baseline_attn, np.ndarray): baseline_attn = np.array(baseline_attn)
        if not isinstance(pruned_attn, np.ndarray): pruned_attn = np.array(pruned_attn)

        max_len = max(baseline_len, pruned_len)
        # Pad shorter attention arrays if lengths differ due to pruning at previous steps
        if baseline_len < max_len:
             # Pad with zeros - assumes non-attended tokens have zero effective weight
            baseline_attn_padded = np.pad(baseline_attn, (0, max_len - baseline_len), 'constant')
        else:
            baseline_attn_padded = baseline_attn[:max_len] # Ensure it's not longer just in case

        if pruned_len < max_len:
            pruned_attn_padded = np.pad(pruned_attn, (0, max_len - pruned_len), 'constant')
        else:
            pruned_attn_padded = pruned_attn[:max_len]


        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle(f"Attention from Query Token at Step {capture_step} to Previous Key Tokens")

        axs[0].bar(range(max_len), baseline_attn_padded)
        axs[0].set_title(f"Baseline (Key Length: {baseline_len})")
        axs[0].set_xlabel("Key Token Index (Previous Tokens)")
        axs[0].set_ylabel("Average Attention Weight")
        axs[0].grid(True, axis='y')
        axs[0].set_xlim(-1, max_len) # Set limits for clarity

        axs[1].bar(range(max_len), pruned_attn_padded)
        axs[1].set_title(f"'{pruned_response_data.get('strategy_name', 'Pruned')}' (Key Length: {pruned_len})")
        axs[1].set_xlabel("Key Token Index (Previous Tokens)")
        axs[1].grid(True, axis='y')
        axs[1].set_xlim(-1, max_len) # Set limits for clarity

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show() # Display the plot
        # Consider saving the plot:
        try:
            save_filename = f"attention_step_{capture_step}_{pruned_response_data.get('strategy_name', 'Pruned')}.png"
            plt.savefig(save_filename)
            print(f"Attention plot saved to {save_filename}")
        except Exception as e:
             print(f"Error saving attention plot: {e}")
    else:
        print("Attention weights not captured or lengths invalid for comparison.")

    # --- Optional: Display generated text for context ---
    print("\n--- Generated Text Snippets (Up to capture step + 5) ---")
    print(f"Baseline Response:\n...\n{baseline_response_data.get('response', 'N/A')}")
    print(f"\nPruned Response ({pruned_response_data.get('strategy_name', 'Pruned')}):\n...\n{pruned_response_data.get('response', 'N/A')}")
    print("-" * 70)


# --- NEW: Visualization Function ---
def visualize_comparison(
   baseline_response_data,
   pruned_response_data,
   tokenizer,
   capture_step,
   top_k=5
   ):
   """Visualizes attention and top-K probability differences."""
   print("\n--- Visualization Comparison ---")
   print(f"Comparing Baseline vs. '{pruned_response_data.get('strategy_name', 'Pruned')}' at Generation Step {capture_step}")

   baseline_capture = baseline_response_data.get("metrics", {}).get("captured_data")
   pruned_capture = pruned_response_data.get("metrics", {}).get("captured_data")

   if not baseline_capture or not pruned_capture:
       print("ERROR: Captured data missing for baseline or pruned run. Cannot visualize.")
       return

   # --- 1. Visualize Top-K Probabilities ---
   print(f"\n--- Top-{top_k} Next Token Probabilities ---")
   baseline_logits = baseline_capture.get("logits")
   pruned_logits = pruned_capture.get("logits")

   if baseline_logits is not None and pruned_logits is not None:
       baseline_probs = F.softmax(baseline_logits.float(), dim=-1).squeeze() # Use float for stability
       pruned_probs = F.softmax(pruned_logits.float(), dim=-1).squeeze()

       baseline_top_p, baseline_top_i = torch.topk(baseline_probs, top_k)
       pruned_top_p, pruned_top_i = torch.topk(pruned_probs, top_k)

       print(f"{'Rank':<5} | {'Baseline Token':<20} {'Prob':<7} | {'Pruned Token':<20} {'Prob':<7}")
       print("-" * 70)
       for i in range(top_k):
           b_tok = tokenizer.decode(baseline_top_i[i].item())
           b_prob = baseline_top_p[i].item()
           p_tok = tokenizer.decode(pruned_top_i[i].item())
           p_prob = pruned_top_p[i].item()
           print(f"{i+1:<5} | {b_tok:<20} {b_prob:<7.2%} | {p_tok:<20} {p_prob:<7.2%}")
   else:
       print("Logits not captured for comparison.")

   # --- 2. Visualize Attention (if captured) ---
   print(f"\n--- Attention Weights (Last Layer Avg Head) ---")
   baseline_attn = baseline_capture.get("attentions")
   pruned_attn = pruned_capture.get("attentions")
   baseline_len = baseline_capture.get("prev_kept_len", 0)
   pruned_len = pruned_capture.get("prev_kept_len", 0)


   if baseline_attn is not None and pruned_attn is not None and baseline_len > 0 and pruned_len > 0:
       # Attention shape is (key_len,) representing attention from last query to all keys
       # We need the corresponding tokens for the keys
       # Note: This assumes capture_step > 0. Handling step 0 needs prompt tokens.
       # We only have the length, getting actual tokens requires more complex tracking.
       # For now, just plot the weights against index.

       max_len = max(baseline_len, pruned_len)
       # Pad shorter attention arrays
       if baseline_len < max_len:
           baseline_attn = np.pad(baseline_attn, (0, max_len - baseline_len))
       if pruned_len < max_len:
           pruned_attn = np.pad(pruned_attn, (0, max_len - pruned_len))


       fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
       fig.suptitle(f"Attention from Query Token at Step {capture_step} to Previous Key Tokens")

       axs[0].bar(range(max_len), baseline_attn)
       axs[0].set_title(f"Baseline (Key Length: {baseline_len})")
       axs[0].set_xlabel("Key Token Index (Previous Tokens)")
       axs[0].set_ylabel("Average Attention Weight")
       axs[0].grid(True, axis='y')

       axs[1].bar(range(max_len), pruned_attn)
       axs[1].set_title(f"Pruned Strategy (Key Length: {pruned_len})")
       axs[1].set_xlabel("Key Token Index (Previous Tokens)")
       axs[1].grid(True, axis='y')

       plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
       plt.show() # Display the plot
       # Consider saving the plot: plt.savefig(f"attention_step_{capture_step}.png")
   else:
       print("Attention weights not captured or lengths invalid for comparison.")

## --- Generation Function (from previous version, uses KVCacheManager) ---
#def generate_response(model, tokenizer, prompt, max_new_tokens=1024, strategy=None):
#    """Generate a response with KV cache pruning and metrics collection"""
#    if not prompt:
#         return "Error: Empty prompt received", {}
#
#    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#    input_token_count = inputs["input_ids"].shape[1]
#    is_cuda = model.device.type == 'cuda'
#
#    if is_cuda:
#        torch.cuda.synchronize()
#        torch.cuda.reset_peak_memory_stats(model.device)
#
#    if strategy is None:
#        strategy = NoOpStrategy()
#    cache_manager = KVCacheManager(strategy)
#
#    step_metrics = {
#        "kv_cache_sizes_bytes": [], "input_lengths": [],
#        "kept_lengths": [], "token_inference_times": []
#    }
#    generated_ids = inputs["input_ids"]
#    past_key_values = None
#    prompt_start_time = time.time()
#
#    try:
#        # --- Generation Loop ---
#        for step in range(max_new_tokens):
#            token_gen_start = time.time()
#            current_input_ids = generated_ids if past_key_values is None else generated_ids[:, -1:]
#
#            with torch.no_grad():
#                outputs = model(
#                    input_ids=current_input_ids,
#                    past_key_values=past_key_values,
#                    use_cache=True
#                )
#
#            if is_cuda: torch.cuda.synchronize()
#            token_time = time.time() - token_gen_start
#            step_metrics["token_inference_times"].append(token_time)
#
#            next_token_logits = outputs.logits[:, -1, :]
#            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
#            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
#
#            # Update KV cache using the manager
#            pruned_past_list, orig_len, kept_len = cache_manager.update(outputs.past_key_values)
#            past_key_values = pack_kv(pruned_past_list) # Pack for the next iteration
#
#            kv_size_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
#                               for k, v in pruned_past_list) if pruned_past_list else 0
#            step_metrics["kv_cache_sizes_bytes"].append(kv_size_bytes)
#            step_metrics["input_lengths"].append(orig_len)
#            step_metrics["kept_lengths"].append(kept_len)
#
#            if next_token_id.item() == tokenizer.eos_token_id: break
#        # --- End Generation Loop ---
#
#        if is_cuda: torch.cuda.synchronize()
#        prompt_total_time = time.time() - prompt_start_time
#        generated_text = tokenizer.decode(generated_ids[0, input_token_count:], skip_special_tokens=True)
#        prompt_peak_mem_bytes = torch.cuda.max_memory_allocated(model.device) if is_cuda else 0
#
#        avg_token_time = np.mean(step_metrics["token_inference_times"]) if step_metrics["token_inference_times"] else 0
#        avg_kv_size_mb = (np.mean(step_metrics["kv_cache_sizes_bytes"]) / (1024*1024)) if step_metrics["kv_cache_sizes_bytes"] else 0
#        avg_input_len = np.mean(step_metrics["input_lengths"]) if step_metrics["input_lengths"] else 0
#        avg_kept_len = np.mean(step_metrics["kept_lengths"]) if step_metrics["kept_lengths"] else 0
#        avg_kept_ratio = avg_kept_len / avg_input_len if avg_input_len > 0 else 0
#
#        final_metrics = {
#            "input_tokens": input_token_count,
#            "output_tokens": generated_ids.shape[1] - input_token_count,
#            "total_tokens_processed": generated_ids.shape[1],
#            "total_time_s": prompt_total_time,
#            "avg_token_time_s": avg_token_time,
#            "avg_kv_cache_size_mb": avg_kv_size_mb,
#            "avg_kept_ratio": avg_kept_ratio,
#            "peak_memory_mb": prompt_peak_mem_bytes / (1024 * 1024),
#        }
#
#    except Exception as e:
#        print(f"Generation error: {e}")
#        import traceback; traceback.print_exc()
#        return "Error during generation", {}
#
#    return generated_text, final_metrics
#
# --- Evaluation Function (Collects metrics per prompt) ---
def evaluate_mtbench(model, tokenizer, dataset, strategy_name="baseline", strategy=None, category=None,num_samples=None):
    """Evaluate a strategy on MT-Bench dataset, collecting detailed metrics per prompt."""
    results = { "strategy": strategy_name, "responses": [], "aggregated_metrics": {} }
    prompt_metrics_list = [] # Store metrics dict from each successful prompt run

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

    for i, question in enumerate(tqdm(questions, desc=f"Evaluating {strategy_name}")):
        prompt = dataset.format_prompt(question, turn_idx=0, model_name=model_name_for_prompt) # First turn only
        if not prompt: continue # Skip if formatting failed

        response_text, prompt_metrics = generate_response(model, tokenizer, prompt, max_new_tokens=1024, strategy=strategy)

        results["responses"].append({
            "question_id": question["question_id"],
            "category": question["category"],
            "prompt": prompt, # Store formatted prompt
            "response": response_text,
            "metrics": prompt_metrics,
        })

        if prompt_metrics: # If generation was successful
            prompt_metrics_list.append(prompt_metrics)

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
    else:
        print(f"Warning: No successful prompt evaluations for strategy {strategy_name}.")
        results["aggregated_metrics"] = {k: 0 for k in [
            "avg_total_time_per_prompt_s", "avg_peak_memory_per_prompt_mb", "avg_token_time_s",
            "avg_kv_cache_size_mb", "avg_kept_ratio", "overall_tokens_per_second"
        ]}

    return results

# --- LLM-as-a-Judge Functions (Adapted from paste.txt / MT-Bench) ---
def get_judge_prompts(question, answer, judge_instruction=""):
    """
    Generate scoring prompts for GPT judge based on FastChat MT-Bench implementation.
    Focuses on single-answer grading.
    """
    # Default instruction based on MT-Bench paper / FastChat repo [8, 10]
    default_judge_instruction = (
        "Please act as an impartial judge and evaluate the quality of the response "
        "provided by an AI assistant to the user query displayed below. Your evaluation should "
        "consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
        "and level of detail of the response. Begin your evaluation by providing a short "
        "explanation.\n"
        "You must rate the response on a scale of 1 to 10 by strictly following this format: "
        "\"[[rating]]\", for example: \"Rating: [[<rating>]]\".\n"
        "Do not use a score of 5. If the response is moderate, use 4 or 6.\n"
        "Your explanation should be concise and provide a clear justification for your rating."
    )

    instruction = judge_instruction if judge_instruction else default_judge_instruction
    prompt = f"[User Query]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]\n\n[System]\n{instruction}"
    return prompt

def submit_to_judge(results_list, api_key, judge_model="gpt-4-turbo", max_retries=3, retry_delay=5):
    """
    Submit results to an LLM judge (e.g., GPT-4) for scoring via OpenAI API.
    Adds 'judge_score' to each response and 'judge_score_avg' to aggregated metrics.
    """
    model_name = "gpt-4o"
    endpoint = "https://ai-wleonardtestp7127ai582819607599.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
    api_version = "2024-12-01-preview"
    deployment = "gpt-4o"
    location = 'East US'
    if OpenAI is None:
        print("OpenAI package not installed. Skipping LLM-as-a-judge scoring.")
        return results_list
    if not api_key:
        print("OpenAI API key not provided (--openai-key). Skipping LLM-as-a-judge scoring.")
        return results_list

    try:
        client = AzureOpenAI(api_version = api_version, azure_endpoint=endpoint,api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return results_list

    print(f"\nSubmitting responses to LLM judge ({judge_model}) for scoring...")

    for result in results_list:
        strategy_name = result["strategy"]
        judge_scores_for_strategy = []
        print(f"Judging strategy: {strategy_name}")

        for i, response_data in enumerate(tqdm(result.get("responses", []), desc=f"Judging {strategy_name}")):
            # Skip if already judged or if response was an error
            if "judge_score" in response_data or response_data.get("response", "").startswith("Error"):
                if "judge_score" in response_data:
                    judge_scores_for_strategy.append(response_data["judge_score"])
                continue

            # Question here should be the original user query, not necessarily the formatted prompt
            # Assuming the original first turn prompt is what we need for judging context
            # Let's retrieve the original question text if possible
            # This part might need adjustment based on how prompts were stored
            original_question_text = response_data.get("prompt", "") # Fallback to formatted prompt if needed
            # Ideally, find the question object from dataset using question_id if stored
            # question_obj = dataset.get_question_by_id(response_data["question_id"])
            # original_question_text = question_obj["turns"][0] if question_obj else response_data.get("prompt", "")

            answer = response_data.get("response", "")

            if not answer or len(answer.strip()) < 5: # Skip very short/empty answers
                print(f"Skipping short/empty response for question {response_data.get('question_id', i)}")
                continue

            judge_prompt = get_judge_prompts(original_question_text, answer)
            extracted_score = None

            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=judge_model,
                        messages=[{"role": "user", "content": judge_prompt}],
                        temperature=0.0, # Low temperature for consistent scoring
                        max_tokens=1024 # Allow sufficient length for explanation + score
                    )
                    judge_response_text = completion.choices[0].message.content

                    # Extract score using the specific format "[[rating]]" [9, 12]
                    match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', judge_response_text)
                    if match:
                        score_val = float(match.group(1))
                        # Clamp score to 1-10 range
                        extracted_score = max(1.0, min(10.0, score_val))
                        response_data["judge_explanation"] = judge_response_text # Store explanation too
                        response_data["judge_score"] = extracted_score
                        judge_scores_for_strategy.append(extracted_score)
                        # print(f"Extracted score: {extracted_score:.1f}") # Debug
                        break # Success, exit retry loop
                    else:
                        print(f"Warning: Could not extract score '[[rating]]' from judge response (attempt {attempt+1}/{max_retries}). Response:\n{judge_response_text}")

                except Exception as e:
                    print(f"Error calling OpenAI API (attempt {attempt+1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            # End retry loop

        # Calculate average judge score for the strategy
        if judge_scores_for_strategy:
            avg_judge_score = np.mean(judge_scores_for_strategy)
            if "aggregated_metrics" not in result: result["aggregated_metrics"] = {}
            result["aggregated_metrics"]["judge_score_avg"] = avg_judge_score
            print(f"Strategy '{strategy_name}' - Average Judge Score: {avg_judge_score:.2f}/10 ({len(judge_scores_for_strategy)} scored)")
        else:
             print(f"Warning: No valid judge scores obtained for strategy '{strategy_name}'.")

    return results_list

# --- Utility and Saving Functions ---
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
        with open(output_file, 'w') as f:
            json.dump(processed_results_list, f, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate KV cache pruning on MT-Bench with LLM-as-a-judge")
    # Model and Data
    # Allow specifying common Llama 2 model sizes easily
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model name or path (e.g., meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-13b-chat-hf, meta-llama/Llama-2-70b-chat-hf)")
    parser.add_argument("--mtbench_data_path", default="mt_bench_data.jsonl", help="Path to MT-Bench questions JSONL file") # Changed default
    # Evaluation Scope
    parser.add_argument("--category", help="Only test specific MT-Bench category (e.g., 'writing', 'reasoning')")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples (questions) to test per strategy")
    # Pruning Strategies
    parser.add_argument("--pruning_ratios", type=str, default="0.5,0.25",
                       help="Comma-separated list of pruning ratios (e.g., 0.5,0.25) for ratio-based strategies")
    parser.add_argument("--max_length_ratio", type=float, default=0.5,
                       help="Ratio for fixed-size strategies (Window, Random, etc.) - applied to estimated max length")
    parser.add_argument("--cache_dir", type=str, default=None, help="Where to store HF model files and tokenizer") # Added cache_dir argument

    # LLM Judge
    parser.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key for judge scoring (or set OPENAI_API_KEY env var)")
    parser.add_argument("--judge-model", default="gpt-4-turbo-preview", help="Which OpenAI model to use as the judge (e.g., gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip the LLM-as-a-judge scoring step")
    # Output
    parser.add_argument("--output_file", type=str, default="mtbench_results_judged.json",
                       help="Output file for detailed JSON results")
    # --- NEW: Visualization Arguments ---
    parser.add_argument("--visualize", action="store_true", help="Enable comparison visualization for one sample.")
    parser.add_argument("--vis_sample_index", type=int, default=0, help="Index of the sample to visualize.")
    parser.add_argument("--vis_step", type=int, default=10, help="Generation step (token index) to capture data for visualization.")
    parser.add_argument("--vis_top_k", type=int, default=5, help="Number of top token probabilities to show.")

    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    pruning_ratios_to_test = [float(r) for r in args.pruning_ratios.split(',')]
    print(f"Testing pruning ratios: {pruning_ratios_to_test}")

    try:
        dataset = MTBenchDataset(data_path=args.mtbench_data_path)
    except Exception as e:
        print(f"Failed to load MT-Bench dataset: {e}"); return

    # --- Load Model & Tokenizer ---
    print(f"Loading model: {args.model}")
    model_load_start = time.time()
    try:
        # --- NEW: Ensure model config allows output_attentions if visualizing ---
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
        if args.visualize:
            config.output_attentions = True
            print("Enabled output_attentions in model config for visualization.")

        tokenizer = LlamaTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # --- Quantization setup ---
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
            config=config, # Pass updated config
            cache_dir=args.cache_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )
        model.eval()
        print(f"Model loaded successfully in {time.time() - model_load_start:.2f}s.")
        if hasattr(model, 'hf_device_map'): print(f"Model device map: {model.hf_device_map}")

    except Exception as e:
        print(f"Error loading model {args.model}: {e}"); return

    # --- Handle Visualization Case ---
    if args.visualize:
        print("\n--- Running Visualization Mode ---")
        if args.vis_sample_index >= len(dataset.questions):
             print(f"Error: --vis_sample_index {args.vis_sample_index} is out of bounds for dataset size {len(dataset.questions)}")
             return

        question = dataset.questions[args.vis_sample_index]
        model_name_for_prompt = model.config._name_or_path if hasattr(model.config, "_name_or_path") else ""
        prompt = dataset.format_prompt(question, turn_idx=0, model_name=model_name_for_prompt)

        if not prompt:
             print(f"Error: Could not format prompt for sample index {args.vis_sample_index}")
             return

        # Find the target pruning strategy to compare against baseline
        target_strategy_name = None
        target_strategy_instance = None
        # Define strategies to choose from for visualization comparison
        estimated_max_total_len = 2048
        max_len_abs = max(1, int(estimated_max_total_len * args.max_length_ratio))
        available_strategies = {"baseline_full": NoOpStrategy()}
        available_strategies[f"window_{max_len_abs}t"] = WindowStrategy(max_length=max_len_abs)
        for ratio in pruning_ratios_to_test:
            ratio_pct = int(ratio * 100)
            available_strategies[f"top_{ratio_pct}pct"] = TopRatioStrategy(pruning_ratio=ratio)
            available_strategies[f"bottom_{ratio_pct}pct"] = BottomRatioStrategy(pruning_ratio=ratio)
            available_strategies[f"both_{ratio_pct}pct"] = BothRatioStrategy(pruning_ratio=ratio)

        # Example: use the first ratio-based 'both' strategy if available
        default_vis_target = f"both_{int(pruning_ratios_to_test[0]*100)}pct" if pruning_ratios_to_test else f"window_{max_len_abs}t"
        target_strategy_name = default_vis_target # Choose a default or let user specify later
        if target_strategy_name in available_strategies:
            target_strategy_instance = available_strategies[target_strategy_name]
            print(f"Visualizing comparison between Baseline and {target_strategy_name}")
        else:
            print(f"Target strategy '{target_strategy_name}' not found for visualization. Available: {list(available_strategies.keys())}. Exiting.")
            return

        print("\nGenerating Baseline...")
        baseline_response, baseline_metrics = generate_response(
            model, tokenizer, prompt,
            strategy=NoOpStrategy(),
            capture_step=args.vis_step,
            output_attentions=True,
            max_new_tokens=args.vis_step + 5
        )
        baseline_data_for_vis = {"metrics": baseline_metrics, "strategy_name": "Baseline", "response": baseline_response} # Add response for context

        # Clear cache before next run
        if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()

        print(f"\nGenerating {target_strategy_name}...")
        pruned_response, pruned_metrics = generate_response(
            model, tokenizer, prompt,
            strategy=target_strategy_instance,
            capture_step=args.vis_step,
            output_attentions=True,
            max_new_tokens=args.vis_step + 5
        )
        pruned_data_for_vis = {"metrics": pruned_metrics, "strategy_name": target_strategy_name, "response": pruned_response} # Add response for context

        # Call visualization
        visualize_comparison(
            baseline_data_for_vis,
            pruned_data_for_vis,
            tokenizer,
            args.vis_step,
            args.vis_top_k
        )
        print("\n--- Visualization Complete ---")
        return # Exit after visualization

    # --- Normal Evaluation Run (if not visualizing) ---
    # --- Define Strategies ---
    all_strategies_to_run = []
    all_strategies_to_run.append(("baseline_full", NoOpStrategy()))
    estimated_max_total_len = 2048
    max_len_abs = max(1, int(estimated_max_total_len * args.max_length_ratio))
    print(f"Using max_len={max_len_abs} for fixed-size strategies")
    all_strategies_to_run.append((f"window_{max_len_abs}t", WindowStrategy(max_length=max_len_abs)))
    for ratio in pruning_ratios_to_test:
        ratio_pct = int(ratio * 100)
        all_strategies_to_run.append((f"top_{ratio_pct}pct", TopRatioStrategy(pruning_ratio=ratio)))
        all_strategies_to_run.append((f"bottom_{ratio_pct}pct", BottomRatioStrategy(pruning_ratio=ratio)))
        all_strategies_to_run.append((f"both_{ratio_pct}pct", BothRatioStrategy(pruning_ratio=ratio)))

    all_results_list = []
    evaluation_start_time = time.time()

    for name, strat_instance in all_strategies_to_run:
        print(f"\n===== EVALUATING STRATEGY: {name} =====")
        strategy_start_time = time.time()
        if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()

        try:
            results = evaluate_mtbench(
                model, tokenizer, dataset,
                strategy_name=name,
                strategy=strat_instance,
                category=args.category,
                num_samples=args.num_samples
            )
            all_results_list.append(results)

            agg_metrics = results.get("aggregated_metrics", {})
            print(f"--- Strategy '{name}' Summary ---")
            print(f"  Avg Total Time/Prompt: {agg_metrics.get('avg_total_time_per_prompt_s', 0):.3f} s")
            print(f"  Avg Peak Mem/Prompt:   {agg_metrics.get('avg_peak_memory_per_prompt_mb', 0):.2f} MB")
            print(f"  Avg KV Size/Step:      {agg_metrics.get('avg_kv_cache_size_mb', 0):.2f} MB")
            print(f"  Avg Kept Ratio/Step:   {agg_metrics.get('avg_kept_ratio', 0):.1%}")
            print(f"  Overall Throughput:    {agg_metrics.get('overall_tokens_per_second', 0):.2f} tokens/s")
            print(f"  Strategy Eval Time:    {time.time() - strategy_start_time:.2f} s")

        except Exception as e:
            print(f"ERROR evaluating strategy {name}: {e}"); import traceback; traceback.print_exc()

    # --- LLM-as-a-Judge Scoring ---
    if not args.skip_judge:
        all_results_list = submit_to_judge(all_results_list, args.openai_key, args.judge_model, dataset=dataset) # Pass dataset here
    else:
        print("\nSkipping LLM-as-a-judge scoring step.")

    evaluation_end_time = time.time()
    print(f"\nTotal evaluation duration (excluding model load): {evaluation_end_time - evaluation_start_time:.2f} s")

    # --- Save Final Results ---
    save_results(all_results_list, args.output_file)

    # --- Print Final Summary Table ---
    print("\n===== FINAL SUMMARY (Aggregated Metrics) =====")
    print(f"{'Strategy':<20} | {'Avg Time/Prompt(s)':<20} | {'Avg Peak Mem(MB)':<18} | {'Avg KV Size(MB)':<16} | {'Avg Kept Ratio':<15} | {'Judge Score':<13} | {'Throughput(tok/s)':<20}")
    print("-" * 135)
    for result in all_results_list:
        strategy = result["strategy"]
        metrics = result.get("aggregated_metrics", {})
        judge_score_str = f"{metrics.get('judge_score_avg', 0):.2f}/10" if 'judge_score_avg' in metrics else "N/A"
        print(f"{strategy:<20} | "
              f"{metrics.get('avg_total_time_per_prompt_s', 0):<20.3f} | "
              f"{metrics.get('avg_peak_memory_per_prompt_mb', 0):<18.2f} | "
              f"{metrics.get('avg_kv_cache_size_mb', 0):<16.2f} | "
              f"{metrics.get('avg_kept_ratio', 0):<15.1%} | "
              f"{judge_score_str:<13} | "
              f"{metrics.get('overall_tokens_per_second', 0):<20.2f}")
    print("-" * 135)
    print(f"Settings: model={args.model}, num_samples={args.num_samples}, category={args.category or 'all'}, device={device}")

if __name__ == "__main__":
    main()

