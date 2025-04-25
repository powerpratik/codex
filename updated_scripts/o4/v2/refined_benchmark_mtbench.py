import argparse
import json
import time
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaCache
from kv_cache_manager import ThresholdEvictionStrategy, measure_cache_mb
from enhanced_strategies import (
    NoOpStrategy,
    WindowStrategy,
    RandomSamplingStrategy,
    StridedStrategy,
    BlockAverageStrategy,
    AttentionScoreStrategy,
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--gen_length", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--memory_threshold_mb", type=float, default=250.0)
    p.add_argument("--data_path", type=str, default='../data/mt_bench_data.jsonl')
    p.add_argument("--output_path", type=str, default="mtbench_results.json")
    p.add_argument("--category", type=str, default=None)
    
    return p.parse_args()


def run_one_strategy(strategy, model, tokenizer, input_ids, gen_length, device):
    past = None
    cache_trace = []
    time_trace = []

    for _ in range(gen_length):
        t0 = time.time()
        outputs = model(
            input_ids,
            past_key_values=past,
            output_attentions=isinstance(strategy, AttentionScoreStrategy),
        )
        t1 = time.time()

        raw_past = outputs.past_key_values

        # apply eviction
        if isinstance(strategy, AttentionScoreStrategy):
            past = strategy.evict(raw_past, attention_scores=outputs.attentions)
        else:
            past = strategy.evict(raw_past)

        # record
        cache_trace.append(measure_cache_mb(past))
        time_trace.append(t1 - t0)

        # greedy next token
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = next_token.to(device)

    return cache_trace, time_trace


def main():
    args = parse_args()
    #Replace HF loader with your local loader
    dataset = MTBenchDataset(data_path=args.data_path)
    # If you’ve specified a category filter:
    questions = dataset.get_questions_by_category(args.category) \
                if args.category else dataset.questions
    # Sample num_samples questions (shuffle already seeded)
    questions = questions[: args.num_samples] \
                if args.num_samples and args.num_samples > 0 else questions

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir='/home/cc/ceph/raptor/hf-hub/'

    # load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        # device_map="auto",
        cache_dir = cache_dir
    ).to(device)
    # Turn each question into a single‐turn prompt
    prompts = [
        dataset.format_prompt(q, turn_idx=0, model_name=model.config._name_or_path)
        for q in questions
    ]
    # prepare all strategies
    strategies = {
        "Baseline": NoOpStrategy(),
        f"Window({args.window_size})": WindowStrategy(args.window_size),
        f"Random({args.window_size})": RandomSamplingStrategy(args.window_size),
        f"Strided({args.window_size})": StridedStrategy(args.window_size),
        f"BlockAvg({args.window_size})": BlockAverageStrategy(args.window_size),
        f"AttentionTop({args.window_size})": AttentionScoreStrategy(args.window_size),
        "ThresholdEvict": ThresholdEvictionStrategy(args.memory_threshold_mb, model.config),
    }

    

    results = {}
    for name, strat in strategies.items():
        print(f"Running strategy → {name}")
        results[name] = {"cache_mb": [], "times_s": []}

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            c, t = run_one_strategy(strat, model, tokenizer, input_ids, args.gen_length, device)
            results[name]["cache_mb"].append(c)
            results[name]["times_s"].append(t)
    # write out results
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved all results to {args.output_path}")


if __name__ == "__main__":
    main()
