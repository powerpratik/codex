# KV Cache Management Experiment

This repository contains code for evaluating different KV cache management strategies with real metrics for Llama-2-7b-chat-hf model.

## Setup

1. Install dependencies:
```bash
pip install torch transformers numpy matplotlib seaborn pandas tqdm
```

2. Update the configuration file:
```bash
# Edit config.json with your model path and dataset location
```

## Running the Experiment

To run the benchmark with all strategies:
```bash
python real_benchmark_with_new_strategies.py --config config.json --generate_dashboard
```

Optional flags:
- `--eval_azure`: Enable Azure evaluation
- `--debug`: Enable debug logging
- `--profile`: Enable memory profiling
- `--limit N`: Limit to N samples from dataset

## Strategies Implemented

1. **Baseline**: No pruning, keeps the entire KV cache
2. **Random**: Randomly selects tokens to keep
3. **AttentionTop**: Keeps tokens with highest attention scores
4. **AttentionBottom**: Keeps tokens with lowest attention scores
5. **HybridNPercent**: Uses weighted combination of recency, attention, and token type
6. **SlidingWindow**: Keeps most recent tokens and some important historical tokens
7. **AdaptiveAttention**: Adapts keep ratio based on layer position

## Metrics Collected

1. **KV Cache Size**:
   - Average cache size
   - Total cache size
   - Step-wise cache sizes for each prompt

2. **Inference Time**:
   - Total inference time
   - Time per token
   - Time to first token generation

3. **Accuracy**:
   - Perplexity
   - MTBench LLM-as-a-judge metric (when Azure is enabled)

## Results

Results are saved to the `benchmark_results` directory (or as specified in config):
- Detailed metrics for each strategy
- Comparative report
- Visualization dashboard

## Files

- `real_benchmark_with_new_strategies.py`: Main benchmark script
- `real_kv_cache_manager_threshold.py`: KV cache manager with threshold-based eviction
- `real_inference_time_tracker.py`: Tracks detailed inference time metrics
- `real_accuracy_evaluator.py`: Evaluates response accuracy
- `additional_strategies.py`: Implements the requested strategies
- `kv_cache_dashboard.py`: Generates visualization dashboard
- `config.json`: Configuration template
