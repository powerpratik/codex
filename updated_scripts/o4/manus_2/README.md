# KV Cache Management Experiment

This package contains all the necessary files to evaluate different KV cache management strategies using real metrics. The implementation properly applies eviction strategies to the actual KV cache and measures their impact with real physical metrics.

## Contents

- `complete_real_kv_cache_manager.py`: KV cache manager with real physical measurements
- `fixed_kv_cache_dashboard_final.py`: Dashboard for visualizing KV cache metrics
- `robust_real_benchmark.py`: Benchmark script for evaluating strategies
- `test_real_metrics.py`: Test script for validating the implementation
- `config.json`: Configuration file for the benchmark

## Key Features

1. **Real KV Cache Size Measurement**:
   - Directly measures the actual KV cache size using PyTorch's memory tracking
   - Applies eviction strategies to the real KV cache structure
   - Tracks detailed metrics including step-wise and layer-wise cache sizes

2. **Real Inference Time Measurement**:
   - Tracks encoding time, generation time, and per-token generation time
   - Measures first token latency
   - Records eviction overhead time

3. **Real Accuracy Measurement**:
   - Calculates perplexity as a proxy for quality
   - Integrates with Azure evaluation when available
   - Tracks detailed per-sample metrics

4. **Comprehensive Strategies**:
   - Baseline (no pruning)
   - Random KV cache management
   - Attention-based top/bottom token selection
   - Hybrid n-percent token selection
   - Sliding window and adaptive attention strategies

5. **Visualization Dashboard**:
   - Generates comprehensive visualizations for all metrics
   - Provides comparative analysis of strategies
   - Helps identify optimal strategies for different use cases

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Matplotlib, Seaborn, NumPy, Pandas

## Quick Start

1. **Update the configuration file**:
   Edit `config.json` with your model path and dataset location:
   ```json
   {
     "model_name": "meta-llama/Llama-2-7b-chat-hf",
     "cache_dir": "/path/to/your/model/cache",
     "output_dir": "benchmark_results",
     "max_gen_tokens": 100,
     "kv_threshold": 20,
     "strategies": [
       "Baseline",
       "Random(keep=0.7)",
       "AttentionTop(keep=0.7)",
       "AttentionBottom(keep=0.7)",
       "HybridNPercent(keep=0.7,r=0.5,a=0.3,t=0.2)",
       "SlidingWindow(window=0.7,important=0.1)",
       "AdaptiveAttention(base_keep=0.7)"
     ],
     "dataset": {
       "local_path": "/path/to/your/mtbench/dataset.json",
       "prompt_field": "turns"
     }
   }
   ```

2. **Run the benchmark**:
   ```bash
   python robust_real_benchmark.py --config config.json --generate_dashboard
   ```

3. **Optional flags**:
   - `--eval_azure`: Enable Azure evaluation (requires Azure config)
   - `--debug`: Enable debug logging
   - `--profile`: Enable memory profiling
   - `--limit N`: Limit to N samples from dataset

4. **Test the implementation**:
   ```bash
   python test_real_metrics.py --cache_dir /path/to/your/model/cache
   ```

## Understanding the Results

After running the benchmark, you'll find:

1. **Results files**: JSON files with detailed metrics for each strategy
2. **Dashboard directory**: Visualizations of all metrics
3. **Comparative report**: Markdown file comparing all strategies

The dashboard includes:
- KV cache sizes
- Inference times
- Token generation times
- Accuracy metrics
- Eviction statistics
- Performance tradeoffs
- Memory usage over time

## Tuning for Your Environment

For optimal results:

1. **Adjust the KV threshold** based on your GPU memory and model size
2. **Tune strategy parameters** to find the best balance for your use case
3. **Limit the dataset size** for initial testing with the `--limit` flag

## Troubleshooting

If you encounter issues:

1. **Check the logs**: Detailed logs are saved in `benchmark.log`
2. **Enable debug mode**: Use the `--debug` flag for more detailed logging
3. **Test individual components**: Run `test_real_metrics.py` to validate each component
4. **Memory issues**: Lower the `kv_threshold` in the config file
5. **Tokenization errors**: Ensure your dataset format matches the expected format

## Strategy Details

1. **Baseline**: No eviction, keeps all tokens in the KV cache
2. **Random**: Randomly selects tokens to keep when the cache exceeds the threshold
3. **AttentionTop**: Keeps tokens with the highest attention scores
4. **AttentionBottom**: Keeps tokens with the lowest attention scores
5. **HybridNPercent**: Combines recency, attention scores, and token type importance
6. **SlidingWindow**: Keeps recent tokens and a small number of important tokens
7. **AdaptiveAttention**: Adjusts the keep ratio based on the attention distribution

## Extending the Framework

To add your own strategies:

1. Create a new strategy class in the benchmark script
2. Implement the strategy in the KV cache manager
3. Add the strategy to your config file
4. Run the benchmark to evaluate your strategy

## Citation

If you use this code in your research, please cite:

```
@software{kv_cache_management,
  author = {Your Name},
  title = {KV Cache Management Strategies for LLMs},
  year = {2025},
  url = {https://github.com/yourusername/kv_cache_management}
}
```
