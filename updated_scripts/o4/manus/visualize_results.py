# visualize_results.py
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from utils import load_config

def load_strategy_results(config):
    """Load results for all strategies"""
    cfg = load_config(config)
    results_dir = Path(cfg['output_dir']) / 'per_strategy'
    
    all_results = {}
    for log_file in sorted(results_dir.glob('*.json')):
        strat = log_file.stem
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            all_results[strat] = data
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return all_results, cfg

def plot_kv_cache_progression(results, cfg, output_dir):
    """Plot how KV cache size progresses during generation for each strategy"""
    plt.figure(figsize=(12, 8))
    
    for strat, data in results.items():
        # Collect all size histories, handling different sequence lengths
        all_size_histories = []
        for sample in data:
            sizes = sample['sizes'].get('history', [])
            if sizes:
                all_size_histories.append(sizes)
        
        if not all_size_histories:
            continue
            
        # Find the longest sequence and fill shorter ones with NaN
        max_len = max(len(hist) for hist in all_size_histories)
        padded_histories = []
        for hist in all_size_histories:
            padded = hist + [np.nan] * (max_len - len(hist))
            padded_histories.append(padded)
        
        # Calculate mean and std dev at each position
        histories_array = np.array(padded_histories)
        mean_sizes = np.nanmean(histories_array, axis=0)
        std_sizes = np.nanstd(histories_array, axis=0)
        
        # Plot mean with shaded std dev area
        x = range(len(mean_sizes))
        plt.plot(x, mean_sizes, label=strat)
        plt.fill_between(x, mean_sizes - std_sizes, mean_sizes + std_sizes, alpha=0.3)
    
    plt.axhline(y=cfg["kv_threshold"], color='r', linestyle='--', label='Threshold')
    plt.xlabel('Generation Step')
    plt.ylabel('KV Cache Size (MB)')
    plt.title('KV Cache Size Progression During Generation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'kv_cache_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_generation_time_comparison(results, output_dir):
    """Plot generation time comparison between strategies"""
    avg_total_times = {}
    avg_per_token_times = {}
    
    for strat, data in results.items():
        total_times = [sample['times']['total'] for sample in data]
        per_token_times = [sample['times']['per_token'] for sample in data]
        
        avg_total_times[strat] = np.mean(total_times)
        avg_per_token_times[strat] = np.mean(per_token_times) * 1000  # Convert to ms
    
    # Plot total time
    plt.figure(figsize=(10, 6))
    plt.bar(avg_total_times.keys(), avg_total_times.values())
    plt.xlabel('Strategy')
    plt.ylabel('Average Total Generation Time (s)')
    plt.title('Average Total Generation Time by Strategy')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'total_generation_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-token time
    plt.figure(figsize=(10, 6))
    plt.bar(avg_per_token_times.keys(), avg_per_token_times.values())
    plt.xlabel('Strategy')
    plt.ylabel('Average Time per Token (ms)')
    plt.title('Average Generation Time per Token by Strategy')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_token_generation_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_comparison(results, output_dir):
    """Plot accuracy comparison if available"""
    has_accuracy = any('accuracy' in sample for data in results.values() for sample in data)
    
    if not has_accuracy:
        print("No accuracy data found. Skipping accuracy comparison.")
        return
    
    accuracies = {}
    for strat, data in results.items():
        scores = [sample.get('accuracy', 0) for sample in data if 'accuracy' in sample]
        if scores:
            accuracies[strat] = scores
    
    if not accuracies:
        return
        
    # Create boxplot of accuracy distributions
    plt.figure(figsize=(12, 8))
    
    data_to_plot = []
    labels = []
    for strat, scores in accuracies.items():
        data_to_plot.append(scores)
        labels.append(strat)
    
    # Fix: set ticks before setting tick labels
    plt.boxplot(data_to_plot)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.xlabel('Strategy')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Distribution by Strategy')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create barplot of average accuracy
    plt.figure(figsize=(10, 6))
    avg_accuracies = {strat: np.mean(scores) for strat, scores in accuracies.items()}
    std_accuracies = {strat: np.std(scores) for strat, scores in accuracies.items()}
    
    plt.bar(avg_accuracies.keys(), avg_accuracies.values(), yerr=list(std_accuracies.values()), capsize=5)
    plt.xlabel('Strategy')
    plt.ylabel('Average Accuracy Score')
    plt.title('Average Accuracy by Strategy')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'average_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

# Fix the matplotlib warnings in plot_comprehensive_dashboard
def plot_comprehensive_dashboard(results, cfg, output_dir):
    """Create a comprehensive analysis dashboard"""
    # Collect summary data
    summary = {}
    for strat, data in results.items():
        avg_time = np.mean([sample['times']['per_token'] for sample in data])
        avg_throughput = 1.0 / avg_time
        avg_peak_mem = np.mean([sample['sizes']['peak'] for sample in data])
        
        eviction_counts = [sample['cache_stats'].get('eviction_count', 0) for sample in data]
        avg_evictions = np.mean(eviction_counts) if eviction_counts else 0
        
        accuracy_data = [sample.get('accuracy', None) for sample in data if 'accuracy' in sample]
        avg_accuracy = np.mean(accuracy_data) if accuracy_data else None
        
        summary[strat] = {
            'throughput': avg_throughput,
            'latency': avg_time * 1000,  # ms
            'memory': avg_peak_mem,
            'evictions': avg_evictions,
            'accuracy': avg_accuracy
        }
    
    # Create a dashboard with all metrics
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    strats = list(summary.keys())
    x_pos = np.arange(len(strats))
    
    # Throughput plot
    ax1 = fig.add_subplot(gs[0, 0])
    throughputs = [summary[s]['throughput'] for s in strats]
    ax1.bar(x_pos, throughputs)
    ax1.set_title('Throughput (tokens/s)')
    # Fix: set ticks first, then labels
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(strats, rotation=45)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Latency plot
    ax2 = fig.add_subplot(gs[0, 1])
    latencies = [summary[s]['latency'] for s in strats]
    ax2.bar(x_pos, latencies)
    ax2.set_title('Average Latency (ms/token)')
    # Fix: set ticks first, then labels
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(strats, rotation=45)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Memory plot
    ax3 = fig.add_subplot(gs[1, 0])
    memories = [summary[s]['memory'] for s in strats]
    ax3.bar(x_pos, memories)
    ax3.set_title('Average Peak Memory (MB)')
    # Fix: set ticks first, then labels
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(strats, rotation=45)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Evictions plot
    ax4 = fig.add_subplot(gs[1, 1])
    evictions = [summary[s]['evictions'] for s in strats]
    ax4.bar(x_pos, evictions)
    ax4.set_title('Average Eviction Count')
    # Fix: set ticks first, then labels
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(strats, rotation=45)
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Accuracy plot (if available)
    accuracies = [summary[s]['accuracy'] for s in strats]
    if any(acc is not None for acc in accuracies):
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.bar(x_pos, accuracies)
        ax5.set_title('Average Accuracy Score')
        # Fix: set ticks first, then labels
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(strats, rotation=45)
        ax5.grid(True, axis='y', alpha=0.3)
    
    # Speed vs Memory tradeoff
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(throughputs, memories, s=100)
    for i, strat in enumerate(strats):
        ax6.annotate(strat, (throughputs[i], memories[i]),
                   xytext=(5, 5), textcoords='offset points')
    ax6.set_xlabel('Throughput (tokens/s)')
    ax6.set_ylabel('Memory (MB)')
    ax6.set_title('Speed vs Memory Tradeoff')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'KV Cache Strategy Performance Dashboard - {cfg["model_name"]}', size=16)
    # Fix: use subplots_adjust instead of tight_layout with rect parameter
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    plt.savefig(output_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_tradeoffs(results, output_dir):
    """Plot performance tradeoffs between speed and accuracy"""
    # Check if we have accuracy data
    has_accuracy = any('accuracy' in sample for data in results.values() for sample in data)
    
    if not has_accuracy:
        print("No accuracy data found. Skipping tradeoff analysis.")
        return
    
    # Collect metrics
    metrics = {}
    for strat, data in results.items():
        avg_speed = np.mean([sample['times']['per_token'] for sample in data])
        avg_accuracy = np.mean([sample.get('accuracy', 0) for sample in data if 'accuracy' in sample])
        peak_memory = np.mean([sample['sizes']['peak'] for sample in data])
        
        metrics[strat] = {
            'speed': 1.0 / avg_speed,  # Tokens per second
            'accuracy': avg_accuracy,
            'memory': peak_memory
        }
    
    # Plot speed vs. accuracy tradeoff
    plt.figure(figsize=(10, 8))
    
    speeds = [m['speed'] for m in metrics.values()]
    accuracies = [m['accuracy'] for m in metrics.values()]
    memories = [m['memory'] for m in metrics.values()]
    
    # Normalize memory for bubble size
    min_mem, max_mem = min(memories), max(memories)
    normalized_sizes = [300 * (m - min_mem) / (max_mem - min_mem + 1e-9) + 100 for m in memories]
    
    plt.scatter(speeds, accuracies, s=normalized_sizes, alpha=0.7)
    
    # Add strategy labels
    for i, strat in enumerate(metrics.keys()):
        plt.annotate(strat, (speeds[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Generation Speed (tokens/s)')
    plt.ylabel('Accuracy Score')
    plt.title('Speed vs. Accuracy Tradeoff (bubble size = memory usage)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_category_performance(results, output_dir):
    """Plot performance breakdowns by category"""
    # Collect category data
    categories = set()
    for data in results.values():
        for sample in data:
            cat = sample.get('category', 'uncategorized')
            categories.add(cat)
    
    categories = sorted(categories)
    if not categories or (len(categories) == 1 and 'uncategorized' in categories):
        print("No meaningful category data found. Skipping category analysis.")
        return
    
    # Collect per-category metrics
    category_metrics = defaultdict(lambda: defaultdict(list))
    for strat, data in results.items():
        for sample in data:
            cat = sample.get('category', 'uncategorized')
            category_metrics[cat][strat + '_time'].append(sample['times']['per_token'])
            category_metrics[cat][strat + '_size'].append(sample['sizes']['peak'])
            if 'accuracy' in sample:
                category_metrics[cat][strat + '_acc'].append(sample['accuracy'])
    
    # Plot time per category
    plt.figure(figsize=(14, 8))
    bar_width = 0.8 / len(results)
    
    for i, cat in enumerate(categories):
        for j, strat in enumerate(results.keys()):
            times = category_metrics[cat][strat + '_time']
            if times:
                pos = i + bar_width * (j - len(results)/2 + 0.5)
                plt.bar(pos, np.mean(times), width=bar_width, label=strat if i == 0 else "")
    
    plt.xlabel('Category')
    plt.ylabel('Average Time per Token (s)')
    plt.title('Generation Time by Category and Strategy')
    plt.xticks(range(len(categories)), categories, rotation=45)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'category_generation_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracy per category if available
    has_accuracy = any('_acc' in metric for metrics in category_metrics.values() for metric in metrics)
    if has_accuracy:
        plt.figure(figsize=(14, 8))
        
        for i, cat in enumerate(categories):
            for j, strat in enumerate(results.keys()):
                acc = category_metrics[cat][strat + '_acc']
                if acc:
                    pos = i + bar_width * (j - len(results)/2 + 0.5)
                    plt.bar(pos, np.mean(acc), width=bar_width, label=strat if i == 0 else "")
        
        plt.xlabel('Category')
        plt.ylabel('Average Accuracy Score')
        plt.title('Accuracy by Category and Strategy')
        plt.xticks(range(len(categories)), categories, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'category_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_latency_distribution(results, output_dir):
    """Plot distribution of token generation times"""
    plt.figure(figsize=(12, 8))
    
    data_to_plot = []
    labels = []
    
    for strat, data in results.items():
        # Collect all token times for this strategy
        all_times = []
        for sample in data:
            # If we have token breakdown, use it
            if 'breakdown' in sample['times']:
                # Skip the first time (prompt encoding)
                all_times.extend(sample['times']['breakdown'][1:])
            else:
                # Otherwise use the average
                all_times.append(sample['times']['per_token'])
        
        if all_times:
            data_to_plot.append(all_times)
            labels.append(strat)
    
    # Create violin plot
    sns.violinplot(data=data_to_plot, inner="quartile")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.xlabel('Strategy')
    plt.ylabel('Token Generation Time (s)')
    plt.title('Distribution of Token Generation Times')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'token_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create CDF plot
    plt.figure(figsize=(12, 8))
    
    for i, (strat, times) in enumerate(zip(labels, data_to_plot)):
        times = np.sort(times)
        # Calculate empirical CDF
        ys = np.arange(1, len(times) + 1) / len(times)
        plt.plot(times, ys, label=strat)
    
    plt.xscale('log')
    plt.xlabel('Token Generation Time (s)')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution of Token Generation Times')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'token_time_cdf.png', dpi=300, bbox_inches='tight')
    plt.close()



def create_validation_report(results, cfg, output_dir):
    """Create a markdown validation report with findings"""
    report = [
        "# KV Cache Strategy Validation Report\n",
        f"## Model: {cfg['model_name']}\n",
        "This report validates the effectiveness of different KV cache management strategies.\n"
    ]
    
    # Collect summary metrics
    metrics = {}
    for strat, data in results.items():
        avg_time = np.mean([sample['times']['per_token'] for sample in data])
        throughput = 1.0 / avg_time
        avg_peak_mem = np.mean([sample['sizes']['peak'] for sample in data])
        max_peak_mem = max([sample['sizes']['peak'] for sample in data])
        
        eviction_data = [sample['cache_stats'].get('eviction_count', 0) for sample in data]
        avg_evictions = np.mean(eviction_data) if eviction_data else 0
        
        accuracy_data = [sample.get('accuracy', None) for sample in data if 'accuracy' in sample]
        avg_accuracy = np.mean(accuracy_data) if accuracy_data and accuracy_data[0] is not None else None
        
        # Compare to baseline if this isn't the baseline
        baseline_strat = "Baseline" if "Baseline" in results else list(results.keys())[0]
        if strat != baseline_strat and baseline_strat in metrics:
            baseline = metrics[baseline_strat]
            time_diff = ((baseline['latency'] - avg_time * 1000) / baseline['latency']) * 100
            memory_diff = ((baseline['memory'] - avg_peak_mem) / baseline['memory']) * 100
            accuracy_diff = (((avg_accuracy or 0) - (baseline['accuracy'] or 0)) / (baseline['accuracy'] or 1)) * 100 if avg_accuracy is not None and baseline['accuracy'] is not None else None
        else:
            time_diff = None
            memory_diff = None
            accuracy_diff = None
        
        metrics[strat] = {
            'throughput': throughput,
            'latency': avg_time * 1000,  # ms
            'memory': avg_peak_mem,
            'max_memory': max_peak_mem,
            'evictions': avg_evictions,
            'accuracy': avg_accuracy,
            'time_diff': time_diff,
            'memory_diff': memory_diff,
            'accuracy_diff': accuracy_diff
        }
    
    # Summary table
    report.append("## Strategy Performance Summary\n")
    report.append("| Strategy | Throughput | Latency | Memory | Max Memory | Evictions | Accuracy |")
    report.append("|----------|------------|---------|--------|------------|-----------|----------|")
    
    for strat, m in metrics.items():
        accuracy_str = f"{m['accuracy']:.2f}" if m['accuracy'] is not None else "N/A"
        
        time_diff_str = ""
        if m['time_diff'] is not None:
            time_diff_str = f" ({'+' if m['time_diff'] < 0 else ''}{-m['time_diff']:.1f}%)"
            
        memory_diff_str = ""
        if m['memory_diff'] is not None:
            memory_diff_str = f" ({'+' if m['memory_diff'] < 0 else ''}{-m['memory_diff']:.1f}%)"
            
        accuracy_diff_str = ""
        if m['accuracy_diff'] is not None:
            accuracy_diff_str = f" ({'+' if m['accuracy_diff'] > 0 else ''}{m['accuracy_diff']:.1f}%)"
        
        report.append(
            f"| {strat} | {m['throughput']:.2f} tok/s | {m['latency']:.2f} ms{time_diff_str} | "
            f"{m['memory']:.2f} MB{memory_diff_str} | {m['max_memory']:.2f} MB | "
            f"{m['evictions']:.1f} | {accuracy_str}{accuracy_diff_str} |"
        )
    
    # Key findings
    report.append("\n## Key Findings\n")
    
    # Find best strategies for different metrics
    best_throughput = max(metrics.items(), key=lambda x: x[1]['throughput'])
    best_memory = min(metrics.items(), key=lambda x: x[1]['memory'])
    
    best_accuracy = None
    if any(m['accuracy'] is not None for m in metrics.values()):
        best_accuracy = max((s for s in metrics.items() if s[1]['accuracy'] is not None), 
                          key=lambda x: x[1]['accuracy'])
    
    # Report best strategies
    report.append(f"- **Fastest Strategy**: {best_throughput[0]} at {best_throughput[1]['throughput']:.2f} tokens/sec")
    report.append(f"- **Most Memory-Efficient**: {best_memory[0]} with average peak usage of {best_memory[1]['memory']:.2f} MB")
    
    if best_accuracy:
        report.append(f"- **Highest Accuracy**: {best_accuracy[0]} with score of {best_accuracy[1]['accuracy']:.2f}")
    
    # Check if any strategy effectively reduces memory
    baseline_memory = metrics.get("Baseline", {"memory": float('inf')})["memory"]
    memory_reduction = [(s, (baseline_memory - m["memory"]) / baseline_memory * 100) 
                       for s, m in metrics.items() if s != "Baseline"]
    
    significant_reductions = [(s, r) for s, r in memory_reduction if r > 10]  # More than 10% reduction
    
    if significant_reductions:
        report.append(f"\n### Memory Reduction Findings")
        for strat, reduction in sorted(significant_reductions, key=lambda x: -x[1]):
            report.append(f"- {strat} reduces memory usage by {reduction:.1f}% compared to baseline")
    
    # Check for performance impact
    baseline_latency = metrics.get("Baseline", {"latency": 0})["latency"]
    performance_impact = [(s, (m["latency"] - baseline_latency) / baseline_latency * 100) 
                         for s, m in metrics.items() if s != "Baseline"]
    
    acceptable_impact = [(s, i) for s, i in performance_impact if abs(i) < 10]  # Less than 10% slowdown
    problematic_impact = [(s, i) for s, i in performance_impact if i >= 10]     # 10% or more slowdown
    
    if acceptable_impact:
        report.append(f"\n### Performance Impact Findings")
        for strat, impact in sorted(acceptable_impact, key=lambda x: x[1]):
            if impact < 0:
                report.append(f"- {strat} is actually {-impact:.1f}% faster than baseline")
            else:
                report.append(f"- {strat} has minimal performance impact ({impact:.1f}% slower)")
    
    if problematic_impact:
        report.append(f"\n### Performance Concerns")
        for strat, impact in sorted(problematic_impact, key=lambda x: x[1]):
            report.append(f"- {strat} shows significant performance degradation ({impact:.1f}% slower)")
    
    # Accuracy impact if available
    if any(m['accuracy'] is not None for m in metrics.values()):
        baseline_acc = metrics.get("Baseline", {"accuracy": 0})["accuracy"]
        if baseline_acc is not None:
            accuracy_impact = [(s, ((m["accuracy"] or 0) - baseline_acc) / baseline_acc * 100) 
                             for s, m in metrics.items() if s != "Baseline" and m["accuracy"] is not None]
            
            report.append(f"\n### Accuracy Impact Findings")
            
            negative_acc = [(s, i) for s, i in accuracy_impact if i < -1]  # More than 1% reduction
            neutral_acc = [(s, i) for s, i in accuracy_impact if abs(i) <= 1]  # Within 1%
            positive_acc = [(s, i) for s, i in accuracy_impact if i > 1]  # More than 1% improvement
            
            if positive_acc:
                for strat, impact in sorted(positive_acc, key=lambda x: -x[1]):
                    report.append(f"- {strat} improves accuracy by {impact:.1f}%")
            
            if neutral_acc:
                report.append(f"- {', '.join(s for s, _ in neutral_acc)} maintain(s) similar accuracy to baseline (within 1%)")
            
            if negative_acc:
                for strat, impact in sorted(negative_acc, key=lambda x: x[1]):
                    report.append(f"- {strat} reduces accuracy by {-impact:.1f}%")
    
    # Overall recommendations
    report.append("\n## Recommendations\n")
    
    # Find the best balanced strategy (good memory reduction with minimal performance impact)
    balanced_candidates = []
    for strat, m in metrics.items():
        if strat == "Baseline":
            continue
        
        memory_reduction = (baseline_memory - m["memory"]) / baseline_memory * 100
        latency_impact = (m["latency"] - baseline_latency) / baseline_latency * 100
        
        # Calculate a simple score: memory_reduction - 2*abs(latency_impact)
        # This prioritizes memory reduction but penalizes performance impact
        score = memory_reduction - 2 * abs(latency_impact)
        balanced_candidates.append((strat, score, memory_reduction, latency_impact))
    
    if balanced_candidates:
        best_balanced = max(balanced_candidates, key=lambda x: x[1])
        report.append(f"- **Best Overall Strategy**: {best_balanced[0]} with {best_balanced[2]:.1f}% memory reduction and {best_balanced[3]:.1f}% latency impact")
    
    report.append("\n### Strategy-Specific Recommendations")
    
    # Make recommendations for each strategy
    for strat, m in metrics.items():
        if strat == "Baseline":
            continue
        
        memory_reduction = (baseline_memory - m["memory"]) / baseline_memory * 100
        latency_impact = (m["latency"] - baseline_latency) / baseline_latency * 100
        
        recommendation = f"- **{strat}**: "
        
        if memory_reduction > 20 and latency_impact < 5:
            recommendation += "Excellent choice for production. Significant memory savings with minimal performance impact."
        elif memory_reduction > 20 and latency_impact < 15:
            recommendation += "Good for production where memory constraints are the primary concern."
        elif memory_reduction > 10 and latency_impact < 0:
            recommendation += "Ideal choice - reduces memory while improving performance."
        elif memory_reduction > 0 and latency_impact < 0:
            recommendation += "Better than baseline in all respects - use this instead."
        elif memory_reduction > 0 and latency_impact < 10:
            recommendation += "Acceptable tradeoff for memory-constrained environments."
        else:
            recommendation += "Not recommended for production use due to unfavorable tradeoffs."
        
        report.append(recommendation)
    
    # Write report to file
    with open(output_dir / "validation_report.md", "w") as f:
        f.write("\n".join(report))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_template.json")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    
    # Load results
    results, cfg = load_strategy_results(args.config)
    
    if not results:
        print("No results found. Run the benchmark first.")
        return
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg['output_dir']) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations in {output_dir}")
    
    # Generate all plots
    plot_kv_cache_progression(results, cfg, output_dir)
    plot_generation_time_comparison(results, output_dir)
    plot_accuracy_comparison(results, output_dir)
    plot_performance_tradeoffs(results, output_dir)
    plot_category_performance(results, output_dir)
    plot_latency_distribution(results, output_dir)
    plot_comprehensive_dashboard(results, cfg, output_dir)
    
    # Create validation report
    create_validation_report(results, cfg, output_dir)
    
    print(f"Visualizations complete. See results in {output_dir}")
    print(f"Detailed validation report available at {output_dir/'validation_report.md'}")

if __name__ == "__main__":
    main()