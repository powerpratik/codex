import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

class KVCacheDashboard:
    """
    Dashboard for visualizing KV cache management metrics.
    Generates comprehensive visualizations for all key metrics:
    - KV cache size
    - Inference time
    - Accuracy
    - Strategy comparisons
    """
    
    def __init__(self, results_dir, dashboard_dir=None):
        """
        Initialize the dashboard.
        
        Args:
            results_dir: Directory containing benchmark results
            dashboard_dir: Directory to save dashboard visualizations (default: results_dir/dashboard)
        """
        self.results_dir = Path(results_dir)
        self.dashboard_dir = Path(dashboard_dir) if dashboard_dir else self.results_dir / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up logger
        self.logger = logging.getLogger("kv_cache_dashboard")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Load results
        self.strategy_data = {}
        self.load_results()
        
        # Set plot style
        plt.style.use('ggplot')
        sns.set_palette("viridis")
    
    def load_results(self):
        """Load benchmark results from the results directory"""
        self.logger.info(f"Loading results from {self.results_dir}")
        
        # Load strategy results
        for strategy_file in self.results_dir.glob("*_results.json"):
            strategy_name = strategy_file.stem.replace("_results", "")
            try:
                with open(strategy_file, 'r') as f:
                    data = json.load(f)
                self.strategy_data[strategy_name] = data
                self.logger.info(f"Loaded results for strategy: {strategy_name}")
            except Exception as e:
                self.logger.error(f"Error loading results for {strategy_name}: {e}")
        
        if not self.strategy_data:
            self.logger.warning("No strategy results found!")
    
    def generate_dashboard(self):
        """Generate all dashboard visualizations"""
        self.logger.info("Generating dashboard visualizations")
        
        # Generate individual visualizations
        self.plot_kv_cache_sizes()
        self.plot_inference_times()
        self.plot_token_generation_times()
        self.plot_accuracy_metrics()
        self.plot_eviction_stats()
        self.plot_performance_tradeoffs()
        self.plot_memory_over_time()
        
        # Generate comparative report
        self.generate_comparative_report()
        
        self.logger.info(f"Dashboard generated at {self.dashboard_dir}")
    
    def plot_kv_cache_sizes(self):
        """Plot KV cache sizes for all strategies"""
        plt.figure(figsize=(12, 8))
        
        # Collect peak KV cache sizes
        strategies = []
        peak_sizes = []
        
        for strat_name, data in self.strategy_data.items():
            peak_size = 0
            for sample in data:
                if "memory" in sample and "peak_memory_mb" in sample["memory"]:
                    peak_size = max(peak_size, sample["memory"]["peak_memory_mb"])
            
            strategies.append(strat_name)
            peak_sizes.append(peak_size)
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(strategies, peak_sizes)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f} MB',
                    ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Strategy')
        plt.ylabel('Peak KV Cache Size (MB)')
        plt.title('Peak KV Cache Size by Strategy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.dashboard_dir / "kv_cache_sizes.png", dpi=300)
        plt.close()
    
    def plot_inference_times(self):
        """Plot inference times for all strategies"""
        plt.figure(figsize=(12, 8))
        
        # Collect inference times
        strategies = []
        total_times = []
        first_token_times = []
        avg_token_times = []
        
        for strat_name, data in self.strategy_data.items():
            total_time = 0
            first_token_time = 0
            token_times = []
            
            for sample in data:
                if "time" in sample:
                    if "total_time" in sample["time"]:
                        total_time += sample["time"]["total_time"]
                    if "first_token_time" in sample["time"]:
                        first_token_time += sample["time"]["first_token_time"]
                    if "token_times" in sample["time"]:
                        # Handle both list and dict formats
                        if isinstance(sample["time"]["token_times"], list):
                            token_times.extend(sample["time"]["token_times"])
                        elif isinstance(sample["time"]["token_times"], dict):
                            token_times.extend(list(sample["time"]["token_times"].values()))
            
            strategies.append(strat_name)
            total_times.append(total_time)
            first_token_times.append(first_token_time)
            avg_token_time = np.mean(token_times) if token_times else 0
            avg_token_times.append(avg_token_time)
        
        # Create grouped bar chart
        x = np.arange(len(strategies))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width, total_times, width, label='Total Time (s)')
        rects2 = ax.bar(x, first_token_times, width, label='First Token Time (s)')
        rects3 = ax.bar(x + width, [t * 1000 for t in avg_token_times], width, label='Avg Token Time (ms)')
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Time')
        ax.set_title('Inference Time Metrics by Strategy')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels
        def autolabel(rects, format_str='{:.2f}'):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(format_str.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        fig.tight_layout()
        
        # Save figure
        plt.savefig(self.dashboard_dir / "inference_times.png", dpi=300)
        plt.close()
    
    def plot_token_generation_times(self):
        """Plot token generation times for all strategies"""
        plt.figure(figsize=(12, 8))
        
        # Collect token times from per-strategy data
        strategy_token_times = {}
        
        for strat_name, data in self.strategy_data.items():
            token_times = []
            for sample in data:
                if "time" in sample and "token_times" in sample["time"]:
                    # Handle both list and dict formats for token_times
                    if isinstance(sample["time"]["token_times"], list):
                        # If it's already a list of numeric values, use it directly
                        for t in sample["time"]["token_times"]:
                            if isinstance(t, (int, float)):
                                token_times.append(t)
                    elif isinstance(sample["time"]["token_times"], dict):
                        # If it's a dict, extract numeric values
                        for _, time_value in sample["time"]["token_times"].items():
                            if isinstance(time_value, (int, float)):
                                token_times.append(time_value)
            
            if token_times:
                strategy_token_times[strat_name] = token_times
        
        # If we have token times, create a violin plot
        if strategy_token_times:
            plt.figure(figsize=(12, 8))
            
            # Prepare data for violin plot
            data_to_plot = []
            labels = []
            
            for strat_name, times in strategy_token_times.items():
                # Convert to ms and ensure all values are numeric
                times_ms = []
                for t in times:
                    if isinstance(t, (int, float)):
                        times_ms.append(t * 1000)
                    elif isinstance(t, dict) and 'time' in t and isinstance(t['time'], (int, float)):
                        times_ms.append(t['time'] * 1000)
                
                if times_ms:  # Only add if we have valid numeric values
                    data_to_plot.append(times_ms)
                    labels.append(strat_name)
            
            if data_to_plot:  # Only create plot if we have valid data
                # Create violin plot
                plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
                
                # Add boxplot inside violin
                plt.boxplot(data_to_plot, widths=0.2)
                
                # Set labels
                plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                plt.ylabel('Token Generation Time (ms)')
                plt.title('Distribution of Token Generation Times by Strategy')
                
                plt.tight_layout()
                
                # Save figure
                plt.savefig(self.dashboard_dir / "token_generation_times.png", dpi=300)
                plt.close()
            else:
                # Create a placeholder figure
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "No valid token generation time data available", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14)
                plt.axis('off')
                plt.savefig(self.dashboard_dir / "token_generation_times.png", dpi=300)
                plt.close()
        else:
            # Create a placeholder figure
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No token generation time data available", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14)
            plt.axis('off')
            plt.savefig(self.dashboard_dir / "token_generation_times.png", dpi=300)
            plt.close()
    
    def plot_accuracy_metrics(self):
        """Plot accuracy metrics for all strategies"""
        plt.figure(figsize=(12, 8))
        
        # Collect accuracy metrics
        strategies = []
        perplexities = []
        azure_scores = []
        
        for strat_name, data in self.strategy_data.items():
            perplexity = 0
            azure_score = 0
            count = 0
            
            for sample in data:
                if "accuracy" in sample:
                    if "perplexity" in sample["accuracy"]:
                        perplexity += sample["accuracy"]["perplexity"]
                        count += 1
                    if "azure_score" in sample["accuracy"]:
                        azure_score += sample["accuracy"]["azure_score"]
            
            strategies.append(strat_name)
            perplexities.append(perplexity / max(1, count))
            azure_scores.append(azure_score / max(1, count) if azure_score > 0 else 0)
        
        # Create bar chart for perplexity
        plt.figure(figsize=(12, 8))
        bars = plt.bar(strategies, perplexities)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Strategy')
        plt.ylabel('Perplexity (lower is better)')
        plt.title('Average Perplexity by Strategy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.dashboard_dir / "perplexity.png", dpi=300)
        plt.close()
        
        # Create bar chart for Azure scores if available
        if any(score > 0 for score in azure_scores):
            plt.figure(figsize=(12, 8))
            bars = plt.bar(strategies, azure_scores)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.xlabel('Strategy')
            plt.ylabel('Azure Evaluation Score (higher is better)')
            plt.title('Average Azure Evaluation Score by Strategy')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(self.dashboard_dir / "azure_scores.png", dpi=300)
            plt.close()
    
    def plot_eviction_stats(self):
        """Plot eviction statistics for all strategies"""
        plt.figure(figsize=(12, 8))
        
        # Collect eviction stats
        strategies = []
        eviction_counts = []
        avg_eviction_times = []
        
        for strat_name, data in self.strategy_data.items():
            eviction_count = 0
            total_eviction_time = 0
            
            for sample in data:
                if "memory" in sample:
                    if "eviction_count" in sample["memory"]:
                        eviction_count += sample["memory"]["eviction_count"]
                    if "total_eviction_time" in sample["memory"] and "eviction_count" in sample["memory"]:
                        if sample["memory"]["eviction_count"] > 0:
                            total_eviction_time += sample["memory"]["total_eviction_time"]
            
            strategies.append(strat_name)
            eviction_counts.append(eviction_count)
            avg_time = (total_eviction_time / eviction_count * 1000) if eviction_count > 0 else 0  # ms
            avg_eviction_times.append(avg_time)
        
        # Create grouped bar chart
        x = np.arange(len(strategies))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot eviction counts on left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Eviction Count', color=color)
        bars1 = ax1.bar(x - width/2, eviction_counts, width, color=color, label='Eviction Count')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot average eviction times on right y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Avg Eviction Time (ms)', color=color)
        bars2 = ax2.bar(x + width/2, avg_eviction_times, width, color=color, label='Avg Eviction Time (ms)')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add value labels
        def autolabel(rects, ax, format_str='{:.0f}'):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(format_str.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(bars1, ax1, '{:.0f}')
        autolabel(bars2, ax2, '{:.2f}')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Eviction Statistics by Strategy')
        plt.xticks(x, strategies, rotation=45, ha='right')
        fig.tight_layout()
        
        # Save figure
        plt.savefig(self.dashboard_dir / "eviction_stats.png", dpi=300)
        plt.close()
    
    def plot_performance_tradeoffs(self):
        """Plot performance tradeoffs between memory, speed, and accuracy"""
        plt.figure(figsize=(12, 10))
        
        # Collect metrics for tradeoff analysis
        strategies = []
        peak_sizes = []
        avg_token_times = []
        perplexities = []
        
        for strat_name, data in self.strategy_data.items():
            peak_size = 0
            token_times = []
            perplexity = 0
            count = 0
            
            for sample in data:
                if "memory" in sample and "peak_memory_mb" in sample["memory"]:
                    peak_size = max(peak_size, sample["memory"]["peak_memory_mb"])
                
                if "time" in sample and "token_times" in sample["time"]:
                    # Handle both list and dict formats
                    if isinstance(sample["time"]["token_times"], list):
                        token_times.extend([t for t in sample["time"]["token_times"] if isinstance(t, (int, float))])
                    elif isinstance(sample["time"]["token_times"], dict):
                        token_times.extend([t for t in sample["time"]["token_times"].values() if isinstance(t, (int, float))])
                
                if "accuracy" in sample and "perplexity" in sample["accuracy"]:
                    perplexity += sample["accuracy"]["perplexity"]
                    count += 1
            
            strategies.append(strat_name)
            peak_sizes.append(peak_size)
            avg_token_time = np.mean(token_times) * 1000 if token_times else 0  # ms
            avg_token_times.append(avg_token_time)
            perplexities.append(perplexity / max(1, count))
        
        # Create scatter plot with bubble size representing memory usage
        plt.figure(figsize=(12, 10))
        
        # Normalize bubble sizes for better visualization
        min_size = min(peak_sizes)
        max_size = max(peak_sizes)
        
        # Fix for ZeroDivisionError: Handle case where all peak sizes are the same
        if max_size == min_size:
            # All strategies have the same peak size, use a constant bubble size
            normalized_sizes = [500 for _ in peak_sizes]
        else:
            # Normal case: normalize between 100 and 1000
            normalized_sizes = [100 + 900 * (size - min_size) / (max_size - min_size) for size in peak_sizes]
        
        # Create scatter plot
        scatter = plt.scatter(avg_token_times, perplexities, s=normalized_sizes, alpha=0.6)
        
        # Add strategy labels
        for i, strat in enumerate(strategies):
            plt.annotate(strat, (avg_token_times[i], perplexities[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Average Token Generation Time (ms)')
        plt.ylabel('Perplexity (lower is better)')
        plt.title('Performance Tradeoffs: Speed vs. Accuracy vs. Memory Usage')
        
        # Add legend for bubble size
        if max_size > min_size:
            handles, labels = scatter.legend_elements(prop="sizes", num=3, alpha=0.6,
                                                    func=lambda s: min_size + (max_size - min_size) * (s - 100) / 900)
            legend = plt.legend(handles, labels, loc="upper right", title="Peak KV Cache (MB)")
        
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.dashboard_dir / "performance_tradeoffs.png", dpi=300)
        plt.close()
    
    def plot_memory_over_time(self):
        """Plot memory usage over time for all strategies"""
        plt.figure(figsize=(12, 8))
        
        # Collect memory over time data
        for strat_name, data in self.strategy_data.items():
            step_memory = []
            
            for sample in data:
                if "memory" in sample and "step_memory" in sample["memory"]:
                    step_memory.extend(sample["memory"]["step_memory"])
            
            if step_memory:
                plt.plot(range(len(step_memory)), step_memory, label=strat_name)
        
        plt.xlabel('Generation Step')
        plt.ylabel('KV Cache Size (MB)')
        plt.title('KV Cache Size Over Generation Steps')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.dashboard_dir / "memory_over_time.png", dpi=300)
        plt.close()
    
    def generate_comparative_report(self):
        """Generate a comparative report of all strategies"""
        # Collect summary metrics for all strategies
        summary_data = []
        
        for strat_name, data in self.strategy_data.items():
            # Initialize metrics
            tokens_per_second = 0
            avg_token_time = 0
            first_token_time = 0
            total_generation_time = 0
            total_tokens_generated = 0
            peak_kv_cache_mb = 0
            eviction_count = 0
            avg_eviction_time = 0
            perplexity = 0
            
            # Calculate metrics
            token_times = []
            perplexity_sum = 0
            perplexity_count = 0
            
            for sample in data:
                # Time metrics
                if "time" in sample:
                    if "total_time" in sample["time"]:
                        total_generation_time += sample["time"]["total_time"]
                    if "first_token_time" in sample["time"]:
                        first_token_time += sample["time"]["first_token_time"]
                    if "token_times" in sample["time"]:
                        # Handle both list and dict formats
                        if isinstance(sample["time"]["token_times"], list):
                            token_times.extend([t for t in sample["time"]["token_times"] if isinstance(t, (int, float))])
                        elif isinstance(sample["time"]["token_times"], dict):
                            token_times.extend([t for t in sample["time"]["token_times"].values() if isinstance(t, (int, float))])
                    if "tokens_generated" in sample["time"]:
                        total_tokens_generated += sample["time"]["tokens_generated"]
                
                # Memory metrics
                if "memory" in sample:
                    if "peak_memory_mb" in sample["memory"]:
                        peak_kv_cache_mb = max(peak_kv_cache_mb, sample["memory"]["peak_memory_mb"])
                    if "eviction_count" in sample["memory"]:
                        eviction_count += sample["memory"]["eviction_count"]
                    if "total_eviction_time" in sample["memory"] and "eviction_count" in sample["memory"]:
                        if sample["memory"]["eviction_count"] > 0:
                            avg_eviction_time += sample["memory"]["total_eviction_time"]
                
                # Accuracy metrics
                if "accuracy" in sample and "perplexity" in sample["accuracy"]:
                    perplexity_sum += sample["accuracy"]["perplexity"]
                    perplexity_count += 1
            
            # Calculate derived metrics
            if total_generation_time > 0 and total_tokens_generated > 0:
                tokens_per_second = total_tokens_generated / total_generation_time
            
            if token_times:
                avg_token_time = np.mean(token_times)
            
            if eviction_count > 0:
                avg_eviction_time /= eviction_count
            
            if perplexity_count > 0:
                perplexity = perplexity_sum / perplexity_count
            
            # Add to summary data
            summary_data.append({
                "name": strat_name,
                "tokens_per_second": tokens_per_second,
                "avg_token_time": avg_token_time,
                "first_token_time": first_token_time / len(data) if data else 0,
                "total_generation_time": total_generation_time,
                "total_tokens_generated": total_tokens_generated,
                "peak_kv_cache_mb": peak_kv_cache_mb,
                "eviction_count": eviction_count,
                "avg_eviction_time": avg_eviction_time,
                "perplexity": perplexity
            })
        
        # Create markdown report
        report_path = self.dashboard_dir / "comparative_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# KV Cache Management Strategy Comparison\n\n")
            
            f.write("## Strategy Performance Comparison\n\n")
            f.write("| Strategy | Tokens/sec | Avg token time (ms) | First token time (ms) | Peak KV Cache (MB) | Evictions | Avg eviction time (ms) |\n")
            f.write("|----------|------------|---------------------|----------------------|-------------------|-----------|------------------------|\n")
            
            for strat in summary_data:
                f.write(f"| {strat['name']} | {strat['tokens_per_second']:.2f} | {strat['avg_token_time']*1000:.2f} | {strat['first_token_time']*1000:.2f} | {strat['peak_kv_cache_mb']:.2f} | {strat['eviction_count']} | {strat['avg_eviction_time']*1000:.2f} |\n")
            
            f.write("\n## Memory Efficiency\n\n")
            f.write("Strategies ranked by memory efficiency (lowest peak KV cache size):\n\n")
            
            # Sort by peak KV cache size
            memory_ranked = sorted(summary_data, key=lambda x: x["peak_kv_cache_mb"])
            for i, strat in enumerate(memory_ranked):
                f.write(f"{i+1}. **{strat['name']}**: {strat['peak_kv_cache_mb']:.2f} MB\n")
            
            f.write("\n## Speed Performance\n\n")
            f.write("Strategies ranked by speed (highest tokens per second):\n\n")
            
            # Sort by tokens per second
            speed_ranked = sorted(summary_data, key=lambda x: x["tokens_per_second"], reverse=True)
            for i, strat in enumerate(speed_ranked):
                f.write(f"{i+1}. **{strat['name']}**: {strat['tokens_per_second']:.2f} tokens/sec\n")
            
            f.write("\n## Accuracy Performance\n\n")
            f.write("Strategies ranked by accuracy (lowest perplexity):\n\n")
            
            # Sort by perplexity
            accuracy_ranked = sorted(summary_data, key=lambda x: x["perplexity"])
            for i, strat in enumerate(accuracy_ranked):
                f.write(f"{i+1}. **{strat['name']}**: {strat['perplexity']:.2f} perplexity\n")
            
            f.write("\n## Overall Recommendations\n\n")
            
            # Calculate a simple combined score (lower is better)
            for strat in summary_data:
                # Normalize each metric to [0, 1] range
                max_memory = max(s["peak_kv_cache_mb"] for s in summary_data) if any(s["peak_kv_cache_mb"] > 0 for s in summary_data) else 1
                max_time = max(s["avg_token_time"] for s in summary_data) if any(s["avg_token_time"] > 0 for s in summary_data) else 1
                max_perplexity = max(s["perplexity"] for s in summary_data) if any(s["perplexity"] > 0 for s in summary_data) else 1
                
                # Avoid division by zero
                if max_memory == 0: max_memory = 1
                if max_time == 0: max_time = 1
                if max_perplexity == 0: max_perplexity = 1
                
                memory_score = strat["peak_kv_cache_mb"] / max_memory if max_memory > 0 else 0
                time_score = strat["avg_token_time"] / max_time if max_time > 0 else 0
                perplexity_score = strat["perplexity"] / max_perplexity if max_perplexity > 0 else 0
                
                # Combined score with equal weights
                strat["combined_score"] = (memory_score + time_score + perplexity_score) / 3
            
            # Sort by combined score
            overall_ranked = sorted(summary_data, key=lambda x: x["combined_score"])
            
            f.write("Based on a combined evaluation of memory efficiency, speed, and accuracy, the recommended strategies are:\n\n")
            
            for i, strat in enumerate(overall_ranked[:3]):
                f.write(f"{i+1}. **{strat['name']}**\n")
                f.write(f"   - Memory: {strat['peak_kv_cache_mb']:.2f} MB\n")
                f.write(f"   - Speed: {strat['tokens_per_second']:.2f} tokens/sec\n")
                f.write(f"   - Accuracy: {strat['perplexity']:.2f} perplexity\n\n")
            
            f.write("\n## Visualization Dashboard\n\n")
            f.write("For detailed visualizations, please refer to the following dashboard images:\n\n")
            f.write("1. [KV Cache Sizes](kv_cache_sizes.png)\n")
            f.write("2. [Inference Times](inference_times.png)\n")
            f.write("3. [Token Generation Times](token_generation_times.png)\n")
            f.write("4. [Perplexity](perplexity.png)\n")
            f.write("5. [Eviction Stats](eviction_stats.png)\n")
            f.write("6. [Performance Tradeoffs](performance_tradeoffs.png)\n")
            f.write("7. [Memory Over Time](memory_over_time.png)\n")
        
        self.logger.info(f"Comparative report generated at {report_path}")
        
        # Also save as JSON for programmatic access
        with open(self.dashboard_dir / "summary_data.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
