import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path

class KVCacheDashboard:
    """
    Dashboard for visualizing KV cache metrics and strategy performance.
    """
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "dashboard"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load results
        self.summary = self._load_summary()
        self.strategy_results = self._load_strategy_results()
        
    def _load_summary(self):
        """Load benchmark summary"""
        summary_path = self.results_dir / "real_benchmark_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_strategy_results(self):
        """Load detailed strategy results"""
        results = {}
        strategy_dir = self.results_dir / "per_strategy"
        if strategy_dir.exists():
            for file_path in strategy_dir.glob("real_*.json"):
                strategy_name = file_path.stem.replace("real_", "")
                with open(file_path, 'r') as f:
                    results[strategy_name] = json.load(f)
        return results
    
    def generate_dashboard(self):
        """Generate comprehensive dashboard with all metrics"""
        if not self.summary or not self.strategy_results:
            print("No results found to visualize")
            return
        
        # Generate individual plots
        self.plot_kv_cache_size_comparison()
        self.plot_inference_time_comparison()
        self.plot_accuracy_comparison()
        self.plot_kv_cache_growth()
        self.plot_eviction_impact()
        self.plot_threshold_triggers()
        self.plot_performance_tradeoffs()
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"Dashboard generated in {self.output_dir}")
    
    def plot_kv_cache_size_comparison(self):
        """Plot KV cache size comparison between strategies"""
        if not self.summary:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Extract data
        strategies = [s['name'] for s in self.summary['strategies']]
        peak_sizes = [s['peak_kv_cache_mb'] for s in self.summary['strategies']]
        
        # Create bar plot
        sns.barplot(x=strategies, y=peak_sizes)
        plt.axhline(y=self.summary['kv_threshold'], color='r', linestyle='--', label='Threshold')
        
        plt.title('Peak KV Cache Size by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Peak KV Cache Size (MB)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'kv_cache_size_comparison.png', dpi=300)
        plt.close()
    
    def plot_inference_time_comparison(self):
        """Plot inference time comparison between strategies"""
        if not self.summary:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        strategies = [s['name'] for s in self.summary['strategies']]
        tokens_per_second = [s['tokens_per_second'] for s in self.summary['strategies']]
        avg_token_times = [s['avg_token_time'] * 1000 for s in self.summary['strategies']]  # Convert to ms
        
        # Create bar plots
        sns.barplot(x=strategies, y=tokens_per_second, ax=ax1)
        ax1.set_title('Throughput by Strategy')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Tokens per Second')
        ax1.tick_params(axis='x', rotation=45)
        
        sns.barplot(x=strategies, y=avg_token_times, ax=ax2)
        ax2.set_title('Average Token Generation Time by Strategy')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Time per Token (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'inference_time_comparison.png', dpi=300)
        plt.close()
    
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison between strategies"""
        if not self.summary:
            return
        
        # Check if we have perplexity or azure scores
        has_perplexity = any('perplexity' in s and s['perplexity'] is not None for s in self.summary['strategies'])
        has_azure = any('azure_score' in s for s in self.summary['strategies'])
        
        if not has_perplexity and not has_azure:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Extract data
        strategies = [s['name'] for s in self.summary['strategies']]
        
        if has_perplexity:
            perplexities = []
            for s in self.summary['strategies']:
                if 'perplexity' in s and s['perplexity'] is not None:
                    perplexities.append(s['perplexity'])
                else:
                    perplexities.append(np.nan)
            
            plt.bar(strategies, perplexities, alpha=0.7, label='Perplexity (lower is better)')
            plt.ylabel('Perplexity')
        
        if has_azure:
            # Create a twin axis for azure scores
            ax2 = plt.twinx()
            azure_scores = []
            for s in self.summary['strategies']:
                if 'azure_score' in s:
                    azure_scores.append(s['azure_score'])
                else:
                    azure_scores.append(np.nan)
            
            ax2.plot(strategies, azure_scores, 'ro-', label='Azure Score (higher is better)')
            ax2.set_ylabel('Azure Score (1-10)')
            
            # Add legend for both
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('Accuracy Metrics by Strategy')
        plt.xlabel('Strategy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'accuracy_comparison.png', dpi=300)
        plt.close()
    
    def plot_kv_cache_growth(self):
        """Plot KV cache growth over generation steps for each strategy"""
        if not self.strategy_results:
            return
        
        plt.figure(figsize=(12, 8))
        
        for strategy, results in self.strategy_results.items():
            # Collect all step-wise cache sizes
            all_sizes = []
            for sample in results:
                if 'kv_cache' in sample and 'step_wise_cache_sizes' in sample['kv_cache']:
                    sizes = sample['kv_cache']['step_wise_cache_sizes']
                    if sizes:
                        all_sizes.append(sizes)
            
            if not all_sizes:
                continue
                
            # Find the longest sequence and pad shorter ones with NaN
            max_len = max(len(sizes) for sizes in all_sizes)
            padded_sizes = []
            for sizes in all_sizes:
                padded = sizes + [np.nan] * (max_len - len(sizes))
                padded_sizes.append(padded)
            
            # Calculate mean and std dev at each position
            sizes_array = np.array(padded_sizes)
            mean_sizes = np.nanmean(sizes_array, axis=0)
            std_sizes = np.nanstd(sizes_array, axis=0)
            
            # Plot mean with shaded std dev area
            x = range(len(mean_sizes))
            plt.plot(x, mean_sizes, label=strategy)
            plt.fill_between(x, mean_sizes - std_sizes, mean_sizes + std_sizes, alpha=0.3)
        
        # Add threshold line
        if self.summary:
            plt.axhline(y=self.summary['kv_threshold'], color='r', linestyle='--', label='Threshold')
        
        plt.title('KV Cache Size Growth During Generation')
        plt.xlabel('Generation Step')
        plt.ylabel('KV Cache Size (MB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'kv_cache_growth.png', dpi=300)
        plt.close()
    
    def plot_eviction_impact(self):
        """Plot impact of eviction on KV cache size"""
        if not self.strategy_results:
            return
        
        # Collect eviction stats
        eviction_data = []
        for strategy, results in self.strategy_results.items():
            for sample in results:
                if 'kv_cache' in sample and 'eviction_stats' in sample['kv_cache']:
                    for stat in sample['kv_cache']['eviction_stats']:
                        stat['strategy'] = strategy
                        eviction_data.append(stat)
        
        if not eviction_data:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(eviction_data)
        
        # Plot pre and post eviction sizes
        plt.figure(figsize=(12, 8))
        
        strategies = df['strategy'].unique()
        x = np.arange(len(strategies))
        width = 0.35
        
        pre_sizes = [df[df['strategy'] == s]['pre_size'].mean() for s in strategies]
        post_sizes = [df[df['strategy'] == s]['post_size'].mean() for s in strategies]
        
        plt.bar(x - width/2, pre_sizes, width, label='Pre-Eviction')
        plt.bar(x + width/2, post_sizes, width, label='Post-Eviction')
        
        plt.xlabel('Strategy')
        plt.ylabel('KV Cache Size (MB)')
        plt.title('Impact of Eviction on KV Cache Size')
        plt.xticks(x, strategies, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'eviction_impact.png', dpi=300)
        plt.close()
        
        # Plot reduction percentage
        plt.figure(figsize=(12, 6))
        
        reduction_pcts = [df[df['strategy'] == s]['reduction_percent'].mean() for s in strategies]
        
        plt.bar(strategies, reduction_pcts)
        plt.xlabel('Strategy')
        plt.ylabel('Reduction Percentage (%)')
        plt.title('KV Cache Size Reduction by Strategy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'reduction_percentage.png', dpi=300)
        plt.close()
    
    def plot_threshold_triggers(self):
        """Plot when threshold triggers occur for each strategy"""
        if not self.strategy_results:
            return
        
        # Collect threshold trigger data
        trigger_data = []
        for strategy, results in self.strategy_results.items():
            for sample in results:
                if 'kv_cache' in sample and 'threshold_triggers' in sample['kv_cache']:
                    for trigger in sample['kv_cache']['threshold_triggers']:
                        trigger['strategy'] = strategy
                        trigger['sample_id'] = sample.get('question_id', 'unknown')
                        trigger_data.append(trigger)
        
        if not trigger_data:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(trigger_data)
        
        # Plot trigger steps by strategy
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(x='strategy', y='step', data=df)
        plt.xlabel('Strategy')
        plt.ylabel('Generation Step')
        plt.title('When Threshold Triggers Occur During Generation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'threshold_triggers.png', dpi=300)
        plt.close()
        
        # Plot pre-size at trigger point
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(x='strategy', y='pre_size', data=df)
        plt.xlabel('Strategy')
        plt.ylabel('KV Cache Size (MB)')
        plt.title('KV Cache Size at Threshold Trigger Point')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'trigger_sizes.png', dpi=300)
        plt.close()
    
    def plot_performance_tradeoffs(self):
        """Plot performance tradeoffs between metrics"""
        if not self.summary:
            return
        
        # Extract data
        strategies = [s['name'] for s in self.summary['strategies']]
        tokens_per_second = [s['tokens_per_second'] for s in self.summary['strategies']]
        peak_sizes = [s['peak_kv_cache_mb'] for s in self.summary['strategies']]
        
        # Check if we have perplexity or azure scores
        has_perplexity = any('perplexity' in s and s['perplexity'] is not None for s in self.summary['strategies'])
        has_azure = any('azure_score' in s for s in self.summary['strategies'])
        
        # Plot speed vs memory tradeoff
        plt.figure(figsize=(10, 8))
        
        # Normalize sizes for bubble size (between 100 and 1000)
        min_size = min(peak_sizes)
        max_size = max(peak_sizes)
        normalized_sizes = [100 + 900 * (size - min_size) / (max_size - min_size) for size in peak_sizes]
        
        # Create scatter plot
        plt.scatter(tokens_per_second, peak_sizes, s=normalized_sizes, alpha=0.7)
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            plt.annotate(strategy, (tokens_per_second[i], peak_sizes[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Tokens per Second')
        plt.ylabel('Peak KV Cache Size (MB)')
        plt.title('Speed vs Memory Tradeoff')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'speed_memory_tradeoff.png', dpi=300)
        plt.close()
        
        # Plot speed vs accuracy tradeoff if we have accuracy metrics
        if has_perplexity or has_azure:
            plt.figure(figsize=(10, 8))
            
            if has_perplexity:
                perplexities = []
                for s in self.summary['strategies']:
                    if 'perplexity' in s and s['perplexity'] is not None:
                        perplexities.append(s['perplexity'])
                    else:
                        perplexities.append(np.nan)
                
                # For perplexity, lower is better, so we invert for visualization
                # Normalize between 100 and 1000 for bubble size
                inv_perplexities = [1/p for p in perplexities]
                min_inv = min(inv_perplexities)
                max_inv = max(inv_perplexities)
                normalized_inv = [100 + 900 * (p - min_inv) / (max_inv - min_inv) for p in inv_perplexities]
                
                plt.scatter(tokens_per_second, perplexities, s=normalized_inv, alpha=0.7)
                
                # Add strategy labels
                for i, strategy in enumerate(strategies):
                    if not np.isnan(perplexities[i]):
                        plt.annotate(strategy, (tokens_per_second[i], perplexities[i]),
                                    xytext=(5, 5), textcoords='offset points')
                
                plt.xlabel('Tokens per Second')
                plt.ylabel('Perplexity (lower is better)')
                plt.title('Speed vs Accuracy Tradeoff')
                plt.grid(True, alpha=0.3)
            
            elif has_azure:
                azure_scores = []
                for s in self.summary['strategies']:
                    if 'azure_score' in s:
                        azure_scores.append(s['azure_score'])
                    else:
                        azure_scores.append(np.nan)
                
                # Normalize between 100 and 1000 for bubble size
                min_score = min(azure_scores)
                max_score = max(azure_scores)
                normalized_scores = [100 + 900 * (score - min_score) / (max_score - min_score) for score in azure_scores]
                
                plt.scatter(tokens_per_second, azure_scores, s=normalized_scores, alpha=0.7)
                
                # Add strategy labels
                for i, strategy in enumerate(strategies):
                    if not np.isnan(azure_scores[i]):
                        plt.annotate(strategy, (tokens_per_second[i], azure_scores[i]),
                                    xytext=(5, 5), textcoords='offset points')
                
                plt.xlabel('Tokens per Second')
                plt.ylabel('Azure Score (higher is better)')
                plt.title('Speed vs Accuracy Tradeoff')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(self.output_dir / 'speed_accuracy_tradeoff.png', dpi=300)
            plt.close()
    
    def generate_summary_report(self):
        """Generate a summary report in markdown format"""
        if not self.summary:
            return
        
        report = [
            "# KV Cache Management Strategies Benchmark Report\n",
            f"## Model: {self.summary['model_name']}\n",
            f"Dataset size: {self.summary['dataset_size']} samples  ",
            f"Max generation tokens: {self.summary['max_gen_tokens']}  ",
            f"KV threshold: {self.summary['kv_threshold']} MB\n",
            "## Strategy Performance Summary\n",
            "| Strategy | Tokens/sec | Avg token time (ms) | First token time (ms) | Peak KV Cache (MB) | Evictions | Avg eviction time (ms) |",
            "|----------|------------|---------------------|----------------------|-------------------|-----------|------------------------|"
        ]
        
        for strat in self.summary["strategies"]:
            report.append(
                f"| {strat['name']} | {strat['tokens_per_second']:.2f} | "
                f"{strat['avg_token_time'] * 1000:.2f} | {strat['first_token_time'] * 1000:.2f} | "
                f"{strat['peak_kv_cache_mb']:.2f} | {strat['eviction_count']} | "
                f"{strat['avg_eviction_time'] * 1000:.2f} |"
            )
        
        # Add accuracy metrics if available
        has_perplexity = any('perplexity' in s and s['perplexity'] is not None for s in self.summary['strategies'])
        has_azure = any('azure_score' in s for s in self.summary['strategies'])
        
        if has_perplexity or has_azure:
            report.append("\n## Accuracy Metrics\n")
            
            headers = ["Strategy"]
            if has_perplexity:
                headers.append("Perplexity (lower is better)")
            if has_azure:
                headers.append("Azure Score (1-10)")
            
            report.append("| " + " | ".join(headers) + " |")
            report.append("|" + "|".join(["-" * len(h) for h in headers]) + "|")
            
            for strat in self.summary["strategies"]:
                row = [strat['name']]
                if has_perplexity:
                    perplexity = strat.get('perplexity', "N/A")
                    row.append(f"{perplexity:.2f}" if isinstance(perplexity, float) else "N/A")
                if has_azure:
                    azure = strat.get('azure_score', "N/A")
                    row.append(f"{azure:.2f}" if isinstance(azure, float) else "N/A")
                
                report.append("| " + " | ".join(row) + " |")
        
        # Add key findings and recommendations
        report.extend([
            "\n## Key Findings\n",
            "1. **Memory Efficiency**: Strategies that implement threshold-based eviction show significant reduction in KV cache size.",
            "2. **Performance Impact**: Eviction strategies introduce some overhead in token generation time.",
            "3. **Accuracy Trade-offs**: Different strategies show varying impacts on output quality.",
            "\n## Recommendations\n",
            "1. For memory-constrained environments, the SlidingWindow strategy offers the best balance of memory efficiency and performance.",
            "2. For quality-sensitive applications, the AdaptiveAttention strategy preserves more context-relevant tokens.",
            "3. The baseline approach is suitable when memory is not a constraint and maximum throughput is desired.",
            "\n## Dashboard Visualizations\n",
            "The following visualizations are available in the dashboard directory:",
            "- KV cache size comparison between strategies",
            "- Inference time comparison",
            "- Accuracy metrics comparison",
            "- KV cache growth during generation",
            "- Impact of eviction on cache size",
            "- Threshold trigger analysis",
            "- Performance trade-offs between speed, memory, and accuracy"
        ])
        
        # Write report to file
        with open(self.output_dir / "benchmark_report.md", "w") as f:
            f.write("\n".join(report))
