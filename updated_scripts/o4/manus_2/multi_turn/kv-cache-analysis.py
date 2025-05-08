import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

def analyze_kv_cache_strategies(data_dir):
    """Analyze KV cache strategies from multiple JSON files in a directory."""
    # ... (initial part of the function remains the same) ...
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    
    # Initialize data structures
    strategy_metrics = defaultdict(list)
    strategy_metrics_by_turn = defaultdict(lambda: defaultdict(list))
    strategies = []
    
    # Track which metrics are populated and their values
    metrics_found = defaultdict(lambda: defaultdict(list))
    
    # Process each JSON file
    for json_file in json_files:
        try:
            print(f"Processing file: {os.path.basename(json_file)}")
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"  Found {len(data)} entries in file")
            
            # Process each entry in the JSON array
            for entry_idx_file, entry_data in enumerate(data): # Renamed to avoid conflict
                # Extract strategy name from the prompt or filename
                strategy = extract_strategy_name(entry_data.get('prompt', ''), os.path.basename(json_file))
                if strategy not in strategies and strategy != "Unknown":
                    strategies.append(strategy)
                
                # Collect metrics and track available metrics/paths
                metrics = extract_metrics(entry_data) # Use entry_data
                
                if metrics and strategy != "Unknown":
                    # Track available metrics and their values
                    for key, value in metrics.items():
                        if value is not None:
                            metrics_found[strategy][key].append(value)
                    
                    # Add to overall metrics
                    strategy_metrics[strategy].append(metrics)
                    
                    # Add to per-turn metrics
                    turn_index = entry_data.get('turn_index', 0) # Use entry_data
                    strategy_metrics_by_turn[strategy][turn_index].append(metrics)
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"Identified strategies: {strategies}")
    
    # Print metrics found for debugging
    print("\nMetrics available by strategy:")
    for strategy_name_debug, metric_values_debug in metrics_found.items(): # Renamed to avoid conflict
        print(f"\n{strategy_name_debug}:")
        for metric_name, values in metric_values_debug.items():
            if values:
                if isinstance(values[0], dict) and len(values[0]) > 10:
                    print(f"  {metric_name}: {len(values)} entries (complex object)")
                else:
                    print(f"  {metric_name}: {len(values)} entries, example: {values[0]}")
    
    # IMPORTANT: Use kv_cache_sizes for memory metrics
    memory_metric_key = "kv_cache_sizes_peak"
    
    # ----- START MODIFIED SECTION for kv_cache_sizes_peak and kv_cache_sizes_avg calculation -----
    print("\nCalculating peak/average memory from kv_cache_sizes for each strategy entry:")
    for strategy, metrics_list_for_strategy in strategy_metrics.items():
        print(f"  Processing strategy: {strategy}")
        for i, m_entry in enumerate(metrics_list_for_strategy): # m_entry is a dict for one raw entry
            # Ensure keys exist before trying to use them
            m_entry[memory_metric_key] = 0.0
            m_entry['kv_cache_sizes_avg'] = 0.0

            if 'kv_cache_sizes' in m_entry and m_entry['kv_cache_sizes']:
                print(f"    Entry {i} for {strategy}: Found {len(m_entry['kv_cache_sizes'])} items in kv_cache_sizes.")
                
                current_kv_cache_data = m_entry['kv_cache_sizes']
                current_keys = list(current_kv_cache_data.keys())
                current_values = list(current_kv_cache_data.values())

                # Calculate Peak Memory
                try:
                    int_keys = [int(t) for t in current_keys if isinstance(t, str) and t.isdigit()]
                    if not int_keys:
                        print(f"      Warning (Peak): No valid integer keys found for entry {i}, strategy {strategy}.")
                    else:
                        max_token_key_int = max(int_keys)
                        max_token_key_str = str(max_token_key_int)
                        
                        if max_token_key_str in current_kv_cache_data:
                            peak_val = float(current_kv_cache_data[max_token_key_str])
                            m_entry[memory_metric_key] = peak_val
                            print(f"      Peak memory for entry {i}: {peak_val:.2f}")
                        else:
                            print(f"      Warning (Peak): Max token key '{max_token_key_str}' not found in original kv_cache_sizes keys for entry {i}, strategy {strategy}.")
                except Exception as e:
                    print(f"      ERROR calculating peak memory for entry {i}, strategy {strategy}: {e}")

                # Calculate Average Memory
                try:
                    float_values = []
                    for v_idx, v_val in enumerate(current_values):
                        try:
                            float_values.append(float(v_val))
                        except (ValueError, TypeError):
                            # This case might be too verbose if many non-float values exist
                            # print(f"      Warning (Avg): Could not convert value '{v_val}' at index {v_idx} to float for entry {i}, strategy {strategy}. Skipping.")
                            pass # Silently skip non-convertible values for average calculation
                    
                    if float_values:
                        avg_val = np.mean(float_values)
                        m_entry['kv_cache_sizes_avg'] = avg_val
                        print(f"      Avg memory for entry {i}: {avg_val:.2f} (from {len(float_values)} values)")
                    else:
                        print(f"      Warning (Avg): No valid float values found for kv_cache_sizes_avg for entry {i}, strategy {strategy}.")
                except Exception as e:
                    print(f"      ERROR calculating average memory for entry {i}, strategy {strategy}: {e}")
            else:
                print(f"    Entry {i} for {strategy}: No/empty 'kv_cache_sizes' found.")
    # ----- END MODIFIED SECTION -----

    # Ensure "Baseline" is the reference point (if it exists)
    if "Baseline" in strategies:
        strategies.remove("Baseline")
        strategies.append("Baseline") # Move to the end for processing order if needed
    
    # Compute average metrics for each strategy
    avg_metrics = {}
    print("\nAggregating average metrics across all entries for each strategy:") # Added print
    for strategy, metrics_list in strategy_metrics.items():
        if not metrics_list:
            print(f"Warning: No metrics found for strategy {strategy} during aggregation")
            continue
            
        # Correctly get all values for the keys from the list of metric dicts
        avg_metrics[strategy] = {
            "total_time": np.mean([m.get("total_time_seconds", 0) for m in metrics_list if m.get("total_time_seconds") is not None]),
            "first_token": np.mean([m.get("first_token_time", 0) for m in metrics_list if m.get("first_token_time") is not None]),
            "memory_usage": np.mean([m.get(memory_metric_key, 0) for m in metrics_list if m.get(memory_metric_key) is not None]), # This is peak
            "memory_avg": np.mean([m.get("kv_cache_sizes_avg", 0) for m in metrics_list if m.get("kv_cache_sizes_avg") is not None]), # This is avg of token averages
            "quality_score": np.mean([m.get("quality_score", 0) for m in metrics_list if m.get("quality_score") is not None]),
            "tokens_per_second": np.mean([m.get("tokens_per_second", 0) for m in metrics_list if m.get("tokens_per_second") is not None]),
            "eviction_count": np.mean([m.get("eviction_count", 0) for m in metrics_list if m.get("eviction_count") is not None]),
            "eviction_time": np.mean([m.get("avg_eviction_time", 0) for m in metrics_list if m.get("avg_eviction_time") is not None])
        }
        
        print(f"  Strategy: {strategy}, Aggregated Avg metrics: { {k: f'{v:.2f}' for k, v in avg_metrics[strategy].items()} }") # Formatted print
    
    # Compute average metrics for each strategy by turn
    avg_metrics_by_turn = {}
    for strategy, turns_data in strategy_metrics_by_turn.items():
        avg_metrics_by_turn[strategy] = {}
        for turn, metrics_list in turns_data.items():
            if not metrics_list:
                continue
                
            avg_metrics_by_turn[strategy][turn] = {
                "total_time": np.mean([m.get("total_time_seconds", 0) for m in metrics_list]),
                "first_token": np.mean([m.get("first_token_time", 0) for m in metrics_list]),
                "memory_usage": np.mean([m.get(memory_metric_key, 0) for m in metrics_list]),
                "memory_avg": np.mean([m.get("kv_cache_sizes_avg", 0) for m in metrics_list]),
                "quality_score": np.mean([m.get("quality_score", 0) for m in metrics_list]),
                "tokens_per_second": np.mean([m.get("tokens_per_second", 0) for m in metrics_list])
            }
    
    # Compute percentage changes relative to baseline
    baseline = avg_metrics.get("Baseline")
    if baseline:
        # Create a list of metrics to iterate over - fixes the dictionary changed size error
        metric_keys = ["total_time", "first_token", "memory_usage", "memory_avg", 
                      "quality_score", "tokens_per_second", "eviction_count", "eviction_time"]
        
        for strategy, metrics in avg_metrics.items():
            if strategy != "Baseline":
                for key in metric_keys:
                    if key in metrics and key in baseline and baseline[key] != 0:
                        metrics[f"{key}_pct"] = ((metrics[key] - baseline[key]) / baseline[key]) * 100
                    else:
                        metrics[f"{key}_pct"] = 0
                        print(f"Warning: Baseline value for {key} is zero or missing, percentage change set to 0")
        
        # Also compute percentage changes for per-turn metrics
        baseline_by_turn = avg_metrics_by_turn.get("Baseline", {})
        turn_metric_keys = ["total_time", "first_token", "memory_usage", "memory_avg", 
                           "quality_score", "tokens_per_second"]
        
        for strategy, turns_data in avg_metrics_by_turn.items():
            if strategy != "Baseline":
                for turn, metrics in turns_data.items():
                    baseline_turn = baseline_by_turn.get(turn)
                    if baseline_turn:
                        for key in turn_metric_keys:
                            if key in metrics and key in baseline_turn and baseline_turn[key] != 0:
                                metrics[f"{key}_pct"] = ((metrics[key] - baseline_turn[key]) / baseline_turn[key]) * 100
                            else:
                                metrics[f"{key}_pct"] = 0
    else:
        print("Warning: No baseline strategy found. Cannot compute percentage changes.")
    
    return strategies, avg_metrics, avg_metrics_by_turn, strategy_metrics, memory_metric_key

def extract_strategy_name(prompt, filename):
    """Extract the KV cache strategy name from the prompt or filename."""
    # Try to extract from prompt first
    strategy_keywords = {
        "adaptive": "AdaptiveAttention",
        "adaptiveattention": "AdaptiveAttention",
        "attention_bottom": "AttentionBottom",
        "attention-bottom": "AttentionBottom",
        "attentionbottom": "AttentionBottom",
        "attention_top": "AttentionTop",
        "attention-top": "AttentionTop",
        "attentiontop": "AttentionTop", 
        "hybrid": "HybridNPercent",
        "npercent": "HybridNPercent",
        "hybridnpercent": "HybridNPercent",
        "random": "Random",
        "sliding": "SlidingWindow",
        "slidingwindow": "SlidingWindow",
        "baseline": "Baseline",
        "base": "Baseline"
    }
    
    # Check prompt
    prompt = prompt.lower()
    for keyword, strategy in strategy_keywords.items():
        if keyword in prompt:
            return strategy
    
    # Check filename
    filename = filename.lower()
    for keyword, strategy in strategy_keywords.items():
        if keyword in filename:
            return strategy
    
    # If unable to determine, check for categories in the filename
    for strategy in strategy_keywords.values():
        if strategy.lower() in filename:
            return strategy
    
    print(f"  Warning: Could not determine strategy from: '{filename}'")
    return "Unknown"

def extract_metrics(entry):
    """Extract key metrics from a JSON entry."""
    metrics = {}
    
    try:
        # Total generation time and throughput
        metrics["total_time_seconds"] = entry.get("total_time_seconds", 0)
        metrics["tokens_per_second"] = entry.get("tokens_per_second", 0)
        metrics["tokens_generated"] = entry.get("tokens_generated", 0)
        
        # First token time
        token_times = entry.get("token_times", {})
        metrics["first_token_time"] = entry.get("first_token_time",0)#float(token_times.get("1", 0)) if token_times and "1" in token_times else 0
        metrics["token_times"] = token_times  # Store all token times for detailed analysis
        
        # KV Cache sizes - this is now our primary memory metric
        metrics["kv_cache_sizes"] = entry.get("kv_cache_sizes", {})
        
        # Extract eviction statistics from memory data if available
        memory = entry.get("memory", {})
        if memory and "kv_cache" in memory and "manager_stats" in memory["kv_cache"]:
            manager_stats = memory["kv_cache"]["manager_stats"]
            metrics["eviction_count"] = manager_stats.get("eviction_count", 0)
            metrics["avg_eviction_time"] = manager_stats.get("avg_eviction_time", 0)
            
            # Store the step memory if available for detailed analysis
            if "step_memory" in manager_stats:
                metrics["step_memory"] = manager_stats["step_memory"]
            
            # Store layer memory if available for detailed analysis
            if "layer_memory" in manager_stats:
                metrics["layer_memory"] = manager_stats["layer_memory"]
        
        # Quality metrics
        accuracy = entry.get("accuracy", {})
        metrics["quality_score"] = accuracy.get("azure_score", 0)
        metrics["perplexity"] = accuracy.get("perplexity", 0)
        
        # Store turn information
        metrics["turn_index"] = entry.get("turn_index", 0)
        metrics["category"] = entry.get("category", "")
        
        return metrics
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return None

def plot_comparison_chart(strategies, metrics, memory_metric_key, output_file=None):
    """Create a bar chart comparing strategies."""
    # Skip if no data
    if not strategies or not metrics:
        print("No data to plot")
        return
    
    # Filter out baseline for display
    display_strategies = [s for s in strategies if s != "Baseline"]
    
    if not display_strategies:
        print("No strategies to display (excluding baseline)")
        return
    
    # Prepare data for plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up the x-axis
    x = np.arange(len(display_strategies))
    width = 0.2  # Width of bars
    
    # Colors for each metric
    colors = {
        "total_time_pct": "#8884d8",  # Purple for total time
        "first_token_pct": "#82ca9d",  # Green for first token
        "memory_usage_pct": "#ffc658",  # Yellow for memory
        "quality_score_pct": "#ff7300"  # Orange for quality
    }
    
    # Labels for legend
    labels = {
        "total_time_pct": "Total Generation Time",
        "first_token_pct": "First Token Latency",
        "memory_usage_pct": "KV Cache Memory",
        "quality_score_pct": "Quality Score"
    }
    
    # Plot bars for each metric
    bars = []
    metric_keys = list(colors.keys())  # Create fixed list to avoid iteration issues
    
    # Print the actual memory values for debugging
    print("\nMemory metric values for bar chart:")
    for strategy in display_strategies:
        if strategy in metrics:
            mem_val = metrics[strategy].get("memory_usage_pct", "N/A")
            print(f"  {strategy}: {mem_val}%")
    
    for i, metric in enumerate(metric_keys):
        values = []
        for strategy in display_strategies:
            if strategy in metrics and metric in metrics[strategy]:
                values.append(metrics[strategy][metric])
            else:
                # If missing, assume 0% change from baseline
                print(f"Warning: No {metric} data for {strategy}, using 0%")
                values.append(0)
        
        bar = ax.bar(x + (i - 1.5) * width, values, width, label=labels[metric], color=colors[metric])
        bars.append(bar)
        
        # Add value labels above/below the bars
        for j, v in enumerate(values):
            # Position text above or below bar depending on value
            va = 'bottom' if v >= 0 else 'top'
            offset = 3 if v >= 0 else -3
            
            ax.text(x[j] + (i - 1.5) * width, v + offset, f"{v:.1f}%", 
                   ha='center', va=va, fontsize=9, fontweight='bold')
    
    # Set chart properties
    title = f'Performance Metrics Comparison (70% KV Cache Retention)'
    title += f'\nUsing {memory_metric_key} for memory metric (derived from kv_cache_sizes)'
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('% Change from Baseline', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_strategies, rotation=0)
    ax.legend()
    
    # Add a note about interpretation
    plt.figtext(0.5, 0.01, 
               "Negative values indicate improvements (faster, less memory, etc.)\n"
               "Positive values indicate degradation (slower, etc.)",
               ha="center", fontsize=10)
    
    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to match reference image
    ax.set_ylim(-60, 60)  # Adjusted to show higher positive values
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Chart saved to {output_file}")
    else:
        plt.show()

def plot_memory_timeline(strategies, raw_metrics):
    """Plot KV cache memory usage over token generation using kv_cache_sizes data."""
    # Collect all KV cache size data
    kv_cache_by_strategy = defaultdict(dict)

    print("\nExtracting data for memory timeline plot:")

    # First pass - collect max token position for each strategy
    max_pos_by_strategy = {}
    for strategy, entries in raw_metrics.items():
        current_max_for_strategy = 0 # Initialize for the strategy
        for entry in entries:
            kv_sizes = entry.get('kv_cache_sizes', {})
            if kv_sizes:
                # Take the highest token position with valid data
                try:
                    # Filter out potentially non-integer keys before finding max
                    token_positions = [int(pos) for pos in kv_sizes.keys() if pos.isdigit()]
                    if token_positions:
                        curr_max = max(token_positions)
                        # Update the strategy's max only if this entry's max is higher
                        if curr_max > current_max_for_strategy:
                            current_max_for_strategy = curr_max
                except (ValueError, TypeError) as e:
                    print(f"  Warning: Error processing keys in kv_cache_sizes for {strategy}: {e}") # Added warning
                    continue
        # Store the overall max position found for this strategy
        if current_max_for_strategy > 0:
             max_pos_by_strategy[strategy] = current_max_for_strategy

        if strategy in max_pos_by_strategy:
            print(f"  {strategy}: Max token position found: {max_pos_by_strategy[strategy]}")
        else:
            print(f"  {strategy}: No valid max token position found.") # Added info

    # Second pass - collect and average data at each position
    print("\n  Processing entries for averaging:") # Added header
    for strategy, entries in raw_metrics.items(): # Use raw_metrics directly here
        if strategy not in max_pos_by_strategy:
            print(f"  Skipping averaging for {strategy} (no max_pos found).") # Added debug
            continue

        # Initialize all positions up to max with empty lists
        positions_data = defaultdict(list)
        max_pos = max_pos_by_strategy[strategy]

        # --- Added Debugging Start ---
        print(f"  Processing {strategy} - Found {len(entries)} entries.")
        if entries:
             sample_kv_cache = entries[0].get('kv_cache_sizes', {})
             print(f"    Sample kv_cache_sizes keys (first entry): {list(sample_kv_cache.keys())[:10]}...") # Print sample keys
             print(f"    Sample kv_cache_sizes values (first entry): {list(sample_kv_cache.values())[:10]}...") # Print sample values
        # --- Added Debugging End ---

        # Collect values for each position
        entry_count_with_kv = 0 # Track entries with kv_cache_sizes
        valid_kv_data_points = 0 # Track valid data points found
        conversion_errors = 0 # Track errors during conversion
        skipped_non_positive = 0 # Track skipped points

        for entry_idx, entry in enumerate(entries):
            kv_sizes = entry.get('kv_cache_sizes', {})
            if not kv_sizes:
                continue # Skip entries with no kv_cache_sizes dictionary

            entry_count_with_kv += 1
            for pos, value in kv_sizes.items():
                try:
                    token_pos = int(pos)
                    mem_value = float(value)
                    if mem_value > 0:  # Only add valid positive values
                        positions_data[token_pos].append(mem_value)
                        valid_kv_data_points += 1
                    else:
                        skipped_non_positive += 1 # Count skipped points
                except (ValueError, TypeError) as e:
                     # print(f"    WARN converting kv_cache_sizes for {strategy}, pos '{pos}', value '{value}': {e}") # Optional: print every error
                     conversion_errors += 1 # Count errors
                     continue # Skip this data point

        print(f"  {strategy}: Found kv_cache_sizes in {entry_count_with_kv}/{len(entries)} entries.") # Added summary
        if conversion_errors > 0:
             print(f"  {strategy}: Encountered {conversion_errors} errors converting pos/value.") # Added error summary
        if skipped_non_positive > 0:
             print(f"  {strategy}: Skipped {skipped_non_positive} non-positive values.") # Added skipped summary
        print(f"  {strategy}: Extracted {valid_kv_data_points} valid data points total.") # Added summary

        # Compute average for each position with data
        kv_cache_by_strategy[strategy] = {} # Ensure it's clean for this strategy
        for pos, values in positions_data.items():
            if values:
                kv_cache_by_strategy[strategy][pos] = np.mean(values)

        print(f"  {strategy}: Calculated averages for {len(kv_cache_by_strategy[strategy])} token positions.") # Changed message slightly

    # Plot the data
    print("\nPlotting data:") # Added header
    plt.figure(figsize=(12, 6))

    # Use consistent colors for strategies
    colors = {
        "HybridNPercent": "#1f77b4",
        "Baseline": "#ff7f0e",
        "Random": "#2ca02c",
        "AttentionTop": "#d62728",
        "AttentionBottom": "#9467bd",
        "SlidingWindow": "#8c564b",
        "AdaptiveAttention": "#e377c2"
    }

    plotted_strategies = [] # Keep track of what is actually plotted
    # Plot each strategy - FORCE plot all strategies even with minimal data
    for strategy in strategies:
        print(f"Attempting to plot: {strategy}") # Added print
        if strategy in kv_cache_by_strategy:
            positions_data = kv_cache_by_strategy[strategy]
            if positions_data:  # Check if there is averaged data for this strategy
                positions = sorted(positions_data.keys())
                values = [positions_data[pos] for pos in positions]

                plt.plot(positions, values,
                       label=strategy,
                       color=colors.get(strategy, None),
                       marker='.' if len(positions) < 50 else None,  # Only use markers for sparse data
                       markersize=2,
                       linewidth=1.5)

                print(f"  -> Plotted {strategy} with {len(positions)} data points") # Confirmation
                plotted_strategies.append(strategy)
            else:
                 print(f"  -> No averaged positions_data found for {strategy} after processing.") # Added reason for not plotting
        else:
             print(f"  -> Strategy {strategy} not found in kv_cache_by_strategy (likely no valid data points extracted).") # Added reason for not plotting

    # --- Rest of the plotting function ---
    plt.xlabel('Token Position')
    plt.ylabel('KV Cache Memory (MB)')
    plt.title('KV Cache Memory Usage Over Generation')
    if not plotted_strategies: # Check if anything was plotted
         plt.text(0.5, 0.5, "No data available for plotting", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
         print("Warning: No strategies had data to plot.")
    elif len(plotted_strategies) < len(strategies):
         print(f"Warning: Plotted data for {len(plotted_strategies)} out of {len(strategies)} identified strategies.")
         plt.legend() # Show legend only if something was plotted
    else:
         plt.legend() # Show legend if all strategies plotted

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("kv_cache_memory_timeline.png", dpi=300)
    print("Memory timeline saved to kv_cache_memory_timeline.png")
    
    

def plot_tokens_per_position(strategies, raw_metrics):
    """Plot token generation time by position to analyze the effect of KV cache strategies."""
    # Collect token time data
    token_times_by_strategy = defaultdict(lambda: defaultdict(list))
    
    print("\nExtracting data for token generation time plot:")
    
    # Count strategies with data
    strategies_with_data = set()
    
    # Extract from raw metrics data
    for strategy, entries in raw_metrics.items():
        entries_with_data = 0
        
        # Collect token times for each token position across all entries
        for entry in entries:
            token_times = entry.get('token_times', {})
            if token_times:
                entries_with_data += 1
                for pos, value in token_times.items():
                    try:
                        token_pos = int(pos)
                        time_value = float(value)
                        if time_value > 0:  # Only add valid values
                            token_times_by_strategy[strategy][token_pos].append(time_value)
                    except (ValueError, TypeError):
                        continue
        
        # Report how many entries contained data
        if entries_with_data > 0:
            print(f"  {strategy}: Found {entries_with_data} entries with token time data")
            positions_with_data = len(token_times_by_strategy[strategy])
            if positions_with_data > 0:
                strategies_with_data.add(strategy)
                print(f"  {strategy}: Successfully extracted data for {positions_with_data} token positions")
            else:
                print(f"  WARNING: No valid token positions found for {strategy}")
        else:
            print(f"  WARNING: No entries with token time data for {strategy}")
    
    # Compute average time at each position for each strategy
    avg_time_by_pos = defaultdict(dict)
    for strategy, positions_data in token_times_by_strategy.items():
        for pos, values in positions_data.items():
            if values:
                avg_time_by_pos[strategy][pos] = np.mean(values)
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    
    # Track if we actually plotted anything
    plotted_any = False
    
    for strategy, positions_data in avg_time_by_pos.items():
        if positions_data:
            positions = sorted(positions_data.keys())
            values = [positions_data[pos] for pos in positions]
            
            # Only plot if we have a reasonable number of points
            if len(positions) > 5:  # At least 5 data points to make a reasonable line
                plt.plot(positions, values, marker='.', markersize=2, label=strategy)
                plotted_any = True
                print(f"Plotting {strategy} token times with {len(positions)} data points")
    
    # If we didn't plot anything, add a message
    if not plotted_any:
        plt.text(0.5, 0.5, "No valid token time data available", 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14)
    
    plt.xlabel('Token Position')
    plt.ylabel('Token Generation Time (seconds)')
    plt.title('Token Generation Time by Position')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("token_generation_time.png", dpi=300)
    print("Token generation time plot saved to token_generation_time.png")

def plot_tradeoff_analysis(strategies, metrics):
    """Create a scatter plot showing memory vs quality tradeoffs with actual data."""
    plt.figure(figsize=(10, 8))
    
    # Setup colors for consistency
    colors = {
        "HybridNPercent": "#1f77b4",
        "Random": "#ff7f0e",
        "AttentionTop": "#2ca02c",
        "AttentionBottom": "#d62728",
        "SlidingWindow": "#9467bd",
        "AdaptiveAttention": "#8c564b"
    }
    
    for strategy in strategies:
        if strategy != "Baseline" and strategy in metrics:
            m = metrics[strategy]
            memory_pct = m.get("memory_usage_pct", 0)
            quality_pct = m.get("quality_score_pct", 0)
            time_pct = m.get("total_time_pct", 0)
            
            # Size proportional to time impact (larger = more slowdown)
            size = 100 + abs(time_pct) * 5
            
            plt.scatter(memory_pct, quality_pct, s=size, 
                      label=strategy, 
                      color=colors.get(strategy, None),
                      alpha=0.7)
            plt.annotate(strategy, (memory_pct, quality_pct), 
                       fontsize=9, ha='center', va='center')
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Memory Usage (% change from baseline)')
    plt.ylabel('Quality Score (% change from baseline)')
    plt.title('Tradeoff Analysis: Memory vs Quality')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("kv_cache_tradeoff_analysis.png", dpi=300)
    print("Tradeoff analysis saved to kv_cache_tradeoff_analysis.png")

def plot_comprehensive_metrics(strategies, metrics):
    """Create a comprehensive comparison of all metrics for each strategy."""
    # Skip if no data
    if not strategies or not metrics:
        print("No data to plot comprehensive metrics")
        return
    
    # Filter out baseline for display
    display_strategies = [s for s in strategies if s != "Baseline"]
    
    if not display_strategies:
        print("No strategies to display (excluding baseline)")
        return
    
    # Define all metrics to compare
    metric_keys = [
        "total_time_pct", 
        "first_token_pct", 
        "memory_usage_pct", 
        "memory_avg_pct",
        "quality_score_pct",
        "tokens_per_second_pct",
        "eviction_count_pct",
        "eviction_time_pct"
    ]
    
    metric_labels = {
        "total_time_pct": "Total Time",
        "first_token_pct": "First Token Latency",
        "memory_usage_pct": "Peak KV Cache Memory",
        "memory_avg_pct": "Avg KV Cache Memory",
        "quality_score_pct": "Quality Score",
        "tokens_per_second_pct": "Tokens/Second",
        "eviction_count_pct": "Eviction Count",
        "eviction_time_pct": "Eviction Time"
    }
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric_key in enumerate(metric_keys):
        ax = axes[i]
        
        # Get values for each strategy
        values = []
        for strategy in display_strategies:
            if strategy in metrics and metric_key in metrics[strategy]:
                values.append(metrics[strategy][metric_key])
            else:
                values.append(0)
        
        # Create bar chart
        bars = ax.bar(display_strategies, values)
        
        # Add value labels
        for j, v in enumerate(values):
            # Position text above or below bar depending on value
            va = 'bottom' if v >= 0 else 'top'
            offset = 1 if v >= 0 else -1
            
            ax.text(j, v + offset, f"{v:.1f}%", 
                   ha='center', va=va, fontsize=9)
        
        # Set title and labels
        ax.set_title(f"{metric_labels[metric_key]} (% change from baseline)", fontsize=10)
        ax.set_ylabel("% Change", fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Set reasonable y-axis limits based on values
        max_abs_val = max(abs(min(values)), abs(max(values)))
        y_limit = max(60, max_abs_val * 1.2)  # Set at least ±60, or 20% more than the max value
        ax.set_ylim(-y_limit, y_limit)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Add overall title
    fig.suptitle("Comprehensive Comparison of KV Cache Management Strategies", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    plt.savefig("kv_cache_comprehensive_metrics.png", dpi=300)
    print("Comprehensive metrics saved to kv_cache_comprehensive_metrics.png")

def plot_raw_metrics_radar(strategies, metrics):
    """Create a radar chart comparing raw metrics for strategies."""
    # Skip if no data
    if not strategies or not metrics:
        print("No data to plot radar chart")
        return
    
    # Filter out baseline for display and include baseline as reference
    display_strategies = [s for s in strategies if s != "Baseline"]
    
    if not display_strategies:
        print("No strategies to display (excluding baseline)")
        return
    
    # Normalize metrics for radar chart
    # Select key raw metrics (not percentages)
    radar_metrics = [
        "total_time", 
        "first_token", 
        "memory_usage", 
        "quality_score",
        "tokens_per_second"
    ]
    
    metric_labels = {
        "total_time": "Total Time (s)",
        "first_token": "First Token (s)",
        "memory_usage": "KV Cache (MB)",
        "quality_score": "Quality Score",
        "tokens_per_second": "Tokens/Second"
    }
    
    # Get baseline values for normalization
    baseline_values = {}
    if "Baseline" in metrics:
        for metric in radar_metrics:
            baseline_values[metric] = metrics["Baseline"].get(metric, 1)  # Default to 1 to avoid division by zero
    else:
        # If no baseline, use maximum values for normalization
        for metric in radar_metrics:
            max_val = max([metrics[s].get(metric, 0) for s in display_strategies])
            baseline_values[metric] = max_val if max_val > 0 else 1
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set angles for each metric
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set radar chart properties
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    plt.xticks(angles[:-1], [metric_labels[m] for m in radar_metrics])
    
    # Draw baseline as reference (circular line at normalized value 1)
    ax.plot(angles, [1] * (len(angles)), '-', linewidth=1, color='gray', alpha=0.5)
    
    # Plot each strategy
    for strategy in display_strategies:
        if strategy in metrics:
            # Normalize values relative to baseline
            values = []
            for metric in radar_metrics:
                raw_value = metrics[strategy].get(metric, 0)
                baseline = baseline_values[metric]
                # Normalize and invert values where lower is better
                if metric in ["total_time", "first_token", "memory_usage"]:
                    # For these metrics, lower values are better, so invert ratio
                    norm_value = baseline / raw_value if raw_value > 0 else 0
                else:
                    # For these metrics, higher values are better
                    norm_value = raw_value / baseline if baseline > 0 else 0
                values.append(norm_value)
            
            # Close the loop for plotting
            values += values[:1]
            
            # Plot the strategy
            ax.plot(angles, values, label=strategy)
            ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("KV Cache Strategy Comparison (Normalized to Baseline)", size=15)
    
    # Save
    plt.tight_layout()
    plt.savefig("kv_cache_radar_chart.png", dpi=300)
    print("Radar chart saved to kv_cache_radar_chart.png")

def main():
    # Directory containing JSON result files
    data_dir = "./results"  # Change this to your data directory
    
    # Allow command line override
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        print(f"Using data directory: {data_dir}")
    
    # Analyze strategies
    strategies, metrics, metrics_by_turn, raw_metrics, memory_metric_key = analyze_kv_cache_strategies(data_dir)
    
    # Print summary
    print("\nStrategy Performance Summary:")
    print("-" * 80)
    metrics_to_show = ["total_time", "first_token", "memory_usage", "memory_avg", "quality_score", "tokens_per_second"]
    header = f"{'Strategy':<20} | " + " | ".join(f"{m.split('_')[0].title():^12}" for m in metrics_to_show)
    print(header)
    print("-" * 80)
    
    for strategy in strategies:
        if strategy in metrics:
            values = metrics[strategy]
            row = f"{strategy:<20} | " + " | ".join(f"{values.get(m, 0):^12.2f}" for m in metrics_to_show)
            print(row)
    
    print("\nPercentage Change from Baseline:")
    print("-" * 80)
    pct_metrics = [f"{m}_pct" for m in metrics_to_show]
    header = f"{'Strategy':<20} | " + " | ".join(f"{m.split('_')[0].title():^12}" for m in metrics_to_show)
    print(header)
    print("-" * 80)
    
    for strategy in strategies:
        if strategy != "Baseline" and strategy in metrics:
            values = metrics[strategy]
            row = f"{strategy:<20} | " + " | ".join(f"{values.get(m+'_pct', 0):^12.2f}%" for m in metrics_to_show)
            print(row)
    
    # Create and save visualizations
    plot_comparison_chart(strategies, metrics, memory_metric_key, "kv_cache_comparison.png")
    plot_memory_timeline(strategies, raw_metrics)
    plot_tokens_per_position(strategies, raw_metrics)
    plot_tradeoff_analysis(strategies, metrics)
    plot_comprehensive_metrics(strategies, metrics)
    plot_raw_metrics_radar(strategies, metrics)
    
    print("\nAnalysis complete! The following files were generated:")
    print("- kv_cache_comparison.png: Main comparison chart of all strategies")
    print("- kv_cache_memory_timeline.png: Memory usage over token generation")
    print("- token_generation_time.png: Token generation time by position")
    print("- kv_cache_tradeoff_analysis.png: Memory vs quality tradeoff analysis")
    print("- kv_cache_comprehensive_metrics.png: Detailed metrics comparison")
    print("- kv_cache_radar_chart.png: Normalized radar chart comparison")

if __name__ == "__main__":
    main()