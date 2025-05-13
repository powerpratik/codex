import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

# --- (extract_strategy_name, extract_metrics functions remain the same as your last provided version) ---

def analyze_kv_cache_strategies(data_dir):
    """Analyze KV cache strategies from multiple JSON files in a directory."""
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {data_dir}")

    strategy_metrics = defaultdict(list)
    strategy_metrics_by_turn = defaultdict(lambda: defaultdict(list))
    strategies = []
    metrics_found = defaultdict(lambda: defaultdict(list))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for entry_data in data:
                strategy = extract_strategy_name(entry_data.get('prompt_this_turn', ''), os.path.basename(json_file))
                if strategy not in strategies and strategy != "Unknown":
                    strategies.append(strategy)
                metrics = extract_metrics(entry_data)
                if metrics and strategy != "Unknown":
                    for key, value in metrics.items():
                        if value is not None: metrics_found[strategy][key].append(value)
                    strategy_metrics[strategy].append(metrics)
                    turn_index = entry_data.get('turn_index', 0)
                    strategy_metrics_by_turn[strategy][turn_index].append(metrics)
        except (json.JSONDecodeError, IOError) as e: print(f"Error processing {json_file}: {e}")

    print(f"Identified strategies: {strategies}")

    memory_metric_key = "kv_cache_sizes_peak"

    for strategy, metrics_list_for_strategy in strategy_metrics.items():
        for m_entry in metrics_list_for_strategy:
            m_entry[memory_metric_key] = 0.0; m_entry['kv_cache_sizes_avg'] = 0.0
            if 'kv_cache_sizes' in m_entry and m_entry['kv_cache_sizes']:
                current_kv_cache_data = m_entry['kv_cache_sizes']
                current_keys = list(current_kv_cache_data.keys())
                current_values = list(current_kv_cache_data.values())
                try:
                    digit_keys = [k for k in current_keys if isinstance(k,str) and k.isdigit()]
                    int_keys = [int(k) for k in digit_keys]
                    if not int_keys:
                        if "kv_cache_end_of_turn_mb" in m_entry and m_entry["kv_cache_end_of_turn_mb"] is not None:
                            m_entry[memory_metric_key] = float(m_entry["kv_cache_end_of_turn_mb"])
                    else:
                        max_token_key_str = str(max(int_keys))
                        if max_token_key_str in current_kv_cache_data:
                            m_entry[memory_metric_key] = float(current_kv_cache_data[max_token_key_str])
                        elif "kv_cache_end_of_turn_mb" in m_entry and m_entry["kv_cache_end_of_turn_mb"] is not None:
                             m_entry[memory_metric_key] = float(m_entry["kv_cache_end_of_turn_mb"])
                except Exception:
                    if "kv_cache_end_of_turn_mb" in m_entry and m_entry["kv_cache_end_of_turn_mb"] is not None:
                        m_entry[memory_metric_key] = float(m_entry["kv_cache_end_of_turn_mb"])
                try:
                    float_values = [float(v) for v in current_values if isinstance(v,(int,float)) or (isinstance(v,str) and v.replace('.','',1).isdigit())]
                    if float_values: m_entry['kv_cache_sizes_avg'] = np.mean(float_values)
                except Exception: pass
            elif "kv_cache_end_of_turn_mb" in m_entry and m_entry["kv_cache_end_of_turn_mb"] is not None:
                 m_entry[memory_metric_key] = float(m_entry["kv_cache_end_of_turn_mb"])

    avg_metrics = {}
    for strategy, metrics_list in strategy_metrics.items():
        if not metrics_list: continue
        def safe_mean(key_name):
            vals = [m.get(key_name) for m in metrics_list if m.get(key_name) is not None]
            return np.mean(vals) if vals else 0.0
        avg_metrics[strategy] = {k: safe_mean(k_map) for k, k_map in {
            "total_time":"total_time_seconds", "first_token":"first_token_time",
            "memory_usage":memory_metric_key, "memory_avg":"kv_cache_sizes_avg",
            "quality_score":"quality_score", "tokens_per_second":"tokens_per_second",
            "tokens_generated":"tokens_generated", "eviction_count":"eviction_count",
            "avg_eviction_time":"avg_eviction_time"}.items()
        }
    
    # Ensure 'strategies' list accurately reflects those with data and correct ordering
    # This filters strategies to only those that have aggregated metrics.
    strategies_with_avg_data = [s for s in strategies if s in avg_metrics]
    if "Baseline" in strategies_with_avg_data: # Ensure Baseline is last if present
        strategies_with_avg_data.remove("Baseline")
        strategies_with_avg_data.append("Baseline")
    strategies = strategies_with_avg_data # Update the main strategies list

    avg_metrics_by_turn = defaultdict(lambda: defaultdict(dict))
    for strategy, turns_data in strategy_metrics_by_turn.items():
        # Only process strategies that are in our refined list (i.e., have overall avg_metrics)
        if strategy not in strategies:
            continue
        for turn, metrics_list_for_turn in turns_data.items():
            if not metrics_list_for_turn: continue
            def safe_mean_turn(key_name):
                vals = [m.get(key_name) for m in metrics_list_for_turn if m.get(key_name) is not None]
                return np.mean(vals) if vals else 0.0 # Return 0.0 for consistency, or np.nan if preferred for plotting gaps

            avg_metrics_by_turn[strategy][turn] = { k: safe_mean_turn(k_map) for k, k_map in {
                "total_time":"total_time_seconds", "first_token":"first_token_time",
                "memory_usage":memory_metric_key, "memory_avg":"kv_cache_sizes_avg",
                "quality_score":"quality_score", "tokens_per_second":"tokens_per_second",
                "tokens_generated":"tokens_generated", # Keep for data structure completeness
                "kv_cache_eot_mb_avg": "kv_cache_end_of_turn_mb" # ADDED for the new plot requirement
                }.items()
            }

    baseline = avg_metrics.get("Baseline")
    if baseline:
        metric_keys_for_pct = ["total_time", "first_token", "memory_usage", "memory_avg",
                               "quality_score", "tokens_per_second", "tokens_generated",
                               "eviction_count", "avg_eviction_time"]
        for strategy, metrics_val in avg_metrics.items():
            if strategy != "Baseline":
                for key in metric_keys_for_pct:
                    baseline_key_val = baseline.get(key)
                    if key in metrics_val and metrics_val[key] is not None and baseline_key_val is not None and baseline_key_val != 0 :
                        metrics_val[f"{key}_pct"] = ((metrics_val[key] - baseline_key_val) / baseline_key_val) * 100
                    elif key in metrics_val and metrics_val[key] is not None and baseline_key_val == 0 and metrics_val[key] != 0:
                         metrics_val[f"{key}_pct"] = float('inf') if metrics_val[key] > 0 else float('-inf')
                    else: metrics_val[f"{key}_pct"] = 0.0
    else: print("Warning: No baseline strategy found. Cannot compute percentage changes.")

    return strategies, avg_metrics, avg_metrics_by_turn, strategy_metrics, memory_metric_key


def extract_strategy_name(prompt_text, filename):
    strategy_keywords = {
        # "adaptive": "AdaptiveAttention", "adaptiveattention": "AdaptiveAttention",
        "attention_bottom": "AttentionBottom", "attention-bottom": "AttentionBottom", "attentionbottom": "AttentionBottom",
        "attention_top": "AttentionTop", "attention-top": "AttentionTop", "attentiontop": "AttentionTop",
        "hybrid": "HybridNPercent", "npercent": "HybridNPercent", "hybridnpercent": "HybridNPercent",
        # "random": "Random",
        "sliding": "SlidingWindow", "slidingwindow": "SlidingWindow",
        "baseline": "Baseline", "base": "Baseline",
        "evictoldes": "EvictOldest", "evictoldest": "EvictOldest", "evict_oldest": "EvictOldest"
    }
    prompt_text_lower = prompt_text.lower(); filename_lower = filename.lower()
    for keyword_map in [prompt_text_lower, filename_lower]: # Check prompt then filename
        for keyword, strategy_name in strategy_keywords.items():
            if keyword in keyword_map: return strategy_name
    for strategy_name_val in strategy_keywords.values(): # Check for full strategy name in filename
        if strategy_name_val.lower() in filename_lower: return strategy_name_val
    return "Unknown"

def extract_metrics(entry):
    metrics = {}
    try:
        metrics["total_time_seconds"] = entry.get("time_total_turn_seconds", 0.0)
        metrics["tokens_per_second"] = entry.get("tokens_per_second_this_turn", 0.0)
        metrics["tokens_generated"] = entry.get("tokens_generated_this_turn", 0)
        metrics["first_token_time"] = entry.get("time_to_first_token_of_turn", 0.0)
        metrics["token_times"] = entry.get("token_gen_times_this_turn", {})
        metrics["kv_cache_sizes"] = entry.get("kv_cache_sizes_during_gen", {})
        memory_stats = entry.get("memory", {})
        eviction_stats = memory_stats.get("eviction_stats_this_turn", {})
        metrics["eviction_count"] = eviction_stats.get("evictions_this_turn", 0)
        metrics["avg_eviction_time"] = eviction_stats.get("avg_eviction_time_this_turn", 0.0)
        metrics["kv_cache_end_of_turn_mb"] = memory_stats.get("kv_cache_end_of_turn_mb") # This is crucial
        accuracy_stats = entry.get("accuracy", {})
        metrics["quality_score"] = accuracy_stats.get("azure_score", 0)
        metrics["turn_index"] = entry.get("turn_index", 0)

        for key, target_type in {
            "total_time_seconds": float, "tokens_per_second": float, "first_token_time": float,
            "avg_eviction_time": float, "kv_cache_end_of_turn_mb": float,
            "tokens_generated": int, "eviction_count": int, "quality_score": int, "turn_index": int
        }.items():
            if metrics.get(key) is not None and not isinstance(metrics[key], target_type):
                try: metrics[key] = target_type(metrics[key])
                except (ValueError, TypeError): metrics[key] = target_type() # Default (0 or 0.0)
        return metrics
    except Exception: return None

# --- (Other plotting functions: plot_comparison_chart, plot_memory_timeline, etc. remain as previously provided) ---
def plot_comparison_chart(strategies, metrics, memory_metric_key, output_file=None):
    if not strategies or not metrics: return
    display_strategies = [s for s in strategies if s != "Baseline" and s in metrics]
    if not display_strategies: return
    fig, ax = plt.subplots(figsize=(18, 10))
    x = np.arange(len(display_strategies))
    metric_keys_to_plot = ["total_time_pct", "first_token_pct", "memory_usage_pct", "quality_score_pct", "tokens_per_second_pct", "tokens_generated_pct"]
    colors = {"total_time_pct": "#8884d8", "first_token_pct": "#82ca9d", "memory_usage_pct": "#ffc658", "quality_score_pct": "#ff7300", "tokens_per_second_pct": "#00A9A5", "tokens_generated_pct": "#FF00FF"}
    labels = {"total_time_pct": "Total Gen. Time", "first_token_pct": "First Token Latency", "memory_usage_pct": "Peak KV Cache", "quality_score_pct": "Quality Score", "tokens_per_second_pct": "Tokens/Second", "tokens_generated_pct": "Total Tokens Gen."}
    num_metrics_to_plot = len(metric_keys_to_plot)
    width = 0.9 / num_metrics_to_plot
    for i, metric in enumerate(metric_keys_to_plot):
        values = [metrics[s].get(metric, 0) for s in display_strategies]
        bar_offset = (i - (num_metrics_to_plot - 1) / 2.0) * width
        ax.bar(x + bar_offset, values, width, label=labels.get(metric, metric), color=colors.get(metric, None))
        for j, v_val in enumerate(values):
            va = 'bottom' if v_val >= 0 else 'top'; text_offset_val = max(abs(v_val)*0.02, 1.5) * (1 if v_val >=0 else -1)
            ax.text(x[j] + bar_offset, v_val + text_offset_val, f"{v_val:.1f}%", ha='center', va=va, fontsize=13, fontweight='bold')
    # title = 'Performance Metrics Comparison (% Change from Baseline)'; ax.set_title(title, fontsize=2)
    ax.set_ylabel('% Change from Baseline', fontsize=22); ax.set_xticks(x)
    
    ax.set_xticklabels(display_strategies, rotation=25, ha="right", fontsize=22)

    ax.legend(fontsize=18, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    all_plot_values = [val for metric in metric_keys_to_plot for s in display_strategies if (val := metrics[s].get(metric)) is not None]
    if all_plot_values: y_abs_max = max(max(abs(v) for v in all_plot_values) if all_plot_values else 0, 60); ax.set_ylim(-y_abs_max * 1.2, y_abs_max * 1.2)
    else: ax.set_ylim(-60, 60)
    # ax.set_yticklabels(y_abs_max,fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.97]);
    if output_file: plt.savefig(output_file, dpi=1000); print(f"Main comparison chart saved to {output_file}")
    else: plt.show()


def plot_memory_timeline(strategies, raw_metrics):
    kv_cache_by_strategy = defaultdict(lambda: defaultdict(list)); max_overall_token_pos = 0
    for strategy, entries in raw_metrics.items():
        for entry in entries:
            kv_sizes = entry.get('kv_cache_sizes', {})
            if kv_sizes:
                current_max_pos_this_entry = 0
                for pos_str, value_str in kv_sizes.items():
                    try: token_pos=int(pos_str); mem_value=float(value_str)
                    except(ValueError,TypeError): continue
                    if mem_value >= 0: kv_cache_by_strategy[strategy][token_pos].append(mem_value)
                    if token_pos > current_max_pos_this_entry: current_max_pos_this_entry = token_pos
                if current_max_pos_this_entry > max_overall_token_pos: max_overall_token_pos = current_max_pos_this_entry
    avg_kv_cache_by_strategy = defaultdict(dict)
    for strategy, positions_data_lists in kv_cache_by_strategy.items():
        for pos, values_list in positions_data_lists.items():
            if values_list: avg_kv_cache_by_strategy[strategy][pos] = np.mean(values_list)
    plt.figure(figsize=(12,7)); colors = {"HybridNPercent":"#1f77b4", "Baseline":"#ff7f0e", "Random":"#2ca02c", "AttentionTop":"#d62728", "AttentionBottom":"#9467bd", "SlidingWindow":"#8c564b", "AdaptiveAttention":"#e377c2", "EvictOldest":"#FF69B4", "Unknown":"#7f7f7f"}
    plotted_count = 0
    active_strategies_for_plot = [s for s in strategies if s in avg_kv_cache_by_strategy and avg_kv_cache_by_strategy[s]]
    for strategy in active_strategies_for_plot:
        pos_data = sorted(avg_kv_cache_by_strategy[strategy].keys()); val_data = [avg_kv_cache_by_strategy[strategy][p] for p in pos_data]
        if len(pos_data)>1: plt.plot(pos_data,val_data,label=strategy,color=colors.get(strategy),marker='.' if len(pos_data)<100 else None, markersize=3,linewidth=1.5); plotted_count+=1
    plt.xlabel('Token Position'); plt.ylabel('Avg KV Cache Memory (MB)'); plt.title('KV Cache Memory Usage During Generation')
    if plotted_count > 0: plt.legend()
    else: plt.text(0.5,0.5,"No data for memory timeline",ha='center',va='center',transform=plt.gca().transAxes)
    plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(left=0, right=max_overall_token_pos or None); plt.tight_layout(); plt.savefig("kv_cache_memory_timeline.png",dpi=300); print("Memory timeline saved.")

def plot_tokens_per_position(strategies, raw_metrics):
    token_times_by_strategy=defaultdict(lambda:defaultdict(list)); max_overall_token_pos=0
    for strategy, entries in raw_metrics.items():
        for entry in entries:
            detailed_token_times = entry.get('token_times',{})
            if detailed_token_times:
                current_max_pos_this_entry=0
                for pos_str, time_str in detailed_token_times.items():
                    try: token_pos=int(pos_str); time_value=float(time_str)
                    except(ValueError,TypeError): continue
                    if time_value > 0: token_times_by_strategy[strategy][token_pos].append(time_value)
                    if token_pos > current_max_pos_this_entry: current_max_pos_this_entry = token_pos
                if current_max_pos_this_entry > max_overall_token_pos: max_overall_token_pos = current_max_pos_this_entry
    avg_time_by_pos=defaultdict(dict)
    for strategy, positions_data_lists in token_times_by_strategy.items():
        for pos, values_list in positions_data_lists.items():
            if values_list: avg_time_by_pos[strategy][pos] = np.mean(values_list)
    plt.figure(figsize=(12,7)); colors = {"HybridNPercent":"#1f77b4", "Baseline":"#ff7f0e", "Random":"#2ca02c", "AttentionTop":"#d62728", "AttentionBottom":"#9467bd", "SlidingWindow":"#8c564b", "AdaptiveAttention":"#e377c2", "EvictOldest":"#FF69B4", "Unknown":"#7f7f7f"}
    plotted_count = 0
    active_strategies_for_plot = [s for s in strategies if s in avg_time_by_pos and avg_time_by_pos[s]]
    for strategy in active_strategies_for_plot:
        pos_data = sorted(avg_time_by_pos[strategy].keys()); val_data = [avg_time_by_pos[strategy][p] for p in pos_data]
        if len(pos_data)>1: plt.plot(pos_data,val_data,label=strategy,color=colors.get(strategy),marker='.' if len(pos_data)<100 else None, markersize=3,linewidth=1.5); plotted_count+=1
    if plotted_count > 0: plt.legend()
    else: plt.text(0.5,0.5,"No data for token times",ha='center',va='center',transform=plt.gca().transAxes)
    plt.xlabel('Token Position'); plt.ylabel('Avg Token Generation Time (s)'); plt.title('Token Generation Time by Position')
    plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(left=0, right=max_overall_token_pos or None); plt.tight_layout(); plt.savefig("token_generation_time.png",dpi=300); print("Token time plot saved.")

def plot_tradeoff_analysis(strategies, metrics):
    plt.figure(figsize=(10,8)); colors = {"HybridNPercent":"#1f77b4", "Baseline":"#ff7f0e", "Random":"#2ca02c", "AttentionTop":"#d62728", "AttentionBottom":"#9467bd", "SlidingWindow":"#8c564b", "AdaptiveAttention":"#e377c2", "EvictOldest":"#FF69B4", "Unknown":"#7f7f7f"}
    plotted_any=False
    active_strategies_for_plot = [s for s in strategies if s != "Baseline" and s in metrics]
    for strategy in active_strategies_for_plot:
        m=metrics[strategy]; memory_pct=m.get("memory_usage_pct"); quality_pct=m.get("quality_score_pct"); time_pct=m.get("total_time_pct",0)
        if memory_pct is not None and quality_pct is not None:
            size=100+abs(time_pct)*5; plt.scatter(memory_pct,quality_pct,s=size,label=strategy,color=colors.get(strategy),alpha=0.7)
            plt.annotate(strategy,(memory_pct,quality_pct),textcoords="offset points",xytext=(0,5),ha='center',fontsize=9); plotted_any=True
    if plotted_any: plt.axhline(y=0,color='gray',linestyle='--',alpha=0.5); plt.axvline(x=0,color='gray',linestyle='--',alpha=0.5); plt.legend()
    else: plt.text(0.5,0.5,"No data for tradeoff analysis",ha='center',va='center',transform=plt.gca().transAxes)
    plt.xlabel('KV Cache Memory Usage (% change)'); plt.ylabel('Quality Score (% change)'); plt.title('Tradeoff Analysis: Memory vs. Quality')
    plt.grid(True, linestyle='--', alpha=0.4); plt.tight_layout(); plt.savefig("kv_cache_tradeoff_analysis.png",dpi=300); print("Tradeoff analysis saved.")

def plot_comprehensive_metrics(strategies, metrics):
    if not strategies or not metrics: return
    display_strategies = [s for s in strategies if s!="Baseline" and s in metrics];
    if not display_strategies: return
    metric_keys = ["total_time_pct", "first_token_pct", "tokens_per_second_pct", "tokens_generated_pct", "memory_usage_pct", "memory_avg_pct", "quality_score_pct", "eviction_count_pct", "avg_eviction_time_pct"]
    metric_labels = {"total_time_pct":"Total Time", "first_token_pct":"First Token Latency", "tokens_per_second_pct":"Tokens/Sec", "tokens_generated_pct":"Total Tokens Gen.", "memory_usage_pct":"Peak KV Cache", "memory_avg_pct":"Avg KV Cache", "quality_score_pct":"Quality Score", "eviction_count_pct":"Eviction Count", "avg_eviction_time_pct":"Avg Eviction Time"}
    num_metrics=len(metric_keys); cols=3; rows=(num_metrics+cols-1)//cols; fig,axes=plt.subplots(rows,cols,figsize=(18,4*rows)); axes=axes.flatten()
    for i,metric_key in enumerate(metric_keys):
        if i>=len(axes): break
        ax=axes[i]; values=[metrics[s].get(metric_key,0) for s in display_strategies]; bar_colors=[plt.cm.get_cmap('viridis')(j/float(len(display_strategies))) for j in range(len(display_strategies))]
        ax.bar(display_strategies,values,color=bar_colors)
        for bar_idx,v_val in enumerate(values): va='bottom' if v_val>=0 else 'top'; offset_val=max(abs(v_val)*0.05,1.0)*(1 if v_val>=0 else -1); ax.text(bar_idx,v_val+offset_val,f"{v_val:.1f}%",ha='center',va=va,fontsize=8)
        ax.set_title(f"{metric_labels.get(metric_key,metric_key)} (% change)",fontsize=10); ax.set_ylabel("%Change",fontsize=8); ax.grid(axis='y',linestyle='--',alpha=0.6)
        y_abs_max=max(max(abs(v) for v in values) if values else 0,10); ax.set_ylim(-y_abs_max*1.25,y_abs_max*1.25); plt.setp(ax.get_xticklabels(),rotation=45,ha='right',fontsize=8)
    for j_ax_idx in range(i+1,len(axes)): fig.delaxes(axes[j_ax_idx])
    fig.suptitle("Comprehensive Comparison: KV Cache Strategies (% Change from Baseline)",fontsize=16,y=0.99); plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig("kv_cache_comprehensive_metrics.png",dpi=300); print("Comprehensive metrics saved.")

def plot_raw_metrics_radar(strategies, metrics):
    if not strategies or not metrics: return
    display_strategies_radar = [s for s in strategies if s in metrics];
    if not display_strategies_radar: return
    radar_metric_keys = ["total_time","first_token","tokens_per_second","tokens_generated","memory_usage","quality_score"]
    metric_labels_radar = {"total_time":"Time(L)","first_token":"1stTok(L)","tokens_per_second":"Tok/s(H)","tokens_generated":"TotalTok(H)","memory_usage":"PeakMem(L)","quality_score":"Quality(H)"}
    normalized_data=defaultdict(dict); min_max_vals={}
    for metric_key_radar in radar_metric_keys:
        all_vals=[v for v in[metrics[s].get(metric_key_radar) for s in display_strategies_radar] if v is not None]
        if not all_vals: min_max_vals[metric_key_radar]=(0,1); continue
        min_val,max_val=min(all_vals),max(all_vals)
        if min_val==max_val: min_val=0 if max_val>0 else -1; max_val=max_val+1 if max_val>=0 else 0
        min_max_vals[metric_key_radar]=(min_val,max_val)
    for strategy in display_strategies_radar:
        for metric_key_radar in radar_metric_keys:
            raw_value=metrics[strategy].get(metric_key_radar)
            if raw_value is None: normalized_data[strategy][metric_key_radar]=0.5; continue
            min_val,max_val=min_max_vals.get(metric_key_radar,(0,1)); current_range=max_val-min_val
            if current_range==0: norm_value=0.5
            elif metric_key_radar in["total_time","first_token","memory_usage"]: norm_value=(max_val-raw_value)/current_range
            else: norm_value=(raw_value-min_val)/current_range
            normalized_data[strategy][metric_key_radar]=max(0,min(1,norm_value))
    num_vars=len(radar_metric_keys); angles=np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(11,11),subplot_kw=dict(polar=True)); ax.set_xticks(angles[:-1]); ax.set_xticklabels([metric_labels_radar[m] for m in radar_metric_keys],fontsize=9)
    ax.set_yticks(np.linspace(0,1,5)); ax.set_yticklabels(["Worst","","Mid","","Best"],fontsize=8); ax.set_rlabel_position(angles[0]*180/np.pi+7)
    strategy_colors_map = {"HybridNPercent":"#1f77b4", "Baseline":"#ff7f0e", "Random":"#2ca02c", "AttentionTop":"#d62728", "AttentionBottom":"#9467bd", "SlidingWindow":"#8c564b", "AdaptiveAttention":"#e377c2", "EvictOldest":"#FF69B4", "Unknown":"#7f7f7f"}
    default_colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i,strategy in enumerate(display_strategies_radar):
        values=[normalized_data[strategy].get(metric,0.5) for metric in radar_metric_keys]; values+=values[:1]
        color=strategy_colors_map.get(strategy,default_colors[i%len(default_colors)])
        ax.plot(angles,values,linewidth=2,linestyle='solid',label=strategy,color=color); ax.fill(angles,values,color=color,alpha=0.25)
    plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.15),ncol=min(3,len(display_strategies_radar)//2+1),fontsize=10)
    ax.set_title("KV Cache Strategy Comparison (Normalized Raw Metrics)",size=16,y=1.1); plt.tight_layout(); plt.savefig("kv_cache_radar_chart.png",dpi=300); print("Radar chart saved.")


# MODIFIED PLOTTING FUNCTION for dual axis
def plot_accuracy_and_kv_cache_over_turns(strategies, avg_metrics_by_turn, output_file="accuracy_kv_cache_over_turns.png"): # Filename changed for clarity
    """Plots average quality score and KV Cache End of Turn (MB) over turns for each strategy."""
    if not avg_metrics_by_turn and not any(avg_metrics_by_turn.values()):
        print("No data available for accuracy and KV Cache EOT over turns plot.")
        # Create an empty plot with a message if no data
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax1.text(0.5, 0.5, "No data to plot for accuracy/KV Cache EOT over turns.", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_xlabel("Turn Index", fontsize=13)
        ax1.set_ylabel("Average Quality Score (Accuracy)", fontsize=13, color='tab:blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel("Average KV Cache End of Turn (MB)", fontsize=13, color='tab:red') # UPDATED Y-axis label for ax2
        fig.suptitle("Accuracy & KV Cache EOT (MB) Over Turns by Strategy", fontsize=16, y=0.98) # UPDATED plot title
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Empty accuracy and KV Cache EOT over turns plot saved to {output_file}")
        else:
            plt.show()
        return

    fig, ax1 = plt.subplots(figsize=(14, 8)) # Main axes for Accuracy
    ax2 = ax1.twinx()  # Create a second y-axis for KV Cache EOT MB

    base_colors = {
        "HybridNPercent": "tab:blue", "Baseline": "tab:orange", "Random": "tab:green",
        "AttentionTop": "tab:red", "AttentionBottom": "tab:purple",
        "SlidingWindow": "tab:brown", "AdaptiveAttention": "tab:pink",
        "EvictOldest": "tab:cyan", "Unknown": "tab:gray"
    }
    default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    max_turn_idx = 0
    plotted_anything = False

    plot_order_strategies = sorted([s for s in strategies if s in avg_metrics_by_turn], key=lambda s: s != "Baseline")


    for idx, strategy in enumerate(plot_order_strategies):
        if strategy in avg_metrics_by_turn: # Should always be true due to plot_order_strategies filter
            turn_data = avg_metrics_by_turn[strategy]
            if turn_data:
                sorted_turns = sorted(turn_data.keys())
                if not sorted_turns: continue

                quality_scores = [turn_data[turn].get("quality_score", np.nan) for turn in sorted_turns]
                # UPDATED: Fetch KV Cache EOT MB data
                kv_cache_eot_mb = [turn_data[turn].get("kv_cache_eot_mb_avg", np.nan) for turn in sorted_turns]

                if not (np.all(np.isnan(quality_scores)) and np.all(np.isnan(kv_cache_eot_mb))):
                    plotted_anything = True
                    max_turn_idx = max(max_turn_idx, sorted_turns[-1] if sorted_turns else 0)
                    color = base_colors.get(strategy, default_color_cycle[idx % len(default_color_cycle)])

                    # Plot Accuracy on ax1
                    if not np.all(np.isnan(quality_scores)):
                        ax1.plot(sorted_turns, quality_scores, label=f"{strategy} - Accuracy", marker='o', linestyle='-', color=color, linewidth=1.5, markersize=4)

                    # Plot KV Cache EOT MB on ax2
                    if not np.all(np.isnan(kv_cache_eot_mb)):
                        ax2.plot(sorted_turns, kv_cache_eot_mb, label=f"{strategy} - KV Cache EOT (MB)", marker='x', linestyle='--', color=color, linewidth=1.5, markersize=4) # UPDATED label


    ax1.set_xlabel("Turn Index", fontsize=13)
    ax1.set_ylabel("Average Quality Score (Accuracy)", fontsize=13, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=11)

    ax2.set_ylabel("Average KV Cache End of Turn (MB)", fontsize=13, color='tab:red') # UPDATED Y-axis label for ax2
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)

    fig.suptitle("Accuracy & KV Cache EOT (MB) Over Turns by Strategy", fontsize=16, y=0.98) # UPDATED plot title

    if plotted_anything:
        handles, labels = [], []
        for ax_in in [ax1, ax2]:
            h, l = ax_in.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        if handles: # Only show legend if there's something to show
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=min(len(plot_order_strategies), 4), fancybox=True, shadow=True, fontsize=10)
    else:
        ax1.text(0.5, 0.5, "No data to plot for accuracy/KV Cache EOT over turns.", ha='center', va='center', transform=ax1.transAxes)
        print("Warning: No data was plotted for accuracy and KV Cache EOT over turns.")

    ax1.grid(True, linestyle=':', alpha=0.7, which='major', axis='y')
    ax2.grid(True, linestyle=':', alpha=0.3, which='major', axis='y')

    if max_turn_idx > 0 or plotted_anything : # Ensure xlim is set even if max_turn_idx is 0 but something was plotted
        ax1.set_xlim(left=-0.5, right=max_turn_idx + 0.5)
    else: # If nothing plotted and max_turn_idx is 0
        ax1.set_xlim(left=-0.5, right=0.5)


    all_quality_scores = [item for strategy in plot_order_strategies if strategy in avg_metrics_by_turn for turn_metrics in avg_metrics_by_turn[strategy].values() if (item := turn_metrics.get("quality_score")) is not None and not np.isnan(item)]
    if all_quality_scores: ax1.set_ylim(bottom=max(0, min(all_quality_scores) - 1), top=min(10.5, max(all_quality_scores) + 1))
    else: ax1.set_ylim(0, 10.5)

    # Dynamic Y-axis scaling for ax2 (KV Cache EOT MB)
    all_kv_cache_eot_mb = [item for strategy in plot_order_strategies if strategy in avg_metrics_by_turn for turn_metrics in avg_metrics_by_turn[strategy].values() if (item := turn_metrics.get("kv_cache_eot_mb_avg")) is not None and not np.isnan(item)]
    if all_kv_cache_eot_mb:
        min_val_kv = min(all_kv_cache_eot_mb)
        max_val_kv = max(all_kv_cache_eot_mb)
        ax2.set_ylim(bottom=max(0, min_val_kv * 0.9 if min_val_kv > 0 else -max_val_kv*0.1), # give some space below if min is not 0
                     top=max_val_kv * 1.1 if max_val_kv > 0 else 10)
    else: ax2.set_ylim(0, 50) # Default if no KV cache data

    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Accuracy and KV Cache EOT over turns plot saved to {output_file}")
    else:
        plt.show()
def plot_kv_cache_single_question(data_dir, question_id=None):
    """Plot KV cache over turns for a single question identified by question_id.
    
    Args:
        data_dir: Directory containing JSON files
        question_id: The question_id to plot data for
    """
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    # Data structure to store KV cache data by strategy and turn
    kv_cache_by_strategy_and_turn = defaultdict(lambda: defaultdict(list))
    strategies = []
    max_turn = 0
    question_ids_found = set()
    
    # First pass: if no question_id provided, collect all available question_ids
    if question_id is None:
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    continue
                
                for entry in data:
                    if 'question_id' in entry:
                        question_ids_found.add(entry['question_id'])
            except (json.JSONDecodeError, IOError):
                pass
        
        if not question_ids_found:
            print("No question_ids found in any of the JSON files.")
            return
        
        # Use the first question_id found if none specified
        question_id = next(iter(question_ids_found))
        print(f"No question_id provided. Available question_ids: {sorted(question_ids_found)}")
        print(f"Using the first question_id found: {question_id}")
    
    # Second pass: collect data for the specified question_id
    print(f"Collecting data for question_id: {question_id}")
    found_entries = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                continue
            
            # Extract strategy from filename
            strategy = extract_strategy_name('', os.path.basename(json_file))
            if strategy == "Unknown":
                continue
            
            # Filter entries for the specified question_id
            question_entries = [entry for entry in data if entry.get('question_id') == question_id]
            
            if not question_entries:
                continue
            
            found_entries += len(question_entries)
            
            if strategy not in strategies:
                strategies.append(strategy)
            
            for entry in question_entries:
                turn_index = entry.get('turn_index', 0)
                max_turn = max(max_turn, turn_index)
                
                memory_data = entry.get('memory', {})
                kv_cache_eot = memory_data.get('kv_cache_end_of_turn_mb')
                
                if kv_cache_eot is not None:
                    try:
                        kv_cache_by_strategy_and_turn[strategy][turn_index].append(float(kv_cache_eot))
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid KV cache value in {json_file}, turn {turn_index}")
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error processing {json_file}: {e}")
    
    if found_entries == 0:
        print(f"No data found for question_id: {question_id}")
        return
    
    print(f"Found {found_entries} entries for question_id: {question_id} across {len(strategies)} strategies")
    # breakpoint()
    
    # Calculate average KV cache sizes for each turn and strategy
    avg_kv_cache_by_strategy = defaultdict(dict)
    for strategy, turn_data in kv_cache_by_strategy_and_turn.items():
        for turn, values in turn_data.items():
            if values:
                avg_kv_cache_by_strategy[strategy][turn] = np.mean(values)
    
    # Plot the data
    plt.figure(figsize=(14, 8))
    
    # Use the same color scheme as other plots in the script
    colors = {"Baseline":"#ff7f0e", "Random":"#2ca02c", 
              "AttentionTop":"#d62728", "AttentionBottom":"#9467bd", 
              "SlidingWindow":"#8c564b", "AdaptiveAttention":"#e377c2", 
              "EvictOldest":"#FF69B4", "Unknown":"#7f7f7f"}
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plotted_count = 0
    # Ensure Baseline is last if present
    if "Baseline" in strategies:
        strategies.remove("Baseline")
        strategies.append("Baseline")
    
    for idx, strategy in enumerate(strategies):
        if strategy in avg_kv_cache_by_strategy and avg_kv_cache_by_strategy[strategy]:
            turn_positions = sorted(avg_kv_cache_by_strategy[strategy].keys())
            values = [avg_kv_cache_by_strategy[strategy][turn] for turn in turn_positions]
            
            if len(turn_positions) > 0:
                color = colors.get(strategy, default_colors[idx % len(default_colors)])
                plt.plot(turn_positions, values, label=strategy, 
                         color=color, marker='o', markersize=6, linewidth=2)
                plotted_count += 1
                
                # Add value annotations if not too many points
                if len(turn_positions) <= 10:
                    for t, v in zip(turn_positions, values):
                        plt.annotate(f"{v:.1f}", (t, v), textcoords="offset points", 
                                    xytext=(0, 7), ha='center', fontsize=8)
    
    plt.xlabel('Turn Index', fontsize=13)
    plt.ylabel('KV Cache End of Turn (MB)', fontsize=13)
    plt.title(f'KV Cache Memory Usage Over Turns for Question ID: {question_id}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if plotted_count > 0:
        plt.legend(fontsize=10)
    else:
        plt.text(0.5, 0.5, f"No KV cache data available for question ID: {question_id}", 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # Set x-axis limits with some padding
    plt.xlim(-0.5, max_turn + 0.5)
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Create a safe filename from the question ID
    safe_question_id = ''.join(c if c.isalnum() else '_' for c in str(question_id))
    output_file = f"kv_cache_question_{safe_question_id}.png"
    plt.savefig(output_file, dpi=300)
    print(f"KV cache plot for question ID {question_id} saved to {output_file}")

def main():
    data_dir = "./results"
    import sys
    if len(sys.argv) > 1: data_dir = sys.argv[1]
    print(f"Using data directory: {data_dir}")
    if not os.path.isdir(data_dir):
        pass
    strategies, metrics, avg_metrics_by_turn, raw_metrics, memory_metric_key = analyze_kv_cache_strategies(data_dir)

    if not strategies or (not metrics and not avg_metrics_by_turn): # Adjusted condition
        print("No strategies or insufficient metrics data found. Exiting visualization.")
        return

    print("\nStrategy Performance Summary (Raw Averages):")
    metrics_to_show = ["total_time", "first_token", "tokens_per_second", "tokens_generated",
                       "memory_usage", "memory_avg", "quality_score", "eviction_count", "avg_eviction_time"]
    header_width = 140; print("-" * header_width)
    header_parts = [f"{'Strategy':<20}"] + [f"{m.replace('_', ' ').title():^12}" for m in metrics_to_show]
    print(" | ".join(header_parts)); print("-" * len(" | ".join(header_parts)))

    # Iterate over the 'strategies' list which is now filtered and ordered
    for strategy_name in strategies:
        if strategy_name in metrics: # Check if metrics were actually computed
            values = metrics[strategy_name]
            row_parts = [f"{strategy_name:<20}"] + [f"{values.get(m, 0):^12.3f}" for m in metrics_to_show]
            print(" | ".join(row_parts))
        # else: # Optional: indicate if a strategy in the list had no avg_metrics
            # print(f" | {strategy_name:<20} | {'No aggregated data':^106} |")


    if "Baseline" in metrics: # Check if Baseline has data
        print("\nPercentage Change from Baseline:"); print("-" * header_width)
        print(" | ".join(header_parts)); print("-" * len(" | ".join(header_parts)))
        for strategy_name in strategies: # Iterate over the filtered 'strategies' list
            if strategy_name != "Baseline" and strategy_name in metrics: # Check if metrics were computed
                values = metrics[strategy_name]
                row_parts_pct = [f"{strategy_name:<20}"]
                for m_key in metrics_to_show:
                    pct_val = values.get(f"{m_key}_pct")
                    if pct_val is not None:
                        if pct_val == float('inf'): row_parts_pct.append(f"{'Inf':^12}")
                        elif pct_val == float('-inf'): row_parts_pct.append(f"{'-Inf':^12}")
                        else: row_parts_pct.append(f"{pct_val:^12.2f}%")
                    else: row_parts_pct.append(f"{'N/A':^12}")
                print(" | ".join(row_parts_pct))
    else: print("\nNo Baseline strategy data found, skipping Percentage Change table.")

    # Call plotting functions with the potentially filtered 'strategies' list
    plot_comparison_chart(strategies, metrics, memory_metric_key, "kv_cache_comparison.png")
    plot_memory_timeline(strategies, raw_metrics)
    plot_tokens_per_position(strategies, raw_metrics)
    plot_tradeoff_analysis(strategies, metrics)
    plot_comprehensive_metrics(strategies, metrics)
    plot_raw_metrics_radar(strategies, metrics)
    plot_kv_cache_single_question(data_dir, "aaTQJBE")
    # UPDATED call to the renamed function (if you choose to rename it) or ensure output_file matches the new purpose
    plot_accuracy_and_kv_cache_over_turns(strategies, avg_metrics_by_turn, "accuracy_kv_cache_over_turns.png")

    print("\nAnalysis complete! Check the generated PNG files in the current directory.")
    # UPDATED print message to reflect the change in the plot's content
    print("- accuracy_kv_cache_over_turns.png: Plot for accuracy and KV Cache End of Turn (MB) over turns.")

if __name__ == "__main__":
    main()
