import json
import matplotlib.pyplot as plt
import numpy as np

def plot_cache(results_path):
    with open(results_path) as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))
    for name, metrics in data.items():
        # average over samples
        cache_arr = np.array(metrics["cache_mb"])        # shape: [num_samples, gen_length]
        mean_cache = cache_arr.mean(axis=0)
        plt.plot(mean_cache, label=name)

    plt.axhline(y=250, color="r", linestyle="--", label="250 MB threshold")
    plt.xlabel("Generation Step")
    plt.ylabel("KV Cache Size (MB)")
    plt.title("MT-Bench: KV Cache Growth by Strategy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_time_per_token(results_path):
    with open(results_path) as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))
    for name, metrics in data.items():
        t_arr = np.array(metrics["times_s"])
        # time per token averaged across samples
        avg_time_per_step = t_arr.mean(axis=0)
        plt.plot(avg_time_per_step, label=name)

    plt.xlabel("Generation Step")
    plt.ylabel("Time per Token (s)")
    plt.title("MT-Bench: Inference Time per Token by Strategy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # adjust the path if needed
    results_file = "mtbench_results.json"
    plot_cache(results_file)
    plot_time_per_token(results_file)
