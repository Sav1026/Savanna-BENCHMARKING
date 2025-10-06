import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your folder
base_path = os.path.expanduser("~/Desktop/6344334")
save_dir = os.path.expanduser("~/Desktop")

# File names
files = {
    "Gemma3-1B": "benchmark_results_EQUALITYgemma3_1b.csv",
    "Gemma3n-E2B": "benchmark_results_EQUALITYgemma3n_e2b.csv",
    "LLaVA-7B": "benchmark_results_EQUALITYllava_7b.csv",
    "LLaVA-Llama3 Latest": "benchmark_results_EQUALITYllava-llama3_latest.csv",
    "Qwen2.5VL-3B": "benchmark_results_EQUALITYqwen2.5vl_3b.csv",
    "Qwen2.5VL-7B": "benchmark_results_EQUALITYqwen2.5vl_7b.csv"
}

# Columns we care about
power_columns = ["avg_tot_w", "max_tot_w", "avg_cpu_gpu_w", "max_cpu_gpu_w"]

plt.style.use("seaborn-darkgrid")

for col in power_columns:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (label, filename) in zip(axes, files.items()):
        file_path = os.path.join(base_path, filename)
        df = pd.read_csv(file_path)

        if col not in df.columns:
            raise ValueError(f"File {filename} does not contain '{col}' column.")

        iterations = range(1, len(df) + 1)
        smoothed = df[col].rolling(window=10, min_periods=1).mean()

        ax.plot(iterations, smoothed, linewidth=2.5, color="tab:blue")

        if col == "avg_tot_w":
            median_val = df[col].median()
            ax.axhline(y=median_val, color="gray", linestyle="--", alpha=0.5)  # optional visual reference
            ax.legend([f"Median = {median_val:.2f} W"], loc="upper right", frameon=True)

        ax.set_title(label, fontsize=12, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

    # Add shared labels
    fig.suptitle(f"Benchmark Comparison for Questions of Type 'EQUALITY': {col}", fontsize=16, weight="bold")
    fig.text(0.5, 0.04, "Iteration", ha="center", fontsize=14)
    fig.text(0.04, 0.5, "Power (Watts)", va="center", rotation="vertical", fontsize=14)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    save_path = os.path.join(save_dir, f"benchmark_{col}_subplots.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

    plt.show()

