import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your folder
base_path = os.path.expanduser("~/Desktop/6344334")
save_path = os.path.expanduser("~/Desktop/benchmark_latency_comparison.png")

# File names
files = {
    "Gemma3-4B": "benchmark_results_gemma3_4b.csv",
    "LLaVA-7B": "benchmark_results_llava_7b.csv",
    "LLaVA-Llama3 Latest": "benchmark_results_llava-llama3_latest.csv",
    "Qwen2.5VL-3B": "benchmark_results_qwen2.5vl:3b.csv"
}

# Use a style that exists in all matplotlib versions
plt.style.use("seaborn-darkgrid")

plt.figure(figsize=(10, 6))

# Markers to help differentiate lines
markers = ["o", "s", "D", "^"]

# Loop through files and plot latency curves
for (label, filename), marker in zip(files.items(), markers):
    file_path = os.path.join(base_path, filename)
    df = pd.read_csv(file_path)
    
    if "latency_sec" not in df.columns:
        raise ValueError(f"File {filename} does not contain 'latency_sec' column.")
    
    iterations = range(1, len(df) + 1)  # 1 through 500
    plt.plot(
        iterations,
        df["latency_sec"],
        label=label,
        linewidth=2,
        marker=marker,
        markersize=4,
        alpha=0.8
    )

# Plot formatting
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Latency Per Prompt (seconds)", fontsize=14)
plt.title("Benchmark Latency Comparison", fontsize=16, weight="bold")

# Legend in top-right, outside plot
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    fontsize=12
)

# Tidy layout for spacing
plt.tight_layout()

# Save the plot
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {save_path}")

# Show plot
plt.show()

