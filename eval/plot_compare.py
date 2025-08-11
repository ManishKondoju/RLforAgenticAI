# eval/plot_compare.py
import pandas as pd
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "plots"
CSV_FILE = os.path.join(PLOTS_DIR, "compare_all.csv")

ORDER = ["Baseline", "DQN", "DQN+Bandit", "DQN+Bandit+PPO"]  # desired order if present

def bar_with_labels(x, y, ylabel, title, outpath):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, y)
    for b, v in zip(bars, y):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    if not os.path.isfile(CSV_FILE):
        raise FileNotFoundError(f"Could not find {CSV_FILE}. Run `python -m eval.compare_all` first.")

    df = pd.read_csv(CSV_FILE)

    # Expect columns from compare_all.py: policy, avg_reward, resolved, violations
    required = {"policy", "avg_reward", "resolved", "violations"}
    if not required.issubset(df.columns):
        raise ValueError(f"{CSV_FILE} must have columns: {', '.join(sorted(required))}")

    # Optional: enforce display order
    df["policy"] = pd.Categorical(df["policy"], categories=[p for p in ORDER if p in df["policy"].unique()], ordered=True)
    df = df.sort_values("policy")

    # Plot 1: Average Reward
    bar_with_labels(
        df["policy"], df["avg_reward"],
        ylabel="Average Reward",
        title="Average Reward by Variant",
        outpath=os.path.join(PLOTS_DIR, "avg_reward_plot.png"),
    )

    # Plot 2: Resolved
    bar_with_labels(
        df["policy"], df["resolved"],
        ylabel="Tasks Resolved per Episode",
        title="Tasks Resolved",
        outpath=os.path.join(PLOTS_DIR, "resolved_plot.png"),
    )

    # Plot 3: Violations
    bar_with_labels(
        df["policy"], df["violations"],
        ylabel="Deadline Violations per Episode",
        title="Deadline Violations",
        outpath=os.path.join(PLOTS_DIR, "violations_plot.png"),
    )

    print("✅ Saved plots:",
          "plots/avg_reward_plot.png, plots/resolved_plot.png, plots/violations_plot.png")

if __name__ == "__main__":
    main()
