"""Save episode reward and length charts from Stable-Baselines3 Monitor logs."""
import argparse
import csv
from pathlib import Path
from training_paths import LOG_DIR, CHART_PATH

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "platformer-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training(log_path=LOG_DIR / "episodes.monitor.csv",
                  output_path=CHART_PATH, step_unit="ticks"):
    with open(log_path, newline="") as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
    rewards = np.asarray([float(row["r"]) for row in rows])
    lengths = np.asarray([int(row["l"]) for row in rows])
    steps = np.cumsum(lengths)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, rewards, alpha=0.5, label="Episode reward")
    if len(rows) >= 10:
        axes[0].plot(steps[9:], np.convolve(rewards, np.ones(10) / 10, "valid"),
                     label="10-episode average")
    if not rows:
        axes[0].text(0.5, 0.5, "No completed episodes yet",
                     ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_ylabel("Total reward")
    axes[0].legend()
    axes[1].plot(steps, lengths)
    axes[1].set_ylabel(f"Episode length ({step_unit})")
    axes[1].set_xlabel("Training steps (completed episodes)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Platformer DQN training")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=180)
    finally:
        plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=str(LOG_DIR / "episodes.monitor.csv"))
    parser.add_argument("--output", default=str(CHART_PATH))
    args = parser.parse_args()
    print(f"Chart saved to {plot_training(args.log, args.output)}")
