"""
generate_plots.py — Training dynamics visualization for Phi-2 QLoRA experiments.

Reads trainer_state.json from one or more experiment output directories and
produces publication-quality figures saved to ./plots/.

Generated figures:
  1. training_loss.png         — Training loss vs step (all runs overlaid)
  2. grad_norm.png             — Gradient norm vs step
  3. learning_rate.png         — Learning rate schedule per run
  4. eval_loss.png             — Evaluation loss vs step (if available)
  5. final_loss_comparison.png — Bar chart of final training loss per run

Usage:
    # Auto-discover all experiment directories + baseline
    python generate_plots.py

    # Specify directories manually
    python generate_plots.py --dirs phi2-qlora-output experiments/run_rank8_cosine

    # Save to custom directory
    python generate_plots.py --output_dir ./my_plots
"""

import argparse
import json
import os
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─────────────────────────── style ───────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#c8ccd8",
    "axes.titlecolor":   "#e0e4f0",
    "xtick.color":       "#8890a6",
    "ytick.color":       "#8890a6",
    "grid.color":        "#2a2e3d",
    "grid.linewidth":    0.8,
    "text.color":        "#c8ccd8",
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4d",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

PALETTE = [
    "#7c83fd", "#fd7c7c", "#7cfda8", "#fdca7c",
    "#c87cfd", "#7cddfd", "#fd7cc8", "#e0e4f0",
]

# ─────────────────────────── data loading ────────────────────────────────────

def load_run(directory: str) -> dict | None:
    """Load trainer_state.json from a run directory and return a structured dict."""
    state_path = os.path.join(directory, "trainer_state.json")
    config_path = os.path.join(directory, "experiment_config.json")

    if not os.path.exists(state_path):
        # Check inside checkpoint subdirectories (take the latest one)
        checkpoints = sorted(glob.glob(os.path.join(directory, "checkpoint-*")))
        if checkpoints:
            state_path = os.path.join(checkpoints[-1], "trainer_state.json")
        if not os.path.exists(state_path):
            print(f"  [Skip] No trainer_state.json found in {directory}")
            return None

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    # Build a friendly label from config if available
    label = Path(directory).name
    config = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        rank     = config.get("rank", "?")
        lr       = config.get("learning_rate", "?")
        sched    = config.get("scheduler", "?")
        label    = f"r={rank} | lr={lr} | {sched}"

    log_history = state.get("log_history", [])
    train_steps, train_loss, grad_norms, learning_rates = [], [], [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
            grad_norms.append(entry.get("grad_norm", None))
            learning_rates.append(entry.get("learning_rate", None))
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

    return {
        "label":          label,
        "directory":      directory,
        "config":         config,
        "train_steps":    train_steps,
        "train_loss":     train_loss,
        "grad_norms":     grad_norms,
        "learning_rates": learning_rates,
        "eval_steps":     eval_steps,
        "eval_loss":      eval_loss,
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_eval_loss":  eval_loss[-1]  if eval_loss  else config.get("final_eval_loss"),
        "max_steps":        state.get("max_steps"),
    }


def discover_runs(base_dirs: list[str]) -> list[dict]:
    """Discover all valid run directories from a list of base paths."""
    runs = []
    for d in base_dirs:
        if os.path.isdir(d):
            run = load_run(d)
            if run:
                runs.append(run)
    return runs

# ─────────────────────────── plotting helpers ─────────────────────────────────

def save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [Saved] {path}")
    plt.close(fig)


def add_watermark(ax, text="Phi-2 QLoRA | SLM-Finetune"):
    ax.text(
        0.99, 0.02, text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="#5a5f72", style="italic",
    )

# ─────────────────────────── individual plots ─────────────────────────────────

def plot_training_loss(runs: list[dict], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, run in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(run["train_steps"], run["train_loss"],
                label=run["label"], color=color, linewidth=1.8, alpha=0.9)
        # Smoothed trend line
        if len(run["train_loss"]) > 4:
            window = max(3, len(run["train_loss"]) // 6)
            smoothed = np.convolve(
                run["train_loss"], np.ones(window) / window, mode="valid"
            )
            smooth_steps = run["train_steps"][window - 1:]
            ax.plot(smooth_steps, smoothed, color=color,
                    linewidth=2.5, alpha=0.4, linestyle="--")

    ax.set_title("Training Loss vs Step", pad=14)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, which="major")
    add_watermark(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "training_loss.png"))


def plot_grad_norm(runs: list[dict], output_dir: str) -> None:
    has_data = any(any(g is not None for g in r["grad_norms"]) for r in runs)
    if not has_data:
        print("  [Skip] No gradient norm data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, run in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        norms = [(s, g) for s, g in zip(run["train_steps"], run["grad_norms"]) if g is not None]
        if not norms:
            continue
        steps, gnorms = zip(*norms)
        ax.plot(steps, gnorms, label=run["label"], color=color, linewidth=1.6, alpha=0.85)

    ax.set_title("Gradient Norm vs Step", pad=14)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Gradient Norm")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True)
    add_watermark(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "grad_norm.png"))


def plot_learning_rate(runs: list[dict], output_dir: str) -> None:
    has_data = any(any(lr is not None for lr in r["learning_rates"]) for r in runs)
    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, run in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        lrs = [(s, lr) for s, lr in zip(run["train_steps"], run["learning_rates"]) if lr is not None]
        if not lrs:
            continue
        steps, lrs_vals = zip(*lrs)
        ax.plot(steps, lrs_vals, label=run["label"], color=color, linewidth=2, alpha=0.9)

    ax.set_title("Learning Rate Schedule", pad=14)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Learning Rate")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True)
    add_watermark(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "learning_rate.png"))


def plot_eval_loss(runs: list[dict], output_dir: str) -> None:
    runs_with_eval = [r for r in runs if r["eval_steps"]]
    if not runs_with_eval:
        print("  [Skip] No evaluation loss data in any run.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, run in enumerate(runs_with_eval):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(run["eval_steps"], run["eval_loss"],
                label=run["label"], color=color,
                linewidth=2, marker="o", markersize=5, alpha=0.9)

    ax.set_title("Evaluation Loss vs Step", pad=14)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Evaluation Loss")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True)
    add_watermark(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "eval_loss.png"))


def plot_final_comparison(runs: list[dict], output_dir: str) -> None:
    labels, train_losses, eval_losses = [], [], []
    for r in runs:
        if r["final_train_loss"] is not None:
            labels.append(r["label"])
            train_losses.append(r["final_train_loss"])
            eval_losses.append(r["final_eval_loss"])

    if not labels:
        print("  [Skip] No final loss data available for comparison.")
        return

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 5))
    bars1 = ax.bar(x - width / 2, train_losses, width,
                   label="Final Train Loss", color=PALETTE[0], alpha=0.85, zorder=3)
    if any(v is not None for v in eval_losses):
        bars2 = ax.bar(x + width / 2,
                       [v if v is not None else 0 for v in eval_losses],
                       width, label="Final Eval Loss", color=PALETTE[1], alpha=0.85, zorder=3)
        for bar, val in zip(bars2, eval_losses):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#c8ccd8")

    for bar, val in zip(bars1, train_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#c8ccd8")

    ax.set_title("Final Loss Comparison Across Runs", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.85)
    ax.grid(True, axis="y", zorder=0)
    add_watermark(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "final_loss_comparison.png"))

# ─────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate training dynamics plots.")
    parser.add_argument("--dirs", nargs="*", default=None,
                        help="Specific run directories to plot. Auto-discovers if not set.")
    parser.add_argument("--output_dir", type=str, default="./plots",
                        help="Directory to save plot images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dirs:
        search_dirs = args.dirs
    else:
        # Auto-discover: baseline + any experiment subdirectories
        search_dirs = ["./phi2-qlora-output"]
        exp_dirs = sorted(glob.glob("./experiments/*/"))
        search_dirs.extend(exp_dirs)
        print(f"[Discover] Found {len(search_dirs)} candidate directories: {search_dirs}")

    runs = discover_runs(search_dirs)
    if not runs:
        print("[Error] No valid run data found. Make sure trainer_state.json exists in the given directories.")
        return

    print(f"\n[Plot] Generating plots for {len(runs)} run(s)...\n")
    plot_training_loss(runs, args.output_dir)
    plot_grad_norm(runs, args.output_dir)
    plot_learning_rate(runs, args.output_dir)
    plot_eval_loss(runs, args.output_dir)
    plot_final_comparison(runs, args.output_dir)

    print(f"\n[Done] All plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
