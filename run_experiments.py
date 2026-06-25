"""
run_experiments.py — Multi-configuration controlled experiment sweep for Phi-2 QLoRA.

Runs four targeted experiments that cover the key axes of parameter-efficiency
tradeoffs: LoRA rank, learning-rate schedule, and peak learning rate.

Results land in ./experiments/<run_id>/ — each directory contains:
  - adapter_model.safetensors
  - experiment_config.json  (hyperparameters + final eval loss)
  - trainer_state.json      (full step-level loss/lr/grad_norm history)
  - runs/                   (TensorBoard event files)

After completion, run  generate_plots.py  to visualise all runs together.

Usage:
    python run_experiments.py

    # To run only specific experiments (by zero-indexed position):
    python run_experiments.py --only 0 2
"""

import argparse
import subprocess
import sys
import os
import json
import time

EXPERIMENTS = [
    {
        "id":            "run_rank8_cosine",
        "description":   "Low-rank (r=8) — cosine schedule — baseline LR",
        "args": {
            "--rank":          "8",
            "--lora_alpha":    "16",
            "--learning_rate": "2e-4",
            "--scheduler":     "cosine",
            "--optimizer":     "paged_adamw_8bit",
        },
    },
    {
        "id":            "run_rank16_cosine",
        "description":   "Standard rank (r=16) — cosine schedule — baseline LR  [matches original run]",
        "args": {
            "--rank":          "16",
            "--lora_alpha":    "32",
            "--learning_rate": "2e-4",
            "--scheduler":     "cosine",
            "--optimizer":     "paged_adamw_8bit",
        },
    },
    {
        "id":            "run_rank16_linear",
        "description":   "Standard rank (r=16) — linear schedule — baseline LR",
        "args": {
            "--rank":          "16",
            "--lora_alpha":    "32",
            "--learning_rate": "2e-4",
            "--scheduler":     "linear",
            "--optimizer":     "paged_adamw_8bit",
        },
    },
    {
        "id":            "run_rank16_low_lr",
        "description":   "Standard rank (r=16) — cosine schedule — low LR (5e-5)",
        "args": {
            "--rank":          "16",
            "--lora_alpha":    "32",
            "--learning_rate": "5e-5",
            "--scheduler":     "cosine",
            "--optimizer":     "paged_adamw_8bit",
        },
    },
]

SHARED_ARGS = {
    "--epochs":           "2",
    "--num_samples":      "2000",
    "--eval_ratio":       "0.1",
    "--batch_size":       "1",
    "--grad_accum_steps": "8",
    "--logging_steps":    "25",
    "--save_steps":       "200",
    "--eval_steps":       "100",
    "--max_seq_length":   "512",
}


def build_cmd(exp: dict) -> list[str]:
    output_dir = os.path.join("experiments", exp["id"])
    cmd = [sys.executable, "train.py", "--output_dir", output_dir]
    for k, v in {**SHARED_ARGS, **exp["args"]}.items():
        cmd.extend([k, v])
    return cmd


def summarize_results(experiments_run: list[str]) -> None:
    """Print a summary table of eval losses from each completed run."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Run ID':<30} {'Rank':>5} {'LR':>8} {'Scheduler':>10} {'Eval Loss':>10}")
    print("-" * 70)
    for exp in EXPERIMENTS:
        if exp["id"] not in experiments_run:
            continue
        config_path = os.path.join("experiments", exp["id"], "experiment_config.json")
        if not os.path.exists(config_path):
            print(f"{exp['id']:<30} {'N/A':>35}")
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        eval_loss = cfg.get("final_eval_loss", "N/A")
        if isinstance(eval_loss, float):
            eval_loss = f"{eval_loss:.4f}"
        rank      = cfg.get("rank", "?")
        lr        = cfg.get("learning_rate", "?")
        scheduler = cfg.get("scheduler", "?")
        print(f"{exp['id']:<30} {rank:>5} {str(lr):>8} {scheduler:>10} {eval_loss:>10}")
    print("=" * 70)
    print("\nRun  python generate_plots.py  to generate training-dynamics plots.")
    print("Run  python eval_model.py --adapter_dir <run_dir>  to evaluate any run.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="*",
                        help="Zero-indexed experiment indices to run (e.g. --only 0 2).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments whose output directory already exists.")
    args = parser.parse_args()

    indices = args.only if args.only is not None else list(range(len(EXPERIMENTS)))
    experiments_run = []

    os.makedirs("experiments", exist_ok=True)

    for idx in indices:
        exp = EXPERIMENTS[idx]
        output_dir = os.path.join("experiments", exp["id"])

        if args.skip_existing and os.path.exists(output_dir):
            print(f"\n[Skip] {exp['id']} — output directory already exists.")
            experiments_run.append(exp["id"])
            continue

        print("\n" + "=" * 70)
        print(f"[{idx + 1}/{len(indices)}] Starting: {exp['id']}")
        print(f"  Description : {exp['description']}")
        print(f"  Output dir  : {output_dir}")
        print(f"  Config      : {exp['args']}")
        print("=" * 70)

        cmd = build_cmd(exp)
        start = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n[ERROR] Experiment {exp['id']} failed with code {result.returncode}.")
            print("Continuing with remaining experiments...\n")
        else:
            print(f"\n[Done] {exp['id']} finished in {elapsed / 60:.1f} min.")
            experiments_run.append(exp["id"])

    summarize_results(experiments_run)


if __name__ == "__main__":
    main()
