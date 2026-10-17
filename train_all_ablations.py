"""
train_all_ablations.py — Helper to run Ablation B and Ablation C sequentially.
Checks if an ablation has already finished (via phase2b_train_summary.json) before running.
"""

import os
import sys
import subprocess

def run_ablation(ablation_name):
    out_dir = f"./phi2-qlora-phase2b-{ablation_name.replace('_', '-')}"
    summary_file = os.path.join(out_dir, "phase2b_train_summary.json")
    if os.path.exists(summary_file):
        print(f"[{ablation_name}] Already completed at {out_dir}. Skipping.")
        return True

    print("\n" + "=" * 80)
    print(f"LAUNCHING TRAINING FOR: {ablation_name}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "-u",
        "train_phase2b_ablation.py",
        "--ablation_name", ablation_name,
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[{ablation_name}] Training failed with return code {res.returncode}")
        return False
    print(f"[{ablation_name}] Training finished successfully.")
    return True

def main():
    ablations = ["masking_ablation", "input_ablation"]
    for ab in ablations:
        success = run_ablation(ab)
        if not success:
            sys.exit(1)
    print("\n[All Remaining Ablations Finished]")

if __name__ == "__main__":
    main()
