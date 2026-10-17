"""
run_phase3_pipeline.py -- End-to-end Phase 3 pipeline orchestrator.

Executes the full Phase 3 workflow in order:
  1. Validate 50-task benchmark (OOD check against train split)
  2. Evaluate Raw Phi-2 baseline on 50-task benchmark (calibration check)
  3. Train Config A (attn-only, 250 steps)
  4. Train Config B (full-attn, 250 steps)
  5. Train Config C (all-linear, 250 steps, OOM fallback r=8)
  6. Train Config D (masking ablation clean re-run, ~4-5h)
  7. Run full evaluation (Sections 3A-3D) on all 6 models
  8. Apply Phase 4 decision rules
  9. Generate reports/phase3_project_report.md

Usage:
  python run_phase3_pipeline.py                    # Full pipeline
  python run_phase3_pipeline.py --skip_training    # Eval only (if models already trained)
  python run_phase3_pipeline.py --only_train a,b   # Train specific configs
  python run_phase3_pipeline.py --only_eval        # Evaluation only
  python run_phase3_pipeline.py --start_from b     # Resume from a specific config
"""

import os
import sys
import json
import time
import datetime
import subprocess
import argparse
import traceback

PIPELINE_LOG_PATH = "reports/phase3_pipeline_log.json"
MASTER_EVAL_PATH = "reports/phase3_master_eval_results.json"
TIMING_LOG_PATH = "reports/phase3_masking_ablation_timing.json"
REPORT_PATH = "reports/phase3_project_report.md"

# Sequential training order
TRAINING_ORDER = ["config_a", "config_b", "config_c", "config_d"]

TRAINING_CONFIG_LABELS = {
    "config_a": "Config A: Attention Only (Baseline Replication)",
    "config_b": "Config B: Full Attention (q/k/v/dense)",
    "config_c": "Config C: All Linear Layers (q/k/v/dense/fc1/fc2)",
    "config_d": "Config D: Masking Ablation Clean Re-run (full-sequence loss)",
}

EXPECTED_OUTPUT_DIRS = {
    "config_a": "phi2-qlora-phase3-attn-only",
    "config_b": "phi2-qlora-phase3-full-attn",
    "config_c": "phi2-qlora-phase3-all-linear",
    "config_d": "phi2-qlora-phase3-masking-ablation-clean",
}


def log_event(log: list, level: str, msg: str):
    ts = datetime.datetime.utcnow().isoformat()
    entry = {"timestamp_utc": ts, "level": level, "message": msg}
    log.append(entry)
    prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "ERROR": "[FAIL]", "START": "[RUN ]", "DONE": "[DONE]"}.get(level, "[....]")
    print(f"[{ts[:19]}] {prefix} {msg}", flush=True)


def save_log(log: list):
    os.makedirs("reports", exist_ok=True)
    with open(PIPELINE_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def check_adapter_exists(config_key: str) -> bool:
    outdir = EXPECTED_OUTPUT_DIRS[config_key]
    adapter_path = os.path.join(outdir, "adapter_model.safetensors")
    return os.path.exists(adapter_path)


def set_system_sleep_prevention():
    """Attempt to prevent system sleep on Windows using powercfg."""
    try:
        # Request SYSTEM power prevention (no sleep while this process runs)
        subprocess.run(
            ["powercfg", "/requestsoverride", "PROCESS", "python.exe", "SYSTEM"],
            capture_output=True, timeout=10,
        )
        print("[Sleep] Power override set: system will not sleep during training.")
    except Exception as e:
        print(f"[Sleep] Could not set power override (non-critical): {e}")


def release_system_sleep_prevention():
    try:
        subprocess.run(
            ["powercfg", "/requestsoverride", "PROCESS", "python.exe"],
            capture_output=True, timeout=10,
        )
        print("[Sleep] Power override released.")
    except Exception:
        pass


def run_training(config_key: str, log: list) -> bool:
    """Run train_phase3.py for the given config. Returns True on success."""
    label = TRAINING_CONFIG_LABELS[config_key]
    log_event(log, "START", f"Training {config_key}: {label}")
    save_log(log)

    cmd = [sys.executable, "train_phase3.py", "--config_name", config_key]
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            timeout=None,   # No timeout -- Config D can take 5+ hours
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            log_event(log, "DONE", f"{config_key} training complete in {elapsed:.0f}s ({elapsed/3600:.2f}h)")
            save_log(log)
            return True
        else:
            log_event(log, "ERROR", f"{config_key} training exited with code {result.returncode} after {elapsed:.0f}s")
            save_log(log)
            return False
    except Exception as e:
        elapsed = time.time() - start
        log_event(log, "ERROR", f"{config_key} training exception after {elapsed:.0f}s: {e}")
        traceback.print_exc()
        save_log(log)
        return False


def run_benchmark_ood_validation(log: list) -> bool:
    """Validate that all 50 benchmark tasks are strictly out-of-distribution vs train split."""
    log_event(log, "START", "Validating 50-task benchmark OOD status vs train split")
    save_log(log)

    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        from datasets import load_dataset
        with open("reports/phase3_benchmark_50tasks.json", "r", encoding="utf-8") as f:
            bench = json.load(f)

        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")
        train_split = ds.train_test_split(test_size=0.1, seed=42)["train"]

        train_instructions = [ex["instruction"].strip().lower() for ex in train_split]

        overlaps = []
        for task in bench:
            instr = task["instruction"].strip().lower()
            for idx, t_inst in enumerate(train_instructions):
                if instr == t_inst or (len(instr) > 30 and instr in t_inst):
                    overlaps.append({"task_id": task["id"], "instruction": task["instruction"], "train_sample_idx": idx})

        ood_report = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "benchmark_task_count": len(bench),
            "train_samples_checked": len(train_split),
            "exact_or_substantial_overlaps": len(overlaps),
            "is_strictly_ood": len(overlaps) == 0,
            "overlaps": overlaps,
        }
        with open("reports/phase3_benchmark_ood_verification.json", "w", encoding="utf-8") as f:
            json.dump(ood_report, f, indent=2)

        if len(overlaps) == 0:
            log_event(log, "DONE", f"OOD Check PASSED: All {len(bench)} tasks are 100% out-of-distribution vs {len(train_split)} train samples")
            save_log(log)
            return True
        else:
            log_event(log, "WARN", f"OOD Check: Found {len(overlaps)} overlaps")
            save_log(log)
            return False
    except Exception as e:
        log_event(log, "ERROR", f"OOD validation error: {e}")
        save_log(log)
        return False


def run_baseline_phi2_eval(log: list) -> dict:
    """Run 50-task benchmark on raw Phi-2 to calibrate difficulty."""
    log_event(log, "START", "Baseline Phi-2 50-task evaluation (calibration check)")
    save_log(log)

    cmd = [
        sys.executable, "eval_phase3.py",
        "--models", "raw_phi2",
        "--benchmark_only",
    ]
    try:
        result = subprocess.run(cmd, timeout=3600, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            log_event(log, "WARN", "Baseline Phi-2 eval returned non-zero exit code")

        # Read partial results
        if os.path.exists(MASTER_EVAL_PATH):
            with open(MASTER_EVAL_PATH) as f:
                data = json.load(f)
            phi2_data = data.get("models", {}).get("raw_phi2", {})
            greedy = phi2_data.get("benchmark_greedy", {})
            pass_rate = greedy.get("core_functional_pass_rate", None)
            if pass_rate is not None:
                log_event(log, "INFO", f"Raw Phi-2 50-task pass rate: {pass_rate*100:.1f}%")
                if pass_rate >= 0.80:
                    log_event(log, "WARN", "Raw Phi-2 passes >= 80% of tasks -- benchmark may be too easy (Section 6 rule applies)")
                elif pass_rate <= 0.50:
                    log_event(log, "INFO", "Raw Phi-2 passes <= 50% of tasks -- benchmark is well-calibrated, proceed as planned")
                return {"pass_rate": pass_rate, "benchmark_calibration": "well_calibrated" if pass_rate <= 0.50 else "too_easy"}
    except Exception as e:
        log_event(log, "ERROR", f"Baseline eval failed: {e}")

    save_log(log)
    return {}


def run_full_evaluation(log: list, models_to_eval: list = None) -> bool:
    """Run complete 6-model evaluation."""
    log_event(log, "START", "Full 6-model evaluation (Sections 3A-3D)")
    save_log(log)

    models_str = ",".join(models_to_eval) if models_to_eval else "raw_phi2,phase2_corrected,config_a,config_b,config_c,config_d"
    cmd = [
        sys.executable, "eval_phase3.py",
        "--models", models_str,
    ]

    try:
        result = subprocess.run(cmd, timeout=None, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode == 0:
            log_event(log, "DONE", "Full evaluation complete")
        else:
            log_event(log, "WARN", f"Evaluation returned exit code {result.returncode}")
    except Exception as e:
        log_event(log, "ERROR", f"Evaluation failed: {e}")
        traceback.print_exc()
        save_log(log)
        return False

    save_log(log)
    return True


def apply_phase4_decision_rules(log: list) -> dict:
    """Load eval results and apply the Section 6 decision logic."""
    log_event(log, "START", "Applying Phase 4 decision rules")

    if not os.path.exists(MASTER_EVAL_PATH):
        log_event(log, "ERROR", f"Master eval results not found at {MASTER_EVAL_PATH}")
        return {}

    with open(MASTER_EVAL_PATH) as f:
        data = json.load(f)

    models = data.get("models", {})

    def get_pass_rate(key):
        m = models.get(key, {})
        return m.get("benchmark_greedy", {}).get("core_functional_pass_rate", None)

    pass_a = get_pass_rate("config_a")
    pass_b = get_pass_rate("config_b")
    pass_c = get_pass_rate("config_c")
    pass_phi2 = get_pass_rate("raw_phi2")

    # ?? LoRA target expansion decision ????????????????????????????????????
    lora_decision = {}
    if pass_a is not None and pass_c is not None:
        delta_c_vs_a = pass_c - pass_a
        delta_b_vs_a = (pass_b - pass_a) if pass_b is not None else None

        if delta_c_vs_a >= 0.05:
            lora_decision = {
                "recommendation": "adopt_all_linear",
                "phase4_action": "Rank sweep (r=8, r=32) on all-linear config",
                "delta_c_vs_a": round(delta_c_vs_a, 4),
                "reason": f"Config C improves 50-task pass rate by {delta_c_vs_a*100:.1f}% over Config A (>= 5% threshold)",
            }
        elif delta_b_vs_a is not None and delta_b_vs_a >= 0.05:
            lora_decision = {
                "recommendation": "adopt_full_attention",
                "phase4_action": "Rank sweep (r=8, r=32) on full-attention config",
                "delta_b_vs_a": round(delta_b_vs_a, 4),
                "reason": f"Config B improves 50-task pass rate by {delta_b_vs_a*100:.1f}% over Config A (>= 5% threshold)",
            }
        else:
            lora_decision = {
                "recommendation": "dataset_scaling",
                "phase4_action": "Scale dataset from 2,000 to 10,000 samples with q_proj,v_proj baseline",
                "delta_c_vs_a": round(delta_c_vs_a, 4) if pass_c is not None else "N/A",
                "reason": "No meaningful gain from LoRA target expansion at this dataset scale",
            }
    else:
        lora_decision = {"recommendation": "insufficient_data", "reason": "One or more config pass rates unavailable"}

    # ?? Masking timing decision ???????????????????????????????????????????
    masking_decision = {}
    timing_data = {}
    if os.path.exists(TIMING_LOG_PATH):
        with open(TIMING_LOG_PATH) as f:
            timing_data = json.load(f)

    speedup_info = timing_data.get("speedup_comparison", {})
    a_time = timing_data.get("config_a", {}).get("wall_clock_seconds")
    d_time = timing_data.get("config_d", {}).get("wall_clock_seconds")

    if a_time and d_time:
        ratio = d_time / a_time
        if ratio >= 1.30:
            masking_decision = {
                "recommendation": "always_use_masking",
                "speedup_ratio": round(ratio, 3),
                "reason": f"Config D (no masking) took {ratio:.2f}x longer than Config A (masking) -- >= 30% penalty",
            }
        else:
            # Compare response-only loss to decide
            resp_a = models.get("config_a", {}).get("held_out_loss", {}).get("response_only_eval_loss")
            resp_d = models.get("config_d", {}).get("held_out_loss", {}).get("response_only_eval_loss")
            if resp_a is not None and resp_d is not None:
                better = "masking" if resp_a <= resp_d else "no_masking"
                masking_decision = {
                    "recommendation": f"use_{better}_based_on_loss",
                    "speedup_ratio": round(ratio, 3),
                    "config_a_resp_loss": resp_a,
                    "config_d_resp_loss": resp_d,
                    "reason": f"Masking neutral on this hardware (ratio={ratio:.2f} < 1.30); selecting based on response loss",
                }
            else:
                masking_decision = {
                    "recommendation": "use_masking_default",
                    "speedup_ratio": round(ratio, 3) if ratio else "N/A",
                    "reason": "Timing and loss data incomplete -- defaulting to masking",
                }
    else:
        masking_decision = {"recommendation": "use_masking_default", "reason": "Timing data unavailable"}

    # ?? Benchmark calibration decision ????????????????????????????????????
    benchmark_decision = {}
    if pass_phi2 is not None:
        if pass_phi2 >= 0.80:
            benchmark_decision = {
                "calibration": "too_easy",
                "raw_phi2_pass_rate": pass_phi2,
                "action": "Add 25 harder tasks (DP, graph algorithms, class hierarchies) before evaluating trained models",
            }
        elif pass_phi2 <= 0.50:
            benchmark_decision = {
                "calibration": "well_calibrated",
                "raw_phi2_pass_rate": pass_phi2,
                "action": "Benchmark is well-calibrated; results are valid as-is",
            }
        else:
            benchmark_decision = {
                "calibration": "marginal",
                "raw_phi2_pass_rate": pass_phi2,
                "action": "Benchmark is marginally calibrated (51-79% raw pass rate); results valid",
            }

    decision_output = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "lora_target_expansion": lora_decision,
        "masking_timing": masking_decision,
        "benchmark_calibration": benchmark_decision,
        "phase4_recommendation": lora_decision.get("phase4_action", "See individual decisions"),
    }

    # Save decisions to master results
    with open(MASTER_EVAL_PATH, "r") as f:
        master = json.load(f)
    master["phase4_decisions"] = decision_output
    with open(MASTER_EVAL_PATH, "w") as f:
        json.dump(master, f, indent=2)

    log_event(log, "DONE", f"Phase 4 decision: {lora_decision.get('recommendation', 'N/A')}")
    log_event(log, "DONE", f"Masking decision: {masking_decision.get('recommendation', 'N/A')}")
    save_log(log)
    return decision_output


def generate_report(log: list):
    """Run generate_phase3_report.py to produce the final markdown report."""
    log_event(log, "START", "Generating Phase 3 project report")
    save_log(log)

    try:
        result = subprocess.run(
            [sys.executable, "generate_phase3_report.py"],
            timeout=120,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            log_event(log, "DONE", f"Report saved to {REPORT_PATH}")
        else:
            log_event(log, "WARN", "Report generator returned non-zero exit code")
    except Exception as e:
        log_event(log, "ERROR", f"Report generation failed: {e}")

    save_log(log)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 pipeline orchestrator")
    parser.add_argument("--skip_training", action="store_true", help="Skip all training, run evaluation only")
    parser.add_argument("--only_train", type=str, default=None,
                        help="Train only specific configs (e.g., 'a,b')")
    parser.add_argument("--only_eval", action="store_true", help="Only run evaluation (no training, no report)")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--start_from", type=str, default=None,
                        help="Start training from a specific config (e.g., 'b' or 'config_b')")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip raw Phi-2 baseline evaluation")
    parser.add_argument("--skip_report", action="store_true", help="Skip report generation")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("reports", exist_ok=True)

    log = []
    pipeline_start = datetime.datetime.utcnow()
    log_event(log, "START", f"Phase 3 Pipeline started at {pipeline_start.isoformat()}")
    save_log(log)

    # Determine which configs to train
    configs_to_train = []
    if not args.skip_training and not args.only_eval:
        if args.only_train:
            specified = [c.strip() for c in args.only_train.split(",")]
            configs_to_train = [f"config_{c}" if not c.startswith("config_") else c for c in specified]
        else:
            configs_to_train = list(TRAINING_ORDER)

        # Apply --start_from
        if args.start_from:
            start_key = args.start_from if args.start_from.startswith("config_") else f"config_{args.start_from}"
            if start_key in configs_to_train:
                idx = configs_to_train.index(start_key)
                skipped = configs_to_train[:idx]
                configs_to_train = configs_to_train[idx:]
                log_event(log, "INFO", f"Resuming from {start_key}. Skipping: {skipped}")

        # Skip configs already trained (adapter exists)
        configs_already_done = [c for c in configs_to_train if check_adapter_exists(c)]
        if configs_already_done:
            log_event(log, "INFO", f"Adapters already exist for: {configs_already_done} -- skipping")
            configs_to_train = [c for c in configs_to_train if not check_adapter_exists(c)]

    # Set sleep prevention for Config D (long run)
    if "config_d" in configs_to_train:
        set_system_sleep_prevention()

    # ?? Step 0: Validate Benchmark OOD Status ????????????????????????????
    ood_ok = run_benchmark_ood_validation(log)
    if not ood_ok:
        log_event(log, "WARN", "Benchmark OOD validation flagged warnings or errors, proceeding with caution")

    # ?? Step 1: Baseline Phi-2 eval ???????????????????????????????????????
    if not args.skip_baseline and not args.only_eval and not args.skip_training:
        baseline_result = run_baseline_phi2_eval(log)
    else:
        baseline_result = {}

    # ?? Step 2: Training ??????????????????????????????????????????????????
    training_results = {}
    for config_key in configs_to_train:
        success = run_training(config_key, log)
        training_results[config_key] = "success" if success else "failed"

        if not success and config_key in ["config_a", "config_d"]:
            log_event(log, "ERROR", f"Critical config {config_key} failed -- halting training sequence")
            break

    if "config_d" in configs_to_train:
        release_system_sleep_prevention()

    # ?? Step 3: Evaluation ????????????????????????????????????????????????
    if not args.skip_eval and not (args.only_train and not args.only_eval):
        # Determine which models have valid adapters to evaluate
        eval_models = []
        for mk, adir in [
            ("raw_phi2", None),
            ("phase2_corrected", "phi2-qlora-phase2-corrected"),
            ("config_a", "phi2-qlora-phase3-attn-only"),
            ("config_b", "phi2-qlora-phase3-full-attn"),
            ("config_c", "phi2-qlora-phase3-all-linear"),
            ("config_d", "phi2-qlora-phase3-masking-ablation-clean"),
        ]:
            if adir is None or os.path.isdir(adir):
                eval_models.append(mk)

        log_event(log, "INFO", f"Evaluating models: {eval_models}")
        run_full_evaluation(log, eval_models)

        # ?? Step 4: Decision rules ????????????????????????????????????????
        decisions = apply_phase4_decision_rules(log)

        # ?? Step 5: Report ????????????????????????????????????????????????
        if not args.skip_report:
            generate_report(log)

    pipeline_end = datetime.datetime.utcnow()
    total_s = (pipeline_end - pipeline_start).total_seconds()
    log_event(log, "DONE", f"Phase 3 Pipeline complete. Total time: {total_s:.0f}s ({total_s/3600:.2f}h)")
    save_log(log)

    print(f"\n{'='*70}")
    print("PHASE 3 PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:    {total_s:.0f}s ({total_s/3600:.2f}h)")
    print(f"  Training:      {training_results}")
    print(f"  Master eval:   {MASTER_EVAL_PATH}")
    print(f"  Report:        {REPORT_PATH}")
    print(f"  Pipeline log:  {PIPELINE_LOG_PATH}")


if __name__ == "__main__":
    main()
