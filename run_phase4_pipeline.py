"""
run_phase4_pipeline.py -- Phase 4 end-to-end pipeline orchestrator.

Executes in order:
  Step 1: Section 1 — Data category audit (CPU-only)
  Step 2: Section 2 — Config E training (GPU, ~1.5-2.5h)
  Step 3: Section 3A — Checkpoint curve greedy eval (GPU)
  Step 4: Section 3B — Best-checkpoint full eval (GPU)
  Step 5: Section 4 — Phase 4 report generation

All Phase 3 artifacts are REUSED without recomputation.
Reuse decisions are logged to reports/phase4_pipeline_log.json.
"""

import os
import sys
import json
import time
import datetime
import subprocess
import argparse
import traceback

PIPELINE_LOG_PATH = "reports/phase4_pipeline_log.json"
CONFIG_E_DIR      = "phi2-qlora-phase4-scaled-5k"
AUDIT_PATH        = "reports/phase4_category_audit.json"
CURVE_PATH        = "reports/phase4_config_e_checkpoint_curve.json"
MASTER_EVAL_PATH  = "reports/phase4_master_eval_results.json"
REPORT_PATH       = "reports/phase4_project_report.md"


def log_event(level: str, message: str):
    ts = datetime.datetime.utcnow().isoformat()
    entry = {"timestamp_utc": ts, "level": level, "message": message}
    os.makedirs("reports", exist_ok=True)
    existing = []
    if os.path.exists(PIPELINE_LOG_PATH):
        try:
            with open(PIPELINE_LOG_PATH) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(entry)
    with open(PIPELINE_LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    prefix = {
        "INFO": "[INFO]", "DONE": "[DONE]", "START": "[RUN ]",
        "ERROR": "[FAIL]", "WARN": "[WARN]", "REUSE": "[REUSE]"
    }.get(level, "[....]")
    print(f"[{ts[:19]}] {prefix} {message}", flush=True)


def run_step(name: str, cmd: list) -> bool:
    log_event("START", f"{name}")
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        elapsed = time.time() - start
        if result.returncode == 0:
            log_event("DONE", f"{name} completed in {elapsed:.0f}s ({elapsed/3600:.2f}h)")
            return True
        else:
            log_event("ERROR", f"{name} exited with code {result.returncode} after {elapsed:.0f}s")
            return False
    except Exception as e:
        elapsed = time.time() - start
        log_event("ERROR", f"{name} exception after {elapsed:.0f}s: {e}")
        traceback.print_exc()
        return False


def adapter_exists() -> bool:
    return os.path.exists(os.path.join(CONFIG_E_DIR, "adapter_model.safetensors"))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 pipeline orchestrator")
    parser.add_argument("--skip_audit",    action="store_true", help="Skip Section 1 category audit")
    parser.add_argument("--skip_train",    action="store_true", help="Skip Config E training (if adapter already exists)")
    parser.add_argument("--skip_curve",    action="store_true", help="Skip Phase A checkpoint curve")
    parser.add_argument("--skip_full_eval",action="store_true", help="Skip Phase B full eval")
    parser.add_argument("--skip_report",   action="store_true", help="Skip report generation")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("reports", exist_ok=True)

    pipeline_start = datetime.datetime.utcnow()
    log_event("START", f"Phase 4 Pipeline started at {pipeline_start.isoformat()}")

    # ── Step 1: Category Audit ──────────────────────────────────────────────
    if not args.skip_audit:
        if os.path.exists(AUDIT_PATH):
            log_event("REUSE", f"SOURCE={AUDIT_PATH} | Category audit already exists — skipping recomputation")
        else:
            ok = run_step("Section 1: Data category audit", [sys.executable, "audit_phase4_categories.py"])
            if not ok:
                log_event("WARN", "Category audit failed — proceeding with plain_sequential sampling")

    # Determine sampling mode from audit result
    balanced_sampling = False
    if os.path.exists(AUDIT_PATH):
        try:
            with open(AUDIT_PATH) as f:
                audit = json.load(f)
            if audit.get("sampling_decision") == "category_balanced":
                balanced_sampling = True
                log_event("INFO", "Sampling decision: category_balanced (File I/O >= 300 threshold)")
            else:
                log_event("INFO", "Sampling decision: plain_sequential (File I/O < 300 threshold)")
        except Exception as e:
            log_event("WARN", f"Could not read audit result: {e}. Defaulting to plain_sequential.")

    # ── Step 2: Training ────────────────────────────────────────────────────
    if not args.skip_train:
        if adapter_exists():
            log_event("REUSE", f"SOURCE={CONFIG_E_DIR}/adapter_model.safetensors | "
                      "Config E adapter already exists — skipping retraining")
        else:
            train_cmd = [sys.executable, "train_phase4.py"]
            if balanced_sampling:
                train_cmd.append("--balanced_sampling")
            ok = run_step("Section 2: Config E training", train_cmd)
            if not ok:
                log_event("ERROR", "Config E training failed — aborting pipeline")
                return

    # ── Step 3A: Checkpoint curve ───────────────────────────────────────────
    if not args.skip_curve:
        if os.path.exists(CURVE_PATH):
            log_event("REUSE", f"SOURCE={CURVE_PATH} | Checkpoint curve already exists — skipping recomputation")
        else:
            ok = run_step("Section 3A: Config E checkpoint curve greedy eval",
                          [sys.executable, "eval_phase4.py", "--phase_a_only"])
            if not ok:
                log_event("WARN", "Checkpoint curve eval had errors — check individual checkpoint results")

    # ── Step 3B: Full eval on best checkpoint ───────────────────────────────
    if not args.skip_full_eval:
        if os.path.exists(MASTER_EVAL_PATH):
            log_event("REUSE", f"SOURCE={MASTER_EVAL_PATH} | Master eval already exists — skipping recomputation")
        else:
            ok = run_step("Section 3B: Config E full eval (best checkpoint only)",
                          [sys.executable, "eval_phase4.py", "--phase_b_only"])
            if not ok:
                log_event("WARN", "Full eval had errors — check master eval JSON")

    # ── Step 4: Report ──────────────────────────────────────────────────────
    if not args.skip_report:
        ok = run_step("Section 4: Phase 4 report generation",
                      [sys.executable, "generate_phase4_report.py"])
        if not ok:
            log_event("WARN", "Report generation had errors")

    # ── Done ────────────────────────────────────────────────────────────────
    pipeline_end = datetime.datetime.utcnow()
    total_s = (pipeline_end - pipeline_start).total_seconds()
    log_event("DONE", f"Phase 4 Pipeline complete. Total time: {total_s:.0f}s ({total_s/3600:.2f}h)")

    print(f"\n{'='*70}")
    print("PHASE 4 PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:   {total_s:.0f}s ({total_s/3600:.2f}h)")
    print(f"  Audit:        {AUDIT_PATH}")
    print(f"  Adapter:      {CONFIG_E_DIR}/adapter_model.safetensors")
    print(f"  Curve:        {CURVE_PATH}")
    print(f"  Master eval:  {MASTER_EVAL_PATH}")
    print(f"  Report:       {REPORT_PATH}")
    print(f"  Pipeline log: {PIPELINE_LOG_PATH}")


if __name__ == "__main__":
    main()
