"""
eval_phase4.py -- Phase 4 Section 3: Config E Evaluation

Phase A: Checkpoint-curve greedy eval on ALL saved checkpoints of Config E.
         Saves reports/phase4_config_e_checkpoint_curve.json.
         Selects best-pass-rate checkpoint per Locked Decision #3.

Phase B: Full eval (greedy + 3x sampled + held-out loss) on BEST CHECKPOINT ONLY.
         Reuses Phase 3 results for all 6 existing models from phase3_master_eval_results.json.
         Saves reports/phase4_master_eval_results.json.

Fixes the per-task pass/fail matrix rendering bug from Phase 3
(all cells rendered as '?' due to broken template); pulls actual
per-task boolean results from the greedy eval dict.
"""

import os
import sys
import re
import ast
import json
import time
import math
import datetime
import hashlib
import traceback
import subprocess
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["HF_HUB_OFFLINE"] = "1"

# ────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────

BENCHMARK_PATH          = "reports/phase3_benchmark_50tasks.json"
PHASE3_MASTER_PATH      = "reports/phase3_master_eval_results.json"
CHECKPOINT_CURVE_PATH   = "reports/phase4_config_e_checkpoint_curve.json"
MASTER_EVAL_PATH        = "reports/phase4_master_eval_results.json"
PIPELINE_LOG_PATH       = "reports/phase4_pipeline_log.json"
CONFIG_E_DIR            = "phi2-qlora-phase4-scaled-5k"
BASE_MODEL_ID           = "microsoft/phi-2"
DATASET_ID              = "iamtarun/python_code_instructions_18k_alpaca"

MAX_NEW_TOKENS_GREEDY   = 300
MAX_NEW_TOKENS_SAMPLED  = 300
SAMPLED_RUNS            = 3
SEED                    = 42

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────

def log_event(level: str, message: str):
    ts = datetime.datetime.utcnow().isoformat()
    entry = {"timestamp_utc": ts, "level": level, "message": message}
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
    prefix = {"INFO": "[INFO]", "DONE": "[DONE]", "START": "[RUN ]",
               "ERROR": "[FAIL]", "WARN": "[WARN]", "REUSE": "[REUSE]"}.get(level, "[....]")
    print(f"[{ts[:19]}] {prefix} {message}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────

def load_qlora_model(adapter_dir=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_dir and os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_device(model):
    return getattr(model, "device", None) or next(model.parameters()).device


# ────────────────────────────────────────────────────────────────────────────
# Sandbox execution
# ────────────────────────────────────────────────────────────────────────────

def run_code_in_sandbox(code_str: str, test_code_str: str, timeout: float = 5.0) -> dict:
    try:
        ast.parse(code_str)
    except SyntaxError as e:
        return {"success": False, "failure_type": "syntax_error", "error": str(e)}

    test_wrapper = f"""
import sys
code = {repr(code_str)}
test_code = {repr(test_code_str)}
scope = {{}}
try:
    exec(code, scope)
    fns = [v for k, v in scope.items() if callable(v) and not k.startswith("__")]
    if not fns:
        print("FAIL:ASSERT:No callable function defined")
        sys.exit(1)
    target = fns[-1]
    test_scope = {{"candidate_fn": target}}
    exec(test_code, test_scope)
    run_test = test_scope.get("run_test")
    if run_test:
        run_test(target)
    print("SUCCESS")
    sys.exit(0)
except AssertionError as e:
    print(f"FAIL:ASSERT:{{e}}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL:RUNTIME:{{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""
    try:
        res = subprocess.run([sys.executable, "-c", test_wrapper],
                             capture_output=True, text=True, timeout=timeout)
        out = (res.stdout or "").strip()
        if "SUCCESS" in out:
            return {"success": True, "failure_type": "success", "error": None}
        elif "FAIL:ASSERT:" in out:
            return {"success": False, "failure_type": "assertion_failure",
                    "error": out.split("FAIL:ASSERT:", 1)[1].strip()}
        elif "FAIL:RUNTIME:" in out:
            return {"success": False, "failure_type": "runtime_error",
                    "error": out.split("FAIL:RUNTIME:", 1)[1].strip()}
        else:
            return {"success": False, "failure_type": "runtime_error",
                    "error": (res.stderr or "Unknown").strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "failure_type": "timeout", "error": f"Timeout >{timeout}s"}
    except Exception as e:
        return {"success": False, "failure_type": "runtime_error", "error": str(e)}


# ────────────────────────────────────────────────────────────────────────────
# Code extraction & EOS
# ────────────────────────────────────────────────────────────────────────────

def extract_core_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    cutoffs = [r"\n###\s+(?:Instruction|Response|Explanation|Output|Exercise|Note)", r"\n```"]
    truncated = text
    for pat in cutoffs:
        m = re.search(pat, truncated, re.IGNORECASE)
        if m:
            truncated = truncated[:m.start()]
    return truncated.strip()


def check_eos(gen_text: str, num_gen_tokens: int, max_tokens: int) -> dict:
    return {
        "eos_terminated": "<|endoftext|>" in gen_text or (num_gen_tokens < max_tokens),
        "max_token_cutoff": num_gen_tokens >= max_tokens,
    }


# ────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ────────────────────────────────────────────────────────────────────────────

def build_prompt(task: dict) -> str:
    inst = (task.get("instruction") or "").strip()
    inp  = (task.get("input")       or "").strip()
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{inst}\n\n### Response:\n"


# ────────────────────────────────────────────────────────────────────────────
# Greedy benchmark evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate_benchmark_greedy(model, tokenizer, benchmark: list, label: str) -> dict:
    model.eval()
    device = get_device(model)
    results = []

    for task in benchmark:
        tid      = task["id"]
        cat      = task["category"]
        test_code = task["test_code"]
        prompt   = build_prompt(task)
        print(f"  [Task {tid:2d}/50] Greedy | {cat}", flush=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GREEDY,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time  = time.time() - t0
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        num_gen    = len(gen_tokens)
        gen_raw    = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        gen_clean  = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        term = check_eos(gen_raw, num_gen, MAX_NEW_TOKENS_GREEDY)

        try:
            ast.parse(gen_clean)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        core_code = extract_core_code(gen_clean)
        try:
            ast.parse(core_code)
            core_syntax = True
        except SyntaxError:
            core_syntax = False

        raw_eval  = run_code_in_sandbox(gen_clean, test_code)
        core_eval = run_code_in_sandbox(core_code,  test_code)

        results.append({
            "task_id":           tid,
            "category":          cat,
            "gen_time_s":        round(gen_time, 2),
            "num_gen_tokens":    num_gen,
            "eos_terminated":    term["eos_terminated"],
            "max_token_cutoff":  term["max_token_cutoff"],
            "syntax_valid":      syntax_valid,
            "core_syntax_valid": core_syntax,
            "raw_functional_pass":  raw_eval["success"],
            "core_functional_pass": core_eval["success"],
            "raw_failure_type":  raw_eval["failure_type"],
            "core_failure_type": core_eval["failure_type"],
        })

    n = len(results)
    return {
        "label": label,
        "num_tasks": n,
        "core_functional_pass_rate":  sum(r["core_functional_pass"]  for r in results) / max(1, n),
        "syntax_validity_rate":       sum(r["syntax_valid"]           for r in results) / max(1, n),
        "eos_termination_rate":       sum(r["eos_terminated"]         for r in results) / max(1, n),
        "max_token_cutoff_rate":      sum(r["max_token_cutoff"]       for r in results) / max(1, n),
        "mean_gen_length_tokens":     sum(r["num_gen_tokens"]         for r in results) / max(1, n),
        "task_results":               results,   # Per-task booleans — fixes Phase 3 matrix bug
    }


# ────────────────────────────────────────────────────────────────────────────
# Sampled benchmark evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate_benchmark_sampled(model, tokenizer, benchmark: list, label: str) -> dict:
    model.eval()
    device = get_device(model)
    per_task_rates = []

    for task in benchmark:
        tid      = task["id"]
        cat      = task["category"]
        test_code = task["test_code"]
        prompt   = build_prompt(task)
        print(f"  [Task {tid:2d}/50] Sampled ({SAMPLED_RUNS} runs) | {cat}", flush=True)

        passes = 0
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for _ in range(SAMPLED_RUNS):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS_SAMPLED,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            gen_clean  = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            core_code  = extract_core_code(gen_clean)
            result     = run_code_in_sandbox(core_code, test_code)
            if result["success"]:
                passes += 1

        per_task_rates.append(passes / SAMPLED_RUNS)

    avg = sum(per_task_rates) / max(1, len(per_task_rates))
    return {"label": label, "avg_sampled_pass_rate": avg, "per_task_sampled_rates": per_task_rates}


# ────────────────────────────────────────────────────────────────────────────
# Held-out loss
# ────────────────────────────────────────────────────────────────────────────

def compute_held_out_loss(model, tokenizer, eval_dataset, max_seq_length=512) -> dict:
    model.eval()
    device = get_device(model)
    response_template = "### Response:\n"
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    rlen = len(response_token_ids)

    total_full_loss  = total_full_tokens  = 0
    total_resp_loss  = total_resp_tokens  = 0

    with torch.no_grad():
        for ex in eval_dataset:
            inst = (ex.get("instruction") or "").strip()
            inp  = (ex.get("input")       or "").strip()
            out  = (ex.get("output")      or "").strip()
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}"

            enc = tokenizer(text, truncation=True, max_length=max_seq_length, return_tensors="pt")
            input_ids  = enc["input_ids"].to(device)
            target_ids = input_ids.clone()
            ids_list   = input_ids[0].tolist()

            match_idx = None
            for i in range(len(ids_list) - rlen + 1):
                if ids_list[i: i + rlen] == response_token_ids:
                    match_idx = i
                    break

            logits       = model(input_ids).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            loss_full = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction="sum")
            total_full_loss   += loss_full.item()
            total_full_tokens += shift_labels.numel()

            if match_idx is not None:
                resp_start = match_idx + rlen
                rs = max(0, resp_start - 1)
                resp_logits = shift_logits[:, rs:, :]
                resp_labels = shift_labels[:, rs:]
                if resp_labels.numel() > 0:
                    loss_resp = F.cross_entropy(
                        resp_logits.view(-1, resp_logits.size(-1)),
                        resp_labels.view(-1), reduction="sum")
                    total_resp_loss   += loss_resp.item()
                    total_resp_tokens += resp_labels.numel()

    avg_full = total_full_loss / max(1, total_full_tokens)
    avg_resp = total_resp_loss / max(1, total_resp_tokens)
    return {
        "full_eval_loss":              round(avg_full, 4),
        "full_perplexity":             round(math.exp(min(20, avg_full)), 2),
        "response_only_eval_loss":     round(avg_resp, 4),
        "response_only_perplexity":    round(math.exp(min(20, avg_resp)), 2),
        "total_eval_samples":          len(eval_dataset),
        "total_response_tokens":       total_resp_tokens,
    }


# ────────────────────────────────────────────────────────────────────────────
# Per-task pass/fail matrix builder (FIXED vs Phase 3)
# ────────────────────────────────────────────────────────────────────────────

def build_pass_fail_matrix(models_data: dict, benchmark: list) -> dict:
    """
    Build pass/fail matrix from task_results lists in greedy eval data.
    This FIXES the Phase 3 bug where all cells rendered as '?' because
    task_results were not stored in the eval output dict.
    """
    matrix = {}
    for model_key, model_data in models_data.items():
        greedy = model_data.get("benchmark_greedy", {})
        task_results = greedy.get("task_results", [])
        # Build a lookup by task_id
        pass_by_id = {}
        for tr in task_results:
            pass_by_id[tr["task_id"]] = tr.get("core_functional_pass", False)
        matrix[model_key] = pass_by_id
    return matrix


# ────────────────────────────────────────────────────────────────────────────
# SHA-256 helper
# ────────────────────────────────────────────────────────────────────────────

def get_sha256(path: str) -> str:
    if not path or not os.path.exists(path):
        return "N/A"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# Phase A — checkpoint curve
# ────────────────────────────────────────────────────────────────────────────

def run_phase_a(benchmark: list) -> dict:
    """Greedy-only eval on every saved checkpoint. Returns curve dict + best ckpt info."""
    log_event("START", "Phase A: Config E checkpoint curve (greedy only)")

    ckpt_dirs = sorted([
        d for d in os.listdir(CONFIG_E_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(CONFIG_E_DIR, d))
    ], key=lambda x: int(x.split("-")[1]))

    if not ckpt_dirs:
        log_event("ERROR", f"No checkpoints found in {CONFIG_E_DIR}")
        return {"curve": [], "best_checkpoint": None}

    print(f"[Phase A] Found {len(ckpt_dirs)} checkpoints: {ckpt_dirs}")

    curve = []
    best_pass_rate = -1.0
    best_step = None
    best_ckpt_path = None

    for ckpt_name in ckpt_dirs:
        step = int(ckpt_name.split("-")[1])
        ckpt_path = os.path.join(CONFIG_E_DIR, ckpt_name)
        print(f"\n[Phase A] Step {step}: Loading from {ckpt_path}", flush=True)

        try:
            model, tokenizer = load_qlora_model(adapter_dir=ckpt_path)
            greedy_result = evaluate_benchmark_greedy(model, tokenizer, benchmark,
                                                      f"config_e_step{step}")
            pass_rate = greedy_result["core_functional_pass_rate"]

            entry = {
                "step":                         step,
                "checkpoint_path":              ckpt_path,
                "core_functional_pass_rate":    pass_rate,
                "syntax_validity_rate":         greedy_result["syntax_validity_rate"],
                "eos_termination_rate":         greedy_result["eos_termination_rate"],
                "max_token_cutoff_rate":        greedy_result["max_token_cutoff_rate"],
                "mean_gen_length_tokens":       greedy_result["mean_gen_length_tokens"],
            }
            curve.append(entry)
            print(f"[Phase A] Step {step}: pass={pass_rate:.1%} EOS={greedy_result['eos_termination_rate']:.1%}")

            if pass_rate > best_pass_rate:
                best_pass_rate = pass_rate
                best_step      = step
                best_ckpt_path = ckpt_path

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Phase A] Error at step {step}: {e}")
            traceback.print_exc()
            curve.append({"step": step, "error": str(e)})

    with open(CHECKPOINT_CURVE_PATH, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"\n[Phase A] Checkpoint curve saved to {CHECKPOINT_CURVE_PATH}")

    best_info = {
        "best_step": best_step,
        "best_checkpoint_path": best_ckpt_path,
        "best_core_functional_pass_rate": round(best_pass_rate, 4),
        "selection_reasoning": (
            f"Selected step {best_step} with {best_pass_rate:.1%} greedy pass rate "
            f"(highest among {len(ckpt_dirs)} checkpoints). Per Locked Decision #3: "
            "best-by-pass-rate, not final step."
        ),
    }

    log_event("DONE", f"Phase A complete. Best checkpoint: step {best_step} at {best_pass_rate:.1%}")
    return {"curve": curve, "best_checkpoint": best_info}


# ────────────────────────────────────────────────────────────────────────────
# Phase B — full eval on best checkpoint
# ────────────────────────────────────────────────────────────────────────────

def run_phase_b(benchmark: list, best_checkpoint_info: dict) -> dict:
    """Full eval (greedy + sampled + held-out loss) on the selected best checkpoint only."""
    best_ckpt_path = best_checkpoint_info["best_checkpoint_path"]
    best_step      = best_checkpoint_info["best_step"]
    log_event("START", f"Phase B: Full eval on best checkpoint step={best_step} at {best_ckpt_path}")

    # Load eval dataset (500 samples, same split as training)
    full_ds = load_dataset(DATASET_ID, split="train")
    slice_ds = full_ds.select(range(min(5000, len(full_ds))))
    split_ds = slice_ds.train_test_split(test_size=0.1, seed=SEED)
    eval_dataset = split_ds["test"]
    print(f"[Phase B] Eval dataset: {len(eval_dataset)} samples")

    # ── Load best checkpoint ──
    print(f"\n[Phase B] Loading best checkpoint from {best_ckpt_path}")
    model, tokenizer = load_qlora_model(adapter_dir=best_ckpt_path)

    # 3A — Greedy
    print(f"\n[3A] Running greedy benchmark on Config E best checkpoint (step {best_step})...")
    greedy_result = evaluate_benchmark_greedy(model, tokenizer, benchmark,
                                              f"config_e (best, step {best_step})")
    print(f"[3A] Core pass rate: {greedy_result['core_functional_pass_rate']:.1%} "
          f"| EOS: {greedy_result['eos_termination_rate']:.1%} "
          f"| Mean len: {greedy_result['mean_gen_length_tokens']:.1f} tok")

    # 3A — Sampled
    print(f"\n[3A] Running sampled benchmark on Config E best checkpoint...")
    sampled_result = evaluate_benchmark_sampled(model, tokenizer, benchmark,
                                                f"config_e (best, step {best_step})")
    print(f"[3A] Avg sampled pass rate: {sampled_result['avg_sampled_pass_rate']:.1%}")

    # 3B — Held-out loss
    print(f"\n[3B] Computing held-out loss on Config E best checkpoint...")
    loss_result = compute_held_out_loss(model, tokenizer, eval_dataset)
    print(f"[3B] Resp loss: {loss_result['response_only_eval_loss']:.4f} "
          f"(PPL {loss_result['response_only_perplexity']:.2f}) "
          f"| Full loss: {loss_result['full_eval_loss']:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load adapter SHA-256 ──
    adapter_sha256 = get_sha256(os.path.join(best_ckpt_path, "adapter_model.safetensors"))

    config_e_result = {
        "label": f"Config E: Scaled 5k (step {best_step}, best checkpoint)",
        "adapter_dir": best_ckpt_path,
        "adapter_sha256": adapter_sha256,
        "best_checkpoint_selection": best_checkpoint_info,
        "benchmark_greedy": greedy_result,
        "benchmark_sampled": sampled_result,
        "held_out_loss": loss_result,
    }

    log_event("DONE", f"Phase B complete. Config E pass rate: {greedy_result['core_functional_pass_rate']:.1%}")
    return config_e_result


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4 evaluation harness")
    parser.add_argument("--phase_a_only", action="store_true",
                        help="Run Phase A (checkpoint curve) only, skip Phase B")
    parser.add_argument("--phase_b_only", action="store_true",
                        help="Run Phase B only using existing checkpoint curve")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)

    # Log reuse decisions
    log_event("REUSE", f"SOURCE={BENCHMARK_PATH} | 50-task benchmark reused without regeneration per CRITICAL RULE")
    log_event("REUSE", f"SOURCE={PHASE3_MASTER_PATH} | All 6 Phase 3 model eval results reused — raw_phi2, phase2_corrected, config_a, config_b, config_c, config_d not re-evaluated per CRITICAL RULE")

    # Load benchmark
    print(f"[Eval] Loading benchmark from {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)
    print(f"[Eval] Loaded {len(benchmark)} benchmark tasks")

    # ── Phase A ──
    if not args.phase_b_only:
        phase_a_results = run_phase_a(benchmark)
        best_checkpoint_info = phase_a_results["best_checkpoint"]
    else:
        # Load from existing curve
        with open(CHECKPOINT_CURVE_PATH) as f:
            curve = json.load(f)
        best_entry = max((e for e in curve if "error" not in e),
                         key=lambda x: x["core_functional_pass_rate"])
        best_checkpoint_info = {
            "best_step": best_entry["step"],
            "best_checkpoint_path": best_entry["checkpoint_path"],
            "best_core_functional_pass_rate": best_entry["core_functional_pass_rate"],
            "selection_reasoning": f"Loaded from existing {CHECKPOINT_CURVE_PATH}",
        }

    if args.phase_a_only:
        print("\n[Done] Phase A complete. Run with --phase_b_only to continue.")
        return

    # ── Phase B ──
    config_e_result = run_phase_b(benchmark, best_checkpoint_info)

    # ── Load Phase 3 results for comparison (REUSE) ──
    print(f"\n[REUSE] Loading Phase 3 results from {PHASE3_MASTER_PATH}")
    with open(PHASE3_MASTER_PATH) as f:
        phase3_data = json.load(f)
    phase3_models = phase3_data.get("models", {})

    # ── Assemble master results ──
    master_results = {
        "evaluation_timestamp": datetime.datetime.utcnow().isoformat(),
        "benchmark_path": BENCHMARK_PATH,
        "reused_phase3_results": True,
        "phase3_master_source": PHASE3_MASTER_PATH,
        "models": {},
    }

    # Insert Phase 3 reused models
    for model_key in ["raw_phi2", "phase2_corrected", "config_a", "config_b", "config_c", "config_d"]:
        if model_key in phase3_models:
            master_results["models"][model_key] = phase3_models[model_key]
            master_results["models"][model_key]["source"] = "REUSED_FROM_PHASE3"

    # Insert Config E new results
    master_results["models"]["config_e"] = config_e_result

    # ── Pass/fail matrix (FIXED) ──
    master_results["pass_fail_matrix"] = build_pass_fail_matrix(
        master_results["models"], benchmark)

    # ── Save ──
    with open(MASTER_EVAL_PATH, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[Done] Master eval results saved to {MASTER_EVAL_PATH}")

    # ── Print summary ──
    print("\n" + "=" * 105)
    print("PHASE 4 MASTER EVALUATION SUMMARY")
    print("=" * 105)
    MODEL_ORDER = ["raw_phi2", "phase2_corrected", "config_a", "config_b", "config_c", "config_d", "config_e"]
    print(f"{'Model':<42} {'Resp Loss':>9} {'Resp PPL':>8} {'Full Loss':>9} {'Core Pass%':>10} {'EOS%':>7} {'AvgLen':>7}")
    print("-" * 105)
    for mk in MODEL_ORDER:
        md = master_results["models"].get(mk, {})
        if "error" in md:
            print(f"{md.get('label', mk):<42} {'ERROR':>9}")
            continue
        hl = md.get("held_out_loss", {})
        bg = md.get("benchmark_greedy", {})
        src = " [REUSED]" if md.get("source") == "REUSED_FROM_PHASE3" else " [NEW]"
        lbl = (md.get("label", mk) + src)[:42]
        print(
            f"{lbl:<42}"
            f" {hl.get('response_only_eval_loss', 'N/A'):>9}"
            f" {hl.get('response_only_perplexity', 'N/A'):>8}"
            f" {hl.get('full_eval_loss', 'N/A'):>9}"
            f" {bg.get('core_functional_pass_rate', 0)*100:>9.1f}%"
            f" {bg.get('eos_termination_rate', 0)*100:>6.1f}%"
            f" {bg.get('mean_gen_length_tokens', 0):>7.1f}"
        )
    print("=" * 105)
    log_event("DONE", "Phase 4 evaluation complete. Master results saved.")


if __name__ == "__main__":
    main()
