"""
eval_phase3.py -- Phase 3 evaluation harness.

Evaluates up to 6 models on:
  3A -- 50-task extended benchmark (greedy + sampled)
  3B -- 200-sample held-out loss (response-only + full-sequence)
  3C -- Checkpoint progression curves (steps 50,100,150,200,250) for Configs A, B, C
  3D -- Masking timing report for Config D

Produces:
  reports/phase3_master_eval_results.json
  reports/phase3_checkpoint_curves.json
  reports/phase3_masking_ablation_timing.json (timing enrichment)

Usage:
  python eval_phase3.py [--skip_base] [--configs_to_eval a,b,c,d] [--benchmark_only] [--checkpoint_only]
"""

import os
import sys
import json
import time
import math
import re
import ast
import argparse
import subprocess
import hashlib
import datetime
import traceback

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["HF_HUB_OFFLINE"] = "1"

# ??????????????????????????? paths ??????????????????????????????????????????

BENCHMARK_PATH = "reports/phase3_benchmark_50tasks.json"
MASTER_EVAL_PATH = "reports/phase3_master_eval_results.json"
CHECKPOINT_CURVES_PATH = "reports/phase3_checkpoint_curves.json"
TIMING_LOG_PATH = "reports/phase3_masking_ablation_timing.json"

BASE_MODEL_ID = "microsoft/phi-2"
DATASET_ID = "iamtarun/python_code_instructions_18k_alpaca"
NUM_SAMPLES = 2000
EVAL_RATIO = 0.1
SEED = 42
MAX_NEW_TOKENS_GREEDY = 300
MAX_NEW_TOKENS_SAMPLED = 300
SAMPLED_RUNS = 3

MODEL_REGISTRY = {
    "raw_phi2": {"adapter_dir": None, "label": "Raw Phi-2 Base"},
    "phase2_corrected": {"adapter_dir": "phi2-qlora-phase2-corrected", "label": "Phase 2 Corrected (Reference)"},
    "config_a": {"adapter_dir": "phi2-qlora-phase3-attn-only", "label": "Config A: Attn-Only"},
    "config_b": {"adapter_dir": "phi2-qlora-phase3-full-attn", "label": "Config B: Full-Attn"},
    "config_c": {"adapter_dir": "phi2-qlora-phase3-all-linear", "label": "Config C: All-Linear"},
    "config_d": {"adapter_dir": "phi2-qlora-phase3-masking-ablation-clean", "label": "Config D: Masking-Ablation"},
}

CHECKPOINT_STEPS = [50, 100, 150, 200, 250]
CHECKPOINT_CONFIGS = ["config_a", "config_b", "config_c"]
CHECKPOINT_DIRS = {
    "config_a": "phi2-qlora-phase3-attn-only",
    "config_b": "phi2-qlora-phase3-full-attn",
    "config_c": "phi2-qlora-phase3-all-linear",
}


# ??????????????????????????? model loading ??????????????????????????????????

def load_qlora_model(adapter_dir=None):
    """Load Phi-2 in 4-bit NF4 with optional PEFT adapter."""
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


# ??????????????????????????? sandbox execution ??????????????????????????????

def run_code_in_sandbox(code_str: str, test_code_str: str, timeout: float = 5.0) -> dict:
    """Run generated code + test harness in an isolated subprocess."""
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
        print("FAIL:ASSERT:No callable function or class defined")
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
        res = subprocess.run(
            [sys.executable, "-c", test_wrapper],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (res.stdout or "").strip()
        if "SUCCESS" in out:
            return {"success": True, "failure_type": "success", "error": None}
        elif "FAIL:ASSERT:" in out:
            return {"success": False, "failure_type": "assertion_failure", "error": out.split("FAIL:ASSERT:", 1)[1].strip()}
        elif "FAIL:RUNTIME:" in out:
            return {"success": False, "failure_type": "runtime_error", "error": out.split("FAIL:RUNTIME:", 1)[1].strip()}
        else:
            return {"success": False, "failure_type": "runtime_error", "error": (res.stderr or "Unknown").strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "failure_type": "timeout", "error": f"Timeout > {timeout}s"}
    except Exception as e:
        return {"success": False, "failure_type": "runtime_error", "error": str(e)}


# ??????????????????????????? code extraction ????????????????????????????????

def extract_core_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    cutoff_patterns = [
        r"\n###\s+(?:Instruction|Response|Explanation|Output|Exercise|Question|Note|Concept)",
        r"\n```",
    ]
    truncated = text
    for pat in cutoff_patterns:
        m = re.search(pat, truncated, re.IGNORECASE)
        if m:
            truncated = truncated[:m.start()]
    return truncated.strip()


def check_eos_termination(gen_text: str, num_gen_tokens: int, max_tokens: int) -> dict:
    return {
        "eos_terminated": "<|endoftext|>" in gen_text or (num_gen_tokens < max_tokens),
        "max_token_cutoff": num_gen_tokens >= max_tokens,
    }


# ??????????????????????????? 3A -- 50-task benchmark ?????????????????????????

def load_benchmark() -> list:
    with open(BENCHMARK_PATH, "r") as f:
        return json.load(f)


def build_prompt(task: dict) -> str:
    inst = (task.get("instruction") or "").strip()
    inp = (task.get("input") or "").strip()
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{inst}\n\n### Response:\n"


def evaluate_benchmark_greedy(model, tokenizer, benchmark: list, model_key: str) -> dict:
    """Run greedy decoding on all 50 tasks."""
    model.eval()
    device = get_device(model)
    results = []

    for task in benchmark:
        tid = task["id"]
        cat = task["category"]
        test_code = task["test_code"]
        prompt = build_prompt(task)
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
        gen_time = time.time() - t0
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        num_gen = len(gen_tokens)
        gen_text_raw = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        gen_text_clean = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        term = check_eos_termination(gen_text_raw, num_gen, MAX_NEW_TOKENS_GREEDY)

        # Syntax check
        syntax_valid = True
        try:
            ast.parse(gen_text_clean)
        except SyntaxError:
            syntax_valid = False

        # Core code
        core_code = extract_core_code(gen_text_clean)
        core_syntax = True
        try:
            ast.parse(core_code)
        except SyntaxError:
            core_syntax = False

        # Functional
        raw_eval = run_code_in_sandbox(gen_text_clean, test_code)
        core_eval = run_code_in_sandbox(core_code, test_code)

        results.append({
            "task_id": tid,
            "category": cat,
            "gen_time_s": round(gen_time, 2),
            "num_gen_tokens": num_gen,
            "syntax_valid": syntax_valid,
            "core_syntax_valid": core_syntax,
            "raw_functional_pass": raw_eval["success"],
            "core_functional_pass": core_eval["success"],
            "raw_failure_type": raw_eval["failure_type"],
            "core_failure_type": core_eval["failure_type"],
            "eos_terminated": term["eos_terminated"],
            "max_token_cutoff": term["max_token_cutoff"],
        })

    n = len(results)
    summary = {
        "model_key": model_key,
        "mode": "greedy",
        "num_tasks": n,
        "syntax_validity_rate": round(sum(r["syntax_valid"] for r in results) / n, 4),
        "core_syntax_validity_rate": round(sum(r["core_syntax_valid"] for r in results) / n, 4),
        "raw_functional_pass_rate": round(sum(r["raw_functional_pass"] for r in results) / n, 4),
        "core_functional_pass_rate": round(sum(r["core_functional_pass"] for r in results) / n, 4),
        "eos_termination_rate": round(sum(r["eos_terminated"] for r in results) / n, 4),
        "max_token_cutoff_rate": round(sum(r["max_token_cutoff"] for r in results) / n, 4),
        "mean_gen_length_tokens": round(sum(r["num_gen_tokens"] for r in results) / n, 1),
        "per_task_results": results,
    }
    return summary


def evaluate_benchmark_sampled(model, tokenizer, benchmark: list, model_key: str) -> dict:
    """Run sampled decoding (3-run average) on all 50 tasks."""
    model.eval()
    device = get_device(model)
    results = []

    for task in benchmark:
        tid = task["id"]
        cat = task["category"]
        test_code = task["test_code"]
        prompt = build_prompt(task)
        print(f"  [Task {tid:2d}/50] Sampled ({SAMPLED_RUNS} runs) | {cat}", flush=True)

        run_passes = []
        for run_idx in range(SAMPLED_RUNS):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
            gen_text_clean = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            core_code = extract_core_code(gen_text_clean)
            result = run_code_in_sandbox(core_code, test_code)
            run_passes.append(result["success"])

        avg_pass = sum(run_passes) / SAMPLED_RUNS
        results.append({
            "task_id": tid,
            "category": cat,
            "sampled_pass_rate": round(avg_pass, 4),
            "run_passes": run_passes,
        })

    n = len(results)
    summary = {
        "model_key": model_key,
        "mode": "sampled",
        "temperature": 0.7,
        "top_p": 0.9,
        "num_runs_per_task": SAMPLED_RUNS,
        "num_tasks": n,
        "avg_sampled_pass_rate": round(sum(r["sampled_pass_rate"] for r in results) / n, 4),
        "per_task_results": results,
    }
    return summary


# ??????????????????????????? 3B -- held-out loss ??????????????????????????????

def load_eval_dataset():
    full = load_dataset(DATASET_ID, split=f"train[:{NUM_SAMPLES}]")
    split = full.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    return split["test"]


def compute_held_out_loss(model, tokenizer, eval_dataset, max_seq_length=512) -> dict:
    """Compute response-only and full-sequence loss on 200 held-out samples."""
    model.eval()
    device = get_device(model)

    response_template = "### Response:\n"
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    rlen = len(response_token_ids)

    total_full_loss = 0.0
    total_full_tokens = 0
    total_resp_loss = 0.0
    total_resp_tokens = 0

    with torch.no_grad():
        for ex in eval_dataset:
            inst = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("output") or "").strip()

            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}"

            enc = tokenizer(text, truncation=True, max_length=max_seq_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            target_ids = input_ids.clone()

            ids_list = input_ids[0].tolist()
            match_idx = None
            for i in range(len(ids_list) - rlen + 1):
                if ids_list[i: i + rlen] == response_token_ids:
                    match_idx = i
                    break

            logits = model(input_ids).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            loss_full = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_full_loss += loss_full.item()
            total_full_tokens += shift_labels.numel()

            if match_idx is not None:
                resp_start = match_idx + rlen
                resp_target_start = max(0, resp_start - 1)
                resp_shift_logits = shift_logits[:, resp_target_start:, :].contiguous()
                resp_shift_labels = shift_labels[:, resp_target_start:].contiguous()
                if resp_shift_labels.numel() > 0:
                    loss_resp = F.cross_entropy(
                        resp_shift_logits.view(-1, resp_shift_logits.size(-1)),
                        resp_shift_labels.view(-1),
                        reduction="sum",
                    )
                    total_resp_loss += loss_resp.item()
                    total_resp_tokens += resp_shift_labels.numel()

    avg_full = total_full_loss / max(1, total_full_tokens)
    avg_resp = total_resp_loss / max(1, total_resp_tokens)
    return {
        "full_eval_loss": round(avg_full, 4),
        "full_perplexity": round(math.exp(min(20, avg_full)), 2),
        "response_only_eval_loss": round(avg_resp, 4),
        "response_only_perplexity": round(math.exp(min(20, avg_resp)), 2),
        "total_eval_samples": len(eval_dataset),
        "total_response_tokens": total_resp_tokens,
    }


# ??????????????????????????? 3C -- checkpoint curves ?????????????????????????

def evaluate_checkpoint_progression(eval_dataset, benchmark: list) -> dict:
    """Evaluate each saved checkpoint at steps 50,100,150,200,250 for Configs A,B,C."""
    curves = {}

    for cfg_key in CHECKPOINT_CONFIGS:
        cfg_dir = CHECKPOINT_DIRS[cfg_key]
        if not os.path.isdir(cfg_dir):
            print(f"[Curves] {cfg_key} dir not found: {cfg_dir} -- skipping")
            continue

        print(f"\n[Curves] Evaluating checkpoints for {cfg_key}...")
        cfg_curve = []

        for step in CHECKPOINT_STEPS:
            ckpt_path = os.path.join(cfg_dir, f"checkpoint-{step}")
            if not os.path.isdir(ckpt_path):
                print(f"  [Step {step}] Checkpoint not found -- skipping")
                continue

            print(f"  [Step {step}] Loading checkpoint from {ckpt_path}")
            try:
                model, tokenizer = load_qlora_model(adapter_dir=ckpt_path)
                loss_metrics = compute_held_out_loss(model, tokenizer, eval_dataset)

                # Quick 50-task benchmark pass rate
                bench_greedy = evaluate_benchmark_greedy(model, tokenizer, benchmark, f"{cfg_key}_step{step}")

                cfg_curve.append({
                    "step": step,
                    "response_only_eval_loss": loss_metrics["response_only_eval_loss"],
                    "response_only_perplexity": loss_metrics["response_only_perplexity"],
                    "full_eval_loss": loss_metrics["full_eval_loss"],
                    "core_functional_pass_rate_50task": bench_greedy["core_functional_pass_rate"],
                    "syntax_validity_rate_50task": bench_greedy["syntax_validity_rate"],
                    "eos_termination_rate": bench_greedy["eos_termination_rate"],
                    "mean_gen_length_tokens": bench_greedy["mean_gen_length_tokens"],
                })

                # Free VRAM
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [Step {step}] Error: {e}")
                cfg_curve.append({"step": step, "error": str(e)})

        curves[cfg_key] = cfg_curve

    return curves


# ??????????????????????????? OOD verification ???????????????????????????????

def verify_ood(benchmark: list) -> dict:
    """Verify benchmark prompts have zero substring overlap with training instructions."""
    print("[OOD] Loading training split for OOD verification...")
    full = load_dataset(DATASET_ID, split=f"train[:{NUM_SAMPLES}]")
    split = full.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    train_texts = [
        (ex.get("instruction") or "").strip().lower()
        for ex in split["train"]
    ]

    violations = []
    for task in benchmark:
        prompt_lower = (task.get("instruction") or "").strip().lower()
        for train_text in train_texts:
            if prompt_lower and prompt_lower in train_text:
                violations.append({"task_id": task["id"], "overlap_with": train_text[:100]})
                break
            if train_text and train_text in prompt_lower:
                violations.append({"task_id": task["id"], "overlap_with": train_text[:100]})
                break

    result = {
        "total_benchmark_tasks": len(benchmark),
        "total_train_samples": len(split["train"]),
        "violations_found": len(violations),
        "ood_verified": len(violations) == 0,
        "violations": violations,
    }
    print(f"[OOD] Violations found: {len(violations)}/{len(benchmark)} tasks")
    return result


# ??????????????????????????? per-task pass/fail matrix ??????????????????????

def build_pass_fail_matrix(model_results: dict, benchmark: list) -> dict:
    """Build 50x6 pass/fail matrix from per-model greedy benchmark results."""
    matrix = {}
    for model_key, model_data in model_results.items():
        greedy = model_data.get("benchmark_greedy", {})
        per_task = {r["task_id"]: r["core_functional_pass"] for r in greedy.get("per_task_results", [])}
        matrix[model_key] = per_task

    # Build category summary
    category_map = {t["id"]: t["category"] for t in benchmark}
    categories = {}
    for task in benchmark:
        cat = task["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(task["id"])

    return {"per_model_per_task": matrix, "task_categories": category_map, "category_task_ids": categories}


# ??????????????????????????? main evaluation loop ???????????????????????????

def get_file_sha256(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_full_evaluation(args):
    os.makedirs("reports", exist_ok=True)

    # Load benchmark
    print(f"\n[Eval] Loading benchmark from {BENCHMARK_PATH}")
    benchmark = load_benchmark()
    print(f"[Eval] Loaded {len(benchmark)} tasks")

    # OOD verification
    ood_result = verify_ood(benchmark)

    # Load held-out eval dataset once
    print("\n[Eval] Loading held-out eval dataset (200 samples)...")
    eval_dataset = load_eval_dataset()

    # Determine which models to evaluate
    models_to_eval = args.models if args.models else list(MODEL_REGISTRY.keys())

    master_results = {
        "evaluation_timestamp": datetime.datetime.utcnow().isoformat(),
        "benchmark_path": BENCHMARK_PATH,
        "ood_verification": ood_result,
        "models": {},
    }

    if os.path.exists(MASTER_EVAL_PATH):
        try:
            with open(MASTER_EVAL_PATH, "r") as f:
                saved = json.load(f)
                if isinstance(saved.get("models"), dict):
                    master_results["models"] = saved["models"]
                    print(f"[Eval] Loaded prior completed models from {MASTER_EVAL_PATH}: {list(master_results['models'].keys())}")
        except Exception as e:
            print(f"[Eval] Could not load prior results from {MASTER_EVAL_PATH}: {e}")

    for model_key in models_to_eval:
        if model_key not in MODEL_REGISTRY:
            print(f"[Eval] Unknown model key: {model_key} -- skipping")
            continue

        info = MODEL_REGISTRY[model_key]
        adapter_dir = info["adapter_dir"]
        label = info["label"]

        if model_key in master_results.get("models", {}) and "error" not in master_results["models"][model_key]:
            existing_m = master_results["models"][model_key]
            if "benchmark_greedy" in existing_m and "held_out_loss" in existing_m:
                print(f"[Eval] Model {label} ({model_key}) already fully evaluated in {MASTER_EVAL_PATH} -- skipping")
                continue

        if adapter_dir and not os.path.isdir(adapter_dir):
            print(f"[Eval] Adapter dir not found: {adapter_dir} -- skipping {model_key}")
            master_results["models"][model_key] = {"label": label, "error": "adapter_dir_not_found"}
            continue

        print(f"\n{'='*70}")
        print(f"[Eval] Model: {label}")
        print(f"{'='*70}")

        try:
            model, tokenizer = load_qlora_model(adapter_dir=adapter_dir)
            device = get_device(model)

            model_result = {
                "label": label,
                "adapter_dir": adapter_dir,
                "adapter_sha256": get_file_sha256(os.path.join(adapter_dir, "adapter_model.safetensors")) if adapter_dir else "N/A",
            }

            # 3A -- Greedy benchmark
            if not args.checkpoint_only:
                print(f"\n[3A] Running greedy 50-task benchmark on {label}...")
                greedy_result = evaluate_benchmark_greedy(model, tokenizer, benchmark, model_key)
                model_result["benchmark_greedy"] = greedy_result
                print(f"[3A] Core pass rate: {greedy_result['core_functional_pass_rate']:.1%} | "
                      f"EOS: {greedy_result['eos_termination_rate']:.1%} | "
                      f"Mean len: {greedy_result['mean_gen_length_tokens']:.1f} tok")

                # 3A -- Sampled benchmark
                print(f"\n[3A] Running sampled 50-task benchmark on {label}...")
                sampled_result = evaluate_benchmark_sampled(model, tokenizer, benchmark, model_key)
                model_result["benchmark_sampled"] = sampled_result
                print(f"[3A] Avg sampled pass rate: {sampled_result['avg_sampled_pass_rate']:.1%}")

                # 3B -- Held-out loss
                print(f"\n[3B] Computing held-out loss on {label}...")
                loss_result = compute_held_out_loss(model, tokenizer, eval_dataset)
                model_result["held_out_loss"] = loss_result
                print(f"[3B] Resp loss: {loss_result['response_only_eval_loss']:.4f} "
                      f"(PPL {loss_result['response_only_perplexity']:.2f}) | "
                      f"Full loss: {loss_result['full_eval_loss']:.4f}")

            master_results["models"][model_key] = model_result

            # Save checkpoint timing for Config D
            if model_key == "config_d" and not args.checkpoint_only:
                _enrich_timing_log(adapter_dir, model_result)

            # Incremental save to disk so completed model evaluations are never lost
            try:
                with open(MASTER_EVAL_PATH, "w") as f:
                    json.dump(master_results, f, indent=2)
                print(f"[Eval] Incremental checkpoint successfully saved to {MASTER_EVAL_PATH}")
            except Exception as e:
                print(f"[Eval] Warning: Failed to save incremental checkpoint: {e}")

            # Free VRAM before next model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Eval] Error evaluating {model_key}: {e}")
            traceback.print_exc()
            master_results["models"][model_key] = {"label": label, "error": str(e)}

    # Build pass/fail matrix
    master_results["pass_fail_matrix"] = build_pass_fail_matrix(master_results["models"], benchmark)

    # 3C -- Checkpoint curves
    if not args.benchmark_only:
        print("\n[3C] Running checkpoint progression evaluation...")
        curves = evaluate_checkpoint_progression(eval_dataset, benchmark)
        master_results["checkpoint_curves"] = curves
        with open(CHECKPOINT_CURVES_PATH, "w") as f:
            json.dump(curves, f, indent=2)
        print(f"[3C] Checkpoint curves saved to {CHECKPOINT_CURVES_PATH}")

    # Save master results
    with open(MASTER_EVAL_PATH, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[Done] Master eval results saved to {MASTER_EVAL_PATH}")

    # Print summary table
    _print_summary_table(master_results)

    return master_results


def _enrich_timing_log(adapter_dir: str, model_result: dict):
    """Load Config A timing from training summary for speedup comparison."""
    summary_path = os.path.join(adapter_dir, "phase3_train_summary.json")
    if not os.path.exists(summary_path):
        return
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        timing_data = {}
        if os.path.exists(TIMING_LOG_PATH):
            with open(TIMING_LOG_PATH) as f:
                timing_data = json.load(f)

        config_a_time = None
        config_a_summary = "phi2-qlora-phase3-attn-only/phase3_train_summary.json"
        if os.path.exists(config_a_summary):
            with open(config_a_summary) as f:
                a_sum = json.load(f)
            config_a_time = a_sum.get("wall_clock_seconds")

        config_d_time = summary.get("wall_clock_seconds")
        if config_a_time and config_d_time:
            speedup = round(config_d_time / config_a_time, 3)
            timing_data["speedup_comparison"] = {
                "config_a_wall_clock_seconds": config_a_time,
                "config_d_wall_clock_seconds": config_d_time,
                "config_d_vs_config_a_ratio": speedup,
                "masking_provides_speedup": speedup > 1.3,
                "recommendation": (
                    "Response-only masking confirmed as meaningful efficiency gain (>30% faster)"
                    if speedup > 1.3 else
                    "Masking is neutral on this hardware at this batch size"
                ),
            }
            with open(TIMING_LOG_PATH, "w") as f:
                json.dump(timing_data, f, indent=2)
    except Exception as e:
        print(f"[Timing] Could not enrich timing log: {e}")


def _print_summary_table(master_results: dict):
    print("\n" + "=" * 100)
    print("PHASE 3 MASTER EVALUATION SUMMARY")
    print("=" * 100)
    print(f"{'Model':<35} {'Resp Loss':>9} {'Resp PPL':>8} {'Full Loss':>9} {'Core Pass%':>10} {'EOS%':>7} {'AvgLen':>7}")
    print("-" * 100)
    for mk, md in master_results.get("models", {}).items():
        if "error" in md:
            print(f"{md.get('label', mk):<35} {'ERROR':>9}")
            continue
        hl = md.get("held_out_loss", {})
        bg = md.get("benchmark_greedy", {})
        print(
            f"{md.get('label', mk):<35}"
            f" {hl.get('response_only_eval_loss', 'N/A'):>9}"
            f" {hl.get('response_only_perplexity', 'N/A'):>8}"
            f" {hl.get('full_eval_loss', 'N/A'):>9}"
            f" {bg.get('core_functional_pass_rate', 0)*100:>9.1f}%"
            f" {bg.get('eos_termination_rate', 0)*100:>6.1f}%"
            f" {bg.get('mean_gen_length_tokens', 0):>7.1f}"
        )
    print("=" * 100)


# ??????????????????????????? CLI ?????????????????????????????????????????????

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 evaluation harness")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys to evaluate (default: all). "
                             "Keys: raw_phi2,phase2_corrected,config_a,config_b,config_c,config_d")
    parser.add_argument("--benchmark_only", action="store_true",
                        help="Run only 3A benchmark + 3B loss, skip 3C checkpoint curves")
    parser.add_argument("--checkpoint_only", action="store_true",
                        help="Run only 3C checkpoint curves")
    parser.add_argument("--skip_sampled", action="store_true",
                        help="Skip sampled decoding (faster evaluation)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.models:
        args.models = [m.strip() for m in args.models.split(",")]

    start = time.time()
    master_results = run_full_evaluation(args)
    elapsed = time.time() - start
    print(f"\n[Done] Total evaluation time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
