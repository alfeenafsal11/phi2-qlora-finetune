"""
run_phase2b_attribution_eval.py — Unified evaluation engine for Phase 2B Intervention Attribution.
Evaluates all 6 models under the exact same standardized evaluation pipeline:
  1. Raw Phi-2 Base (No Adapter)
  2. Broken Baseline Adapter (Phase 1)
  3. Phase 2 Final Corrected Adapter (Reference Condition)
  4. Ablation A: EOS Ablation (phi2-qlora-phase2b-eos-ablation)
  5. Ablation B: Masking Ablation (phi2-qlora-phase2b-masking-ablation)
  6. Ablation C: Input Ablation (phi2-qlora-phase2b-input-ablation)

Computes:
  - Standardized Full-Sequence Eval Loss & Perplexity on 200 held-out samples
  - Standardized Response-Only Eval Loss & Perplexity on 200 held-out samples (Primary Likelihood Metric)
  - 12-Task Benchmark Deterministic Evaluation (greedy: temp=0.0, max_new_tokens=256)
  - 12-Task Benchmark Sampled Evaluation (secondary: temp=0.7, top_p=0.9, max_new_tokens=256)
  - Raw vs Core syntax and functional pass rates
  - Termination metrics (EOS termination vs max token cutoff)
  - Generation length statistics (mean, median, min, max)
  - Full failure taxonomy breakdown
  - Contamination & secondary prompt template leakage rates
"""

import os
import sys
import json
import time
import math
import hashlib
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from eval_harness import (
    evaluate_model_on_benchmark,
    compute_independent_eval_loss,
    BENCHMARK_PROMPTS,
)

os.environ["HF_HUB_OFFLINE"] = "1"


def get_file_sha256(path):
    if not os.path.exists(path):
        return "MISSING"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_qlora_model(adapter_dir=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate_sampled_benchmark(model, tokenizer, max_new_tokens=256, temperature=0.7, top_p=0.9, device=None):
    """Secondary sampled evaluation on the 12 benchmark tasks."""
    model.eval()
    if device is None:
        device = getattr(model, "device", None) or next(model.parameters()).device

    results = []
    for item in BENCHMARK_PROMPTS:
        pid = item["id"]
        cat = item["category"]
        inst = item["instruction"]
        inp = item["input"]
        test_code = item["test_code"]

        if inp:
            prompt_str = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt_str = f"### Instruction:\n{inst}\n\n### Response:\n"

        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        clean_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

        from eval_harness import run_code_in_sandbox, extract_core_code
        core_code = extract_core_code(clean_text)
        core_eval = run_code_in_sandbox(core_code, test_code)

        results.append({
            "prompt_id": pid,
            "category": cat,
            "core_functional_pass": core_eval["success"],
            "core_failure_type": core_eval["failure_type"],
            "num_gen_tokens": len(gen_tokens),
            "eos_terminated": "<|endoftext|>" in raw_text or len(gen_tokens) < max_new_tokens,
        })

    total = len(results)
    pass_rate = sum(1 for r in results if r["core_functional_pass"]) / total
    eos_rate = sum(1 for r in results if r["eos_terminated"]) / total
    return {
        "sampled_functional_pass_rate": round(pass_rate, 4),
        "sampled_eos_termination_rate": round(eos_rate, 4),
        "temperature": temperature,
        "top_p": top_p,
        "results": results,
    }


def main():
    print("=" * 80)
    print("PHASE 2B — UNIFIED INTERVENTION ATTRIBUTION EVALUATION SUITE")
    print("=" * 80)

    # Held-out dataset
    dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
    full_dataset = load_dataset(dataset_name, split="train[:2000]")
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = split_dataset["test"]
    print(f"Loaded held-out evaluation dataset: {len(eval_dataset)} samples.")

    models = [
        {
            "id": "raw_phi2",
            "name": "Raw Phi-2 Base (No Adapter)",
            "adapter_dir": None,
            "role": "Historical Baseline",
        },
        {
            "id": "broken_baseline",
            "name": "Broken Baseline Adapter (Phase 1)",
            "adapter_dir": "phi2-qlora-output",
            "role": "Historical Control",
        },
        {
            "id": "phase2_corrected",
            "name": "Phase 2 Corrected Adapter",
            "adapter_dir": "phi2-qlora-phase2-corrected",
            "role": "Reference Condition",
        },
        {
            "id": "phase2b_eos_ablation",
            "name": "Ablation A: EOS Ablation (Input=YES, EOS=NO, Masking=YES)",
            "adapter_dir": "phi2-qlora-phase2b-eos-ablation",
            "role": "Ablation A",
        },
        {
            "id": "phase2b_masking_ablation",
            "name": "Ablation B: Masking Ablation (Input=YES, EOS=YES, Masking=NO)",
            "adapter_dir": "phi2-qlora-phase2b-masking-ablation",
            "role": "Ablation B",
        },
        {
            "id": "phase2b_input_ablation",
            "name": "Ablation C: Input Ablation (Input=NO, EOS=YES, Masking=YES)",
            "adapter_dir": "phi2-qlora-phase2b-input-ablation",
            "role": "Ablation C",
        },
    ]

    all_results = {}

    for m_cfg in models:
        mid = m_cfg["id"]
        mname = m_cfg["name"]
        adir = m_cfg["adapter_dir"]
        role = m_cfg["role"]

        print("\n" + "=" * 75)
        print(f"EVALUATING SYSTEM [{role}]: {mname}")
        print("=" * 75)

        # Check adapter existence if not base model
        adapter_sha256 = None
        if adir:
            adapter_file = os.path.join(adir, "adapter_model.safetensors")
            if not os.path.exists(adapter_file):
                print(f"[Warning] Adapter not found at {adapter_file}. Skipping {mid}.")
                continue
            adapter_sha256 = get_file_sha256(adapter_file)
            print(f"[Adapter Checksum] {adapter_file}: {adapter_sha256}")

        t0 = time.time()
        model, tokenizer = load_qlora_model(adir)
        print(f"Loaded model in {time.time() - t0:.1f}s")

        # ── 1. Standardized Loss Evaluation (200 held-out samples) ────────────
        print("\n[1/3] Computing standardized eval loss (Full & Response-Only)...")
        t_loss = time.time()
        loss_metrics = compute_independent_eval_loss(model, tokenizer, eval_dataset, max_seq_length=512)
        print(f"  Completed in {time.time() - t_loss:.1f}s")
        print(f"  Full Eval Loss:         {loss_metrics['full_eval_loss']} (PPL: {loss_metrics['full_perplexity']})")
        print(f"  Response-Only Loss:     {loss_metrics['response_only_eval_loss']} (PPL: {loss_metrics['response_only_perplexity']})")

        # ── 2. Primary Deterministic Generation Benchmark (Greedy) ───────────
        print("\n[2/3] Evaluating 12-task benchmark (Deterministic Greedy)...")
        t_bench = time.time()
        bench_det = evaluate_model_on_benchmark(model, tokenizer, prompt_format="standardized_phase2", max_new_tokens=256)
        print(f"  Completed in {time.time() - t_bench:.1f}s")
        print(f"  Raw Syntax:      {bench_det['raw_syntax_validity_rate']*100:.1f}%")
        print(f"  Raw Func Pass:   {bench_det['raw_functional_pass_rate']*100:.1f}%")
        print(f"  Core Syntax:     {bench_det['core_syntax_validity_rate']*100:.1f}%")
        print(f"  Core Func Pass:  {bench_det['core_functional_pass_rate']*100:.1f}% ({int(bench_det['core_functional_pass_rate']*12)}/12)")
        print(f"  EOS Term Rate:   {bench_det['eos_termination_rate']*100:.1f}%")
        print(f"  Max-Token Cut:   {bench_det['max_token_termination_rate']*100:.1f}%")
        print(f"  Sec. Prompt Rate:{bench_det['secondary_prompt_rate']*100:.1f}%")

        # Compute length statistics
        lengths = [r["num_gen_tokens"] for r in bench_det["results"]]
        mean_len = float(np.mean(lengths))
        median_len = float(np.median(lengths))
        min_len = int(np.min(lengths))
        max_len = int(np.max(lengths))
        print(f"  Gen Lengths:     Mean={mean_len:.1f}, Median={median_len:.1f}, Min={min_len}, Max={max_len}")

        # Failure taxonomy counts
        tax_counts = {}
        for r in bench_det["results"]:
            ft = r["core_failure_type"]
            tax_counts[ft] = tax_counts.get(ft, 0) + 1

        # ── 3. Secondary Sampled Generation Benchmark ─────────────────────────
        print("\n[3/3] Evaluating 12-task benchmark (Sampled: temp=0.7, top_p=0.9)...")
        t_sample = time.time()
        bench_sampled = evaluate_sampled_benchmark(model, tokenizer, max_new_tokens=256, temperature=0.7, top_p=0.9)
        print(f"  Completed in {time.time() - t_sample:.1f}s")
        print(f"  Sampled Core Pass: {bench_sampled['sampled_functional_pass_rate']*100:.1f}%")
        print(f"  Sampled EOS Rate:  {bench_sampled['sampled_eos_termination_rate']*100:.1f}%")

        all_results[mid] = {
            "id": mid,
            "name": mname,
            "role": role,
            "adapter_dir": adir,
            "adapter_sha256": adapter_sha256,
            "loss_metrics": loss_metrics,
            "benchmark_deterministic": bench_det,
            "benchmark_sampled": bench_sampled,
            "length_stats": {
                "mean": mean_len,
                "median": median_len,
                "min": min_len,
                "max": max_len,
            },
            "failure_taxonomy_counts": tax_counts,
        }

        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Save complete evaluation output
    out_file = "reports/phase2b_attribution_eval_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Evaluation Complete] Master results saved to {out_file}")


if __name__ == "__main__":
    main()
