"""
run_full_phase2_eval.py — Complete evaluation suite for Phase 2.
Evaluates:
  1. Raw Base Phi-2
  2. Broken Phase 1 Adapter (phi2-qlora-output)
  3. Phase 2 Checkpoint-200
  4. Phase 2 Checkpoint-400
  5. Phase 2 Final Adapter (phi2-qlora-phase2-corrected)

Computes:
  - Independent Full-Sequence Eval Loss & Perplexity on 200 held-out samples (Control H)
  - Independent Response-Only Eval Loss & Perplexity on 200 held-out samples (Control H)
  - 12-Prompt Benchmark Evaluation under Native Format (Control G)
  - 12-Prompt Benchmark Evaluation under Standardized Format (Control G)
  - Raw vs Core syntax and functional pass rates (Control K)
  - Full failure taxonomy (Control J)
  - Contamination and termination metrics (Control L)
"""

import os
import sys
import json
import time
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


def main():
    print("=" * 80)
    print("PHASE 2 — COMPREHENSIVE MULTI-MODEL EVALUATION SUITE")
    print("=" * 80)

    # 1. Load the fixed 200 held-out evaluation samples
    print("\nLoading held-out evaluation dataset (200 samples, seed 42)...")
    dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
    full_dataset = load_dataset(dataset_name, split="train[:2000]")
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = split_dataset["test"]
    print(f"Loaded {len(eval_dataset)} held-out eval samples.")

    models_to_evaluate = [
        {"id": "raw_phi2", "name": "Raw Phi-2 Base (No Adapter)", "adapter_dir": None, "native_format": "standardized_phase2"},
        {"id": "broken_baseline", "name": "Broken Baseline Adapter (Phase 1)", "adapter_dir": "phi2-qlora-output", "native_format": "native_phase1"},
        {"id": "phase2_ckpt200", "name": "Phase 2 Checkpoint-200", "adapter_dir": "phi2-qlora-phase2-corrected/checkpoint-200", "native_format": "standardized_phase2"},
        {"id": "phase2_ckpt400", "name": "Phase 2 Checkpoint-400", "adapter_dir": "phi2-qlora-phase2-corrected/checkpoint-400", "native_format": "standardized_phase2"},
        {"id": "phase2_final", "name": "Phase 2 Final Corrected Adapter", "adapter_dir": "phi2-qlora-phase2-corrected", "native_format": "standardized_phase2"},
    ]

    all_results = {}

    for m_cfg in models_to_evaluate:
        mid = m_cfg["id"]
        mname = m_cfg["name"]
        adir = m_cfg["adapter_dir"]
        native_fmt = m_cfg["native_format"]

        print("\n" + "=" * 70)
        print(f"EVALUATING SYSTEM: {mname}")
        print("=" * 70)

        t_load_start = time.time()
        model, tokenizer = load_qlora_model(adir)
        print(f"Loaded model in {time.time() - t_load_start:.1f}s")

        # ── A. Compute Independent Eval Losses (Control H) ─────────────────
        print("\n[1/3] Computing independent full-sequence and response-only eval losses...")
        t_loss_start = time.time()
        loss_metrics = compute_independent_eval_loss(model, tokenizer, eval_dataset, max_seq_length=512)
        print(f"  Completed in {time.time() - t_loss_start:.1f}s")
        print(f"  Full Eval Loss:         {loss_metrics['full_eval_loss']} (PPL: {loss_metrics['full_perplexity']})")
        print(f"  Response-Only Loss:     {loss_metrics['response_only_eval_loss']} (PPL: {loss_metrics['response_only_perplexity']})")

        # ── B. Benchmark Generation (Native Format) ───────────────────────
        print(f"\n[2/3] Evaluating 12 benchmark prompts (Native Format: {native_fmt})...")
        t_bench_start = time.time()
        bench_native = evaluate_model_on_benchmark(model, tokenizer, prompt_format=native_fmt, max_new_tokens=256)
        print(f"  Native Raw Syntax Validity:  {bench_native['raw_syntax_validity_rate']*100:.1f}%")
        print(f"  Native Raw Functional Pass:  {bench_native['raw_functional_pass_rate']*100:.1f}%")
        print(f"  Native Core Syntax Validity: {bench_native['core_syntax_validity_rate']*100:.1f}%")
        print(f"  Native Core Functional Pass: {bench_native['core_functional_pass_rate']*100:.1f}%")
        print(f"  Native EOS Termination:      {bench_native['eos_termination_rate']*100:.1f}%")
        print(f"  Native Max-Token Cutoff:     {bench_native['max_token_termination_rate']*100:.1f}%")

        # ── C. Benchmark Generation (Standardized Format) ─────────────────
        # For Phase 2 models, native == standardized, so avoid redundant execution
        if native_fmt != "standardized_phase2":
            print(f"\n[3/3] Evaluating 12 benchmark prompts (Standardized Format)...")
            bench_standard = evaluate_model_on_benchmark(model, tokenizer, prompt_format="standardized_phase2", max_new_tokens=256)
            print(f"  Std Raw Syntax Validity:   {bench_standard['raw_syntax_validity_rate']*100:.1f}%")
            print(f"  Std Raw Functional Pass:   {bench_standard['raw_functional_pass_rate']*100:.1f}%")
            print(f"  Std Core Functional Pass:  {bench_standard['core_functional_pass_rate']*100:.1f}%")
        else:
            bench_standard = bench_native

        all_results[mid] = {
            "id": mid,
            "name": mname,
            "adapter_dir": adir,
            "loss_metrics": loss_metrics,
            "benchmark_native": bench_native,
            "benchmark_standardized": bench_standard,
        }

        # Free GPU memory before next model
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Save comprehensive evaluation results
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/phase2_full_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Saved all evaluation results to {out_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
