"""
test_eval_regression.py — Evaluation Harness Regression Test (Addendum Control F).
Runs the Phase 1 broken adapter (phi2-qlora-output) through eval_harness.py
and compares directly with the recorded Phase 1 baseline in reports/eval_audit_full_results.json.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from eval_harness import evaluate_model_on_benchmark, compute_independent_eval_loss
from datasets import load_dataset

os.environ["HF_HUB_OFFLINE"] = "1"

def main():
    print("=" * 80)
    print("CONTROL F: EVALUATION HARNESS REGRESSION TEST")
    print("=" * 80)

    adapter_dir = "phi2-qlora-output"
    model_id = "microsoft/phi-2"

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model {model_id} in 4-bit NF4 (GPU-resident)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Attaching broken baseline adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("\nRunning benchmark evaluation (greedy decoding, native Phase 1 format)...")
    summary = evaluate_model_on_benchmark(model, tokenizer, prompt_format="native_phase1", max_new_tokens=256)

    print("\nBenchmark Summary Results:")
    print(f"  Raw Syntax Validity Rate:   {summary['raw_syntax_validity_rate']*100:.1f}%")
    print(f"  Raw Functional Pass Rate:   {summary['raw_functional_pass_rate']*100:.1f}%")
    print(f"  Core Syntax Validity Rate:  {summary['core_syntax_validity_rate']*100:.1f}%")
    print(f"  Core Functional Pass Rate:  {summary['core_functional_pass_rate']*100:.1f}%")
    print(f"  EOS Termination Rate:       {summary['eos_termination_rate']*100:.1f}%")
    print(f"  Max-Token Termination Rate: {summary['max_token_termination_rate']*100:.1f}%")
    print(f"  Secondary Prompt Rate:      {summary['secondary_prompt_rate']*100:.1f}%")
    print(f"  Repeated Explanations:      {summary['total_repeated_explanation_blocks']}")

    # Load Phase 1 recorded baseline
    recorded = json.load(open("reports/eval_audit_full_results.json"))["final_adapter"]
    rec_syntax = recorded["rates_greedy"]["syntax_pass_rate"]
    rec_func = recorded["rates_greedy"]["func_pass_rate"]

    print("\nComparison with Recorded Phase 1 Benchmark:")
    print(f"  Recorded Syntax Validity:   {rec_syntax*100:.1f}% vs Measured: {summary['raw_syntax_validity_rate']*100:.1f}%")
    print(f"  Recorded Functional Pass:   {rec_func*100:.1f}% vs Measured: {summary['raw_functional_pass_rate']*100:.1f}%")

    # Save regression report
    os.makedirs("reports", exist_ok=True)
    reg_path = "reports/eval_harness_regression_report.json"
    with open(reg_path, "w") as f:
        json.dump({
            "recorded_phase1": {
                "syntax_validity_rate": rec_syntax,
                "functional_pass_rate": rec_func,
            },
            "measured_harness": summary,
        }, f, indent=2)
    print(f"\nSaved regression report to {reg_path}")
    print("=" * 80)
    print("REGRESSION TEST COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    main()
