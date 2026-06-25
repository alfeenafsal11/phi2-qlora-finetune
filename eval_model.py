"""
eval_model.py — Structured qualitative + quantitative evaluation for Phi-2 QLoRA adapters.

Runs inference on a fixed held-out test bench of 12 instruction prompts, generates
responses from the fine-tuned model, and produces:
  - <output_dir>/eval_report.json   : Machine-readable results with metadata
  - <output_dir>/eval_report.md     : Human-readable markdown table

Usage:
    # Evaluate the default baseline run
    python eval_model.py

    # Evaluate a specific experiment
    python eval_model.py --adapter_dir ./experiments/run_rank8_cosine

    # Compare baseline vs fine-tuned (also generate base model responses)
    python eval_model.py --adapter_dir ./phi2-qlora-output --compare_base

    # Change output location
    python eval_model.py --adapter_dir ./phi2-qlora-output --report_dir ./reports
"""

import argparse
import json
import os
import textwrap
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─────────────────────────── evaluation prompts ──────────────────────────────

EVAL_PROMPTS = [
    # --- String manipulation ---
    {
        "category":    "String Manipulation",
        "instruction": "Write a Python function that reverses a string without using slicing.",
    },
    {
        "category":    "String Manipulation",
        "instruction": "Write a Python function that checks if a string is a palindrome.",
    },
    {
        "category":    "String Manipulation",
        "instruction": "Write a Python function that counts the frequency of each character in a string and returns a dictionary.",
    },
    # --- Data structures ---
    {
        "category":    "Data Structures",
        "instruction": "Write a Python function that removes duplicates from a list while preserving order.",
    },
    {
        "category":    "Data Structures",
        "instruction": "Write a Python function that flattens a nested list of arbitrary depth.",
    },
    {
        "category":    "Data Structures",
        "instruction": "Write a Python function that merges two dictionaries and sums values for duplicate keys.",
    },
    # --- Sorting / Searching ---
    {
        "category":    "Sorting & Searching",
        "instruction": "Implement binary search in Python. The function should return the index of the target or -1 if not found.",
    },
    {
        "category":    "Sorting & Searching",
        "instruction": "Write a Python function that sorts a list of dictionaries by a specified key.",
    },
    # --- Recursion ---
    {
        "category":    "Recursion",
        "instruction": "Write a recursive Python function to compute the nth Fibonacci number.",
    },
    {
        "category":    "Recursion",
        "instruction": "Write a recursive Python function to compute the factorial of a number.",
    },
    # --- Bug fixing ---
    {
        "category":    "Bug Fixing",
        "instruction": (
            "The following Python function is supposed to return the sum of all even numbers "
            "in a list, but it has a bug. Fix it:\n\n"
            "def sum_evens(lst):\n    total = 0\n    for x in lst:\n        if x % 2 = 0:\n            total += x\n    return total"
        ),
    },
    # --- File I/O ---
    {
        "category":    "File I/O",
        "instruction": "Write a Python function that reads a CSV file and returns a list of dictionaries, one per row.",
    },
]

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ─────────────────────────── model loading ───────────────────────────────────

def load_model(adapter_dir: str, device: str):
    """Load the fine-tuned adapter model over the Phi-2 base."""
    print(f"[Load] Loading base model (microsoft/phi-2)...")
    base = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"[Load] Attaching adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def load_base_model(device: str):
    """Load the raw Phi-2 base model without any LoRA adapter."""
    print("[Load] Loading raw base model (no adapter)...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


# ─────────────────────────── inference ───────────────────────────────────────

def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt tokens, return only newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ─────────────────────────── report generation ───────────────────────────────

def write_markdown_report(results: list[dict], report_path: str, adapter_dir: str, compare_base: bool) -> None:
    lines = [
        "# Phi-2 QLoRA — Evaluation Report",
        "",
        f"**Adapter:** `{adapter_dir}`",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Prompts evaluated:** {len(results)}",
        f"**Base model comparison:** {'Yes' if compare_base else 'No'}",
        "",
        "---",
        "",
    ]

    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        lines.append(f"## {cat}")
        lines.append("")
        for r in cat_results:
            lines.append(f"### Prompt")
            lines.append(f"> {r['instruction']}")
            lines.append("")
            lines.append("**Fine-tuned Model Response:**")
            lines.append("```python")
            lines.append(r["finetuned_response"])
            lines.append("```")
            if compare_base and "base_response" in r:
                lines.append("")
                lines.append("**Base Model Response (no adapter):**")
                lines.append("```python")
                lines.append(r["base_response"])
                lines.append("```")
            lines.append("")
            lines.append(f"*Inference time: {r['inference_time_s']:.2f}s*")
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Markdown report written to: {report_path}")


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Phi-2 QLoRA adapter.")
    parser.add_argument("--adapter_dir",  type=str, default="./phi2-qlora-output",
                        help="Path to the directory containing adapter_model.safetensors.")
    parser.add_argument("--report_dir",   type=str, default=None,
                        help="Directory to write eval reports (defaults to adapter_dir).")
    parser.add_argument("--compare_base", action="store_true",
                        help="Also generate responses from the raw base model for comparison.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    report_dir = args.report_dir or args.adapter_dir
    os.makedirs(report_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")

    # Load fine-tuned model
    ft_model, ft_tokenizer, _ = load_model(args.adapter_dir, device)

    # Optionally load base model
    base_model, base_tokenizer = None, None
    if args.compare_base:
        base_model, base_tokenizer, _ = load_base_model(device)

    results = []
    print(f"\n[Eval] Running inference on {len(EVAL_PROMPTS)} prompts...\n")

    for i, prompt_data in enumerate(EVAL_PROMPTS):
        print(f"  [{i+1}/{len(EVAL_PROMPTS)}] {prompt_data['category']}: {prompt_data['instruction'][:60]}...")
        entry = {
            "id":          i + 1,
            "category":    prompt_data["category"],
            "instruction": prompt_data["instruction"],
        }

        t0 = time.time()
        entry["finetuned_response"] = generate_response(
            ft_model, ft_tokenizer, prompt_data["instruction"], args.max_new_tokens
        )
        entry["inference_time_s"] = round(time.time() - t0, 2)

        if base_model is not None:
            entry["base_response"] = generate_response(
                base_model, base_tokenizer, prompt_data["instruction"], args.max_new_tokens
            )

        results.append(entry)

    # ── Write JSON report ──────────────────────────────────────────────────
    json_path = os.path.join(report_dir, "eval_report.json")
    report_data = {
        "adapter_dir":    args.adapter_dir,
        "timestamp":      datetime.now().isoformat(),
        "device":         device,
        "compare_base":   args.compare_base,
        "num_prompts":    len(results),
        "results":        results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Report] JSON report written to: {json_path}")

    # ── Write Markdown report ──────────────────────────────────────────────
    md_path = os.path.join(report_dir, "eval_report.md")
    write_markdown_report(results, md_path, args.adapter_dir, args.compare_base)

    print(f"\n[Done] Evaluation complete. {len(results)} prompts processed.")


if __name__ == "__main__":
    main()
