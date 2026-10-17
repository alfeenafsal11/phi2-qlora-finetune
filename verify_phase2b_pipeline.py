"""
verify_phase2b_pipeline.py — Verification script for Phase 2B data pipeline and label masking.
Checks 5 representative dataset samples across:
  1. Reference Condition (Input=YES, EOS=YES, Masking=YES)
  2. Ablation A: EOS Ablation (Input=YES, EOS=NO, Masking=YES)
  3. Ablation B: Masking Ablation (Input=YES, EOS=YES, Masking=NO)
  4. Ablation C: Input Ablation (Input=NO, EOS=YES, Masking=YES)
"""

import os
import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

os.environ["HF_HUB_OFFLINE"] = "1"

class Phase2CompletionCollator(DataCollatorForCompletionOnlyLM):
    """Preserves terminal EOS token supervision when pad_token == eos_token."""
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i, ex in enumerate(examples):
            raw_ids = ex["input_ids"] if isinstance(ex, dict) else ex
            if len(raw_ids) > 0 and raw_ids[-1] == self.tokenizer.eos_token_id:
                batch["labels"][i, len(raw_ids) - 1] = self.tokenizer.eos_token_id
        return batch


class Phase2FullSequenceCollator:
    """
    Collator for full causal LM sequence loss (Masking=NO).
    Labels = input_ids.clone().
    Pad tokens beyond sequence length are set to -100, but genuine sequence EOS is preserved.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids_list = [torch.tensor(ex["input_ids"]) if not isinstance(ex["input_ids"], torch.Tensor) else ex["input_ids"] for ex in examples]
        max_len = max(len(ids) for ids in input_ids_list)
        batch_size = len(input_ids_list)
        
        padded_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            seq_len = len(ids)
            padded_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = ids

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def format_prompt(instruction: str, input_text: str, output_text: str, input_mode: str, eos_mode: str, eos_token: str) -> str:
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    out = (output_text or "").strip()

    eos_str = eos_token if eos_mode == "with_eos" else ""

    if input_mode == "with_input" and inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{eos_str}"
    else:
        return f"### Instruction:\n{inst}\n\n### Response:\n{out}{eos_str}"


def main():
    print("=" * 80)
    print("PHASE 2B DATA PIPELINE & LABEL MASKING AUDIT")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id

    # Load dataset exactly as in training
    dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]

    # Pick 5 representative samples: mix of empty and non-empty inputs
    test_indices = [0, 1, 2, 3, 4]
    samples = [train_dataset[i] for i in test_indices]

    conditions = [
        {
            "name": "Reference Condition (Phase 2 Corrected)",
            "input_mode": "with_input",
            "eos_mode": "with_eos",
            "masking_mode": "response_only",
        },
        {
            "name": "Ablation A (EOS Ablation)",
            "input_mode": "with_input",
            "eos_mode": "no_eos",
            "masking_mode": "response_only",
        },
        {
            "name": "Ablation B (Masking Ablation)",
            "input_mode": "with_input",
            "eos_mode": "with_eos",
            "masking_mode": "full_sequence",
        },
        {
            "name": "Ablation C (Input Ablation)",
            "input_mode": "omit_input",
            "eos_mode": "with_eos",
            "masking_mode": "response_only",
        },
    ]

    response_template = "### Response:\n"
    resp_collator = Phase2CompletionCollator(response_template=response_template, tokenizer=tokenizer, mlm=False, ignore_index=-100)
    full_collator = Phase2FullSequenceCollator(tokenizer=tokenizer)

    audit_report = {}

    for cond in conditions:
        cname = cond["name"]
        imode = cond["input_mode"]
        emode = cond["eos_mode"]
        mmode = cond["masking_mode"]

        print(f"\n" + "=" * 70)
        print(f"CONDITION: {cname}")
        print(f"Configuration: input={imode}, eos={emode}, masking={mmode}")
        print("=" * 70)

        collator = resp_collator if mmode == "response_only" else full_collator
        cond_samples_log = []

        for idx, sample in enumerate(samples):
            inst = sample["instruction"]
            inp = sample["input"]
            out = sample["output"]

            formatted_text = format_prompt(inst, inp, out, imode, emode, tokenizer.eos_token)
            enc = tokenizer(formatted_text, truncation=True, max_length=512)
            input_ids = enc["input_ids"]

            # Run through collator as single-example batch
            batch = collator([{"input_ids": input_ids}])
            labels = batch["labels"][0].tolist()
            b_input_ids = batch["input_ids"][0].tolist()

            total_tokens = len(b_input_ids)
            active_tokens = sum(1 for l in labels if l != -100)
            masked_tokens = sum(1 for l in labels if l == -100)
            active_ratio = active_tokens / max(1, total_tokens)

            has_eos_token = (eos_token_id in b_input_ids)
            eos_position = b_input_ids.index(eos_token_id) if has_eos_token else None
            is_eos_at_end = (b_input_ids[-1] == eos_token_id) if total_tokens > 0 else False
            eos_label = labels[-1] if total_tokens > 0 else None

            # Find response boundary
            resp_ids = tokenizer.encode(response_template, add_special_tokens=False)
            boundary_idx = None
            for i in range(len(b_input_ids) - len(resp_ids) + 1):
                if b_input_ids[i : i + len(resp_ids)] == resp_ids:
                    boundary_idx = i
                    break

            sample_info = {
                "sample_idx": idx,
                "has_raw_input": bool(inp.strip() if inp else False),
                "instruction_preview": inst[:60] + "..." if len(inst) > 60 else inst,
                "input_preview": (inp[:40] + "...") if inp else "(empty)",
                "output_preview": out[:60].replace("\n", "\\n") + "...",
                "formatted_text_preview": formatted_text[:100].replace("\n", "\\n") + "...",
                "total_tokens": total_tokens,
                "active_tokens": active_tokens,
                "masked_tokens": masked_tokens,
                "active_ratio": round(active_ratio, 4),
                "has_eos": has_eos_token,
                "eos_at_end": is_eos_at_end,
                "eos_position": eos_position,
                "eos_label": eos_label,
                "response_boundary_idx": boundary_idx,
            }
            cond_samples_log.append(sample_info)

            print(f"\n  [Sample {idx}] has_input={sample_info['has_raw_input']} | Total tokens: {total_tokens} | Active: {active_tokens} ({active_ratio*100:.1f}%) | Masked: {masked_tokens}")
            print(f"    EOS in text: {has_eos_token} (at end: {is_eos_at_end}, label: {eos_label})")
            print(f"    Response boundary at token index: {boundary_idx}")
            if mmode == "response_only":
                if boundary_idx is not None:
                    prompt_labels = labels[:boundary_idx + len(resp_ids)]
                    all_prompt_masked = all(l == -100 for l in prompt_labels)
                    print(f"    Strict prompt masking verified: {all_prompt_masked}")
            else:
                if boundary_idx is not None:
                    prompt_labels = labels[:boundary_idx + len(resp_ids)]
                    all_prompt_active = all(l != -100 for l in prompt_labels)
                    print(f"    Full causal LM supervision verified: {all_prompt_active}")

        audit_report[cname] = {
            "condition": cond,
            "samples": cond_samples_log,
        }

    os.makedirs("reports", exist_ok=True)
    report_file = "reports/phase2b_data_pipeline_verification.json"
    with open(report_file, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n[Audit Complete] Saved audit verification report to {report_file}")


if __name__ == "__main__":
    main()
