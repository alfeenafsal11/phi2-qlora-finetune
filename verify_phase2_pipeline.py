"""
verify_phase2_pipeline.py — Comprehensive pre-training verification script for Phase 2.
Implements all checks required by Phase 2 core protocol and Addendum Controls A-E.
"""

import sys
import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from trl import DataCollatorForCompletionOnlyLM

def main():
    print("=" * 80)
    print("PHASE 2 — PRE-TRAINING VERIFICATION SUITE")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CONTROL A: PAD / EOS TOKEN VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 1. PAD / EOS TOKEN VERIFICATION ---")
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    print(f"tokenizer.eos_token:       {repr(tokenizer.eos_token)}")
    print(f"tokenizer.eos_token_id:    {tokenizer.eos_token_id}")
    print(f"tokenizer.pad_token:       {repr(tokenizer.pad_token)}")
    print(f"tokenizer.pad_token_id:    {tokenizer.pad_token_id}")
    print(f"model.config.eos_token_id: {config.eos_token_id}")
    print(f"model.config.pad_token_id: {config.pad_token_id}")

    # Baseline convention in train.py: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token
    print(f"After baseline setup (pad_token = eos_token):")
    print(f"  tokenizer.pad_token:    {repr(tokenizer.pad_token)}")
    print(f"  tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    pad_eos_shared = (tokenizer.pad_token_id == tokenizer.eos_token_id)
    print(f"  PAD and EOS share token ID: {pad_eos_shared}")

    assert tokenizer.eos_token is not None, "FATAL: EOS token is None"
    assert tokenizer.eos_token_id is not None, "FATAL: EOS token ID is None"

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DATASET AUDIT & PROMPT FORMATTING (CASE A & CASE B)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 2. DATASET AUDIT & PROMPT FORMATTING (FIX #1) ---")
    dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
    raw_dataset = None
    for attempt in range(5):
        try:
            print(f"Loading dataset {dataset_name} (attempt {attempt+1}/5)...")
            raw_dataset = load_dataset(dataset_name, split="train[:2000]")
            break
        except Exception as e:
            print(f"Network glitch loading dataset: {e}. Retrying in 3s...")
            import time
            time.sleep(3)
    if raw_dataset is None:
        raise RuntimeError("Failed to load dataset after 5 attempts.")
    total_examples = len(raw_dataset)

    # Train/eval split matching baseline: 1800 train, 200 eval (test_size=0.1, seed=42)
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Total dataset slice: {total_examples}")
    print(f"Train samples:       {len(train_dataset)}")
    print(f"Eval samples:        {len(eval_dataset)}")

    def format_example_phase2(instruction: str, input_text: str, output_text: str, eos_token: str) -> str:
        """Controlled Case A / Case B formatter."""
        inst = (instruction or "").strip()
        inp = (input_text or "").strip()
        out = (output_text or "").strip()
        if inp:
            # Case A: Non-empty input
            return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{eos_token}"
        else:
            # Case B: Empty input
            return f"### Instruction:\n{inst}\n\n### Response:\n{out}{eos_token}"

    def format_sample_batch(sample):
        """Batch formatter for SFTTrainer."""
        results = []
        for inst, inp, out in zip(sample["instruction"], sample["input"], sample["output"]):
            results.append(format_example_phase2(inst, inp, out, tokenizer.eos_token))
        return results

    # Count input distribution across all 2000 samples
    non_empty_count = sum(1 for x in raw_dataset if (x.get("input") or "").strip() != "")
    empty_count = total_examples - non_empty_count
    pct_non_empty = (non_empty_count / total_examples) * 100

    print(f"Non-empty 'input' examples: {non_empty_count} ({pct_non_empty:.2f}%)")
    print(f"Empty 'input' examples:     {empty_count} ({100 - pct_non_empty:.2f}%)")

    # Verify that formatter preserves input
    lost_input_count = 0
    preserved_input_count = 0
    for ex in raw_dataset:
        inp = (ex.get("input") or "").strip()
        formatted = format_example_phase2(ex["instruction"], ex["input"], ex["output"], tokenizer.eos_token)
        if inp:
            if f"### Input:\n{inp}" in formatted:
                preserved_input_count += 1
            else:
                lost_input_count += 1

    print(f"Preserved non-empty inputs: {preserved_input_count} / {non_empty_count} ({preserved_input_count/non_empty_count*100:.2f}%)")
    print(f"Lost non-empty inputs:      {lost_input_count}")
    assert lost_input_count == 0, "FATAL: Formatter dropped non-empty input!"

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PRINT 5 REPRESENTATIVE EXAMPLES (RAW vs FORMATTED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 3. 5 REPRESENTATIVE EXAMPLES (RAW vs FORMATTED) ---")
    
    # 1: Non-empty input
    idx_non_empty = next(i for i, x in enumerate(raw_dataset) if len((x.get("input") or "").strip()) > 30 and "def " not in x.get("input"))
    # 2: Empty input
    idx_empty = next(i for i, x in enumerate(raw_dataset) if not (x.get("input") or "").strip())
    # 3: Short input
    idx_short = next(i for i, x in enumerate(raw_dataset) if 0 < len((x.get("input") or "").strip()) <= 20)
    # 4: Longer input
    idx_longer = next(i for i, x in enumerate(raw_dataset) if len((x.get("input") or "").strip()) > 100)
    # 5: Code-related instruction with input
    idx_code = next(i for i, x in enumerate(raw_dataset) if "function" in x.get("instruction", "").lower() and (x.get("input") or "").strip())

    representative_indices = [
        ("1. Non-empty input", idx_non_empty),
        ("2. Empty input", idx_empty),
        ("3. Short input", idx_short),
        ("4. Longer input", idx_longer),
        ("5. Code-related instruction", idx_code),
    ]

    for label, idx in representative_indices:
        ex = raw_dataset[idx]
        formatted = format_example_phase2(ex["instruction"], ex["input"], ex["output"], tokenizer.eos_token)
        print(f"\n[{label}] (Index {idx}):")
        print("RAW:")
        print(f"  instruction: {repr(ex['instruction'][:120])}")
        print(f"  input:       {repr(ex['input'][:120])}")
        print(f"  output:      {repr(ex['output'][:120])}")
        print("FORMATTED (truncated preview):")
        lines = formatted.split("\n")
        preview = "\n  ".join(lines[:8])
        if len(lines) > 8:
            preview += f"\n  ... ({len(lines)-8} more lines)"
        print(f"  {preview}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FIX #2 & CONTROL E: EOS VERIFICATION & RETENTION UNDER TRUNCATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 4. EOS VERIFICATION & TRUNCATION RETENTION (FIX #2 & CONTROL E) ---")
    max_seq_length = 512

    # Measure across train and eval sets
    for split_name, ds in [("Train", train_dataset), ("Eval", eval_dataset)]:
        n_total = len(ds)
        n_ends_with_eos_str = 0
        n_eos_retained_after_trunc = 0
        n_eos_truncated = 0
        n_response_truncated = 0
        n_prompt_truncated = 0
        seq_lengths = []

        for ex in ds:
            formatted = format_example_phase2(ex["instruction"], ex["input"], ex["output"], tokenizer.eos_token)
            if formatted.endswith(tokenizer.eos_token):
                n_ends_with_eos_str += 1

            tokens = tokenizer.encode(formatted, add_special_tokens=False)
            seq_len = len(tokens)
            seq_lengths.append(seq_len)

            # Prompt boundary
            prompt_str = format_example_phase2(ex["instruction"], ex["input"], "", "")
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
            n_prompt_tokens = len(prompt_tokens)

            if n_prompt_tokens >= max_seq_length:
                n_prompt_truncated += 1

            if seq_len > max_seq_length:
                n_eos_truncated += 1
                n_response_truncated += 1
            else:
                # Sequence fits inside max_seq_length
                if tokens[-1] == tokenizer.eos_token_id:
                    n_eos_retained_after_trunc += 1
                else:
                    n_eos_truncated += 1

        print(f"[{split_name} Set ({n_total} samples)]:")
        print(f"  EOS token:                             {repr(tokenizer.eos_token)}")
        print(f"  EOS token ID:                          {tokenizer.eos_token_id}")
        print(f"  Formatted samples ending with EOS str: {n_ends_with_eos_str} / {n_total} ({n_ends_with_eos_str/n_total*100:.2f}%)")
        print(f"  Mean sequence length:                  {np.mean(seq_lengths):.1f} tokens (min={min(seq_lengths)}, max={max(seq_lengths)})")
        print(f"  Samples > {max_seq_length} (truncated):           {n_eos_truncated} / {n_total} ({n_eos_truncated/n_total*100:.2f}%)")
        print(f"  EOS retained after truncation:         {n_eos_retained_after_trunc} / {n_total} ({n_eos_retained_after_trunc/n_total*100:.2f}%)")
        print(f"  Prompt truncated (>={max_seq_length}):            {n_prompt_truncated} / {n_total} ({n_prompt_truncated/n_total*100:.2f}%)")
        print(f"  Response truncated:                    {n_response_truncated} / {n_total} ({n_response_truncated/n_total*100:.2f}%)")

        assert n_ends_with_eos_str == n_total, f"FATAL: Not all {split_name} samples end with EOS string!"

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CONTROL B: RESPONSE TEMPLATE TOKEN-LEVEL VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 5. RESPONSE TEMPLATE TOKEN-LEVEL VERIFICATION (CONTROL B) ---")
    response_template_str = "### Response:\n"
    response_token_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
    print(f"Response template string:    {repr(response_template_str)}")
    print(f"Response template token IDs: {response_token_ids}")
    print(f"Decoded token-by-token:      {[tokenizer.decode([t]) for t in response_token_ids]}")

    # Verify that in all representative samples and a 200-sample test batch,
    # the response template matches exactly once
    print("\nVerifying single-match invariant across 200 random samples...")
    single_match_count = 0
    multi_match_count = 0
    zero_match_count = 0

    for i in range(200):
        ex = raw_dataset[i]
        formatted = format_example_phase2(ex["instruction"], ex["input"], ex["output"], tokenizer.eos_token)
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        
        matches = []
        rlen = len(response_token_ids)
        for idx in range(len(tokens) - rlen + 1):
            if tokens[idx : idx + rlen] == response_token_ids:
                matches.append(idx)

        if len(matches) == 1:
            single_match_count += 1
        elif len(matches) > 1:
            multi_match_count += 1
        else:
            zero_match_count += 1

    print(f"Exact 1 match:  {single_match_count} / 200 ({single_match_count/200*100:.1f}%)")
    print(f"Multi-match:    {multi_match_count} / 200")
    print(f"Zero-match:     {zero_match_count} / 200")
    assert zero_match_count == 0, "FATAL: Response template not found in some examples!"
    assert multi_match_count == 0, f"FATAL: Response template matched multiple times in {multi_match_count} examples!"

    # ─────────────────────────────────────────────────────────────────────────
    # 6. FIX #3 & CONTROL D: RESPONSE-ONLY LOSS MASKING & LABEL STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 6. RESPONSE-ONLY LOSS MASKING & LABEL-MASK STATISTICS (FIX #3 & CONTROL D) ---")
    class Phase2CompletionCollator(DataCollatorForCompletionOnlyLM):
        def torch_call(self, examples):
            batch = super().torch_call(examples)
            for i, ex in enumerate(examples):
                raw_ids = ex["input_ids"] if isinstance(ex, dict) else ex
                if len(raw_ids) > 0 and raw_ids[-1] == self.tokenizer.eos_token_id:
                    # Terminal token of unpadded sequence is genuine EOS; supervise it!
                    batch["labels"][i, len(raw_ids) - 1] = self.tokenizer.eos_token_id
            return batch

    collator = Phase2CompletionCollator(
        response_template=response_template_str,
        tokenizer=tokenizer,
        mlm=False,
        ignore_index=-100,
    )

    print(f"DataCollator initialized with:")
    print(f"  response_template: {repr(collator.response_template)}")
    print(f"  response_token_ids:{collator.response_token_ids}")
    print(f"  ignore_index:      {collator.ignore_index}")

    # Test representative examples through collator
    for label, idx in representative_indices:
        ex = raw_dataset[idx]
        formatted = format_example_phase2(ex["instruction"], ex["input"], ex["output"], tokenizer.eos_token)
        
        tokenized = tokenizer(
            formatted,
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None,
            add_special_tokens=False,
        )
        
        # SFTTrainer formats batch as list of dicts with 'input_ids'
        batch = collator([{"input_ids": tokenized["input_ids"]}])
        
        input_ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()
        
        total_tokens = len(input_ids)
        masked_tokens = sum(1 for l in labels if l == -100)
        unmasked_tokens = sum(1 for l in labels if l != -100)

        # Locate response template in input_ids
        rlen = len(response_token_ids)
        match_idx = None
        for i in range(len(input_ids) - rlen + 1):
            if input_ids[i : i + rlen] == response_token_ids:
                match_idx = i
                break

        assert match_idx is not None, f"Response template not found in {label}"
        response_start_idx = match_idx + rlen

        prompt_tokens_count = response_start_idx
        response_and_eos_tokens_count = total_tokens - response_start_idx

        # Verify mask invariant:
        # Prompt tokens up to response_start_idx must ALL be -100
        prompt_labels = labels[:response_start_idx]
        response_labels = labels[response_start_idx:]
        response_input_ids = input_ids[response_start_idx:]

        assert all(l == -100 for l in prompt_labels), f"FATAL: Prompt token unmasked in {label}!"
        assert response_labels == response_input_ids, f"FATAL: Response token label mismatch in {label}!"
        
        is_truncated = (len(tokenizer.encode(formatted, add_special_tokens=False)) > max_seq_length)
        if not is_truncated:
            assert labels[-1] == tokenizer.eos_token_id, f"FATAL: Last label is not EOS token ID in non-truncated {label}!"
            eos_status = f"1 (ID={labels[-1]}) [Retained]"
        else:
            eos_status = "Truncated (> 512 tokens)"

        print(f"\n[{label}] Label-Mask Statistics:")
        print(f"  Total tokens (clipped to {max_seq_length}): {total_tokens}")
        print(f"  Prompt tokens:                     {prompt_tokens_count}")
        print(f"  Response tokens:                   {response_and_eos_tokens_count - (1 if not is_truncated else 0)}")
        print(f"  EOS token:                         {eos_status}")
        print(f"  Masked tokens:                     {masked_tokens} (-100)")
        print(f"  Unmasked tokens:                   {unmasked_tokens} (active loss tokens)")
        print(f"  Invariant holds:                   unmasked ({unmasked_tokens}) == response+EOS ({response_and_eos_tokens_count}) -> TRUE")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. HUMAN-READABLE TOKEN/LABEL DIAGNOSTIC AROUND BOUNDARIES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 7. TOKEN / LABEL ALIGNMENT DIAGNOSTIC AROUND BOUNDARIES ---")
    # Use Case A representative example (non-empty input)
    ex_case_a = raw_dataset[idx_non_empty]
    formatted_case_a = format_example_phase2(ex_case_a["instruction"], ex_case_a["input"], ex_case_a["output"], tokenizer.eos_token)
    tokenized_case_a = tokenizer(formatted_case_a, truncation=True, max_length=max_seq_length, add_special_tokens=False)
    batch_case_a = collator([{"input_ids": tokenized_case_a["input_ids"]}])
    ids_a = batch_case_a["input_ids"][0].tolist()
    labels_a = batch_case_a["labels"][0].tolist()

    # Find boundaries
    rlen = len(response_token_ids)
    resp_match = next(i for i in range(len(ids_a) - rlen + 1) if ids_a[i : i + rlen] == response_token_ids)

    print("\nA. Prompt Header Boundary (First 6 tokens):")
    for i in range(min(6, len(ids_a))):
        tok_str = tokenizer.decode([ids_a[i]])
        print(f"  Pos {i:3d} | ID {ids_a[i]:5d} | Label {labels_a[i]:5d} | Token: {repr(tok_str)}")

    print(f"\nB. Response Template Boundary (Positions {resp_match - 2} to {resp_match + rlen + 4}):")
    for i in range(max(0, resp_match - 2), min(len(ids_a), resp_match + rlen + 5)):
        tok_str = tokenizer.decode([ids_a[i]])
        marker = " <== RESPONSE BOUNDARY START" if i == resp_match + rlen else ""
        print(f"  Pos {i:3d} | ID {ids_a[i]:5d} | Label {labels_a[i]:5d} | Token: {repr(tok_str)}{marker}")

    print(f"\nC. EOS Termination Boundary (Last 4 tokens):")
    for i in range(max(0, len(ids_a) - 4), len(ids_a)):
        tok_str = tokenizer.decode([ids_a[i]])
        marker = " <== EOS TARGET TOKEN" if i == len(ids_a) - 1 else ""
        print(f"  Pos {i:3d} | ID {ids_a[i]:5d} | Label {labels_a[i]:5d} | Token: {repr(tok_str)}{marker}")

    print("\n" + "=" * 80)
    print("ALL PRE-TRAINING VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
