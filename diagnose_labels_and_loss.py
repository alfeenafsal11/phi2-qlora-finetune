import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"

print("=" * 80)
print("PART 3: LABEL COMPARISON & LOSS FORMULATION (Sections 7, 8, 9)")
print("=" * 80)

dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")
split_data = dataset.train_test_split(test_size=0.1, seed=42)
eval_data = split_data["test"]

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Inspect Example 3 (which has non-empty input)
ex = eval_data[3]
inst = ex["instruction"].strip()
inp = ex["input"].strip()
out = ex["output"].strip()

# 1. Phase 1 format (No Input, No EOS)
p1_text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
p1_ids = tokenizer(p1_text, truncation=True, max_length=512)["input_ids"]

# 2. Phase 2 format (With Input, With EOS)
p2_text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
p2_ids = tokenizer(p2_text, truncation=True, max_length=512)["input_ids"]

print(f"Example 3 has non-empty input: '{inp[:40]}...'")
print(f"Phase 1 text length: {len(p1_text)} chars, {len(p1_ids)} tokens")
print(f"Phase 2 text length: {len(p2_text)} chars, {len(p2_ids)} tokens")

# Label construction:
# In Phase 1 Full-Sequence: shift_labels = input_ids[1:] (all active)
# In Phase 1 Response-Only:
# Prompt tokens masked to -100, response tokens active
rep_ids = tokenizer.encode("### Response:\n", add_special_tokens=False)
rlen = len(rep_ids)

# Find response start in Phase 1
p1_match = None
for i in range(len(p1_ids) - rlen + 1):
    if p1_ids[i:i+rlen] == rep_ids:
        p1_match = i
        break
p1_resp_start = p1_match + rlen

# Find response start in Phase 2
p2_match = None
for i in range(len(p2_ids) - rlen + 1):
    if p2_ids[i:i+rlen] == rep_ids:
        p2_match = i
        break
p2_resp_start = p2_match + rlen

print(f"\nPhase 1 Response Delimiter match position: token index {p1_match}")
print(f"Phase 1 Response start position:           token index {p1_resp_start}")
print(f"Phase 2 Response Delimiter match position: token index {p2_match}")
print(f"Phase 2 Response start position:           token index {p2_resp_start}")

print("\n--- TOKEN REGION ACTIVE/MASKED STATUS (Section 7) ---")
regions = [
    ("Instruction header ('### Instruction:\\n')", 0, 4),
    ("Instruction body", 4, p1_match),
    ("Input header & body ('### Input:\\n...')", None, None),
    ("Response header ('### Response:\\n')", p1_match, p1_resp_start),
    ("Response body (Python code)", p1_resp_start, len(p1_ids)),
    ("Terminal EOS token (<|endoftext|>)", None, None),
]

print(f"{'Token Region':<40} | {'Phase 1 Full':<15} | {'Phase 2 Full':<15} | {'Phase 1 Resp-Only':<18} | {'Phase 2 Resp-Only':<18}")
print("-" * 115)
print(f"{'Instruction header':<40} | {'ACTIVE':<15} | {'ACTIVE':<15} | {'MASKED (-100)':<18} | {'MASKED (-100)':<18}")
print(f"{'Instruction body':<40} | {'ACTIVE':<15} | {'ACTIVE':<15} | {'MASKED (-100)':<18} | {'MASKED (-100)':<18}")
print(f"{'Input header & body':<40} | {'OMITTED (NOT IN P1)':<15} | {'ACTIVE':<15} | {'OMITTED (NOT IN P1)':<18} | {'MASKED (-100)':<18}")
print(f"{'Response header':<40} | {'ACTIVE':<15} | {'ACTIVE':<15} | {'MASKED (-100)':<18} | {'MASKED (-100)':<18}")
print(f"{'Response body (code)':<40} | {'ACTIVE':<15} | {'ACTIVE':<15} | {'ACTIVE':<18} | {'ACTIVE':<18}")
print(f"{'Terminal EOS (<|endoftext|>)':<40} | {'OMITTED (NOT IN P1)':<15} | {'ACTIVE':<15} | {'OMITTED (NOT IN P1)':<18} | {'ACTIVE':<18}")

print("\n--- LOSS FORMULA ANALYSIS (Section 8) ---")
print("Formula 1: Token-weighted mean (Total Sum of Cross-Entropy Losses / Total Evaluated Tokens)")
print("  Used in eval_harness.py:")
print("    loss_full = F.cross_entropy(shift_logits, shift_labels, reduction='sum')")
print("    avg_full_loss = total_full_loss / total_full_tokens")
print("  Denominator: sum of all non-ignored tokens across the entire evaluation set.")

print("\nFormula 2: Sample-weighted mean (Mean of Per-Sample Mean Losses)")
print("  Used in standard HuggingFace Trainer eval loop (if batch_size=1):")
print("    sample_loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')")
print("    avg_loss = mean(sample_losses)")
print("  Denominator: 200 (equal weight to short and long samples).")
