import os
import json
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"

print("=" * 80)
print("PART 1: DATASET IDENTITY & MANIFEST CHECK (Section 13)")
print("=" * 80)

dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")
split_data = dataset.train_test_split(test_size=0.1, seed=42)
eval_data = split_data["test"]
train_data = split_data["train"]

print(f"Total loaded: {len(dataset)}")
print(f"Train samples: {len(train_data)}")
print(f"Eval samples:  {len(eval_data)}")

manifest = []
for idx, ex in enumerate(eval_data):
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    
    content = f"INST:{inst}###INP:{inp}###OUT:{out}"
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest.append({
        "eval_idx": idx,
        "sha256": h,
        "has_input": bool(inp),
        "inst_len": len(inst),
        "inp_len": len(inp),
        "out_len": len(out),
    })

os.makedirs("reports", exist_ok=True)
with open("reports/eval_dataset_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved manifest for {len(manifest)} eval samples to reports/eval_dataset_manifest.json")
has_input_count = sum(1 for m in manifest if m["has_input"])
print(f"Samples with non-empty input: {has_input_count} / {len(manifest)} ({has_input_count/len(manifest)*100:.1f}%)")

print("\n" + "=" * 80)
print("PART 2: TOKENIZATION & TRUNCATION COMPARISON (Sections 6 & 14)")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Compare 5 examples in detail
print("\n--- DETAILED COMPARISON FOR 5 EXAMPLES (Section 6) ---")
for idx in range(5):
    ex = eval_data[idx]
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    
    # Phase 1 format
    p1_text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
    p1_enc = tokenizer(p1_text, truncation=False)
    p1_ids = p1_enc["input_ids"]
    
    # Phase 2 format
    if inp:
        p2_text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
    else:
        p2_text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}"
    p2_enc = tokenizer(p2_text, truncation=False)
    p2_ids = p2_enc["input_ids"]
    
    print(f"\nExample ID: {idx}")
    print(f"RAW FIELDS:")
    print(f"  instruction: {inst[:60]}...")
    print(f"  input:       {inp[:60] if inp else '(EMPTY)'}...")
    print(f"  output:      {out[:60]}...")
    print(f"PHASE 1 TOKEN COUNT: {len(p1_ids)}")
    print(f"PHASE 2 TOKEN COUNT: {len(p2_ids)}")
    print(f"TOKEN COUNT DIFFERENCE: {len(p2_ids) - len(p1_ids)} tokens")
    print(f"PHASE 1 FIRST 10 TOKEN IDS: {p1_ids[:10]}")
    print(f"PHASE 2 FIRST 10 TOKEN IDS: {p2_ids[:10]}")
    print(f"PHASE 1 LAST 5 TOKEN IDS:   {p1_ids[-5:]}")
    print(f"PHASE 2 LAST 5 TOKEN IDS:   {p2_ids[-5:]} (Last token is {p2_ids[-1]} = {tokenizer.decode([p2_ids[-1]])})")

# Full 200 samples tokenization & truncation stats
p1_lens_raw = []
p2_lens_raw = []
p1_truncated = 0
p2_truncated = 0

p1_shift_tokens_512 = 0
p1_resp_tokens_512 = 0
p2_shift_tokens_512 = 0
p2_resp_tokens_512 = 0

rep_ids = tokenizer.encode("### Response:\n", add_special_tokens=False)
rlen = len(rep_ids)

for ex in eval_data:
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    
    # Phase 1
    p1_text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
    p1_full_ids = tokenizer(p1_text, truncation=False)["input_ids"]
    p1_lens_raw.append(len(p1_full_ids))
    if len(p1_full_ids) > 512:
        p1_truncated += 1
    
    p1_ids_512 = tokenizer(p1_text, truncation=True, max_length=512)["input_ids"]
    p1_shift_tokens_512 += len(p1_ids_512) - 1
    for i in range(len(p1_ids_512) - rlen + 1):
        if p1_ids_512[i:i+rlen] == rep_ids:
            resp_start = i + rlen
            p1_resp_tokens_512 += max(0, len(p1_ids_512) - resp_start)
            break
            
    # Phase 2
    if inp:
        p2_text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
    else:
        p2_text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}"
    p2_full_ids = tokenizer(p2_text, truncation=False)["input_ids"]
    p2_lens_raw.append(len(p2_full_ids))
    if len(p2_full_ids) > 512:
        p2_truncated += 1
        
    p2_ids_512 = tokenizer(p2_text, truncation=True, max_length=512)["input_ids"]
    p2_shift_tokens_512 += len(p2_ids_512) - 1
    for i in range(len(p2_ids_512) - rlen + 1):
        if p2_ids_512[i:i+rlen] == rep_ids:
            resp_start = i + rlen
            p2_resp_tokens_512 += max(0, len(p2_ids_512) - resp_start)
            break

print("\n--- 200 SAMPLES AGGREGATE TOKEN & TRUNCATION COMPARISON (Section 14) ---")
print(f"Phase 1 (No Input, No EOS):")
print(f"  Total raw tokens (no truncation): {sum(p1_lens_raw)}")
print(f"  Samples > 512 tokens:             {p1_truncated} / 200 ({p1_truncated/200*100:.1f}%)")
print(f"  Total evaluated tokens (max=512): {p1_shift_tokens_512} (Shifted target tokens)")
print(f"  Total response tokens (max=512):  {p1_resp_tokens_512}")

print(f"\nPhase 2 (With Input, With EOS):")
print(f"  Total raw tokens (no truncation): {sum(p2_lens_raw)}")
print(f"  Samples > 512 tokens:             {p2_truncated} / 200 ({p2_truncated/200*100:.1f}%)")
print(f"  Total evaluated tokens (max=512): {p2_shift_tokens_512} (Shifted target tokens)")
print(f"  Total response tokens (max=512):  {p2_resp_tokens_512}")

print(f"\nExact Token Differences:")
print(f"  Target tokens difference:   {p2_shift_tokens_512 - p1_shift_tokens_512} tokens")
print(f"  Response tokens difference: {p2_resp_tokens_512 - p1_resp_tokens_512} tokens")
