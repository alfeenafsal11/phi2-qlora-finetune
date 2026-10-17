"""
train_phase4.py -- Phase 4 Config E: Scaled Dataset QLoRA Training for Phi-2

Config E:
  - Dataset: iamtarun/python_code_instructions_18k_alpaca, train[:5000]
  - 90/10 split => 4,500 train / 500 eval, seed=42
  - target_modules: q_proj, v_proj (Locked Decision #2)
  - Loss masking: response-only, Phase2CompletionCollator (Locked Decision #1)
  - LoRA: r=16, alpha=32, dropout=0.05
  - Optimizer: adamw_torch, LR=2e-4, cosine schedule, warmup_ratio=0.03
  - Batch size: 1, grad_accum=8 (effective 8)
  - num_train_epochs=2 over 4,500 samples => ~1,125 steps
  - save_steps=100, eval_steps=100
  - Supports category-balanced sampling via --balanced_sampling flag
  - VRAM logged every checkpoint step
  - Sleep prevention via powercfg
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import platform
import datetime
import subprocess
import argparse
import numpy as np
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ────────────────────────────────────────────────────────────────────────────
# Locked constants — do NOT change without explicit Phase 5 brief
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR       = "phi2-qlora-phase4-scaled-5k"
TARGET_MODULES   = ["q_proj", "v_proj"]
MASKING_MODE     = "response_only"       # Locked Decision #1
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
NUM_SAMPLES      = 5000
EVAL_RATIO       = 0.1
NUM_EPOCHS       = 2
BATCH_SIZE       = 1
GRAD_ACCUM       = 8
LR               = 2e-4
SCHEDULER        = "cosine"
OPTIMIZER        = "adamw_torch"
WARMUP_RATIO     = 0.03
MAX_SEQ_LENGTH   = 512
SAVE_STEPS       = 100
EVAL_STEPS       = 100
LOGGING_STEPS    = 50
SEED             = 42
MODEL_ID         = "microsoft/phi-2"
DATASET_ID       = "iamtarun/python_code_instructions_18k_alpaca"
VRAM_LOG_PATH    = "reports/phase4_vram_log.json"
PIPELINE_LOG_PATH = "reports/phase4_pipeline_log.json"
SUMMARY_PATH     = os.path.join(OUTPUT_DIR, "phase4_train_summary.json")

CATEGORY_AUDIT_PATH = "reports/phase4_category_audit.json"

# VRAM checkpoint log steps — every save_steps
VRAM_LOG_STEPS_SET = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1125}


# ────────────────────────────────────────────────────────────────────────────
# Collators (identical to Phase 2/3 implementations)
# ────────────────────────────────────────────────────────────────────────────

class Phase2CompletionCollator(DataCollatorForCompletionOnlyLM):
    """Response-only loss masking with terminal EOS supervision."""
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i, ex in enumerate(examples):
            raw_ids = ex["input_ids"] if isinstance(ex, dict) else ex
            if len(raw_ids) > 0 and raw_ids[-1] == self.tokenizer.eos_token_id:
                batch["labels"][i, len(raw_ids) - 1] = self.tokenizer.eos_token_id
        return batch


# ────────────────────────────────────────────────────────────────────────────
# VRAM callback
# ────────────────────────────────────────────────────────────────────────────

class VRAMTrackingCallback(TrainerCallback):
    def __init__(self, vram_log_path: str):
        self.vram_log_path = vram_log_path
        self.max_vram_allocated_mb = 0.0
        self.max_vram_reserved_mb  = 0.0
        self.step_vram_log = []

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        res   = torch.cuda.max_memory_reserved()  / (1024 ** 2)
        self.max_vram_allocated_mb = max(self.max_vram_allocated_mb, alloc)
        self.max_vram_reserved_mb  = max(self.max_vram_reserved_mb,  res)

        step = state.global_step
        if step in VRAM_LOG_STEPS_SET or (step > 0 and step % SAVE_STEPS == 0):
            entry = {
                "config": "config_e",
                "step": step,
                "vram_allocated_mb": round(alloc, 2),
                "vram_reserved_mb":  round(res,   2),
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            }
            self.step_vram_log.append(entry)
            print(f"[VRAM] Step {step}: Alloc={alloc:.1f}MB, Reserved={res:.1f}MB", flush=True)
            self._flush(entry)

    def _flush(self, entry: dict):
        os.makedirs(os.path.dirname(self.vram_log_path), exist_ok=True)
        existing = []
        if os.path.exists(self.vram_log_path):
            try:
                with open(self.vram_log_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(entry)
        with open(self.vram_log_path, "w") as f:
            json.dump(existing, f, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# Prompt formatter (Phase 2 standard)
# ────────────────────────────────────────────────────────────────────────────

def create_formatter(tokenizer):
    def format_sample(sample):
        output = []
        for inst, inp, out in zip(sample["instruction"], sample["input"], sample["output"]):
            inst = (inst or "").strip()
            inp  = (inp  or "").strip()
            out  = (out  or "").strip()
            eos  = tokenizer.eos_token
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{eos}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{eos}"
            output.append(text)
        return output
    return format_sample


# ────────────────────────────────────────────────────────────────────────────
# Category-balanced sampling
# ────────────────────────────────────────────────────────────────────────────

CATEGORY_PATTERNS = [
    ("Bug Fixing", [r"\bfix\b", r"\bbug\b", r"\bdebug\b", r"\bbroken\b", r"\bwrong\b", r"\bincorrect\b", r"\bdefect\b"]),
    ("File I/O & Exception Handling", [r"\bfile\b", r"\bexception\b", r"\btry.*except\b", r"\berror handling\b", r"\bwith open\b", r"\bos\.path\b", r"\bpathlib\b", r"\bcsv\b", r"\bpickle\b"]),
    ("OOP & Class Definitions", [r"\bclass\b", r"\binheritanc\b", r"\bsubclass\b", r"\bpolymorphi\b", r"\b__init__\b", r"\binstanc\b", r"\boop\b"]),
    ("Recursion & Dynamic Programming", [r"\brecursi\b", r"\bdynamic programming\b", r"\bmemoiz\b", r"\bbacktrack\b", r"\bknapsack\b", r"\bfibonacci\b"]),
    ("Sorting & Searching", [r"\bsort\b", r"\bbinary search\b", r"\bmerge sort\b", r"\bquick sort\b", r"\bbubble sort\b", r"\blinear search\b"]),
    ("Data Structures", [r"\blinked list\b", r"\bstack\b", r"\bqueue\b", r"\btree\b", r"\bgraph\b", r"\bhash\b", r"\bheap\b", r"\bmatrix\b", r"\bdeque\b", r"\btrie\b"]),
    ("Algorithm Complexity & Edge Cases", [r"\bcomplexity\b", r"\btime complexity\b", r"\bbig.?o\b", r"\bedge case\b", r"\boptimiz\b"]),
    ("String Manipulation", [r"\bstring\b", r"\bcharacter\b", r"\bsubstring\b", r"\bpalindrome\b", r"\banagram\b", r"\bregex\b", r"\bsplit\b", r"\bjoin\b"]),
]


def classify_instruction(instruction: str) -> str:
    text = instruction.lower()
    for category, patterns in CATEGORY_PATTERNS:
        for pat in patterns:
            if re.search(pat, text):
                return category
    return "Uncategorized"


def build_balanced_dataset(full_ds, target_n: int, seed: int = 42) -> Dataset:
    """
    Build a balanced dataset of target_n examples by oversampling underrepresented
    categories (up to 2x their natural frequency) then filling remainder sequentially.
    """
    rng = random.Random(seed)

    # Classify all 18k+ samples
    categorized = {}
    for i, ex in enumerate(full_ds):
        cat = classify_instruction(ex.get("instruction", ""))
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(i)

    # Drop Uncategorized from balancing pool; they fill naturally
    cats = [c for c in categorized if c != "Uncategorized"]
    target_per_cat = target_n // max(1, len(cats))

    selected_indices = []
    for cat in cats:
        pool = categorized[cat]
        if len(pool) >= target_per_cat:
            selected_indices.extend(rng.sample(pool, target_per_cat))
        else:
            # Oversample with replacement (capped at 2x natural count)
            max_oversample = min(target_per_cat, len(pool) * 2)
            sampled = pool.copy()
            while len(sampled) < max_oversample:
                sampled.extend(pool)
            sampled = sampled[:max_oversample]
            rng.shuffle(sampled)
            selected_indices.extend(sampled)

    # Fill any remaining slots from the full dataset sequentially
    used_set = set(selected_indices)
    for i in range(len(full_ds)):
        if len(selected_indices) >= target_n:
            break
        if i not in used_set:
            selected_indices.append(i)
            used_set.add(i)

    selected_indices = selected_indices[:target_n]
    rng.shuffle(selected_indices)
    return full_ds.select(selected_indices)


# ────────────────────────────────────────────────────────────────────────────
# Sleep prevention
# ────────────────────────────────────────────────────────────────────────────

def set_sleep_prevention():
    try:
        subprocess.run(["powercfg", "/requestsoverride", "PROCESS", "python.exe", "SYSTEM"],
                       capture_output=True, timeout=10)
        print("[Sleep] Power override set: system will not sleep during training.")
    except Exception as e:
        print(f"[Sleep] Could not set power override (non-critical): {e}")


def release_sleep_prevention():
    try:
        subprocess.run(["powercfg", "/requestsoverride", "PROCESS", "python.exe"],
                       capture_output=True, timeout=10)
        print("[Sleep] Power override released.")
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────────
# Pipeline logging
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
               "ERROR": "[FAIL]", "WARN": "[WARN]"}.get(level, "[....]")
    print(f"[{ts[:19]}] {prefix} {message}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# Main training routine
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Config E training")
    parser.add_argument("--balanced_sampling", action="store_true",
                        help="Use category-balanced sampling (set per Section 1 audit)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    set_sleep_prevention()

    print("=" * 80)
    print("PHASE 4 TRAINING: CONFIG E (Scaled Dataset, Attention-Only Baseline)")
    print(f"  Output:         {OUTPUT_DIR}")
    print(f"  Target modules: {TARGET_MODULES}")
    print(f"  Masking:        {MASKING_MODE} (Locked Decision #1)")
    print(f"  Dataset:        train[:{NUM_SAMPLES}] ({int(NUM_SAMPLES*0.9)} train / {int(NUM_SAMPLES*0.1)} eval)")
    print(f"  Epochs:         {NUM_EPOCHS}  |  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Save/Eval every {SAVE_STEPS} steps")
    print("=" * 80)

    # Seed everything
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    # Environment manifest
    import transformers, peft, bitsandbytes, datasets as ds_lib, trl, accelerate
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "gpu_total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2) if torch.cuda.is_available() else 0,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "bitsandbytes_version": bitsandbytes.__version__,
        "datasets_version": ds_lib.__version__,
        "trl_version": trl.__version__,
        "accelerate_version": accelerate.__version__,
    }

    log_event("START", "Phase 4 Config E training starting")

    # ── Load tokenizer & model ──
    print(f"\n[Model] Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"[Model] Loading base model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"[LoRA] r={LORA_RANK}, alpha={LORA_ALPHA}, targets={TARGET_MODULES}")
    model.print_trainable_parameters()

    # ── Load & prepare dataset ──
    print(f"\n[Data] Loading dataset: {DATASET_ID}, train[:{NUM_SAMPLES}]")
    full_ds = load_dataset(DATASET_ID, split="train")

    if args.balanced_sampling and os.path.exists(CATEGORY_AUDIT_PATH):
        print(f"[Data] Category-balanced sampling enabled")
        slice_ds = build_balanced_dataset(full_ds, NUM_SAMPLES, seed=SEED)
    else:
        print(f"[Data] Plain sequential sampling (train[:{NUM_SAMPLES}])")
        slice_ds = full_ds.select(range(min(NUM_SAMPLES, len(full_ds))))

    split_ds = slice_ds.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    train_dataset = split_ds["train"]
    eval_dataset  = split_ds["test"]
    print(f"[Data] Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    log_event("INFO", f"Dataset split: {len(train_dataset)} train / {len(eval_dataset)} eval | balanced={args.balanced_sampling}")

    # ── Collator ──
    collator = Phase2CompletionCollator(
        response_template="### Response:\n",
        tokenizer=tokenizer,
        mlm=False,
        ignore_index=-100,
    )
    print("[Collator] Phase2CompletionCollator (response-only masking, Locked Decision #1)")

    # ── Compute expected steps ──
    steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
    expected_total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"[Steps] {steps_per_epoch} steps/epoch × {NUM_EPOCHS} epochs = ~{expected_total_steps} steps")
    log_event("INFO", f"Expected training steps: ~{expected_total_steps} ({steps_per_epoch}/epoch × {NUM_EPOCHS} epochs)")

    # ── Training arguments ──
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        max_steps=-1,                    # epoch-based, not step-capped
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        lr_scheduler_type=SCHEDULER,
        optim=OPTIMIZER,
        warmup_ratio=WARMUP_RATIO,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
        seed=SEED,
    )

    vram_cb = VRAMTrackingCallback(vram_log_path=VRAM_LOG_PATH)
    formatter = create_formatter(tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=training_args,
        formatting_func=formatter,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        callbacks=[vram_cb],
    )

    # ── Train ──
    print(f"\n[Train] Starting Config E...")
    start_ts = datetime.datetime.utcnow()
    print(f"[Train] Start UTC: {start_ts.isoformat()}")
    train_result = trainer.train()
    end_ts = datetime.datetime.utcnow()
    print(f"[Train] End UTC:   {end_ts.isoformat()}")
    wall_clock_s = (end_ts - start_ts).total_seconds()
    log_event("DONE", f"Config E training complete: {wall_clock_s:.0f}s ({wall_clock_s/3600:.2f}h), steps={train_result.global_step}")

    # ── Save adapter ──
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[Done] Adapter saved to {OUTPUT_DIR}")

    # ── SHA-256 ──
    adapter_file = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    adapter_sha256 = "MISSING"
    adapter_bytes  = 0
    if os.path.exists(adapter_file):
        with open(adapter_file, "rb") as f:
            content = f.read()
            adapter_sha256 = hashlib.sha256(content).hexdigest()
            adapter_bytes  = len(content)
        print(f"[Artifact] SHA-256: {adapter_sha256} ({adapter_bytes} bytes)")

    # ── Final eval loss ──
    eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
    final_eval_loss = eval_entries[-1]["eval_loss"] if eval_entries else None
    final_eval_step = eval_entries[-1]["step"]      if eval_entries else None

    # ── Summary ──
    run_summary = {
        "config_name": "config_e",
        "description": "Config E: Scaled Dataset (5k), Attention-Only Baseline",
        "locked_decision_masking": "response_only (Locked Decision #1)",
        "locked_decision_targets": "q_proj,v_proj (Locked Decision #2)",
        "target_modules": TARGET_MODULES,
        "lora_rank": LORA_RANK,
        "num_samples": NUM_SAMPLES,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "balanced_sampling": args.balanced_sampling,
        "num_epochs": NUM_EPOCHS,
        "expected_steps": expected_total_steps,
        "actual_steps": train_result.global_step,
        "training_start_utc": start_ts.isoformat(),
        "training_end_utc":   end_ts.isoformat(),
        "wall_clock_seconds": round(wall_clock_s, 2),
        "wall_clock_hours":   round(wall_clock_s / 3600, 3),
        "peak_vram_allocated_mb": round(vram_cb.max_vram_allocated_mb, 2),
        "peak_vram_reserved_mb":  round(vram_cb.max_vram_reserved_mb,  2),
        "train_loss": train_result.training_loss,
        "final_eval_loss": final_eval_loss,
        "final_eval_step": final_eval_step,
        "adapter_safetensors_sha256": adapter_sha256,
        "adapter_safetensors_bytes": adapter_bytes,
        "metrics": train_result.metrics,
        "log_history": trainer.state.log_history,
        "environment": env_info,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"[Summary] Saved to {SUMMARY_PATH}")

    release_sleep_prevention()

    print("\n" + "=" * 80)
    print("PHASE 4 CONFIG E TRAINING COMPLETE")
    print(f"  Wall-clock:  {wall_clock_s:.1f}s ({wall_clock_s/3600:.2f}h)")
    print(f"  Steps:       {train_result.global_step} (expected ~{expected_total_steps})")
    print(f"  Train loss:  {train_result.training_loss:.4f}")
    print(f"  Peak VRAM:   {vram_cb.max_vram_allocated_mb:.1f} MB allocated")
    print("=" * 80)


if __name__ == "__main__":
    main()
