"""
train_phase2.py — Controlled Phase 2 QLoRA training script for Microsoft Phi-2 (2.7B).
Integrates:
  1. Case A / Case B prompt formatting preserving 'input' field.
  2. Terminal EOS token (<|endoftext|>) appended to all responses.
  3. Response-only loss masking via Phase2CompletionCollator.
  4. Mandatory packing=False.
  5. Controlled adamw_torch optimizer and fp16 precision matching baseline.
  6. Saving checkpoints at step 200, 400, and final adapter to ./phi2-qlora-phase2-corrected.
"""

import os
import sys
import json
import time
import argparse
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ─────────────────────────── custom collator ─────────────────────────────────

class Phase2CompletionCollator(DataCollatorForCompletionOnlyLM):
    """
    Subclass of DataCollatorForCompletionOnlyLM that preserves terminal EOS token supervision.
    When pad_token == eos_token (50256 in Phi-2), DataCollatorForLanguageModeling blindly
    masks all 50256 tokens to -100. This collator restores the loss label on the unpadded
    terminal EOS token so the model is actively trained to generate EOS and terminate.
    """
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i, ex in enumerate(examples):
            raw_ids = ex["input_ids"] if isinstance(ex, dict) else ex
            if len(raw_ids) > 0 and raw_ids[-1] == self.tokenizer.eos_token_id:
                # Terminal token of unpadded sequence is genuine EOS; supervise it!
                batch["labels"][i, len(raw_ids) - 1] = self.tokenizer.eos_token_id
        return batch


# ─────────────────────────── prompt formatter ────────────────────────────────

def format_example_phase2(instruction: str, input_text: str, output_text: str, eos_token: str) -> str:
    """Format example into Alpaca prompt with Case A / Case B logic and EOS termination."""
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    out = (output_text or "").strip()
    if inp:
        # Case A: Non-empty input
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{eos_token}"
    else:
        # Case B: Empty input
        return f"### Instruction:\n{inst}\n\n### Response:\n{out}{eos_token}"


def create_batch_formatter(tokenizer):
    def format_sample(sample):
        output = []
        for inst, inp, out in zip(sample["instruction"], sample["input"], sample["output"]):
            output.append(format_example_phase2(inst, inp, out, tokenizer.eos_token))
        return output
    return format_sample


# ─────────────────────────── memory tracking callback ────────────────────────

class VRAMTrackingCallback(TrainerCallback):
    def __init__(self):
        self.max_vram_allocated_mb = 0.0
        self.max_vram_reserved_mb = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            res = torch.cuda.max_memory_reserved() / (1024 ** 2)
            self.max_vram_allocated_mb = max(self.max_vram_allocated_mb, alloc)
            self.max_vram_reserved_mb = max(self.max_vram_reserved_mb, res)


# ─────────────────────────── main training routine ───────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 QLoRA fine-tuning for Phi-2")
    parser.add_argument("--model_id",         type=str,   default="microsoft/phi-2")
    parser.add_argument("--dataset",          type=str,   default="iamtarun/python_code_instructions_18k_alpaca")
    parser.add_argument("--num_samples",      type=int,   default=2000)
    parser.add_argument("--eval_ratio",       type=float, default=0.1)
    parser.add_argument("--output_dir",       type=str,   default="./phi2-qlora-phase2-corrected")

    # LoRA hyperparameters (strictly matched to Phase 1 broken baseline)
    parser.add_argument("--rank",             type=int,   default=16)
    parser.add_argument("--lora_alpha",       type=int,   default=32)
    parser.add_argument("--lora_dropout",     type=float, default=0.05)
    parser.add_argument("--target_modules",   type=str,   default="q_proj,v_proj")

    # Training hyperparameters (strictly matched to Phase 1 broken baseline)
    parser.add_argument("--epochs",           type=int,   default=2)
    parser.add_argument("--learning_rate",    type=float, default=2e-4)
    parser.add_argument("--scheduler",        type=str,   default="cosine")
    parser.add_argument("--optimizer",        type=str,   default="adamw_torch")
    parser.add_argument("--warmup_ratio",     type=float, default=0.03)
    parser.add_argument("--batch_size",       type=int,   default=1)
    parser.add_argument("--grad_accum_steps", type=int,   default=8)
    parser.add_argument("--max_seq_length",   type=int,   default=512)
    parser.add_argument("--logging_steps",    type=int,   default=25)
    parser.add_argument("--save_steps",       type=int,   default=200)
    parser.add_argument("--eval_steps",       type=int,   default=100)
    parser.add_argument("--seed",             type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 80)
    print("PHASE 2 — CONTROLLED TRAINING EXECUTION")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Set all random seeds for strict reproducibility (Addendum Control M)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Save initial experiment configuration
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[Config] Saved initial configuration to {config_path}")

    # ── 4-bit NF4 Quantization ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"[Model] Loading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer EOS: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print(f"  Tokenizer PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})")

    # ── Base Model ────────────────────────────────────────────────────────
    print(f"[Model] Loading base model in 4-bit NF4 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA Adapter ──────────────────────────────────────────────────────
    target_modules = [m.strip() for m in args.target_modules.split(",")]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"[LoRA] Trainable parameters:")
    model.print_trainable_parameters()

    # ── Dataset (train/eval split) ─────────────────────────────────────────
    print(f"[Data] Loading dataset: {args.dataset} ({args.num_samples} samples)")
    full_dataset = load_dataset(args.dataset, split=f"train[:{args.num_samples}]")
    split_dataset = full_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset  = split_dataset["test"]
    print(f"[Data] Train: {len(train_dataset)} samples | Eval: {len(eval_dataset)} samples")

    # ── Collator ──────────────────────────────────────────────────────────
    response_template = "### Response:\n"
    collator = Phase2CompletionCollator(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
        ignore_index=-100,
    )

    # ── Training Arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler,
        optim=args.optimizer,
        warmup_ratio=args.warmup_ratio,
        fp16=True,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
        seed=args.seed,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    vram_callback = VRAMTrackingCallback()
    formatting_func = create_batch_formatter(tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=False,  # Addendum Control C
        callbacks=[vram_callback],
    )

    print("[Train] Starting Phase 2 fine-tuning...")
    train_result = trainer.train()
    training_time_s = time.time() - start_time

    # ── Save Final Adapter Weights ─────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Done] Adapter saved to {args.output_dir}")

    # Extract final evaluation metrics
    final_eval_loss = None
    final_eval_step = None
    if trainer.state.log_history:
        eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
        if eval_entries:
            final_eval = eval_entries[-1]
            final_eval_loss = final_eval["eval_loss"]
            final_eval_step = final_eval["step"]
            print(f"[Eval] Final eval loss: {final_eval_loss:.4f} at step {final_eval_step}")

    # Save complete run summary metadata
    run_summary = {
        "training_time_seconds": round(training_time_s, 2),
        "peak_vram_allocated_mb": round(vram_callback.max_vram_allocated_mb, 2),
        "peak_vram_reserved_mb": round(vram_callback.max_vram_reserved_mb, 2),
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "final_eval_loss": final_eval_loss,
        "final_eval_step": final_eval_step,
        "metrics": train_result.metrics,
        "config": vars(args),
    }
    summary_path = os.path.join(args.output_dir, "phase2_train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"[Summary] Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
