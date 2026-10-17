"""
train_phase2b_ablation.py — Parameterized training script for Phase 2B Intervention Attribution.
Supports controlled ablations isolating:
  1. EOS ablation (input=YES, eos=NO, masking=YES)
  2. Masking ablation (input=YES, eos=YES, masking=NO)
  3. Input ablation (input=NO, eos=YES, masking=YES)
All other hyperparameters, seeds, and configurations are frozen to the Phase 2 baseline.
"""

import os
import sys
import json
import time
import hashlib
import platform
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


# ─────────────────────────── collators ───────────────────────────────────────

class Phase2CompletionCollator(DataCollatorForCompletionOnlyLM):
    """
    Collator for response-only loss masking with terminal EOS supervision.
    When pad_token == eos_token (50256 in Phi-2), this unmasks the genuine terminal EOS.
    """
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
    Pad tokens beyond sequence length are set to -100, genuine sequence tokens and EOS are supervised.
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


# ─────────────────────────── prompt formatters ───────────────────────────────

def format_example(instruction: str, input_text: str, output_text: str, input_mode: str, eos_mode: str, eos_token: str) -> str:
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    out = (output_text or "").strip()

    eos_str = eos_token if eos_mode == "with_eos" else ""

    if input_mode == "with_input" and inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{eos_str}"
    else:
        return f"### Instruction:\n{inst}\n\n### Response:\n{out}{eos_str}"


def create_batch_formatter(tokenizer, input_mode: str, eos_mode: str):
    def format_sample(sample):
        output = []
        for inst, inp, out in zip(sample["instruction"], sample["input"], sample["output"]):
            output.append(format_example(inst, inp, out, input_mode, eos_mode, tokenizer.eos_token))
        return output
    return format_sample


# ─────────────────────────── VRAM tracking callback ──────────────────────────

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


# ─────────────────────────── argument parsing ────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2B QLoRA fine-tuning for Phi-2")
    parser.add_argument("--ablation_name",    type=str,   required=True,
                        choices=["eos_ablation", "masking_ablation", "input_ablation", "reference_replica"],
                        help="Name of the ablation condition.")
    parser.add_argument("--output_dir",       type=str,   default=None,
                        help="Output directory for checkpoints and final adapter.")
    parser.add_argument("--input_mode",       type=str,   default=None,
                        choices=["with_input", "omit_input"],
                        help="Input preservation mode.")
    parser.add_argument("--eos_mode",         type=str,   default=None,
                        choices=["with_eos", "no_eos"],
                        help="Terminal EOS mode.")
    parser.add_argument("--masking_mode",     type=str,   default=None,
                        choices=["response_only", "full_sequence"],
                        help="Loss masking mode.")

    # Immutable Hyperparameters (Identical to Phase 2 Baseline)
    parser.add_argument("--model_id",         type=str,   default="microsoft/phi-2")
    parser.add_argument("--dataset",          type=str,   default="iamtarun/python_code_instructions_18k_alpaca")
    parser.add_argument("--num_samples",      type=int,   default=2000)
    parser.add_argument("--eval_ratio",       type=float, default=0.1)
    parser.add_argument("--rank",             type=int,   default=16)
    parser.add_argument("--lora_alpha",       type=int,   default=32)
    parser.add_argument("--lora_dropout",     type=float, default=0.05)
    parser.add_argument("--target_modules",   type=str,   default="q_proj,v_proj")
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


def resolve_ablation_config(args):
    """Sets default ablation parameters based on ablation_name if not explicitly overridden."""
    if args.ablation_name == "eos_ablation":
        args.input_mode = args.input_mode or "with_input"
        args.eos_mode = args.eos_mode or "no_eos"
        args.masking_mode = args.masking_mode or "response_only"
        args.output_dir = args.output_dir or "./phi2-qlora-phase2b-eos-ablation"
    elif args.ablation_name == "masking_ablation":
        args.input_mode = args.input_mode or "with_input"
        args.eos_mode = args.eos_mode or "with_eos"
        args.masking_mode = args.masking_mode or "full_sequence"
        args.output_dir = args.output_dir or "./phi2-qlora-phase2b-masking-ablation"
    elif args.ablation_name == "input_ablation":
        args.input_mode = args.input_mode or "omit_input"
        args.eos_mode = args.eos_mode or "with_eos"
        args.masking_mode = args.masking_mode or "response_only"
        args.output_dir = args.output_dir or "./phi2-qlora-phase2b-input-ablation"
    elif args.ablation_name == "reference_replica":
        args.input_mode = args.input_mode or "with_input"
        args.eos_mode = args.eos_mode or "with_eos"
        args.masking_mode = args.masking_mode or "response_only"
        args.output_dir = args.output_dir or "./phi2-qlora-phase2b-reference-replica"
    return args


def main():
    raw_args = parse_args()
    args = resolve_ablation_config(raw_args)
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 80)
    print(f"PHASE 2B TRAINING: {args.ablation_name.upper()}")
    print(f"Output directory: {args.output_dir}")
    print(f"Intervention Config: Input={args.input_mode} | EOS={args.eos_mode} | Masking={args.masking_mode}")
    print("=" * 80)

    # ── Reproducibility & Environment Manifest ────────────────────────────
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    import transformers, peft, bitsandbytes, datasets, trl, accelerate
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "bitsandbytes_version": bitsandbytes.__version__,
        "datasets_version": datasets.__version__,
        "trl_version": trl.__version__,
        "accelerate_version": accelerate.__version__,
    }

    # Save experiment configuration and environment info
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump({"config": vars(args), "environment": env_info}, f, indent=2)
    print(f"[Config] Saved initial configuration and environment to {config_path}")

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
    if args.masking_mode == "response_only":
        collator = Phase2CompletionCollator(
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
            ignore_index=-100,
        )
        print("[Collator] Using Phase2CompletionCollator (Response-only loss masking)")
    else:
        collator = Phase2FullSequenceCollator(tokenizer=tokenizer)
        print("[Collator] Using Phase2FullSequenceCollator (Full-sequence causal LM loss)")

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
    formatting_func = create_batch_formatter(tokenizer, args.input_mode, args.eos_mode)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=False,
        callbacks=[vram_callback],
    )

    print(f"[Train] Starting {args.ablation_name} fine-tuning...")
    train_result = trainer.train()
    training_time_s = time.time() - start_time

    # ── Save Final Adapter Weights ─────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Done] Adapter saved to {args.output_dir}")

    # Compute SHA-256 of the newly created adapter_model.safetensors
    adapter_file = os.path.join(args.output_dir, "adapter_model.safetensors")
    adapter_sha256 = "MISSING"
    adapter_bytes = 0
    if os.path.exists(adapter_file):
        with open(adapter_file, "rb") as f:
            content = f.read()
            adapter_sha256 = hashlib.sha256(content).hexdigest()
            adapter_bytes = len(content)
        print(f"[Artifact] adapter_model.safetensors SHA-256: {adapter_sha256} ({adapter_bytes} bytes)")

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
        "ablation_name": args.ablation_name,
        "input_mode": args.input_mode,
        "eos_mode": args.eos_mode,
        "masking_mode": args.masking_mode,
        "training_time_seconds": round(training_time_s, 2),
        "peak_vram_allocated_mb": round(vram_callback.max_vram_allocated_mb, 2),
        "peak_vram_reserved_mb": round(vram_callback.max_vram_reserved_mb, 2),
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "final_eval_loss": final_eval_loss,
        "final_eval_step": final_eval_step,
        "adapter_safetensors_sha256": adapter_sha256,
        "adapter_safetensors_bytes": adapter_bytes,
        "metrics": train_result.metrics,
        "config": vars(args),
        "log_history": trainer.state.log_history,
    }
    summary_path = os.path.join(args.output_dir, "phase2b_train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"[Summary] Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
