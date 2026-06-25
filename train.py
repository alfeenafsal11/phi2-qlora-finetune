"""
train.py — Parameterized QLoRA fine-tuning for Microsoft Phi-2 (2.7B)

Usage:
    python train.py [OPTIONS]

Example (default baseline):
    python train.py --rank 16 --learning_rate 2e-4 --scheduler cosine --output_dir ./phi2-qlora-output

Example (low-rank experiment):
    python train.py --rank 8 --output_dir ./experiments/run_rank8_cosine
"""

import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ─────────────────────────── argument parsing ────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Phi-2")
    parser.add_argument("--model_id",         type=str,   default="microsoft/phi-2")
    parser.add_argument("--dataset",          type=str,   default="iamtarun/python_code_instructions_18k_alpaca")
    parser.add_argument("--num_samples",      type=int,   default=2000,
                        help="Total number of dataset samples (train + eval).")
    parser.add_argument("--eval_ratio",       type=float, default=0.1,
                        help="Fraction of samples reserved for evaluation.")
    parser.add_argument("--output_dir",       type=str,   default="./phi2-qlora-output")

    # LoRA hyperparameters
    parser.add_argument("--rank",             type=int,   default=16,
                        help="LoRA rank (r). Controls adapter parameter count.")
    parser.add_argument("--lora_alpha",       type=int,   default=32,
                        help="LoRA scaling factor (alpha).")
    parser.add_argument("--lora_dropout",     type=float, default=0.05)
    parser.add_argument("--target_modules",   type=str,   default="q_proj,v_proj",
                        help="Comma-separated list of modules to apply LoRA to.")

    # Training hyperparameters
    parser.add_argument("--epochs",           type=int,   default=2)
    parser.add_argument("--learning_rate",    type=float, default=2e-4)
    parser.add_argument("--scheduler",        type=str,   default="cosine",
                        choices=["cosine", "linear", "constant", "cosine_with_restarts"])
    parser.add_argument("--optimizer",        type=str,   default="paged_adamw_8bit",
                        choices=["paged_adamw_8bit", "adamw_torch", "adamw_hf"])
    parser.add_argument("--warmup_ratio",     type=float, default=0.03)
    parser.add_argument("--batch_size",       type=int,   default=1)
    parser.add_argument("--grad_accum_steps", type=int,   default=8)
    parser.add_argument("--max_seq_length",   type=int,   default=512)
    parser.add_argument("--logging_steps",    type=int,   default=25)
    parser.add_argument("--save_steps",       type=int,   default=200)
    parser.add_argument("--eval_steps",       type=int,   default=100)
    return parser.parse_args()

# ─────────────────────────── formatting ──────────────────────────────────────

def format_sample(sample):
    """Format instruction-response pairs into a structured prompt template."""
    output = []
    for instruction, response in zip(sample["instruction"], sample["output"]):
        output.append(
            f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        )
    return output

# ─────────────────────────── main ────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save experiment configuration for reproducibility and traceability
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[Config] Saved experiment config to {config_path}")

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
    model.print_trainable_parameters()

    # ── Dataset (train/eval split) ─────────────────────────────────────────
    print(f"[Data] Loading dataset: {args.dataset} ({args.num_samples} samples)")
    full_dataset = load_dataset(args.dataset, split=f"train[:{args.num_samples}]")
    split_dataset = full_dataset.train_test_split(test_size=args.eval_ratio, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset  = split_dataset["test"]
    print(f"[Data] Train: {len(train_dataset)} samples | Eval: {len(eval_dataset)} samples")

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
        report_to="tensorboard",
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        formatting_func=format_sample,
        max_seq_length=args.max_seq_length,
    )

    print("[Train] Starting training...")
    trainer.train()

    # ── Save adapter weights ───────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Done] Adapter saved to {args.output_dir}")

    # Append final eval loss to config if available
    if trainer.state.log_history:
        eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
        if eval_entries:
            final_eval = eval_entries[-1]
            config_data = vars(args)
            config_data["final_eval_loss"] = final_eval["eval_loss"]
            config_data["final_eval_step"] = final_eval["step"]
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            print(f"[Eval] Final eval loss: {final_eval['eval_loss']:.4f}")


if __name__ == "__main__":
    main()