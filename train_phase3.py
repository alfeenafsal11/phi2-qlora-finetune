"""
train_phase3.py -- Unified Phase 3 QLoRA training script for Microsoft Phi-2 (2.7B).

Supports four controlled configurations:
  Config A -- Attention Only (Baseline Replication): q_proj, v_proj, response-only masking
  Config B -- Full Attention:  q_proj, k_proj, v_proj, dense, response-only masking
  Config C -- All Linear:      q_proj, k_proj, v_proj, dense, fc1, fc2, response-only masking
  Config D -- Masking Ablation (clean re-run): q_proj, v_proj, full-sequence loss

All non-intervention hyperparameters are frozen to Phase 2 corrected values.
max_steps=250 replaces epoch-based training (saturation confirmed at step ~200 in Phase 2).
Checkpoints saved every 50 steps; evaluation every 50 steps.
Per-step VRAM logged at steps 1, 50, 100, 150, 200, 250.
Config C has automatic OOM recovery: reduces r from 16 to 8 and retries.
Config D records wall-clock timestamps to reports/phase3_masking_ablation_timing.json.
"""

import os
import sys
import json
import time
import hashlib
import platform
import argparse
import datetime
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
from datasets import load_dataset
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


# ??????????????????????????? collators ???????????????????????????????????????

class Phase2CompletionCollator(DataCollatorForCompletionOnlyLM):
    """
    Response-only loss masking with terminal EOS supervision.
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
    Full causal LM sequence loss (Masking=NO).
    Labels = input_ids.clone(). Pad tokens set to -100; genuine tokens supervised.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids_list = [
            torch.tensor(ex["input_ids"]) if not isinstance(ex["input_ids"], torch.Tensor)
            else ex["input_ids"]
            for ex in examples
        ]
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


# ??????????????????????????? prompt formatter ????????????????????????????????

def create_batch_formatter(tokenizer, eos_mode="with_eos"):
    """Phase 2 standardized prompt format with Case A / Case B logic."""
    def format_sample(sample):
        output = []
        for inst, inp, out in zip(sample["instruction"], sample["input"], sample["output"]):
            inst = (inst or "").strip()
            inp = (inp or "").strip()
            out = (out or "").strip()
            eos_str = tokenizer.eos_token if eos_mode == "with_eos" else ""
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{eos_str}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{eos_str}"
            output.append(text)
        return output
    return format_sample


# ??????????????????????????? VRAM tracking callback ??????????????????????????

VRAM_LOG_STEPS = {1, 50, 100, 150, 200, 250}


class VRAMTrackingCallback(TrainerCallback):
    """Logs peak VRAM at steps 1, 50, 100, 150, 200, 250 and tracks overall peak."""
    def __init__(self, config_name: str, vram_log_path: str):
        self.config_name = config_name
        self.vram_log_path = vram_log_path
        self.max_vram_allocated_mb = 0.0
        self.max_vram_reserved_mb = 0.0
        self.step_vram_log = []

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        res = torch.cuda.max_memory_reserved() / (1024 ** 2)
        self.max_vram_allocated_mb = max(self.max_vram_allocated_mb, alloc)
        self.max_vram_reserved_mb = max(self.max_vram_reserved_mb, res)

        step = state.global_step
        if step in VRAM_LOG_STEPS:
            entry = {
                "config": self.config_name,
                "step": step,
                "vram_allocated_mb": round(alloc, 2),
                "vram_reserved_mb": round(res, 2),
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            }
            self.step_vram_log.append(entry)
            print(f"[VRAM] Step {step}: Allocated={alloc:.1f}MB, Reserved={res:.1f}MB")
            # Append to the shared VRAM log file
            self._flush_to_log(entry)

    def _flush_to_log(self, entry: dict):
        os.makedirs(os.path.dirname(self.vram_log_path), exist_ok=True)
        existing = []
        if os.path.exists(self.vram_log_path):
            try:
                with open(self.vram_log_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(entry)
        with open(self.vram_log_path, "w") as f:
            json.dump(existing, f, indent=2)


# ??????????????????????????? config presets ??????????????????????????????????

CONFIG_PRESETS = {
    "config_a": {
        "output_dir": "phi2-qlora-phase3-attn-only",
        "target_modules": "q_proj,v_proj",
        "masking_mode": "response_only",
        "description": "Config A: Attention Only (Baseline Replication)",
    },
    "config_b": {
        "output_dir": "phi2-qlora-phase3-full-attn",
        "target_modules": "q_proj,k_proj,v_proj,dense",
        "masking_mode": "response_only",
        "description": "Config B: Full Attention (adds k_proj + dense)",
    },
    "config_c": {
        "output_dir": "phi2-qlora-phase3-all-linear",
        "target_modules": "q_proj,k_proj,v_proj,dense,fc1,fc2",
        "masking_mode": "response_only",
        "description": "Config C: All Linear Layers (full MLP coverage)",
    },
    "config_d": {
        "output_dir": "phi2-qlora-phase3-masking-ablation-clean",
        "target_modules": "q_proj,v_proj",
        "masking_mode": "full_sequence",
        "description": "Config D: Masking Ablation (clean re-run, full-sequence loss)",
    },
}


# ??????????????????????????? argument parsing ????????????????????????????????

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 QLoRA fine-tuning for Phi-2")
    # Config selection
    parser.add_argument("--config_name", type=str, required=True,
                        choices=list(CONFIG_PRESETS.keys()),
                        help="Phase 3 configuration preset.")
    # Overrides (optional)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--target_modules", type=str, default=None)
    parser.add_argument("--masking_mode", type=str, default=None,
                        choices=["response_only", "full_sequence"])
    parser.add_argument("--rank", type=int, default=16)

    # Immutable hyperparameters (Phase 2 corrected values)
    parser.add_argument("--model_id",         type=str,   default="microsoft/phi-2")
    parser.add_argument("--dataset",          type=str,   default="iamtarun/python_code_instructions_18k_alpaca")
    parser.add_argument("--num_samples",      type=int,   default=2000)
    parser.add_argument("--eval_ratio",       type=float, default=0.1)
    parser.add_argument("--lora_alpha",       type=int,   default=32)
    parser.add_argument("--lora_dropout",     type=float, default=0.05)
    parser.add_argument("--max_steps",        type=int,   default=250)
    parser.add_argument("--learning_rate",    type=float, default=2e-4)
    parser.add_argument("--scheduler",        type=str,   default="cosine")
    parser.add_argument("--optimizer",        type=str,   default="adamw_torch")
    parser.add_argument("--warmup_ratio",     type=float, default=0.03)
    parser.add_argument("--batch_size",       type=int,   default=1)
    parser.add_argument("--grad_accum_steps", type=int,   default=8)
    parser.add_argument("--max_seq_length",   type=int,   default=512)
    parser.add_argument("--logging_steps",    type=int,   default=25)
    parser.add_argument("--save_steps",       type=int,   default=50)
    parser.add_argument("--eval_steps",       type=int,   default=50)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--vram_log_path",    type=str,   default="reports/phase3_vram_log.json")
    parser.add_argument("--timing_log_path",  type=str,   default="reports/phase3_masking_ablation_timing.json")
    return parser.parse_args()


def resolve_config(args):
    """Merge preset defaults with any CLI overrides."""
    preset = CONFIG_PRESETS[args.config_name]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]
    if args.target_modules is None:
        args.target_modules = preset["target_modules"]
    if args.masking_mode is None:
        args.masking_mode = preset["masking_mode"]
    args.config_description = preset["description"]
    return args


# ??????????????????????????? training routine ????????????????????????????????

def build_and_train(args, rank_override=None):
    """Build model + trainer and run training. Returns (trainer, train_result, vram_callback, start_ts, end_ts)."""
    effective_rank = rank_override if rank_override is not None else args.rank

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    # ?? 4-bit NF4 Quantization ????????????????????????????????????????????
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ?? Tokenizer ??????????????????????????????????????????????????????????
    print(f"[Model] Loading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  EOS: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print(f"  PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})")

    # ?? Base Model ????????????????????????????????????????????????????????
    print(f"[Model] Loading base model in 4-bit NF4 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ?? LoRA Adapter ??????????????????????????????????????????????????????
    target_modules = [m.strip() for m in args.target_modules.split(",")]
    lora_config = LoraConfig(
        r=effective_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"[LoRA] Trainable parameters (r={effective_rank}):")
    model.print_trainable_parameters()

    # ?? Dataset ???????????????????????????????????????????????????????????
    print(f"[Data] Loading dataset: {args.dataset} ({args.num_samples} samples)")
    full_dataset = load_dataset(args.dataset, split=f"train[:{args.num_samples}]")
    split_dataset = full_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset  = split_dataset["test"]
    print(f"[Data] Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ?? Collator ??????????????????????????????????????????????????????????
    if args.masking_mode == "response_only":
        collator = Phase2CompletionCollator(
            response_template="### Response:\n",
            tokenizer=tokenizer,
            mlm=False,
            ignore_index=-100,
        )
        print("[Collator] Phase2CompletionCollator (response-only loss masking)")
    else:
        collator = Phase2FullSequenceCollator(tokenizer=tokenizer)
        print("[Collator] Phase2FullSequenceCollator (full-sequence causal LM loss)")

    # ?? Training Arguments ????????????????????????????????????????????????
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=1000,        # sentinel; max_steps takes precedence
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

    # ?? Callbacks ?????????????????????????????????????????????????????????
    vram_callback = VRAMTrackingCallback(
        config_name=args.config_name,
        vram_log_path=args.vram_log_path,
    )
    formatting_func = create_batch_formatter(tokenizer, eos_mode="with_eos")

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

    # ?? Training ??????????????????????????????????????????????????????????
    print(f"\n[Train] Starting {args.config_description}...")
    start_ts = datetime.datetime.utcnow()
    print(f"[Train] Start UTC: {start_ts.isoformat()}")
    train_result = trainer.train()
    end_ts = datetime.datetime.utcnow()
    print(f"[Train] End UTC:   {end_ts.isoformat()}")

    return trainer, train_result, vram_callback, start_ts, end_ts, effective_rank, tokenizer, model


def main():
    args = parse_args()
    args = resolve_config(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=" * 80)
    print(f"PHASE 3 TRAINING: {args.config_name.upper()}")
    print(f"  {args.config_description}")
    print(f"  Output: {args.output_dir}")
    print(f"  Target modules: {args.target_modules}")
    print(f"  Masking: {args.masking_mode}")
    print(f"  max_steps: {args.max_steps} | save_steps: {args.save_steps} | eval_steps: {args.eval_steps}")
    print("=" * 80)

    # ?? Environment manifest ???????????????????????????????????????????????
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
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump({"config": vars(args), "environment": env_info}, f, indent=2)
    print(f"[Config] Saved configuration to {config_path}")

    # ?? Train with OOM recovery for Config C ??????????????????????????????
    rank_used = args.rank
    oom_recovery_triggered = False

    try:
        trainer, train_result, vram_cb, start_ts, end_ts, rank_used, tokenizer, model = build_and_train(args)
    except torch.cuda.OutOfMemoryError as oom:
        if args.config_name == "config_c":
            print(f"\n[OOM] Config C hit OutOfMemoryError at r={args.rank}. Recovering: reducing r to 8.")
            print(f"[OOM] Error: {oom}")
            torch.cuda.empty_cache()
            oom_recovery_triggered = True
            rank_used = 8
            trainer, train_result, vram_cb, start_ts, end_ts, rank_used, tokenizer, model = build_and_train(args, rank_override=8)
        elif args.config_name == "config_b":
            print(f"\n[OOM] Config B hit OutOfMemoryError. Recovering: reducing grad_accum_steps from {args.grad_accum_steps} to 4.")
            torch.cuda.empty_cache()
            oom_recovery_triggered = True
            args.grad_accum_steps = 4
            trainer, train_result, vram_cb, start_ts, end_ts, rank_used, tokenizer, model = build_and_train(args)
        else:
            print(f"\n[OOM] Unexpected OOM for {args.config_name}. Logging and halting.")
            error_path = os.path.join(args.output_dir, "oom_error.txt")
            with open(error_path, "w") as f:
                f.write(f"OOM Error during {args.config_name}:\n{oom}\n")
            raise

    wall_clock_s = (end_ts - start_ts).total_seconds()

    # ?? Save adapter ??????????????????????????????????????????????????????
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Done] Adapter saved to {args.output_dir}")

    # ?? Adapter SHA-256 ???????????????????????????????????????????????????
    adapter_file = os.path.join(args.output_dir, "adapter_model.safetensors")
    adapter_sha256 = "MISSING"
    adapter_bytes = 0
    if os.path.exists(adapter_file):
        with open(adapter_file, "rb") as f:
            content = f.read()
            adapter_sha256 = hashlib.sha256(content).hexdigest()
            adapter_bytes = len(content)
        print(f"[Artifact] SHA-256: {adapter_sha256} ({adapter_bytes} bytes)")

    # ?? Final eval loss ???????????????????????????????????????????????????
    final_eval_loss = None
    final_eval_step = None
    if trainer.state.log_history:
        eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
        if eval_entries:
            last_eval = eval_entries[-1]
            final_eval_loss = last_eval["eval_loss"]
            final_eval_step = last_eval["step"]
            print(f"[Eval] Final trainer eval loss: {final_eval_loss:.4f} at step {final_eval_step}")

    # ?? Run summary ???????????????????????????????????????????????????????
    run_summary = {
        "config_name": args.config_name,
        "config_description": args.config_description,
        "target_modules": args.target_modules,
        "masking_mode": args.masking_mode,
        "rank_requested": args.rank,
        "rank_used": rank_used,
        "oom_recovery_triggered": oom_recovery_triggered,
        "training_start_utc": start_ts.isoformat(),
        "training_end_utc": end_ts.isoformat(),
        "wall_clock_seconds": round(wall_clock_s, 2),
        "peak_vram_allocated_mb": round(vram_cb.max_vram_allocated_mb, 2),
        "peak_vram_reserved_mb": round(vram_cb.max_vram_reserved_mb, 2),
        "step_vram_log": vram_cb.step_vram_log,
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "final_eval_loss": final_eval_loss,
        "final_eval_step": final_eval_step,
        "adapter_safetensors_sha256": adapter_sha256,
        "adapter_safetensors_bytes": adapter_bytes,
        "metrics": train_result.metrics,
        "config": vars(args),
        "log_history": trainer.state.log_history,
        "environment": env_info,
    }
    summary_path = os.path.join(args.output_dir, "phase3_train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"[Summary] Saved to {summary_path}")

    # ?? Config D: masking timing log ??????????????????????????????????????
    if args.config_name == "config_d":
        timing_entry = {
            "config_name": "config_d",
            "description": "Clean re-run of Phase 2B Ablation B (masking=NO, full-sequence loss)",
            "training_start_utc": start_ts.isoformat(),
            "training_end_utc": end_ts.isoformat(),
            "wall_clock_seconds": round(wall_clock_s, 2),
            "wall_clock_hours": round(wall_clock_s / 3600, 3),
            "global_step": train_result.global_step,
            "note": "Uninterrupted run. See phase3_master_eval_results.json for Config A timing for comparison.",
        }
        # Load or create the timing log
        existing_timing = {}
        if os.path.exists(args.timing_log_path):
            try:
                with open(args.timing_log_path, "r") as f:
                    existing_timing = json.load(f)
            except Exception:
                existing_timing = {}
        existing_timing["config_d"] = timing_entry
        with open(args.timing_log_path, "w") as f:
            json.dump(existing_timing, f, indent=2)
        print(f"[Timing] Config D wall-clock log saved to {args.timing_log_path}")
        print(f"[Timing] Total wall-clock time: {wall_clock_s:.1f}s ({wall_clock_s/3600:.2f}h)")

    # ?? Config A: also save timing for comparison ?????????????????????????
    if args.config_name == "config_a":
        timing_entry = {
            "config_name": "config_a",
            "description": "Attention-only baseline (masking=YES, response-only loss)",
            "training_start_utc": start_ts.isoformat(),
            "training_end_utc": end_ts.isoformat(),
            "wall_clock_seconds": round(wall_clock_s, 2),
            "wall_clock_hours": round(wall_clock_s / 3600, 3),
            "global_step": train_result.global_step,
        }
        existing_timing = {}
        if os.path.exists(args.timing_log_path):
            try:
                with open(args.timing_log_path, "r") as f:
                    existing_timing = json.load(f)
            except Exception:
                existing_timing = {}
        existing_timing["config_a"] = timing_entry
        with open(args.timing_log_path, "w") as f:
            json.dump(existing_timing, f, indent=2)
        print(f"[Timing] Config A timing saved to {args.timing_log_path} (for speedup comparison vs Config D)")

    print("\n" + "=" * 80)
    print(f"PHASE 3 TRAINING COMPLETE: {args.config_name.upper()}")
    print(f"  Wall-clock: {wall_clock_s:.1f}s ({wall_clock_s/3600:.2f}h)")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Peak VRAM:  {vram_cb.max_vram_allocated_mb:.1f} MB allocated")
    if oom_recovery_triggered:
        print(f"  [WARN]  OOM recovery triggered -- rank reduced from {args.rank} to {rank_used}")
    print("=" * 80)


if __name__ == "__main__":
    main()
