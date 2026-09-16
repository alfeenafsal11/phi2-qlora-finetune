import os
import sys
import json
import math
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["HF_HUB_OFFLINE"] = "1"

def get_model_fingerprint(model, adapter_path=None):
    """Computes a deterministic checksum of model/adapter parameters."""
    fingerprint = {}
    if adapter_path:
        # Check adapter weight file directly
        weight_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(weight_file):
            with open(weight_file, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            fingerprint["adapter_safetensors_sha256"] = h
            fingerprint["adapter_safetensors_bytes"] = os.path.getsize(weight_file)
    
    # Checksum a sample of active parameter tensors
    total_params = 0
    trainable_params = 0
    sample_hash = hashlib.sha256()
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if "lora" in name or "model.layers.0.self_attn.q_proj" in name:
            sample_hash.update(param.data.cpu().numpy().tobytes())
            
    fingerprint["total_params"] = total_params
    fingerprint["trainable_params"] = trainable_params
    fingerprint["sample_param_sha256"] = sample_hash.hexdigest()
    return fingerprint


def evaluate_format_losses(model, tokenizer, eval_dataset, prompt_mode="phase2", max_seq_length=512, device="cuda"):
    """
    Evaluates 200 samples under a specific prompt format.
    prompt_mode options:
      - 'phase1': No input, No EOS
      - 'phase2': With input (Case A/B), With EOS
      - 'phase1_with_eos': No input, With EOS
      - 'phase2_no_eos': With input (Case A/B), No EOS
    Returns token-weighted metrics, sample-weighted metrics, and per-example arrays.
    """
    model.eval()
    rep_ids = tokenizer.encode("### Response:\n", add_special_tokens=False)
    rlen = len(rep_ids)

    per_sample_full_loss = []
    per_sample_resp_loss = []
    
    total_full_sum_loss = 0.0
    total_full_tokens = 0
    total_resp_sum_loss = 0.0
    total_resp_tokens = 0

    with torch.no_grad():
        for ex in eval_dataset:
            inst = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("output") or "").strip()

            if prompt_mode == "phase1":
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            elif prompt_mode == "phase2":
                if inp:
                    text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
                else:
                    text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}"
            elif prompt_mode == "phase1_with_eos":
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}"
            elif prompt_mode == "phase2_no_eos":
                if inp:
                    text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                else:
                    text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"

            enc = tokenizer(text, truncation=True, max_length=max_seq_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            target_ids = input_ids.clone()

            ids_list = input_ids[0].tolist()
            match_idx = None
            for i in range(len(ids_list) - rlen + 1):
                if ids_list[i : i + rlen] == rep_ids:
                    match_idx = i
                    break

            logits = model(input_ids).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            # Full sequence loss
            loss_full_sum = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum").item()
            n_full = shift_labels.numel()
            total_full_sum_loss += loss_full_sum
            total_full_tokens += n_full
            per_sample_full_loss.append(loss_full_sum / max(1, n_full))

            # Response only loss
            if match_idx is not None:
                resp_start = match_idx + rlen
                resp_target_start = max(0, resp_start - 1)
                resp_shift_logits = shift_logits[:, resp_target_start:, :].contiguous()
                resp_shift_labels = shift_labels[:, resp_target_start:].contiguous()
                n_resp = resp_shift_labels.numel()
                if n_resp > 0:
                    loss_resp_sum = F.cross_entropy(resp_shift_logits.view(-1, resp_shift_logits.size(-1)), resp_shift_labels.view(-1), reduction="sum").item()
                    total_resp_sum_loss += loss_resp_sum
                    total_resp_tokens += n_resp
                    per_sample_resp_loss.append(loss_resp_sum / n_resp)
                else:
                    per_sample_resp_loss.append(0.0)
            else:
                per_sample_resp_loss.append(0.0)

    # Token-weighted (Sum Loss / Sum Tokens)
    tok_full_loss = total_full_sum_loss / max(1, total_full_tokens)
    tok_resp_loss = total_resp_sum_loss / max(1, total_resp_tokens)

    # Sample-weighted (Mean of per-sample means)
    samp_full_loss = float(np.mean(per_sample_full_loss))
    samp_resp_loss = float(np.mean(per_sample_resp_loss))

    # Per-example distributions
    full_dist = {
        "mean": float(np.mean(per_sample_full_loss)),
        "std": float(np.std(per_sample_full_loss)),
        "median": float(np.median(per_sample_full_loss)),
        "min": float(np.min(per_sample_full_loss)),
        "max": float(np.max(per_sample_full_loss)),
        "p25": float(np.percentile(per_sample_full_loss, 25)),
        "p75": float(np.percentile(per_sample_full_loss, 75)),
    }
    resp_dist = {
        "mean": float(np.mean(per_sample_resp_loss)),
        "std": float(np.std(per_sample_resp_loss)),
        "median": float(np.median(per_sample_resp_loss)),
        "min": float(np.min(per_sample_resp_loss)),
        "max": float(np.max(per_sample_resp_loss)),
        "p25": float(np.percentile(per_sample_resp_loss, 25)),
        "p75": float(np.percentile(per_sample_resp_loss, 75)),
    }

    return {
        "prompt_mode": prompt_mode,
        "token_weighted": {
            "full_loss": round(tok_full_loss, 4),
            "full_ppl": round(math.exp(min(20, tok_full_loss)), 2),
            "resp_loss": round(tok_resp_loss, 4),
            "resp_ppl": round(math.exp(min(20, tok_resp_loss)), 2),
            "total_full_tokens": total_full_tokens,
            "total_resp_tokens": total_resp_tokens,
        },
        "sample_weighted": {
            "full_loss": round(samp_full_loss, 4),
            "full_ppl": round(math.exp(min(20, samp_full_loss)), 2),
            "resp_loss": round(samp_resp_loss, 4),
            "resp_ppl": round(math.exp(min(20, samp_resp_loss)), 2),
        },
        "full_distribution": full_dist,
        "resp_distribution": resp_dist,
    }


def main():
    print("=" * 80)
    print("PHASE 2 FORENSIC FOLLOW-UP: CONTROLLED LOSS RECONCILIATION")
    print("=" * 80)

    dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")
    eval_dataset = dataset.train_test_split(test_size=0.1, seed=42)["test"]
    print(f"Loaded {len(eval_dataset)} evaluation samples.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("\n[Loading Base Model microsoft/phi-2...]")
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ── 1. EVALUATE RAW BASE MODEL ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATING SYSTEM 1: RAW PHI-2 BASE")
    print("=" * 60)
    raw_fp = get_model_fingerprint(base_model)
    print("Model Fingerprint:", raw_fp)

    modes = ["phase1", "phase2", "phase1_with_eos", "phase2_no_eos"]
    raw_results = {"fingerprint": raw_fp, "evaluations": {}}
    for m in modes:
        print(f"\nEvaluating Raw Base under mode: '{m}'...")
        res = evaluate_format_losses(base_model, tokenizer, eval_dataset, prompt_mode=m)
        raw_results["evaluations"][m] = res
        tw = res["token_weighted"]
        sw = res["sample_weighted"]
        print(f"  [Token-Weighted] Full: {tw['full_loss']} (PPL {tw['full_ppl']}) | Resp: {tw['resp_loss']} (PPL {tw['resp_ppl']}) | Toks: {tw['total_full_tokens']}")
        print(f"  [Sample-Weighted] Full: {sw['full_loss']} (PPL {sw['full_ppl']}) | Resp: {sw['resp_loss']} (PPL {sw['resp_ppl']})")
    results["raw_phi2"] = raw_results

    # ── 2. EVALUATE BROKEN PHASE 1 ADAPTER ──────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATING SYSTEM 2: BROKEN PHASE 1 ADAPTER (phi2-qlora-output)")
    print("=" * 60)
    adapter_model = PeftModel.from_pretrained(base_model, "phi2-qlora-output")
    adapter_model.eval()
    broken_fp = get_model_fingerprint(adapter_model, adapter_path="phi2-qlora-output")
    print("Adapter Fingerprint:", broken_fp)

    broken_results = {"fingerprint": broken_fp, "evaluations": {}}
    for m in modes:
        print(f"\nEvaluating Broken Adapter under mode: '{m}'...")
        res = evaluate_format_losses(adapter_model, tokenizer, eval_dataset, prompt_mode=m)
        broken_results["evaluations"][m] = res
        tw = res["token_weighted"]
        sw = res["sample_weighted"]
        print(f"  [Token-Weighted] Full: {tw['full_loss']} (PPL {tw['full_ppl']}) | Resp: {tw['resp_loss']} (PPL {tw['resp_ppl']}) | Toks: {tw['total_full_tokens']}")
        print(f"  [Sample-Weighted] Full: {sw['full_loss']} (PPL {sw['full_ppl']}) | Resp: {sw['resp_loss']} (PPL {sw['resp_ppl']})")
    results["broken_phase1"] = broken_results

    # Save complete forensic report
    out_file = "reports/forensic_reconciliation_raw_data.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved complete reconciliation data to {out_file}")

if __name__ == "__main__":
    main()
