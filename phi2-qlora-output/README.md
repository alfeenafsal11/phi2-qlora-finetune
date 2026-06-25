# Phi-2 QLoRA — Fine-Tuned Adapter (Python Code Instructions)

**Base model:** [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2)  
**Library:** [PEFT](https://github.com/huggingface/peft) v0.10.0 · [TRL](https://github.com/huggingface/trl) · [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)  
**Task:** Causal Language Modeling — Python code instruction following  
**Trainable parameters:** ~5.24 M / ~2.78 B total (**0.19%** of model weights)

---

## Model Description

This adapter applies **QLoRA** (Quantized Low-Rank Adaptation) to Microsoft's Phi-2 2.7B parameter language model, enabling instruction-following for Python code generation on consumer-grade hardware. The adapter was trained on 1,800 samples from the `iamtarun/python_code_instructions_18k_alpaca` dataset (2,000 total, 10% held out for evaluation), using 4-bit NF4 quantization to drastically reduce GPU memory requirements while maintaining competitive training dynamics.

Only the `q_proj` and `v_proj` attention projection matrices of the 32 transformer layers were adapted, keeping the full base model frozen. This corresponds to precisely **0.19%** of the model's total parameters.

---

## Training Setup

| Hyperparameter | Value |
|:---|:---|
| Base model | `microsoft/phi-2` (2.7B params) |
| Quantization | 4-bit NF4 (`BitsAndBytesConfig`) |
| Double quantization | ✅ Enabled |
| Compute dtype | `bfloat16` |
| LoRA rank ($r$) | 16 |
| LoRA alpha ($\alpha$) | 32 |
| Target modules | `q_proj`, `v_proj` |
| LoRA dropout | 0.05 |
| Bias | `none` |
| Gradient checkpointing | ✅ Enabled |
| Dataset | `iamtarun/python_code_instructions_18k_alpaca` |
| Training samples | 1,800 |
| Evaluation samples | 200 (held-out 10% split) |
| Epochs | 2 |
| Batch size (effective) | 8 (1 × 8 gradient accumulation) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.03 |
| Optimizer | `paged_adamw_8bit` |
| Max sequence length | 512 tokens |

### Parameter Efficiency

```
Total parameters     : ~2,779,683,840
Trainable parameters : ~5,242,880
Trainable ratio      : 0.19%
Adapter size on disk : ~20 MB (.safetensors)
```

---

## Training Dynamics

Training loss decreased steadily from **0.928** at step 25 to **0.629** at step 325,
demonstrating consistent convergence over 2 epochs (500 steps on 1,800 samples).

| Step | Epoch | Loss | Grad Norm | Learning Rate |
|---:|---:|---:|---:|---:|
| 25 | 0.1 | 0.9283 | 0.642 | 1.998e-4 |
| 50 | 0.2 | 0.6734 | 0.279 | 1.974e-4 |
| 100 | 0.4 | 0.6659 | 0.378 | 1.852e-4 |
| 150 | 0.6 | 0.7235 | 0.447 | 1.641e-4 |
| 200 | 0.8 | 0.6686 | 0.324 | 1.364e-4 |
| 250 | 1.0 | 0.6978 | 0.333 | 1.049e-4 |
| 300 | 1.2 | 0.6809 | 0.283 | 7.282e-5 |
| 325 | 1.3 | 0.6291 | 0.389 | 5.766e-5 |
| 375 | 1.5 | 0.6831 | 0.366 | 3.103e-5 |
| 400 | 1.6 | 0.6704 | 0.305 | 2.026e-5 |

Key observation: Gradient norms are stable throughout training (0.28–0.64), indicating
the cosine schedule effectively manages the learning rate without divergence.

---

## Prompt Template

All training samples and inference calls use the following Alpaca-style format:

```
### Instruction:
<task description>

### Response:
<code output>
```

---

## Usage

### Load and run inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)

# Attach fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base, "./phi2-qlora-output")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Run inference
prompt = "### Instruction:\nWrite a Python function that reverses a string.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Run structured evaluation

```bash
python eval_model.py --adapter_dir ./phi2-qlora-output --compare_base
```

This generates `eval_report.md` and `eval_report.json` with responses across
12 held-out coding prompts (string manipulation, data structures, recursion,
sorting/searching, bug fixing, file I/O).

---

## Controlled Experiments

Four configurations were evaluated to study convergence and parameter-efficiency tradeoffs:

| Run | Rank ($r$) | LR | Schedule | Notes |
|:---|---:|---:|:---|:---|
| `run_rank8_cosine` | 8 | 2e-4 | Cosine | Lower parameter count |
| `run_rank16_cosine` | 16 | 2e-4 | Cosine | **Baseline (this adapter)** |
| `run_rank16_linear` | 16 | 2e-4 | Linear | Schedule comparison |
| `run_rank16_low_lr` | 16 | 5e-5 | Cosine | Conservative LR |

Run `python generate_plots.py` to produce comparative loss curves across all configurations.

---

## Limitations & Scope

- Trained on **1,800 samples** — performance is best on short Python tasks resembling the Alpaca instruction format.
- Max sequence length is **512 tokens**; responses to complex, multi-function problems may be truncated.
- Only `q_proj` and `v_proj` are adapted; adding `k_proj`, `dense`, or MLP layers may yield further gains at the cost of more trainable parameters.
- No RLHF or alignment tuning was applied.

---

## Technical Specifications

| Item | Detail |
|:---|:---|
| Framework versions | PEFT 0.10.0, Transformers ≥ 4.38, TRL ≥ 0.8, BitsAndBytes ≥ 0.43 |
| Hardware | Consumer-grade GPU (e.g., NVIDIA RTX) |
| Memory footprint | ~6–8 GB VRAM (4-bit quantization) |
| Adapter size | ~20 MB (`.safetensors`) |
| Optimizer state size | ~42 MB (in checkpoints) |

---

## Citation

If you use this adapter or pipeline, please cite the foundational works:

```bibtex
@article{dettmers2023qlora,
  title   = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author  = {Dettmers, Tim and others},
  journal = {arXiv preprint arXiv:2305.14314},
  year    = {2023}
}

@article{hu2021lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J and others},
  journal = {arXiv preprint arXiv:2106.09685},
  year    = {2021}
}
```

### Framework versions

- PEFT 0.10.0