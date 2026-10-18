"""
Phi-2 QLoRA Side-by-Side Model Comparator
Standalone Gradio application comparing Microsoft's Phi-2 base model against
its fine-tuned QLoRA code generation adapter (alfeenafsal/phi2-qlora-finetune).
"""

import os
import time
import threading
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "microsoft/phi-2"
# Priority: use local checkpoint if available, otherwise download from HF Hub
LOCAL_ADAPTER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "phi2-qlora-phase4-scaled-5k", "checkpoint-100")
)
HUB_ADAPTER_ID = "alfeenafsal/phi2-qlora-finetune"
ADAPTER_PATH = LOCAL_ADAPTER_DIR if os.path.exists(LOCAL_ADAPTER_DIR) else HUB_ADAPTER_ID

inference_lock = threading.Lock()
loaded_model = None
loaded_tokenizer = None
model_device_info = "Not initialized"


def load_model_and_tokenizer():
    global loaded_model, loaded_tokenizer, model_device_info
    if loaded_model is not None and loaded_tokenizer is not None:
        return loaded_model, loaded_tokenizer

    print(f"Loading tokenizer from {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")

    if has_cuda:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("Loading base model in 4-bit NF4...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model_device_info = f"CUDA (4-bit NF4 on {torch.cuda.get_device_name(0)})"
        except Exception as e:
            print(f"4-bit loading failed ({e}), falling back to float16...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model_device_info = "CUDA (float16)"
    else:
        print("Loading base model on CPU (float32)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model_device_info = "CPU (float32)"

    print(f"Attaching LoRA adapter from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    loaded_model = model
    loaded_tokenizer = tokenizer
    print("Model and adapter successfully initialized.")
    return loaded_model, loaded_tokenizer


def compare_models(instruction: str, input_text: str, temperature: float, max_new_tokens: int):
    """
    Runs deterministic or sampled inference on both raw base model and QLoRA adapter
    under equivalent prompt formatting and decoding settings.
    """
    if not instruction or not instruction.strip():
        return (
            "Please provide a programming instruction or select an example below.",
            "Please provide a programming instruction or select an example below.",
            "Status: Ready",
            "",
            "",
        )

    # Format prompt adhering to training schema
    inst = instruction.strip()
    inp = input_text.strip() if input_text else ""
    if inp:
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n"

    with inference_lock:
        try:
            model, tokenizer = load_model_and_tokenizer()
            device = getattr(model, "device", None) or next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]

            do_sample = temperature > 0.0
            gen_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": do_sample,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = 0.95

            # 1. Base Model Inference (with adapter disabled)
            t0 = time.time()
            with model.disable_adapter():
                with torch.no_grad():
                    base_outputs = model.generate(**inputs, **gen_kwargs)
            base_time = time.time() - t0
            base_gen_tokens = base_outputs[0][prompt_len:]
            base_text = tokenizer.decode(base_gen_tokens, skip_special_tokens=True)
            base_eos = (len(base_gen_tokens) > 0 and base_outputs[0][-1].item() == tokenizer.eos_token_id)
            base_meta = (
                f"⏱️ **Latency**: {base_time:.2f}s | "
                f"🔤 **Tokens**: {len(base_gen_tokens)} ({len(base_gen_tokens)/max(0.01, base_time):.1f} tok/s) | "
                f"{'✅ Clean EOS Stop' if base_eos else '⚠️ Cut off by token limit / runaway loop'}"
            )

            # 2. Fine-Tuned Model Inference (with adapter enabled)
            t1 = time.time()
            with torch.no_grad():
                adapted_outputs = model.generate(**inputs, **gen_kwargs)
            adapted_time = time.time() - t1
            adapted_gen_tokens = adapted_outputs[0][prompt_len:]
            adapted_text = tokenizer.decode(adapted_gen_tokens, skip_special_tokens=True)
            adapted_eos = (len(adapted_gen_tokens) > 0 and adapted_outputs[0][-1].item() == tokenizer.eos_token_id)
            adapted_meta = (
                f"⏱️ **Latency**: {adapted_time:.2f}s | "
                f"🔤 **Tokens**: {len(adapted_gen_tokens)} ({len(adapted_gen_tokens)/max(0.01, adapted_time):.1f} tok/s) | "
                f"{'✅ Clean EOS Stop' if adapted_eos else '⚠️ Cut off by token limit'}"
            )

            status_msg = f"Inference complete on {model_device_info}. Base: {len(base_gen_tokens)} tok, Adapted: {len(adapted_gen_tokens)} tok."
            return base_text, adapted_text, status_msg, base_meta, adapted_meta

        except Exception as e:
            err_msg = f"Inference Error: {str(e)}"
            print(err_msg)
            return err_msg, err_msg, f"Error: {e}", "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Example Prompts across Benchmark Categories
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    [
        "Write a Python function that checks if two strings are anagrams of each other (case-insensitive).",
        "",
        0.0,
        150,
    ],
    [
        "Write a Python function that removes duplicates from a list while preserving the original element order.",
        "",
        0.0,
        150,
    ],
    [
        "Write a recursive Python function to compute the nth Fibonacci number.",
        "",
        0.0,
        150,
    ],
    [
        "Write a Python class representing a BankAccount with deposit, withdraw, and get_balance methods.",
        "",
        0.0,
        200,
    ],
    [
        "The following function is supposed to return the sum of all even numbers in a list, but contains a syntax/logic bug. Fix it:\n\ndef sum_evens(lst):\n    total = 0\n    for x in lst:\n        if x % 2 = 0:\n            total += x\n    return total",
        "",
        0.0,
        150,
    ],
]

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI Construction
# ─────────────────────────────────────────────────────────────────────────────

custom_css = """
.container { max-width: 1200px; margin: auto; padding-top: 15px; }
.header-box { text-align: center; margin-bottom: 20px; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600; margin: 3px; }
.badge-blue { background-color: #2563eb; color: white; }
.badge-purple { background-color: #7c3aed; color: white; }
.badge-green { background-color: #059669; color: white; }
.meta-info { font-size: 0.88em; color: #64748b; margin-top: 5px; }
.disclaimer-box { background: #f8fafc; border-left: 4px solid #f59e0b; padding: 12px; margin-top: 15px; border-radius: 4px; }
"""

with gr.Blocks(title="Phi-2 QLoRA Code Model Comparator", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="header-box">
        <h1 style="font-size: 2.2rem; font-weight: 700; margin-bottom: 8px;">
            ⚡ Microsoft Phi-2 (2.7B) vs. QLoRA Adapter Comparator
        </h1>
        <p style="font-size: 1.05rem; color: #475569; max-width: 800px; margin: auto;">
            Compare generations from the frozen <strong>microsoft/phi-2</strong> base model and the fine-tuned <strong>QLoRA adapter</strong> (Config E, Step 100) under identical decoding settings.
        </p>
        <div style="margin-top: 10px;">
            <span class="badge badge-blue">LoRA Rank 16</span>
            <span class="badge badge-purple">5.24M Trainable Params (0.19%)</span>
            <span class="badge badge-green">68.0% OOD Pass Rate (+16%)</span>
            <span class="badge badge-blue">96% Clean EOS Stop</span>
        </div>
        <p style="margin-top: 10px; font-size: 0.95em;">
            🔗 <a href="https://github.com/alfeenafsal11/phi2-qlora-finetune" target="_blank"><strong>GitHub Repository</strong></a> |
            🤗 <a href="https://huggingface.co/alfeenafsal/phi2-qlora-finetune" target="_blank"><strong>Hugging Face Model Card</strong></a> |
            📄 <a href="https://github.com/alfeenafsal11/phi2-qlora-finetune/blob/main/reports/phase4_project_report.md" target="_blank"><strong>Phase 4 Evaluation Report</strong></a>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            instruction_input = gr.Textbox(
                label="Instruction / Programming Task",
                placeholder="e.g. Write a Python function that checks if two strings are anagrams of each other.",
                lines=4,
            )
            input_context = gr.Textbox(
                label="Optional Input Context (if applicable)",
                placeholder="Optional input arguments, docstrings, or test inputs...",
                lines=2,
            )
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Temperature (0.0 = Deterministic Greedy)",
                )
                max_tokens_slider = gr.Slider(
                    minimum=32,
                    maximum=350,
                    value=150,
                    step=16,
                    label="Max New Tokens",
                )

            compare_btn = gr.Button("⚡ Generate & Compare", variant="primary", size="lg")
            status_output = gr.Markdown("Status: Ready to compare models.")

        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Benchmark Category Examples")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[instruction_input, input_context, temperature_slider, max_tokens_slider],
                label="Click an example prompt to load it into the comparator:",
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏛️ Raw Base Model (`microsoft/phi-2`)")
            base_output = gr.Code(
                label="Un-adapted Base Model Completion",
                language="python",
                lines=12,
            )
            base_meta_output = gr.Markdown(value="*Latency and token stats will appear here.*")

        with gr.Column():
            gr.Markdown("### 🚀 Fine-Tuned Model (`Phi-2 + QLoRA Adapter`)")
            adapted_output = gr.Code(
                label="QLoRA Adapted Completion (Config E, Step 100)",
                language="python",
                lines=12,
            )
            adapted_meta_output = gr.Markdown(value="*Latency and token stats will appear here.*")

    compare_btn.click(
        fn=compare_models,
        inputs=[instruction_input, input_context, temperature_slider, max_tokens_slider],
        outputs=[base_output, adapted_output, status_output, base_meta_output, adapted_meta_output],
    )

    with gr.Accordion("📊 Verified 50-Task Out-of-Distribution Benchmark Evidence", open=False):
        gr.Markdown("""
        ### Empirical Benchmark Performance (50 Leak-Free OOD Tasks)
        Evaluated on an independent, unit-tested 50-task algorithmic benchmark spanning 8 core domains.

        | Model / Configuration | Training Scope | Target Modules | Resp Loss | Core Pass% (Greedy) | EOS Term% | Max-Token Cutoff% | Mean Length (tok) |
        |:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        | **Raw Phi-2 Base** | Pretrained (2.78B) | None | 0.5792 | 52.0% (26/50) | 18.0% | 84.0% | 284.6 |
        | **Config E Step 100 (Adapter)** | 5,000 samples | `q_proj, v_proj` | **0.4803** | **68.0% (34/50)** | **96.0%** | **4.0%** | **115.4** |

        #### Domain Breakdown:
        - **String Manipulation**: 3/6 (50%) vs 2/6 (33%) raw
        - **Data Structures**: 5/8 (63%) vs 3/8 (38%) raw
        - **Sorting & Searching**: 5/6 (83%) vs 3/6 (50%) raw
        - **Recursion & Dynamic Programming**: 7/8 (88%) vs 5/8 (63%) raw
        - **OOP & Class Definitions**: 4/6 (67%) vs 3/6 (50%) raw
        - **File I/O & Exception Handling**: 2/4 (50%) vs 1/4 (25%) raw
        - **Algorithm Complexity & Edge Cases**: 3/6 (50%) vs 5/6 (83%) raw
        - **Bug Fixing**: 5/6 (83%) vs 4/6 (67%) raw
        """)

    with gr.Accordion("⚙️ Architecture & Methodological Details", open=False):
        gr.Markdown("""
        - **Base Model**: `microsoft/phi-2` (2.78B parameters, frozen)
        - **Quantization**: 4-bit NormalFloat4 (NF4) with double quantization and bfloat16 compute dtype
        - **LoRA Configuration**: Rank $r=16$, Alpha $\alpha=32$, Dropout $0.05$, Target modules: `q_proj`, `v_proj`
        - **Trainable Footprint**: 5,242,880 parameters (~0.19% of base model), ~20 MB safetensors
        - **Loss Masking**: Response-only loss masking (`Phase2CompletionCollator`) computing cross-entropy solely on response tokens
        - **Termination Supervision**: Explicit `<|endoftext|>` token loss supervision resolving runaway text generation
        """)

    gr.HTML("""
    <div class="disclaimer-box">
        <strong>⚠️ Experimental Research Demonstration & Disclaimer:</strong><br/>
        This model is an experimental research fine-tune developed for studying parameter-efficient adaptation, termination dynamics, and response-only loss masking on consumer hardware (peak training VRAM ~3.1 GB). It is <strong>not</strong> a production-grade code generator. Outputs may contain logical flaws, security bugs, or edge-case failures. Always sandbox and verify generated code before use.
    </div>
    """)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
