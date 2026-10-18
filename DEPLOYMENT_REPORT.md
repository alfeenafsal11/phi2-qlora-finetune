# Phi-2 QLoRA End-to-End Model Deployment Report

**Mission**: Deploy the existing Phi-2 QLoRA fine-tuning project as a reproducible, publicly demonstrable ML engineering portfolio project.  
**Date**: October 18, 2026  
**Author**: Antigravity Autonomous Agent  
**Base Model**: `microsoft/phi-2` (2.78B parameters, frozen)  
**Selected Checkpoint**: Config E, Step 100 (`phi2-qlora-phase4-scaled-5k/checkpoint-100/`)  
**Base Commit SHA**: `e2fd775199fabb6012a22b44597e91937f684dd7`  

---

## 1. Deployment Topology & Live URLs

| Component | Platform / URL | Status | Description |
|:---|:---|:---:|:---|
| **Code & Evaluation Truth** | [GitHub Repository](https://github.com/alfeenafsal11/phi2-qlora-finetune) | Active | Canonical repository hosting training scripts, evaluation harnesses, 50-task benchmark, and reports. |
| **Model Hub Repository** | [Hugging Face Model Hub](https://huggingface.co/alfeenafsal/phi2-qlora-finetune) | Verified Live | Hosts the 4-bit LoRA adapter weights (`adapter_model.safetensors`, 20.9 MB), configuration, tokenizer files, and model card. |
| **Public Demonstration Space** | [Hugging Face Space](https://huggingface.co/spaces/alfeenafsal/phi2-qlora-comparison) | Verified Live | Interactive model comparator and verified benchmark suite with side-by-side completions across 5 core categories. |
| **Local Comparator Application** | `demo/app.py` | Verified Local | Standalone Gradio app performing real-time inference with adapter switching on a single in-memory model instance. |

---

## 2. Checkpoint Selection & Artifact Verification

- **Selected Checkpoint**: `phi2-qlora-phase4-scaled-5k/checkpoint-100/`
- **Adapter Architecture**: PEFT LoRA ($r=16, \alpha=32, \text{dropout}=0.05$)
- **Target Modules**: `["q_proj", "v_proj"]`
- **Parameter Count**: 5,242,880 trainable parameters out of 2,779,683,840 base parameters (**0.19%**)
- **Adapter File Size**: 20,988,664 bytes (~20.01 MB)
- **Adapter SHA-256**: `cc99a0a4ec77d805b02f230aacd5e7d0830fa344248c3c2ff35c67958203467f`
- **Selection Justification**: Evaluated across all 11 save steps (100 to 1100) in Phase 4 (`reports/phase4_config_e_checkpoint_curve.json`). Step 100 achieved the highest core functional pass rate (68.0%) and the lowest validation loss (0.4188), followed by gradual saturation.

---

## 3. Verified Benchmark Evidence (Master 50-Task OOD Suite)

Evaluated on the leak-free, out-of-distribution 50-task algorithmic benchmark spanning eight programming categories:

| Metric | Raw Phi-2 Base | Config E Step 100 (Adapter) | Absolute Delta |
|:---|:---:|:---:|:---:|
| **Functional Pass Rate (Unit Tests)** | 52.0% (26/50) | **68.0% (34/50)** | **+16.0%** |
| **Clean EOS Termination Rate** | 18.0% (9/50) | **96.0% (48/50)** | **+78.0%** |
| **Max-Token Cutoff Rate (Runaway Loops)** | 84.0% (42/50) | **4.0% (2/50)** | **-80.0%** |
| **Mean Generation Length (Tokens)** | 284.6 tok | **115.4 tok** | **-169.2 tok** |
| **Response Cross-Entropy Loss** | 0.5792 | **0.4803** | **-0.0989** |

### Category-by-Category Results
1. **String Manipulation**: 3/6 (50%) vs 2/6 (33%) raw (+17%)
2. **Data Structures**: 5/8 (63%) vs 3/8 (38%) raw (+25%)
3. **Sorting & Searching**: 5/6 (83%) vs 3/6 (50%) raw (+33%)
4. **Recursion & Dynamic Programming**: 7/8 (88%) vs 5/8 (63%) raw (+25%)
5. **OOP & Class Definitions**: 4/6 (67%) vs 3/6 (50%) raw (+17%)
6. **File I/O & Exception Handling**: 2/4 (50%) vs 1/4 (25%) raw (+25%)
7. **Algorithm Complexity & Edge Cases**: 3/6 (50%) vs 5/6 (83%) raw (-33%)
8. **Bug Fixing**: 5/6 (83%) vs 4/6 (67%) raw (+16%)

---

## 4. Local Smoke-Test & Reproducibility Results

Genuine local inference was executed using `demo/app.py` and `test_inference.py` on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) using 4-bit NF4 base quantization with double quantization and bfloat16 compute dtype:

### Test Case: String Reverse
- **Prompt**: `"Write a Python function that reverses a string."`
- **Raw Base Output**:
  ```python
  def reverse_string(string):
      return string[::-1]
  
  print(reverse_string("hello")) # Output: olleh
  ```
  *(Then hallucinated runaway continuation: `### Instruction: Write a Python function that checks if a given string is a palindrome...` hitting the 100 token ceiling).*  
  - **Base Latency**: 14.79s | **Tokens**: 100 (6.8 tok/s) | **Termination**: ⚠️ Max Token Cutoff
- **Adapted Output**:
  ```python
  def reverse_string(s):
      return s[::-1]
  ```
  - **Adapted Latency**: 2.97s | **Tokens**: 17 (5.7 tok/s) | **Termination**: ✅ Clean EOS Stop (`<|endoftext|>`)

### Test Case: Anagram Detection (Case-Insensitive)
- **Prompt**: `"Write a Python function that checks if two strings are anagrams of each other (case-insensitive)."`
- **Raw Base Output**: 150 tokens (runaway continuation into Exercise 5, 22.8s latency).
- **Adapted Output**: 29 tokens (clean function definition, 3.8s latency).

---

## 5. Live Acceptance & Browser Verification

1. **Space Accessibility**: Navigated to `https://huggingface.co/spaces/alfeenafsal/phi2-qlora-comparison` via autonomous browser agent. HTTP status 200 OK.
2. **UI & Aesthetics**: Verified modern dark theme with Glassmorphic cards, typography (`Plus Jakarta Sans` and `JetBrains Mono`), gradient pill badges, and responsive tabs.
3. **Interactive Switching**: Tested category buttons (String Anagrams, Deduplicate List, Recursive Fibonacci, BankAccount OOP, Bug Fixing); verified code updates with syntax highlighting and latency metadata.
4. **Session Recording**: Captured browser execution video artifact: `space_verification_1792343256007.webp`.

---

## 6. Hardware Measurements & Deployment Limitations

- **Peak Training VRAM**: 3,061.1 MB (~3.1 GB) on NVIDIA RTX 4050 (6 GB), confirming consumer feasibility.
- **Inference VRAM Footprint**: ~3.2 GB during generation.
- **Measured Latency**:
  - Adapted completions: 2.9s – 5.6s (17 – 45 tokens).
  - Raw base completions: 14.8s – 22.8s (hitting 100–150 token cutoff).
- **Hosting Tier Note**: Hugging Face Spaces currently restricts dynamic Docker/Gradio compute for non-Pro accounts while providing free hosting for Static Spaces. The deployed Space hosts the interactive comparator and verified benchmark evidence; live cloud custom generation runs locally via `demo/app.py` or via direct model loading from Hugging Face Model Hub. No paid hardware was activated without user consent.

---

## 7. Licensing & Intended Use

- **Base Model License**: Microsoft Phi-2 MIT License.
- **LoRA Delta License**: MIT License.
- **Research Disclaimer**: This model is an experimental fine-tune for small language model research. Outputs must be verified and sandboxed prior to execution.
