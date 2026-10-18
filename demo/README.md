---
title: Phi-2 QLoRA Model Comparison & Benchmark Suite
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
---

# Phi-2 QLoRA Side-by-Side Model Comparator & Benchmark Suite

This Hugging Face Space hosts an empirical comparison between **Microsoft's Phi-2 (2.7B)** raw base model and its **4-bit QLoRA fine-tuned adapter** (`alfeenafsal/phi2-qlora-finetune`).

- **Base Model**: `microsoft/phi-2` (2.78B parameters, frozen)
- **Fine-Tuned Adapter**: [alfeenafsal/phi2-qlora-finetune](https://huggingface.co/alfeenafsal/phi2-qlora-finetune) (5.24M parameters / 0.19%)
- **Canonical Repository**: [https://github.com/alfeenafsal11/phi2-qlora-finetune](https://github.com/alfeenafsal11/phi2-qlora-finetune)
- **50-Task OOD Pass Rate**: 68.0% (Adapted) vs 52.0% (Base)
- **EOS Termination Rate**: 96.0% (Adapted) vs 18.0% (Base)

*Note: This is an experimental research model demonstration. Generated outputs may contain errors and must be sandboxed and verified before execution.*
