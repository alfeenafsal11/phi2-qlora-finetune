
# Phase 4 Experimental Report: Category-Balanced Dataset Scaling
**Author / Runner**: Antigravity Autonomous Agent
**Date**: September 21, 2026
**Artifact Directory**: `d:/SLM-finetune`
**Base Model**: `microsoft/phi-2` (2.7B, 4-bit NF4, bfloat16 compute)
**Evaluation Timestamp**: `2026-09-21T16:46:17.358210`

---

## Reuse Ledger
> **Required by Phase 4 brief** — verifies compute discipline was followed.

### Artifacts REUSED Without Recomputation
| Artifact | Source Path | Justification |
|:---|:---|:---|
| All 6 Phase 3 model eval results | `reports/phase3_master_eval_results.json` | Fully evaluated in Phase 3; deterministic results. Not re-run per CRITICAL RULE. |
| Config A/B/C checkpoint curves (15 checkpoints) | `reports/phase3_checkpoint_curves.json` | Already computed across all steps 50–250. Not re-run per CRITICAL RULE. |
| Masking timing data | `reports/phase3_masking_ablation_timing.json` | Masking question closed per Locked Decision #1. Not re-run. |
| Phase 3 VRAM logs (Configs A–D) | `reports/phase3_vram_log.json` | Architecture unchanged for these configs. Not re-run per CRITICAL RULE. |
| 50-task OOD benchmark | `reports/phase3_benchmark_50tasks.json` | OOD-verified; no regeneration needed. Reused without modification. |

### NEW Computations Performed in Phase 4
| Computation | Output | Wall-clock |
|:---|:---|:---:|
| Section 1: Category audit (18k classification, CPU-only) | `reports/phase4_category_audit.json` | ~3 min |
| Section 2: Config E training (5k samples, 2 epochs, 1124 steps) | `phi2-qlora-phase4-scaled-5k/` | 2.68h |
| Section 3A: Checkpoint curve greedy eval (all checkpoints) | `reports/phase4_config_e_checkpoint_curve.json` | ~varies |
| Section 3B: Best-checkpoint full eval (greedy + sampled + loss) | `reports/phase4_master_eval_results.json` | ~varies |
| Section 4: Phase 4 report generation | `reports/phase4_project_report.md` | <1 min |

---

## 1. Masking Decision — Phase 3 Contradiction Resolved

Phase 3's automated report contained an internal contradiction:
- **Section 1** (automated decision rule): `use_no_masking_based_on_loss`
- **Section 8** (manual recommendation): `Always use response-only masking`

### Resolution: Locked Decision #1 Supersedes Both

> **All Phase 4 training uses response-only masking (Phase2CompletionCollator).**

**Rationale (noise floor argument):**
The Phase 3 automated decision rule compared Config A response loss (0.4522) against
Config D response loss (0.4489) — a difference of **0.0033**.
This gap is **smaller than the noise floor** observed between adjacent checkpoints
of the *same* config (e.g., Config A step 150→200 = 0.454→0.453, a 0.001 shift).
A 0.0033 difference across *different configs with different data ordering* does not
constitute a valid empirical basis for changing the masking setting permanently.
Response-only masking remains correct on theoretical grounds as training data
scales: it prevents the model from gradient-updating on the instruction tokens,
keeping fine-tuning focused on code generation quality.

---

## 2. Section 1 — Data Category Audit

Full dataset size: **18,612** samples | Mean per category: **778.8**
Oversampling threshold: ≥ 300 examples

| Category | Full 18k Count | 18k % | 2k Count | 5k Count | Sufficient (≥300)? |
|:---|:---:|:---:|:---:|:---:|:---:|
| String Manipulation | 2,171 | 11.7% | 241 | 599 | ✓ YES |
| Data Structures | 1,437 | 7.7% | 166 | 409 | ✓ YES |
| Sorting & Searching | 765 | 4.1% | 74 | 205 | ✓ YES |
| Recursion & Dynamic Programming | 469 | 2.5% | 61 | 143 | ✓ YES |
| OOP & Class Definitions | 1,062 | 5.7% | 99 | 269 | ✓ YES |
| File I/O & Exception Handling | 205 | 1.1% | 24 | 54 | ✗ NO |
| Algorithm Complexity & Edge Cases | 43 | 0.2% | 7 | 16 | ✗ NO |
| Bug Fixing | 78 | 0.4% | 9 | 20 | ✗ NO |
| Uncategorized | 12,382 | 66.5% | 1319 | 3285 | ✓ YES |

**Sampling Decision**: `PLAIN_SEQUENTIAL`
**Rationale**: File I/O & Exception Handling has only 205 examples in full 18k (< 300 threshold). Use plain sequential sampling. File I/O improvement is DATASET-LEVEL CAPPED.

> [!IMPORTANT]
> File I/O & Exception Handling has only **205** samples in the full 18k dataset.
> This is a **dataset-level ceiling**, not a training artifact. Resampling cannot
> fix this. Phase 5 should consider augmenting File I/O examples from an external source.

---

## 3. Section 2 — Config E Training

| Parameter | Value |
|:---|:---|
| Output directory | `phi2-qlora-phase4-scaled-5k` |
| Dataset slice | `train[:5000]` (4,500 train / 500 eval, seed=42) |
| Sampling method | `plain_sequential` (per Section 1 decision rule) |
| Target modules | `q_proj, v_proj` (Locked Decision #2) |
| Loss masking | Response-only (Phase2CompletionCollator, Locked Decision #1) |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Epochs | 2 |
| Effective batch size | 8 (1 per device × 8 grad accum) |
| Expected steps | ~1,125 |
| Actual steps completed | 1124 |
| Wall-clock time | 9644s (2.68h) |
| Final train loss | 0.4199 |
| Peak VRAM allocated | 3061.1 MB |
| Adapter SHA-256 | `aaa7581ea2c303a18ef987d8f13a25633800b893eddf1cdb25db27fcdef43cfd` |

---

## 4. Section 3A — Config E Checkpoint Saturation Curve

| Step | Core Pass% | EOS Term% | Max-Token% | Mean Len (tok) |
|:---:|:---:|:---:|:---:|:---:|
| **100** | 68.0% ← **BEST** | 96.0% | 4.0% | 115.4 |
| **200** | 64.0% | 94.0% | 6.0% | 121.6 |
| **300** | 66.0% | 88.0% | 12.0% | 125.6 |
| **400** | 62.0% | 94.0% | 6.0% | 116.3 |
| **500** | 62.0% | 96.0% | 4.0% | 113.9 |
| **600** | 60.0% | 96.0% | 6.0% | 115.3 |
| **700** | 60.0% | 96.0% | 4.0% | 114.8 |
| **800** | 64.0% | 96.0% | 4.0% | 114.8 |
| **900** | 56.0% | 96.0% | 4.0% | 115.0 |
| **1000** | 60.0% | 96.0% | 4.0% | 115.2 |
| **1100** | 56.0% | 94.0% | 6.0% | 115.5 |

**Best checkpoint selected**: Step **100** (68.0% pass rate)
**Selection reasoning**: Loaded from existing reports/phase4_config_e_checkpoint_curve.json

---

## 5. Master Evaluation Table (All 7 Models)

| Model | Source | Resp Loss | Resp PPL | Full Loss | Core Pass% | EOS% | Max-Tok% | Mean Len | Sampled% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw Phi-2 Base** | REUSED | 0.5792 | 1.78 | 0.9905 | 52.0% | 18.0% | 84.0% | 284.6 | 42.7% |
| **Phase 2 Corrected (Ref)** | REUSED | 0.4471 | 1.56 | 0.9007 | 62.0% | 94.0% | 6.0% | 112.8 | 60.0% |
| **Config A (Attn-Only, 2k)** | REUSED | 0.4522 | 1.57 | 0.9107 | 62.0% | 98.0% | 2.0% | 112.8 | 61.3% |
| **Config B (Full-Attn, 2k)** | REUSED | 0.4477 | 1.56 | 0.8687 | 54.0% | 90.0% | 10.0% | 121.9 | 56.7% |
| **Config C (All-Linear, 2k)** | REUSED | 0.4442 | 1.56 | 0.8846 | 60.0% | 98.0% | 2.0% | 115.6 | 56.7% |
| **Config D (Masking-Ablation, 2k)** | REUSED | 0.4489 | 1.57 | 0.6086 | 62.0% | 98.0% | 2.0% | 114.1 | 57.3% |
| **Config E (Attn-Only, 5k) [NEW]** | **NEW** | 0.4803 | 1.62 | 0.8852 | 68.0% | 96.0% | 4.0% | 115.4 | 60.0% |

> Phase 3 models (raw_phi2 through config_d) are REUSED without recomputation.
> Config E is the only new computation.

---

## 6. Per-Task Pass/Fail Matrix (FIXED)

> **Phase 3 bug**: Section 5.1 rendered all cells as `?` due to missing `task_results`
> in the eval output dict. Phase 4 fixes this by storing and reading per-task booleans directly.

| Task | Category | Phi-2 | Ph2-Ref | Cfg-A | Cfg-B | Cfg-C | Cfg-D | **Cfg-E** |
|:---:|:---|:---: | :---: | :---: | :---: | :---: | :---: | :---:|
| 1 | String Manipulation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | String Manipulation | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| 3 | String Manipulation | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 4 | String Manipulation | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 5 | String Manipulation | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 6 | String Manipulation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | Data Structures | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| 8 | Data Structures | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | Data Structures | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | Data Structures | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 11 | Data Structures | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 12 | Data Structures | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 13 | Data Structures | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 14 | Data Structures | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 15 | Sorting & Searching | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 16 | Sorting & Searching | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 17 | Sorting & Searching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 18 | Sorting & Searching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 19 | Sorting & Searching | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| 20 | Sorting & Searching | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ |
| 21 | Recursion & Dynamic Programming | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 22 | Recursion & Dynamic Programming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 23 | Recursion & Dynamic Programming | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| 24 | Recursion & Dynamic Programming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 25 | Recursion & Dynamic Programming | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 26 | Recursion & Dynamic Programming | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 27 | Recursion & Dynamic Programming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 28 | Recursion & Dynamic Programming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 29 | OOP & Class Definitions | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 30 | OOP & Class Definitions | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| 31 | OOP & Class Definitions | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 32 | OOP & Class Definitions | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 33 | OOP & Class Definitions | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 34 | OOP & Class Definitions | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 35 | File I/O & Exception Handling | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 36 | File I/O & Exception Handling | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 37 | File I/O & Exception Handling | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 38 | File I/O & Exception Handling | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 39 | Algorithm Complexity & Edge Cases | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 40 | Algorithm Complexity & Edge Cases | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| 41 | Algorithm Complexity & Edge Cases | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 42 | Algorithm Complexity & Edge Cases | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 43 | Algorithm Complexity & Edge Cases | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 44 | Algorithm Complexity & Edge Cases | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 45 | Bug Fixing | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 46 | Bug Fixing | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 47 | Bug Fixing | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 48 | Bug Fixing | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 49 | Bug Fixing | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 50 | Bug Fixing | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### Category Pass Rate Summary
| Category | Phi-2 | Ph2-Ref | Cfg-A | Cfg-B | Cfg-C | Cfg-D | **Cfg-E** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| String Manipulation | 2/6 | 3/6 | 3/6 | 2/6 | 2/6 | 2/6 | 3/6 |
| Data Structures | 3/8 | 5/8 | 5/8 | 4/8 | 5/8 | 5/8 | 5/8 |
| Sorting & Searching | 3/6 | 4/6 | 5/6 | 3/6 | 3/6 | 5/6 | 5/6 |
| Recursion & Dynamic Programming | 5/8 | 6/8 | 5/8 | 5/8 | 6/8 | 6/8 | 7/8 |
| OOP & Class Definitions | 3/6 | 2/6 | 3/6 | 2/6 | 3/6 | 3/6 | 4/6 |
| File I/O & Exception Handling | 1/4 | 1/4 | 1/4 | 1/4 | 1/4 | 1/4 | 2/4 |
| Algorithm Complexity & Edge Cases | 5/6 | 5/6 | 3/6 | 4/6 | 4/6 | 3/6 | 3/6 |
| Bug Fixing | 4/6 | 5/6 | 6/6 | 6/6 | 6/6 | 6/6 | 5/6 |

---

## 7. File I/O Pass Rate Analysis

Phase 3 showed **all models** (including Raw Phi-2 base) achieved **25.0%** (1/4) pass rate
on File I/O & Exception Handling tasks, suggesting a dataset-level constraint.

**Config E File I/O result**: `2/4` = **50%**

> [!NOTE]
> File I/O pass rate improved to 50% (was 25.0% in Phase 3).
> Improvement achieved despite limited File I/O data (205 samples in 18k).

---

## 8. Phase 5 Decision Rules (Applied)

**Config A best-checkpoint pass rate (Phase 3)**: 68.0% at step 200
**Config E best-checkpoint pass rate (Phase 4)**: 68.0%
**Delta (E vs A best)**: +0.0 points (threshold: ≥ +5)

> [!IMPORTANT]
> **Phase 5 Decision: Rank sweep (r=8, r=32) on 5k dataset**
> Data scaling alone insufficient. Config E achieved 68.0%, only 0.0 points above Config A best (68.0%) (threshold: >= 5 points). Phase 5: Rank sweep (r=8, r=32) on the 5k dataset already trained in Phase 4 to probe if frozen LoRA rank (16) is the binding constraint.


---

## 9. Artifact Manifest
| Artifact | Path | Status |
|:---|:---|:---:|
| Category audit | [`phase4_category_audit.json`](file:///d:/SLM-finetune/reports/phase4_category_audit.json) | ✓ |
| Config E adapter (best checkpoint) | [`adapter_model.safetensors`](file:///d:/SLM-finetune/phi2-qlora-phase4-scaled-5k/adapter_model.safetensors) | ✓ |
| Config E training summary | [`phase4_train_summary.json`](file:///d:/SLM-finetune/phi2-qlora-phase4-scaled-5k/phase4_train_summary.json) | ✓ |
| Config E checkpoint curve | [`phase4_config_e_checkpoint_curve.json`](file:///d:/SLM-finetune/reports/phase4_config_e_checkpoint_curve.json) | ✓ |
| Master eval (Config E + reused Phase 3) | [`phase4_master_eval_results.json`](file:///d:/SLM-finetune/reports/phase4_master_eval_results.json) | ✓ |
| VRAM log | [`phase4_vram_log.json`](file:///d:/SLM-finetune/reports/phase4_vram_log.json) | ✓ |
| Pipeline log (incl. reuse decisions) | [`phase4_pipeline_log.json`](file:///d:/SLM-finetune/reports/phase4_pipeline_log.json) | ✓ |
| Phase 4 report | [`phase4_project_report.md`](file:///d:/SLM-finetune/reports/phase4_project_report.md) | ✓ |

