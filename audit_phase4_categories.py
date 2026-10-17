"""
audit_phase4_categories.py -- Phase 4 Section 1: Data Category Audit

Classifies all ~18,000 samples from iamtarun/python_code_instructions_18k_alpaca
into the 8 Phase 3 benchmark categories using keyword/pattern heuristics on the
instruction field.

Outputs: reports/phase4_category_audit.json
         Appends reuse ledger entries to reports/phase4_pipeline_log.json
"""

import os
import re
import json
import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
from datasets import load_dataset

REPORT_PATH = "reports/phase4_category_audit.json"
PIPELINE_LOG_PATH = "reports/phase4_pipeline_log.json"

CATEGORY_PATTERNS = [
    ("Bug Fixing", [
        r"\bfix\b", r"\bbug\b", r"\bdebug\b", r"\bbroken\b",
        r"\bwrong\b", r"\bincorrect\b", r"\bdefect\b", r"\bpatch\b",
        r"\bcorrect\b.*\bcode\b", r"\brepair\b",
    ]),
    ("File I/O & Exception Handling", [
        r"\bfile\b", r"\bread.*file\b", r"\bwrite.*file\b",
        r"\bopen.*file\b", r"\bexception\b", r"\btry.*except\b",
        r"\berror handling\b", r"\bwith open\b", r"\bos\.path\b",
        r"\bpathlib\b", r"\bcsv\b", r"\bpickle\b", r"\bshutil\b",
    ]),
    ("OOP & Class Definitions", [
        r"\bclass\b", r"\bobject.orient\b", r"\binheritanc\b",
        r"\bsubclass\b", r"\bpolymorphi\b", r"\bencapsulat\b",
        r"\b__init__\b", r"\binstanc\b", r"\boverrid\b",
        r"\babstract.*class\b", r"\boop\b",
    ]),
    ("Recursion & Dynamic Programming", [
        r"\brecursi\b", r"\bdynamic programming\b",
        r"\bmemoiz\b", r"\btop.?down\b", r"\bbottom.?up\b",
        r"\bfibonacci\b", r"\bfactorial\b", r"\bbacktrack\b",
        r"\bdivide.and.conquer\b", r"\bknapsack\b",
    ]),
    ("Sorting & Searching", [
        r"\bsort\b", r"\bbinary search\b",
        r"\bquick sort\b", r"\bmerge sort\b", r"\bbubble sort\b",
        r"\binsertion sort\b", r"\bheap sort\b", r"\blinear search\b",
        r"\bfind.*element\b",
    ]),
    ("Data Structures", [
        r"\blinked list\b", r"\bstack\b", r"\bqueue\b",
        r"\btree\b", r"\bgraph\b", r"\bhash\b", r"\bheap\b",
        r"\barray\b", r"\bmatrix\b", r"\bdeque\b",
        r"\btrie\b", r"\bbst\b", r"\bbinary tree\b",
    ]),
    ("Algorithm Complexity & Edge Cases", [
        r"\bcomplexity\b", r"\btime complexity\b", r"\bspace complexity\b",
        r"\bbig.?o\b", r"\bedge case\b", r"\boptimiz\b",
        r"\befficient.*algorithm\b", r"\bworst.case\b",
    ]),
    ("String Manipulation", [
        r"\bstring\b", r"\bcharacter\b", r"\bsubstring\b",
        r"\bpalindrome\b", r"\banagram\b", r"\breverse.*string\b",
        r"\bcount.*char\b", r"\bformat.*string\b",
        r"\bsplit\b", r"\bjoin\b", r"\bregex\b",
    ]),
]

CATEGORY_ORDER = [
    "String Manipulation", "Data Structures", "Sorting & Searching",
    "Recursion & Dynamic Programming", "OOP & Class Definitions",
    "File I/O & Exception Handling", "Algorithm Complexity & Edge Cases",
    "Bug Fixing", "Uncategorized",
]
OVERSAMPLING_THRESHOLD = 300


def classify_instruction(instruction: str) -> str:
    text = instruction.lower()
    for category, patterns in CATEGORY_PATTERNS:
        for pat in patterns:
            if re.search(pat, text):
                return category
    return "Uncategorized"


def count_categories(samples) -> dict:
    counts = {cat: 0 for cat in CATEGORY_ORDER}
    for ex in samples:
        cat = classify_instruction(ex.get("instruction", ""))
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def log_event(level: str, message: str, pipeline_log: list):
    ts = datetime.datetime.utcnow().isoformat()
    entry = {"timestamp_utc": ts, "level": level, "message": message}
    pipeline_log.append(entry)
    prefix = {"INFO": "[INFO]", "DONE": "[DONE]", "START": "[RUN ]",
               "ERROR": "[FAIL]", "WARN": "[WARN]", "REUSE": "[REUSE]"}.get(level, "[....]")
    print(f"[{ts[:19]}] {prefix} {message}", flush=True)


def save_pipeline_log(pipeline_log: list):
    os.makedirs("reports", exist_ok=True)
    existing = []
    if os.path.exists(PIPELINE_LOG_PATH):
        try:
            with open(PIPELINE_LOG_PATH, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.extend(pipeline_log)
    with open(PIPELINE_LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    pipeline_log = []
    log_event("START", "Phase 4 Section 1: Data category audit", pipeline_log)

    REUSE_ARTIFACTS = [
        ("reports/phase3_master_eval_results.json",
         "All 6 Phase 3 model evaluation results reused without recomputation per CRITICAL RULE"),
        ("reports/phase3_checkpoint_curves.json",
         "Config A/B/C checkpoint curves reused without recomputation per CRITICAL RULE"),
        ("reports/phase3_masking_ablation_timing.json",
         "Masking timing closed per Locked Decision #1 — reused without recomputation"),
        ("reports/phase3_vram_log.json",
         "Phase 3 VRAM logs for configs A-D reused without recomputation per CRITICAL RULE"),
        ("reports/phase3_benchmark_50tasks.json",
         "50-task OOD-verified benchmark reused without regeneration per CRITICAL RULE"),
    ]
    for path, justification in REUSE_ARTIFACTS:
        log_event("REUSE", f"SOURCE={path} | {justification}", pipeline_log)

    log_event("START", "Loading full 18k dataset", pipeline_log)
    full_ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    total_samples = len(full_ds)
    log_event("INFO", f"Dataset loaded: {total_samples} samples", pipeline_log)

    log_event("START", "Classifying full 18k dataset", pipeline_log)
    full_counts = count_categories(full_ds)

    log_event("START", "Classifying train[:2000] slice", pipeline_log)
    counts_2k = count_categories(full_ds.select(range(min(2000, total_samples))))

    log_event("START", "Classifying train[:5000] slice", pipeline_log)
    counts_5k = count_categories(full_ds.select(range(min(5000, total_samples))))

    sufficient_flags = {cat: full_counts.get(cat, 0) >= OVERSAMPLING_THRESHOLD
                        for cat in CATEGORY_ORDER}

    file_io_count = full_counts.get("File I/O & Exception Handling", 0)
    file_io_sufficient = file_io_count >= OVERSAMPLING_THRESHOLD

    categorized = {k: v for k, v in full_counts.items() if k != "Uncategorized"}
    mean_count = sum(categorized.values()) / max(1, len(categorized))

    if file_io_sufficient:
        sampling_decision = "category_balanced"
        sampling_rationale = (
            f"File I/O & Exception Handling has {file_io_count} examples in full 18k "
            f"(>= {OVERSAMPLING_THRESHOLD} threshold). Use category-balanced sampling."
        )
    else:
        sampling_decision = "plain_sequential"
        sampling_rationale = (
            f"File I/O & Exception Handling has only {file_io_count} examples in full 18k "
            f"(< {OVERSAMPLING_THRESHOLD} threshold). Use plain sequential sampling. "
            "File I/O improvement is DATASET-LEVEL CAPPED."
        )

    category_details = []
    for cat in CATEGORY_ORDER:
        c18k = full_counts.get(cat, 0)
        c2k  = counts_2k.get(cat, 0)
        c5k  = counts_5k.get(cat, 0)
        category_details.append({
            "category": cat,
            "count_full_18k":  c18k,
            "pct_full_18k":    round(c18k / max(1, total_samples) * 100, 2),
            "count_train_2k":  c2k,
            "pct_train_2k":    round(c2k  / max(1, 2000) * 100, 2),
            "count_train_5k":  c5k,
            "pct_train_5k":    round(c5k  / max(1, 5000) * 100, 2),
            "sufficient_for_oversampling": sufficient_flags.get(cat, False),
            "below_mean_18k":  (c18k < mean_count) if cat != "Uncategorized" else None,
        })

    print("\n" + "=" * 95)
    print(f"PHASE 4 CATEGORY AUDIT — FULL 18K (n={total_samples})")
    print("=" * 95)
    print(f"{'Category':<42} {'18k':>7} {'18k%':>6} {'2k':>6} {'5k':>6} {'>=300?':>8}")
    print("-" * 95)
    for d in category_details:
        suf = "YES" if d["sufficient_for_oversampling"] else "NO"
        print(f"{d['category']:<42} {d['count_full_18k']:>7} "
              f"{d['pct_full_18k']:>5.1f}% {d['count_train_2k']:>6} "
              f"{d['count_train_5k']:>6} {suf:>8}")
    print("=" * 95)
    print(f"\nDecision: {sampling_decision.upper()} — {sampling_rationale}\n")

    audit_result = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "total_dataset_samples": total_samples,
        "oversampling_threshold": OVERSAMPLING_THRESHOLD,
        "mean_category_count_18k": round(mean_count, 1),
        "sampling_decision": sampling_decision,
        "sampling_rationale": sampling_rationale,
        "file_io_sufficient_for_oversampling": file_io_sufficient,
        "category_details": category_details,
        "category_counts_full_18k": full_counts,
        "category_counts_train_2k": counts_2k,
        "category_counts_train_5k": counts_5k,
    }

    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(audit_result, f, indent=2)
    print(f"[Audit] Saved to {REPORT_PATH}")
    log_event("DONE", f"Category audit done. Decision: {sampling_decision}", pipeline_log)
    save_pipeline_log(pipeline_log)
    return audit_result


if __name__ == "__main__":
    main()
