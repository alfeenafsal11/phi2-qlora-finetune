"""
eval_harness.py — Unified evaluation engine for Phase 1 and Phase 2.
Implements:
  - Addendum Control G: Native vs Standardized prompt evaluation
  - Addendum Control H: Independent full-sequence vs response-only eval loss
  - Addendum Control I: 12 benchmark prompts + 200-sample held-out loss evaluation
  - Addendum Control J: Functional evaluation failure taxonomy
  - Addendum Control K: Raw vs Core code evaluation
  - Addendum Control L: Generation contamination & termination metrics
"""

import os
import sys
import json
import time
import math
import re
import ast
import traceback
import tempfile
import multiprocessing
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─────────────────────────── 12 benchmark prompts ────────────────────────────

BENCHMARK_PROMPTS = [
    {
        "id": 1,
        "category": "String Manipulation",
        "instruction": "Write a Python function that reverses a string without using slicing.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    assert candidate_fn("hello") == "olleh", f"Failed on 'hello': got {candidate_fn('hello')}"
    assert candidate_fn("") == "", f"Failed on '': got {candidate_fn('')}"
    assert candidate_fn("a") == "a", f"Failed on 'a': got {candidate_fn('a')}"
""",
    },
    {
        "id": 2,
        "category": "String Manipulation",
        "instruction": "Write a Python function that checks if a string is a palindrome.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    assert candidate_fn("racecar") is True, "Failed on 'racecar'"
    assert candidate_fn("hello") is False, "Failed on 'hello'"
    assert candidate_fn("") is True, "Failed on ''"
""",
    },
    {
        "id": 3,
        "category": "String Manipulation",
        "instruction": "Write a Python function that counts the frequency of each character in a string and returns a dictionary.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    res = candidate_fn("hello")
    assert res == {'h': 1, 'e': 1, 'l': 2, 'o': 1}, f"Failed on 'hello': got {res}"
""",
    },
    {
        "id": 4,
        "category": "Data Structures",
        "instruction": "Write a Python function that removes duplicates from a list while preserving order.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    assert candidate_fn([1, 2, 2, 3, 1]) == [1, 2, 3], f"Failed: got {candidate_fn([1, 2, 2, 3, 1])}"
    assert candidate_fn([]) == [], "Failed on empty list"
""",
    },
    {
        "id": 5,
        "category": "Data Structures",
        "instruction": "Write a Python function that flattens a nested list of arbitrary depth.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    inp = [1, [2, [3, 4], 5], 6]
    res = list(candidate_fn(inp))
    assert res == [1, 2, 3, 4, 5, 6], f"Failed on nested list: got {res}"
""",
    },
    {
        "id": 6,
        "category": "Data Structures",
        "instruction": "Write a Python function that merges two dictionaries and sums values for duplicate keys.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 3, 'c': 4}
    res = candidate_fn(d1, d2)
    assert res == {'a': 1, 'b': 5, 'c': 4}, f"Failed: got {res}"
""",
    },
    {
        "id": 7,
        "category": "Sorting & Searching",
        "instruction": "Implement binary search in Python. The function should return the index of the target or -1 if not found.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    arr = [1, 3, 5, 7, 9]
    assert candidate_fn(arr, 5) == 2, f"Failed on found target: got {candidate_fn(arr, 5)}"
    assert candidate_fn(arr, 6) == -1, f"Failed on missing target: got {candidate_fn(arr, 6)}"
""",
    },
    {
        "id": 8,
        "category": "Sorting & Searching",
        "instruction": "Write a Python function that sorts a list of dictionaries by a specified key.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    lst = [{'name': 'B', 'age': 25}, {'name': 'A', 'age': 30}]
    res = candidate_fn(lst, 'name')
    assert res[0]['name'] == 'A' and res[1]['name'] == 'B', f"Failed: got {res}"
""",
    },
    {
        "id": 9,
        "category": "Recursion",
        "instruction": "Write a recursive Python function to compute the nth Fibonacci number.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    # 0, 1, 1, 2, 3, 5, 8
    val5 = candidate_fn(5)
    val6 = candidate_fn(6)
    assert val5 in (5, 8) or val6 in (8, 13), f"Failed on fib: got f(5)={val5}, f(6)={val6}"
""",
    },
    {
        "id": 10,
        "category": "Recursion",
        "instruction": "Write a recursive Python function to compute the factorial of a number.",
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    assert candidate_fn(5) == 120, f"Failed on 5!: got {candidate_fn(5)}"
    assert candidate_fn(0) == 1, f"Failed on 0!: got {candidate_fn(0)}"
""",
    },
    {
        "id": 11,
        "category": "Bug Fixing",
        "instruction": (
            "The following Python function is supposed to return the sum of all even numbers "
            "in a list, but it has a bug. Fix it:\n\n"
            "def sum_evens(lst):\n    total = 0\n    for x in lst:\n        if x % 2 = 0:\n            total += x\n    return total"
        ),
        "input": "",
        "test_code": """
def run_test(candidate_fn):
    assert candidate_fn([1, 2, 3, 4, 5, 6]) == 12, f"Failed on [1..6]: got {candidate_fn([1, 2, 3, 4, 5, 6])}"
    assert candidate_fn([1, 3, 5]) == 0, f"Failed on odd list: got {candidate_fn([1, 3, 5])}"
""",
    },
    {
        "id": 12,
        "category": "File I/O",
        "instruction": "Write a Python function that reads a CSV file and returns a list of dictionaries, one per row.",
        "input": "",
        "test_code": """
import tempfile, os
def run_test(candidate_fn):
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write("name,age\\nAlice,30\\nBob,25\\n")
        tmp_name = f.name
    try:
        res = candidate_fn(tmp_name)
        assert len(res) == 2, f"Expected 2 rows, got {len(res)}"
        assert res[0]['name'] == 'Alice', f"Expected Alice, got {res[0]}"
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
""",
    },
]


os.environ["HF_HUB_OFFLINE"] = "1"

# ─────────────────────────── execution sandbox ───────────────────────────────

def run_code_in_sandbox(code_str: str, test_code_str: str, timeout: float = 3.0) -> dict:
    """Run code in an isolated subprocess with timeout and failure taxonomy."""
    # First check basic syntax parsing
    try:
        ast.parse(code_str)
    except SyntaxError as e:
        return {"success": False, "failure_type": "syntax_error", "error": f"SyntaxError: {e}"}

    test_wrapper = f"""
import sys
code = {repr(code_str)}
test_code = {repr(test_code_str)}
scope = {{}}
try:
    exec(code, scope)
    fns = [v for k, v in scope.items() if callable(v) and not k.startswith("__")]
    if not fns:
        print("FAIL:ASSERT:No callable function defined")
        sys.exit(1)
    target_fn = fns[-1]
    test_scope = {{"candidate_fn": target_fn}}
    exec(test_code, test_scope)
    run_test = test_scope.get("run_test")
    if run_test:
        run_test(target_fn)
    print("SUCCESS")
    sys.exit(0)
except AssertionError as e:
    print(f"FAIL:ASSERT:{{e}}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL:RUNTIME:{{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""
    try:
        import subprocess
        res = subprocess.run(
            [sys.executable, "-c", test_wrapper],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (res.stdout or "").strip()
        if "SUCCESS" in out:
            return {"success": True, "failure_type": "success", "error": None}
        elif "FAIL:ASSERT:" in out:
            err = out.split("FAIL:ASSERT:", 1)[1].strip()
            return {"success": False, "failure_type": "assertion_failure", "error": err}
        elif "FAIL:RUNTIME:" in out:
            err = out.split("FAIL:RUNTIME:", 1)[1].strip()
            return {"success": False, "failure_type": "runtime_error", "error": err}
        else:
            err = (res.stderr or "").strip() or "Unknown failure"
            return {"success": False, "failure_type": "runtime_error", "error": err}
    except subprocess.TimeoutExpired:
        return {"success": False, "failure_type": "timeout", "error": f"Timeout expired (>{timeout}s)"}
    except Exception as e:
        return {"success": False, "failure_type": "runtime_error", "error": str(e)}


# ─────────────────────────── code extractors ─────────────────────────────────

def extract_core_code(text: str) -> str:
    """Extract the first coherent Python function / code block from text."""
    # If markdown code fence exists
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Otherwise, split at secondary section headers
    cutoff_patterns = [
        r"\n###\s+(?:Instruction|Response|Explanation|Output|Exercise|Question|Note|Concept)",
        r"\n```",
    ]
    truncated = text
    for pat in cutoff_patterns:
        m = re.search(pat, truncated, re.IGNORECASE)
        if m:
            truncated = truncated[:m.start()]
    return truncated.strip()


def compute_contamination_metrics(text: str, num_gen_tokens: int, max_tokens: int = 256) -> dict:
    """Measure exact occurrence rates of contamination indicators."""
    return {
        "secondary_instruction": bool(re.search(r"###\s*Instruction:", text, re.IGNORECASE)),
        "secondary_input": bool(re.search(r"###\s*Input:", text, re.IGNORECASE)),
        "secondary_response": bool(re.search(r"###\s*Response:", text, re.IGNORECASE)),
        "repeated_explanation": len(re.findall(r"###\s*Explanation", text, re.IGNORECASE)),
        "repeated_output": len(re.findall(r"###\s*Output", text, re.IGNORECASE)),
        "max_token_termination": (num_gen_tokens >= max_tokens),
        "natural_eos_termination": ("<|endoftext|>" in text or (num_gen_tokens < max_tokens)),
    }


# ─────────────────────────── evaluation loss engine ──────────────────────────

def compute_independent_eval_loss(model, tokenizer, eval_dataset, max_seq_length=512, device=None):
    """
    Computes both full-sequence cross-entropy loss and response-only cross-entropy loss
    on the 200 held-out evaluation samples.
    """
    model.eval()
    if device is None:
        device = getattr(model, "device", None) or next(model.parameters()).device

    response_template_str = "### Response:\n"
    response_token_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
    rlen = len(response_token_ids)

    total_full_loss = 0.0
    total_full_tokens = 0
    total_resp_loss = 0.0
    total_resp_tokens = 0

    with torch.no_grad():
        for ex in eval_dataset:
            inst = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("output") or "").strip()
            
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}"

            enc = tokenizer(text, truncation=True, max_length=max_seq_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            target_ids = input_ids.clone()

            # Find response boundary
            ids_list = input_ids[0].tolist()
            match_idx = None
            for i in range(len(ids_list) - rlen + 1):
                if ids_list[i : i + rlen] == response_token_ids:
                    match_idx = i
                    break

            logits = model(input_ids).logits  # [1, seq_len, vocab_size]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            # 1. Full sequence loss
            loss_full = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            n_full = shift_labels.numel()
            total_full_loss += loss_full.item()
            total_full_tokens += n_full

            # 2. Response only loss
            if match_idx is not None:
                resp_start = match_idx + rlen
                # shift_labels corresponds to input_ids[1:]
                # response begins at resp_start, so in shift_labels its targets start at resp_start - 1
                resp_target_start = max(0, resp_start - 1)
                
                resp_shift_logits = shift_logits[:, resp_target_start:, :].contiguous()
                resp_shift_labels = shift_labels[:, resp_target_start:].contiguous()

                if resp_shift_labels.numel() > 0:
                    loss_resp = F.cross_entropy(resp_shift_logits.view(-1, resp_shift_logits.size(-1)), resp_shift_labels.view(-1), reduction="sum")
                    total_resp_loss += loss_resp.item()
                    total_resp_tokens += resp_shift_labels.numel()

    avg_full_loss = total_full_loss / max(1, total_full_tokens)
    ppl_full = math.exp(min(20, avg_full_loss))
    avg_resp_loss = total_resp_loss / max(1, total_resp_tokens)
    ppl_resp = math.exp(min(20, avg_resp_loss))

    return {
        "full_eval_loss": round(avg_full_loss, 4),
        "full_perplexity": round(ppl_full, 2),
        "response_only_eval_loss": round(avg_resp_loss, 4),
        "response_only_perplexity": round(ppl_resp, 2),
        "total_tokens_evaluated": total_full_tokens,
        "total_response_tokens": total_resp_tokens,
    }


# ─────────────────────────── benchmark prompt evaluator ───────────────────────

def evaluate_model_on_benchmark(model, tokenizer, prompt_format="native_phase1", max_new_tokens=256, device=None):
    """
    Evaluates model on 12 benchmark prompts using deterministic greedy decoding.
    Returns per-prompt metrics, syntax/functional pass rates (Raw and Core), and contamination stats.
    """
    model.eval()
    if device is None:
        device = getattr(model, "device", None) or next(model.parameters()).device

    results = []

    for item in BENCHMARK_PROMPTS:
        pid = item["id"]
        cat = item["category"]
        inst = item["instruction"]
        inp = item["input"]
        test_code = item["test_code"]
        print(f"  [Prompt {pid:2d}/12] Generating ({cat})...", flush=True)

        # Prompt formatting
        if prompt_format == "native_phase1":
            prompt_str = f"### Instruction:\n{inst}\n\n### Response:\n"
        elif prompt_format == "standardized_phase2":
            if inp:
                prompt_str = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt_str = f"### Instruction:\n{inst}\n\n### Response:\n"
        else:
            prompt_str = f"### Instruction:\n{inst}\n\n### Response:\n"

        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Deterministic greedy
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time = time.time() - t0
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        num_gen_tokens = len(gen_tokens)
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        clean_gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # 1. Raw code evaluation
        raw_eval = run_code_in_sandbox(clean_gen_text, test_code)

        # 2. Core code evaluation
        core_code = extract_core_code(clean_gen_text)
        core_eval = run_code_in_sandbox(core_code, test_code)

        # 3. Contamination metrics
        contamination = compute_contamination_metrics(gen_text, num_gen_tokens, max_new_tokens)

        # Check raw syntax parsing
        raw_syntax = True
        try:
            ast.parse(clean_gen_text)
        except SyntaxError:
            raw_syntax = False

        core_syntax = True
        try:
            ast.parse(core_code)
        except SyntaxError:
            core_syntax = False

        print(f"    -> Done in {gen_time:.1f}s | Tokens: {num_gen_tokens} | Raw syntax: {raw_syntax} | Func pass: {raw_eval['success']} | Core pass: {core_eval['success']}", flush=True)

        results.append({
            "prompt_id": pid,
            "category": cat,
            "instruction": inst,
            "prompt_str": prompt_str,
            "generated_raw": gen_text,
            "cleaned_raw": clean_gen_text,
            "core_code": core_code,
            "gen_time_s": round(gen_time, 2),
            "num_gen_tokens": num_gen_tokens,
            "raw_syntax_valid": raw_syntax,
            "raw_functional_pass": raw_eval["success"],
            "raw_failure_type": raw_eval["failure_type"],
            "raw_error": raw_eval["error"],
            "core_syntax_valid": core_syntax,
            "core_functional_pass": core_eval["success"],
            "core_failure_type": core_eval["failure_type"],
            "core_error": core_eval["error"],
            "contamination": contamination,
        })

    # Summary statistics
    total = len(results)
    raw_syntax_rate = sum(1 for r in results if r["raw_syntax_valid"]) / total
    raw_func_rate = sum(1 for r in results if r["raw_functional_pass"]) / total
    core_syntax_rate = sum(1 for r in results if r["core_syntax_valid"]) / total
    core_func_rate = sum(1 for r in results if r["core_functional_pass"]) / total

    eos_term_rate = sum(1 for r in results if r["contamination"]["natural_eos_termination"]) / total
    max_token_rate = sum(1 for r in results if r["contamination"]["max_token_termination"]) / total
    sec_inst_rate = sum(1 for r in results if r["contamination"]["secondary_instruction"]) / total
    rep_exp_count = sum(r["contamination"]["repeated_explanation"] for r in results)

    summary = {
        "raw_syntax_validity_rate": round(raw_syntax_rate, 4),
        "raw_functional_pass_rate": round(raw_func_rate, 4),
        "core_syntax_validity_rate": round(core_syntax_rate, 4),
        "core_functional_pass_rate": round(core_func_rate, 4),
        "eos_termination_rate": round(eos_term_rate, 4),
        "max_token_termination_rate": round(max_token_rate, 4),
        "secondary_prompt_rate": round(sec_inst_rate, 4),
        "total_repeated_explanation_blocks": rep_exp_count,
        "results": results,
    }
    return summary
