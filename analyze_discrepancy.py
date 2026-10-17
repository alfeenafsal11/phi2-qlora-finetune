import json

rec = json.load(open("reports/eval_audit_full_results.json"))["final_adapter"]["prompt_results_greedy"]
harness = json.load(open("reports/eval_harness_regression_report.json"))["measured_harness"]["results"]

print("=" * 80)
print("PROMPT-BY-PROMPT REGRESSION DISCREPANCY ANALYSIS (Control F)")
print("=" * 80)

for r, h in zip(rec, harness):
    pid = r["prompt_id"]
    r_syn = r["syntax_valid"]
    h_syn = h["raw_syntax_valid"]
    r_pass = r["passed"]
    h_pass = h["raw_functional_pass"]
    
    print(f"\nPrompt {pid:2d} ({r['category']}):")
    print(f"  Recorded (FP16 offload):  Syntax={r_syn!s:5s} | FuncPass={r_pass!s:5s} | Error: {r.get('syntax_error') or r.get('exec_error')}")
    print(f"  Harness (4-bit NF4 GPU):  Syntax={h_syn!s:5s} | FuncPass={h_pass!s:5s} | Error: {h.get('raw_error')}")
    if r_syn != h_syn or r_pass != h_pass:
        print("  --> DISCREPANCY DETECTED:")
        print(f"      Recorded text preview:\n      {repr(r['generated_output'][:150])}")
        print(f"      Harness text preview:\n      {repr(h['generated_raw'][:150])}")
