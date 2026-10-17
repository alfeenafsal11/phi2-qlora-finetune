import json

harness = json.load(open("reports/eval_harness_regression_report.json"))["measured_harness"]["results"]
for h in harness:
    print(f"=== P{h['prompt_id']} ({h['category']}) ===")
    print("Raw Syntax:", h["raw_syntax_valid"], "| Func Pass:", h["raw_functional_pass"], "| Core Pass:", h["core_functional_pass"])
    print("Generated Text Preview:")
    print(h["cleaned_raw"][:250])
    print("-" * 50)
