import json

data = json.load(open("reports/eval_audit_full_results.json"))
fa = data["final_adapter"]
for p in fa["prompt_results_greedy"]:
    print("=" * 60)
    print(f"Prompt {p['prompt_id']}: {p['instruction']}")
    print(f"Passed: {p['passed']}, Syntax: {p['syntax_valid']}, Failure: {p['failure_type']}")
    print(f"Cleaned Code:\n{p.get('cleaned_code')}")
    print(f"Raw Output:\n{p.get('generated_output')[:200]}...")
