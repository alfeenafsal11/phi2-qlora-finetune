import json

data = json.load(open("reports/eval_audit_full_results.json"))
fa = data["final_adapter"]
print("Rates greedy:", fa["rates_greedy"])
print("Rates sampled:", fa["rates_sampled"])
print("Loss metrics:", fa["loss_metrics"])
print("\nPrompt results greedy:")
for p in fa["prompt_results_greedy"]:
    err = repr(p.get("exec_error") or p.get("syntax_error"))[:80]
    print(f"P{p['prompt_id']:2d}: syntax={p['syntax_valid']!s:5s} passed={p['passed']!s:5s} fail_type={p['failure_type']} error={err}")
