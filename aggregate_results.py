import json

with open("reports/phase2_full_eval_results.json") as f:
    data = json.load(f)

print(f"{'System':<32} | {'FullLoss':<8} | {'RespLoss':<8} | {'RespPPL':<8} | {'RawSyn':<7} | {'RawPass':<7} | {'CoreSyn':<7} | {'CorePass':<8} | {'EOSTerm':<7} | {'AvgTok':<7} | {'SecPrompt':<9}")
print("-" * 135)

for sid, sdata in data.items():
    sname = sdata["name"]
    lm = sdata["loss_metrics"]
    floss = lm["full_eval_loss"]
    rloss = lm["response_only_eval_loss"]
    rppl = lm["response_only_perplexity"]
    
    bn = sdata["benchmark_native"]
    rsyn = bn["raw_syntax_validity_rate"] * 100
    rpass = bn["raw_functional_pass_rate"] * 100
    csyn = bn.get("core_syntax_validity_rate", 1.0) * 100
    cpass = bn["core_functional_pass_rate"] * 100
    eos = bn["eos_termination_rate"] * 100
    sec = bn.get("secondary_prompt_rate", 0.0) * 100
    avg_tok = sum(r["num_gen_tokens"] for r in bn["results"]) / len(bn["results"])
    
    print(f"{sname:<32} | {floss:<8.4f} | {rloss:<8.4f} | {rppl:<8.2f} | {rsyn:<6.1f}% | {rpass:<6.1f}% | {csyn:<6.1f}% | {cpass:<7.1f}% | {eos:<6.1f}% | {avg_tok:<7.1f} | {sec:<8.1f}%")

    if sid == "broken_baseline":
        bs = sdata["benchmark_standardized"]
        rsyn_s = bs["raw_syntax_validity_rate"] * 100
        rpass_s = bs["raw_functional_pass_rate"] * 100
        csyn_s = bs.get("core_syntax_validity_rate", 1.0) * 100
        cpass_s = bs["core_functional_pass_rate"] * 100
        eos_s = bs["eos_termination_rate"] * 100
        sec_s = bs.get("secondary_prompt_rate", 0.0) * 100
        avg_tok_s = sum(r["num_gen_tokens"] for r in bs["results"]) / len(bs["results"])
        print(f"  -> (Standardized Format)      | {'-':<8} | {'-':<8} | {'-':<8} | {rsyn_s:<6.1f}% | {rpass_s:<6.1f}% | {csyn_s:<6.1f}% | {cpass_s:<7.1f}% | {eos_s:<6.1f}% | {avg_tok_s:<7.1f} | {sec_s:<8.1f}%")

print("\n" + "=" * 100)
print("TAXONOMY BREAKDOWN (Core Failure Types)")
print("=" * 100)
for sid, sdata in data.items():
    print(f"\nSystem: {sdata['name']}")
    for r in sdata["benchmark_native"]["results"]:
        status = "PASS" if r["core_functional_pass"] else f"FAIL ({r['core_failure_type']}: {str(r['core_error'])[:40]})"
        print(f"  P{r['prompt_id']:02d} [{r['category']}]: {status} (toks: {r['num_gen_tokens']}, time: {r['gen_time_s']:.1f}s)")
