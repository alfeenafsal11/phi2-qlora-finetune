import json

with open("reports/phase2b_attribution_eval_results.json") as f:
    data = json.load(f)

print("=" * 145)
print(f"{'Model / Condition':<40} | {'RespLoss':<8} | {'RespPPL':<7} | {'FullLoss':<8} | {'CorePass':<8} | {'RawPass':<7} | {'EOSTerm':<7} | {'MaxCut':<7} | {'SecPrompt':<9} | {'MeanTok':<7} | {'SampledPass':<11}")
print("=" * 145)

for mid, d in data.items():
    name = d["name"]
    lm = d["loss_metrics"]
    rloss = lm["response_only_eval_loss"]
    rppl = lm["response_only_perplexity"]
    floss = lm["full_eval_loss"]
    
    bd = d["benchmark_deterministic"]
    cpass = bd["core_functional_pass_rate"] * 100
    rpass = bd["raw_functional_pass_rate"] * 100
    eos = bd["eos_termination_rate"] * 100
    max_cut = bd["max_token_termination_rate"] * 100
    sec = bd["secondary_prompt_rate"] * 100
    mean_len = d["length_stats"]["mean"]
    
    bs = d["benchmark_sampled"]
    spass = bs["sampled_functional_pass_rate"] * 100
    
    print(f"{name:<40} | {rloss:<8.4f} | {rppl:<7.2f} | {floss:<8.4f} | {cpass:<7.1f}% | {rpass:<6.1f}% | {eos:<6.1f}% | {max_cut:<6.1f}% | {sec:<8.1f}% | {mean_len:<7.1f} | {spass:<10.1f}%")

print("=" * 145)

print("\n\n" + "=" * 80)
print("INTERVENTION ATTRIBUTION DECOMPOSITION")
print("=" * 80)

ref = data["phase2_corrected"]
ab_a = data["phase2b_eos_ablation"]
ab_b = data["phase2b_masking_ablation"]
ab_c = data["phase2b_input_ablation"]
base = data["raw_phi2"]
broken = data["broken_baseline"]

print(f"\n1. ISOLATING TERMINAL EOS SUPERVISION (Reference vs Ablation A [EOS=NO]):")
print(f"   - EOS Termination Rate:       {ref['benchmark_deterministic']['eos_termination_rate']*100:.1f}% vs {ab_a['benchmark_deterministic']['eos_termination_rate']*100:.1f}% (Delta: {ab_a['benchmark_deterministic']['eos_termination_rate']*100 - ref['benchmark_deterministic']['eos_termination_rate']*100:+.1f}%)")
print(f"   - Max Token Cutoff Rate:      {ref['benchmark_deterministic']['max_token_termination_rate']*100:.1f}% vs {ab_a['benchmark_deterministic']['max_token_termination_rate']*100:.1f}% (Delta: {ab_a['benchmark_deterministic']['max_token_termination_rate']*100 - ref['benchmark_deterministic']['max_token_termination_rate']*100:+.1f}%)")
print(f"   - Mean Generation Length:     {ref['length_stats']['mean']:.1f} vs {ab_a['length_stats']['mean']:.1f} tokens")
print(f"   - Core Functional Pass Rate:  {ref['benchmark_deterministic']['core_functional_pass_rate']*100:.1f}% vs {ab_a['benchmark_deterministic']['core_functional_pass_rate']*100:.1f}% (Delta: {ab_a['benchmark_deterministic']['core_functional_pass_rate']*100 - ref['benchmark_deterministic']['core_functional_pass_rate']*100:+.1f}%)")
print(f"   - Response-Only Loss:         {ref['loss_metrics']['response_only_eval_loss']:.4f} vs {ab_a['loss_metrics']['response_only_eval_loss']:.4f} (Delta: {ab_a['loss_metrics']['response_only_eval_loss'] - ref['loss_metrics']['response_only_eval_loss']:+.4f})")

print(f"\n2. ISOLATING RESPONSE-ONLY LOSS MASKING (Reference vs Ablation B [Masking=NO]):")
print(f"   - Response-Only Loss:         {ref['loss_metrics']['response_only_eval_loss']:.4f} vs {ab_b['loss_metrics']['response_only_eval_loss']:.4f} (Delta: {ab_b['loss_metrics']['response_only_eval_loss'] - ref['loss_metrics']['response_only_eval_loss']:+.4f})")
print(f"   - Full Sequence Loss:         {ref['loss_metrics']['full_eval_loss']:.4f} vs {ab_b['loss_metrics']['full_eval_loss']:.4f} (Delta: {ab_b['loss_metrics']['full_eval_loss'] - ref['loss_metrics']['full_eval_loss']:+.4f})")
print(f"   - Core Functional Pass Rate:  {ref['benchmark_deterministic']['core_functional_pass_rate']*100:.1f}% vs {ab_b['benchmark_deterministic']['core_functional_pass_rate']*100:.1f}% (Delta: {ab_b['benchmark_deterministic']['core_functional_pass_rate']*100 - ref['benchmark_deterministic']['core_functional_pass_rate']*100:+.1f}%)")
print(f"   - Training Time:              3,202s vs 16,891s (5.27x slowdown due to backpropping entire sequence)")

print(f"\n3. ISOLATING INPUT PRESERVATION (Reference vs Ablation C [Input=NO]):")
print(f"   - Response-Only Loss:         {ref['loss_metrics']['response_only_eval_loss']:.4f} vs {ab_c['loss_metrics']['response_only_eval_loss']:.4f} (Delta: {ab_c['loss_metrics']['response_only_eval_loss'] - ref['loss_metrics']['response_only_eval_loss']:+.4f})")
print(f"   - Full Sequence Loss:         {ref['loss_metrics']['full_eval_loss']:.4f} vs {ab_c['loss_metrics']['full_eval_loss']:.4f} (Delta: {ab_c['loss_metrics']['full_eval_loss'] - ref['loss_metrics']['full_eval_loss']:+.4f})")
print(f"   - Core Functional Pass Rate:  {ref['benchmark_deterministic']['core_functional_pass_rate']*100:.1f}% vs {ab_c['benchmark_deterministic']['core_functional_pass_rate']*100:.1f}% (Delta: {ab_c['benchmark_deterministic']['core_functional_pass_rate']*100 - ref['benchmark_deterministic']['core_functional_pass_rate']*100:+.1f}%)")
