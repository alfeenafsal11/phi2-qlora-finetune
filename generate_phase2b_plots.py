import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Dark modern theme
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#c8ccd8",
    "axes.titlecolor":   "#e0e4f0",
    "xtick.color":       "#8890a6",
    "ytick.color":       "#8890a6",
    "grid.color":        "#2a2e3d",
    "grid.linewidth":    0.8,
    "text.color":        "#c8ccd8",
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4d",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})

os.makedirs("plots", exist_ok=True)

# Load master evaluation results
with open("reports/phase2b_attribution_eval_results.json") as f:
    results = json.load(f)

# Load training histories
train_histories = {}
for ab_name, pth in [
    ("Phase 2 Corrected", "phi2-qlora-phase2-corrected/phase2_train_summary.json"),
    ("Ablation A (EOS=NO)", "phi2-qlora-phase2b-eos-ablation/phase2b_train_summary.json"),
    ("Ablation B (Masking=NO)", "phi2-qlora-phase2b-masking-ablation/phase2b_train_summary.json"),
    ("Ablation C (Input=NO)", "phi2-qlora-phase2b-input-ablation/phase2b_train_summary.json"),
]:
    if os.path.exists(pth):
        with open(pth) as f:
            train_histories[ab_name] = json.load(f)

# 1. Figure 1: Attribution Matrix (4-panel)
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Phase 2B — Factorial Intervention Attribution Matrix (Phi-2 QLoRA)", fontsize=15, fontweight="bold", color="#f0f4fc", y=0.98)

labels = [
    "Raw Phi-2\n(Base)",
    "Broken Base\n(Phase 1)",
    "Phase 2 Ref\n(Corrected)",
    "Ablation A\n(EOS=NO)",
    "Ablation B\n(Masking=NO)",
    "Ablation C\n(Input=NO)",
]
keys = ["raw_phi2", "broken_baseline", "phase2_corrected", "phase2b_eos_ablation", "phase2b_masking_ablation", "phase2b_input_ablation"]
colors = ["#8890a6", "#fd7c7c", "#7cfda8", "#fdca7c", "#7cddfd", "#c87cfd"]

# Panel 1: Core Functional Pass Rate
ax1 = axs[0, 0]
pass_rates = [results[k]["benchmark_deterministic"]["core_functional_pass_rate"] * 100 for k in keys]
bars1 = ax1.bar(labels, pass_rates, color=colors, edgecolor="#ffffff22", width=0.55)
ax1.set_title("Core Functional Pass Rate (%) — 12-Task Benchmark", pad=12)
ax1.set_ylabel("Pass Rate (%)")
ax1.set_ylim(0, 115)
ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
for bar, val in zip(bars1, pass_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 2.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#ffffff")

# Panel 2: Termination Dynamics (EOS vs Max Token)
ax2 = axs[0, 1]
x = np.arange(len(labels))
w = 0.35
eos_rates = [results[k]["benchmark_deterministic"]["eos_termination_rate"] * 100 for k in keys]
cut_rates = [results[k]["benchmark_deterministic"]["max_token_termination_rate"] * 100 for k in keys]
r1 = ax2.bar(x - w/2, eos_rates, width=w, label="EOS Termination (%)", color="#7cfda8", edgecolor="#ffffff22")
r2 = ax2.bar(x + w/2, cut_rates, width=w, label="Max-Token Cutoff (%)", color="#fd7c7c", edgecolor="#ffffff22")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_title("Termination Mode: EOS Supervision vs Runaway Cutoff", pad=12)
ax2.set_ylabel("Percentage (%)")
ax2.set_ylim(0, 115)
ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
ax2.legend(loc="upper right")

# Panel 3: Response-Only Loss & Perplexity
ax3 = axs[1, 0]
resp_loss = [results[k]["loss_metrics"]["response_only_eval_loss"] for k in keys]
bars3 = ax3.bar(labels, resp_loss, color=colors, edgecolor="#ffffff22", width=0.55)
ax3.set_title("Standardized Response-Only Loss (Held-out 200 samples)", pad=12)
ax3.set_ylabel("Cross-Entropy Loss")
ax3.set_ylim(0, 0.70)
ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
for bar, val in zip(bars3, resp_loss):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#ffffff")

# Panel 4: Mean Generation Length (Tokens)
ax4 = axs[1, 1]
mean_toks = [results[k]["length_stats"]["mean"] for k in keys]
bars4 = ax4.bar(labels, mean_toks, color=colors, edgecolor="#ffffff22", width=0.55)
ax4.set_title("Mean Generation Length (Tokens) — Runaway vs Concise", pad=12)
ax4.set_ylabel("Mean Generated Tokens")
ax4.set_ylim(0, 300)
ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
for bar, val in zip(bars4, mean_toks):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 5, f"{val:.1f}t", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#ffffff")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/phase2b_attribution_matrix.png", dpi=300)
plt.close()
print("Saved plots/phase2b_attribution_matrix.png")

# 2. Figure 2: Training Loss Dynamics Curves
fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Phase 2B Training & Eval Loss Curves across Factorial Ablations", fontsize=14, fontweight="bold", color="#f0f4fc")

curve_colors = {
    "Phase 2 Corrected": "#7cfda8",
    "Ablation A (EOS=NO)": "#fdca7c",
    "Ablation B (Masking=NO)": "#7cddfd",
    "Ablation C (Input=NO)": "#c87cfd",
}

for name, hist in train_histories.items():
    logs = hist.get("log_history", [])
    steps = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]
    eval_steps = [l["step"] for l in logs if "eval_loss" in l]
    eval_losses = [l["eval_loss"] for l in logs if "eval_loss" in l]
    
    col = curve_colors.get(name, "#ffffff")
    if steps:
        ax_train.plot(steps, losses, label=name, color=col, linewidth=2.0)
    if eval_steps:
        ax_eval.plot(eval_steps, eval_losses, label=name, color=col, linewidth=2.0, marker="o", markersize=5)

ax_train.set_title("Training Loss vs Optimization Step (450 Steps, 2 Epochs)", pad=10)
ax_train.set_xlabel("Global Step")
ax_train.set_ylabel("Training Loss")
ax_train.grid(True, linestyle="--", alpha=0.5)
ax_train.legend()

ax_eval.set_title("Periodic Evaluation Loss vs Step (Held-out Split)", pad=10)
ax_eval.set_xlabel("Global Step")
ax_eval.set_ylabel("Eval Loss")
ax_eval.grid(True, linestyle="--", alpha=0.5)
ax_eval.legend()

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("plots/phase2b_training_curves.png", dpi=300)
plt.close()
print("Saved plots/phase2b_training_curves.png")
