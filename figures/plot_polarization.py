#!/usr/bin/env python3
"""
The polarization -> repair figure: per-cluster agent YES rates after SFT, DPO
and GRPO against the cluster annotator-mean targets, in all four settings.

Reads only committed files (artifacts/results/, artifacts/clustering/), so the
figure regenerates on a fresh clone. Requires matplotlib:

    pip install matplotlib
    python3 figures/plot_polarization.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

CLUSTERS = ("cluster1", "cluster2", "cluster3")
SETTINGS = ["GPT-EN", "GPT-ES", "Qwen-EN", "Qwen-ES"]
STAGE_KEYS = {
    "GPT-EN": {"SFT": "sft_41mini", "DPO": "dpo_41mini",
               "GRPO": "grpo_41mini_alpha02"},
    "GPT-ES": {"SFT": "gpt41mini_sft_es", "DPO": "gpt41mini_dpo_es",
               "GRPO": "gpt41mini_grpo_a20_es"},
    "Qwen-EN": {"SFT": "local_qwen3_sft_en", "DPO": "local_qwen3_dpo_en",
                "GRPO": "local_qwen3_grpo_a20_en"},
    "Qwen-ES": {"SFT": "local_qwen3_sft_es", "DPO": "local_qwen3_dpo_es",
                "GRPO": "local_qwen3_grpo_a20_es"},
}
STYLE = {  # stage -> (color, marker)
    "SFT": ("#9CA3AF", "o"),
    "DPO": ("#B91C1C", "x"),
    "GRPO": ("#1E3A5F", "s"),
}


def yes_rates(key):
    with open(Path(config.RESULTS_DIR) / f"{key}_results.json") as f:
        res = config.normalize_result(json.load(f))
    return [res[f"yes_rate_{c}"] * 100 for c in CLUSTERS]


def targets(lang):
    with open(ROOT / "artifacts" / "clustering" / lang / "cluster_summary.json") as f:
        s = json.load(f)
    return [s["features"][c]["yes_rate"]["mean"] * 100 for c in CLUSTERS]


def main():
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    x = range(len(CLUSTERS))
    for ax, setting in zip(axes, SETTINGS):
        lang = "en" if setting.endswith("EN") else "es"
        tgt = targets(lang)
        ax.plot(x, tgt, color="black", ls="--", lw=1.2, marker="_",
                ms=16, label="annotator target", zorder=2)
        for stage in ("SFT", "DPO", "GRPO"):
            color, marker = STYLE[stage]
            ax.plot(x, yes_rates(STAGE_KEYS[setting][stage]), color=color,
                    marker=marker, ls="", ms=7, mew=2, label=stage, zorder=3)
        ax.set_title(setting, fontsize=11)
        ax.set_xticks(list(x))
        ax.set_xticklabels(["c1", "c2", "c3"])
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("agent YES rate (%)")
    axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False)
    fig.suptitle("Individual-only DPO polarizes the outer agents; "
                 "a team signal pulls them back to their cluster", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "figures" / "polarization.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
