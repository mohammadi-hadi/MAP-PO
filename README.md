# MAP-PO

**Learning Sexism Detection Using Multi-Agent Perspectivist Preference Optimization** (MAP-PO), on EXIST 2024.

Annotators are clustered by labeling behavior (YES rate, agreement rate, label
entropy) into three groups per language. Three LLM agents, one per cluster,
are fine-tuned with SFT, DPO / Mars-PO, and GRPO (rejection sampling with a
convex individual/team reward) and vote on the final label. Clusters use
neutral names: `cluster1`/`cluster2`/`cluster3`, ordered by ascending YES rate
(display names come from `src/config.py:CLUSTER_DISPLAY`; legacy names in old
artifacts are normalized at load time).

Four settings cross two backbones with two languages:
`gpt-4.1-mini` via the OpenAI fine-tuning API (GPT-EN, GPT-ES) and Qwen3-8B
trained locally with MLX LoRA on Apple Silicon (Qwen-EN, Qwen-ES).

## Results (Team F1-macro, full test sets)

| Stage | GPT-EN | GPT-ES | Qwen-EN | Qwen-ES |
|-------|--------|--------|---------|---------|
| Zero-shot ensemble | 64.7 | 57.4 | 76.8 | 72.9 |
| SFT (mixed) | 89.1 | 84.6 | **84.7** | **81.0** |
| DPO (individual-only) | 85.8 | **85.5** | 78.1 | 80.5 |
| GRPO (α=0.2 team reward) | **89.9** | 85.1 | 80.9 | 80.6 |

GPT-EN also has Mars-PO (88.8) and SFT-pure (88.4); Qwen-EN has a
label-balanced SFT ablation (83.4). Source: `artifacts/results/*.json`.
**[RESULTS.md](RESULTS.md)** presents every experiment (all tables, sweeps,
ablations, fidelity, clustering) in one generated document — refresh with
`python3 src/evaluation/generate_results_md.py`.

**Key findings**

- **Zero-shot collapse.** Without fine-tuning the three agents are copies:
  0.0% disagreement in both Qwen settings (identical YES rates 0.427 EN /
  0.461 ES), 3.2% on GPT-EN.
- **DPO polarization (4 of 4 settings).** Individual-only DPO pins the
  outer agents' YES rates to the extremes (0.000/1.000 on Qwen,
  0.000/0.997 on GPT-ES, 0.015/0.953 on GPT-EN); disagreement saturates at
  94–100% and per-cluster calibration is destroyed everywhere, even when
  the headline Team F1 survives (GPT-ES).
- **Team signal restores calibration every time.** GRPO's convex reward
  brings mean YES-rate calibration error back to 2.0–8.0 pp and
  disagreement to 34–46% in every setting.
- **Capacity nuance.** Whether the team reward also lifts headline
  accuracy depends on the backbone: GRPO beats SFT on GPT-EN (+0.8 F1)
  and ties it on GPT-ES, but trails SFT on Qwen3-8B.
- **Label balancing hurts.** Balanced-SFT (Qwen-EN) drops Team F1
  84.7 → 83.4 and inflates calibration error 1.7 → 10.9 pp; imbalance is
  handled at the metric level (F1-macro, balanced accuracy) instead.

## Quick start (local, no API key)

```bash
# one-time: create the MLX venv (Apple Silicon)
python3 -m venv .venv-mlx && .venv-mlx/bin/pip install -r requirements.txt mlx-lm

# full bilingual pipeline: clustering -> data prep -> SFT/DPO/GRPO -> evals
.venv-mlx/bin/python src/pipeline/run_local_experiments.py            # both languages
.venv-mlx/bin/python src/pipeline/run_local_experiments.py --lang es  # one language
```

The runner is restart-safe (state in `artifacts/local_state.json`, logs in
`artifacts/logs/`).

## OpenAI suite

```bash
export OPENAI_API_KEY="your-key-here"
python3 src/training/openai_es_suite.py           # full Spanish gpt-4.1-mini suite (resume-safe)
python3 src/training/openai_es_suite.py --status
```

The legacy English OpenAI experiments (SFT / DPO / Mars-PO / GRPO sweeps) are
documented step by step in `PIPELINE.md`.

## Repository layout

```
MAP-PO/
├── src/                         All pipeline code; scripts run from repo root
│   ├── config.py                Paths, hyperparameters, cluster naming
│   ├── preprocessing/           clustering, ablation, data prep, MLX formatting
│   ├── models/                  Agent / MultiAgentSystem
│   ├── training/                sft, dpo, grpo, mlx_*, openai_es_suite
│   ├── evaluation/              evaluate_agents, evaluate_local, metrics, fidelity, plots
│   └── pipeline/                run_local_experiments, orchestrator, verify_pipeline
│
├── data/
│   ├── raw/                     EXIST2024_training.json (and EXIST2023)
│   ├── behavioral_features_{en,es}.csv   Clustering output per language
│   ├── splits/{en,es}/          train / test / split_info / cluster_meta
│   ├── preference/{en,es}/      SFT / DPO / GRPO training JSONL per language
│   └── mlx/{en,es}/             mlx-lm-formatted datasets
│
├── artifacts/
│   ├── model_ids/               OpenAI fine-tune IDs + job records
│   ├── results/                 Per-method evaluation JSONs (all four settings)
│   ├── predictions/             Per-text test predictions per method
│   ├── adapters/                Local LoRA adapters
│   └── clustering/              Clustering figures + per-language reports
│
├── paper/                       ACL LaTeX paper, figures, tables, Overleaf zip
├── report_assets/               Editable per-figure / per-table sources + notebook
├── presentation/                pptx slide deck
└── references/                  Reference PDFs + legacy scripts
```

## Verification before every commit

```bash
python3 src/pipeline/verify_pipeline.py
```

Checks clustering outputs (both languages), data formats, splits, model-ID
files, result JSONs for all four settings, the synthetic-data gate, paper
figures, `paper/main.pdf`, and git hygiene. Exits non-zero if anything
drifted.
