#!/usr/bin/env python3
"""
Export artifacts/results/<lang>_core_backfill.json: the aggregate metrics that
generate_results_md.py needs for columns whose original result files predate
the full metric block (the GPT-EN runs).

The export must run in a checkout that has data/splits/<lang>/ — i.e. after
regenerating the splits from the EXIST source — because the gold labels are
not distributed with this repository. Only aggregate metrics are written, so
the exported file is safe to commit.

Run from the repo root:
    python3 src/evaluation/export_core_backfill.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from evaluation.evaluate_local import core_metrics, subset_metrics  # noqa: E402

LANG = "en"
CORE_KEYS = [
    "always_no", "base_41mini", "persona_41mini", "dpo_41mini",
    "sft_pure_41mini", "sft_41mini", "marspo_41mini", "grpo_41mini_alpha02",
]
SUBSET_KEYS = {"sft_41mini", "dpo_41mini", "grpo_41mini_alpha02"}


def main():
    splits = Path(config.SPLITS_DIR) / LANG
    if not (splits / "test_set.json").exists():
        sys.exit(f"missing {splits}/test_set.json — regenerate the splits "
                 "from the EXIST source first (see PIPELINE.md)")
    with open(splits / "test_set.json") as f:
        records = json.load(f)
    with open(splits / "cluster_meta.json") as f:
        expected = json.load(f)["expected_yes_rates"]

    out = {}
    for key in CORE_KEYS:
        with open(Path(config.PREDICTIONS_DIR) / f"{key}_predictions.json") as f:
            preds = json.load(f)
        preds = [{**x, "agent_predictions":
                  config.normalize_cluster_keyed(x["agent_predictions"])} for x in preds]
        entry = {"core": core_metrics(preds, records, expected)}
        if key in SUBSET_KEYS:
            entry["all_clusters_subset"] = subset_metrics(preds, records, expected, 3)
        out[key] = entry

    dest = Path(config.RESULTS_DIR) / f"{LANG}_core_backfill.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
