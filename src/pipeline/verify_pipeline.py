#!/usr/bin/env python3
"""
Pipeline Integrity Checker
===========================
Verifies the entire MAP-PO pipeline is in a consistent state.
Run before every commit/push to catch regressions.

Exit codes:
  0 = all checks passed
  1 = one or more checks failed
"""

import json
import os
import subprocess
import sys

# Make `import config` and `from models.agent ...` work from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(name, condition, detail=""):
    status = OK if condition else FAIL
    print(f"  {status} {name}" + (f" - {detail}" if detail else ""))
    return condition


def warn(name, detail):
    print(f"  {WARN} {name} - {detail}")


def main():
    print("=" * 60)
    print("MAP-PO Pipeline Verification")
    print("=" * 60)

    all_passed = True

    # Sections 1-4 need data/, which is regenerated locally from the EXIST
    # source and never distributed; on a fresh clone they are skipped.
    has_data = os.path.isdir("data")
    if not has_data:
        print("\n1-4. Data checks")
        warn("data/ not present",
             "skipped — regenerate it from the EXIST source first (see README)")

    if has_data:
        # 1. behavioral_features.csv has cluster column
        print("\n1. Clustering output")
        try:
            import csv
            with open("data/behavioral_features.csv") as f:
                rows = list(csv.DictReader(f))
            has_cluster = bool(rows) and "cluster" in rows[0]
            all_passed &= check("cluster column exists", has_cluster)
            if has_cluster:
                clusters = {r["cluster"] for r in rows}
                expected = {"Conservative", "Mainstream", "Sensitive"}
                all_passed &= check(
                    "3 expected clusters",
                    clusters == expected,
                    f"got {clusters}",
                )
                all_passed &= check("348 annotators", len(rows) == 348, f"got {len(rows)}")
        except Exception as e:
            all_passed &= check("CSV loadable", False, str(e))

        # 1b. Per-language clustering outputs (neutral cluster keys)
        print("\n1b. Per-language clustering outputs")
        import csv
        expected_rows = {"en": 348, "es": 390}
        for lang, n_expected in expected_rows.items():
            path = f"data/behavioral_features_{lang}.csv"
            exists = os.path.exists(path)
            all_passed &= check(path, exists)
            if not exists:
                continue
            try:
                with open(path) as f:
                    rows = list(csv.DictReader(f))
                clusters = {r["cluster"] for r in rows}
                all_passed &= check(
                    f"  {lang}: cluster values are cluster1..3",
                    clusters == {"cluster1", "cluster2", "cluster3"},
                    f"got {sorted(clusters)}",
                )
                all_passed &= check(
                    f"  {lang}: {n_expected} annotators",
                    len(rows) == n_expected,
                    f"got {len(rows)}",
                )
            except Exception as e:
                all_passed &= check(f"  {lang}: CSV loadable", False, str(e))

        # 2. SFT JSONL files
        print("\n2. SFT training data")
        for cluster in ["conservative", "mainstream", "sensitive"]:
            path = f"data/preference/sft_{cluster}_train.jsonl"
            exists = os.path.exists(path)
            all_passed &= check(f"{path}", exists)
            if exists:
                with open(path) as f:
                    lines = sum(1 for _ in f)
                all_passed &= check(f"  {cluster}: >500 examples", lines > 500, f"{lines}")

        # 3. DPO JSONL files with correct format
        print("\n3. DPO preference pairs (format check)")
        for cluster in ["conservative", "mainstream", "sensitive"]:
            path = f"data/preference/dpo_{cluster}_train.jsonl"
            exists = os.path.exists(path)
            all_passed &= check(f"{path}", exists)
            if exists:
                with open(path) as f:
                    first = json.loads(f.readline())
                has_messages_dict = (
                    isinstance(first.get("input"), dict)
                    and "messages" in first["input"]
                )
                all_passed &= check(
                    f"  {cluster}: input.messages format",
                    has_messages_dict,
                )

        # 4. Test/Train split
        print("\n4. Data splits")
        for f in ["data/splits/test_set.json", "data/splits/train_set.json"]:
            all_passed &= check(f"{f}", os.path.exists(f))

        # 4b. Per-language splits
        print("\n4b. Per-language data splits")
        for lang in ["en", "es"]:
            for name in ["test_set", "train_set", "cluster_meta"]:
                f = f"data/splits/{lang}/{name}.json"
                all_passed &= check(f"{f}", os.path.exists(f))

    # 5. Saved model IDs (all 12 experiments + legacy GRPO iter1).
    # Not distributed: the IDs embed the fine-tuning account name.
    print("\n5. Model IDs")
    if not os.path.isdir("artifacts/model_ids"):
        warn("artifacts/model_ids not present",
             "skipped (not distributed with this repository)")
    model_files = [] if not os.path.isdir("artifacts/model_ids") else [
        "artifacts/model_ids/sft_model_ids.json",
        "artifacts/model_ids/sft_41mini_model_ids.json",
        "artifacts/model_ids/sft_pure_41mini_model_ids.json",
        "artifacts/model_ids/dpo_41mini_model_ids.json",
        "artifacts/model_ids/dpo_bb03_41mini_model_ids.json",
        "artifacts/model_ids/dpo_bb05_41mini_model_ids.json",
        "artifacts/model_ids/marspo_41mini_model_ids.json",
        "artifacts/model_ids/grpo_iter1_model_ids.json",
        "artifacts/model_ids/grpo_41mini_alpha00_model_ids.json",
        "artifacts/model_ids/grpo_41mini_alpha02_model_ids.json",
        "artifacts/model_ids/grpo_41mini_alpha05_model_ids.json",
        "artifacts/model_ids/grpo_41mini_alpha10_model_ids.json",
    ]
    for f in model_files:
        exists = os.path.exists(f)
        all_passed &= check(f"{f}", exists)
        if exists:
            with open(f) as fh:
                ids = json.load(fh)
            all_passed &= check(f"  {os.path.basename(f)}: 3 IDs", len(ids) == 3, f"got {len(ids)}")

    # 6. Evaluation results (all 12 experiments + baselines + stats)
    print("\n6. Evaluation results")
    result_files = [
        "artifacts/results/baseline_results.json",
        "artifacts/results/sft_results.json",
        "artifacts/results/grpo_results.json",
        "artifacts/results/sft_41mini_results.json",
        "artifacts/results/sft_pure_41mini_results.json",
        "artifacts/results/dpo_41mini_results.json",
        "artifacts/results/dpo_bb03_41mini_results.json",
        "artifacts/results/dpo_bb05_41mini_results.json",
        "artifacts/results/marspo_41mini_results.json",
        "artifacts/results/grpo_41mini_alpha00_results.json",
        "artifacts/results/grpo_41mini_alpha02_results.json",
        "artifacts/results/grpo_41mini_alpha05_results.json",
        "artifacts/results/grpo_41mini_alpha10_results.json",
        "artifacts/results/confidence_intervals.json",
        "artifacts/results/significance_tests.json",
    ]
    for f in result_files:
        exists = os.path.exists(f)
        all_passed &= check(f"{f}", exists)
        if exists:
            try:
                with open(f) as fh:
                    json.load(fh)
            except Exception as e:
                all_passed &= check(f"  {os.path.basename(f)}: well-formed JSON", False, str(e))

    # 6b. Every non-baseline result reports the core metrics the paper cites
    print("\n6b. Result metric coverage")
    required_keys = ["mva", "f1_macro", "disagreement_rate"]
    fine_tuned = [
        "artifacts/results/sft_41mini_results.json",
        "artifacts/results/sft_pure_41mini_results.json",
        "artifacts/results/dpo_41mini_results.json",
        "artifacts/results/dpo_bb03_41mini_results.json",
        "artifacts/results/dpo_bb05_41mini_results.json",
        "artifacts/results/marspo_41mini_results.json",
        "artifacts/results/grpo_41mini_alpha00_results.json",
        "artifacts/results/grpo_41mini_alpha02_results.json",
        "artifacts/results/grpo_41mini_alpha05_results.json",
        "artifacts/results/grpo_41mini_alpha10_results.json",
    ]
    for f in fine_tuned:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            d = json.load(fh)
        missing = [k for k in required_keys if k not in d]
        all_passed &= check(
            f"  {os.path.basename(f)}: mva/f1_macro/disagreement_rate",
            not missing,
            f"missing {missing}" if missing else "",
        )

    # 6c. Paper figures (only in a checkout that carries the paper)
    print("\n6c. Paper figures")
    if os.path.isdir("paper"):
        for fig in ["team_f1", "cluster_f1", "cluster_vs_team",
                    "balance_scatter", "cluster_f1_breakdown",
                    "cluster_coverage", "yes_rates", "agreement",
                    "alpha_sweep", "beta_sweep",
                    "crosslingual_team_f1", "behavioral_fidelity",
                    "subset_comparison"]:
            p = f"paper/figures/{fig}.pdf"
            all_passed &= check(p, os.path.exists(p))
    else:
        warn("paper/ not present", "skipped (not distributed with this repository)")

    # 6d. Bilingual / local experiment results
    print("\n6d. Bilingual + local results")
    local_gpt_results = []
    for lang in ["en", "es"]:
        for stage in ["base", "sft", "dpo", "grpo_a20"]:
            local_gpt_results.append(f"artifacts/results/local_qwen3_{stage}_{lang}_results.json")
    local_gpt_results.append("artifacts/results/local_qwen3_sft_balanced_en_results.json")
    for stage in ["sft", "dpo", "grpo_a20"]:
        local_gpt_results.append(f"artifacts/results/gpt41mini_{stage}_es_results.json")
    # Same-base baselines on gpt-4.1-mini (paper tables)
    local_gpt_results.append("artifacts/results/base_41mini_results.json")
    local_gpt_results.append("artifacts/results/persona_41mini_results.json")
    local_gpt_results.append("artifacts/results/gpt41mini_base_es_results.json")
    analysis_results = [
        "artifacts/results/behavioral_fidelity_en.json",
        "artifacts/results/behavioral_fidelity_es.json",
        "artifacts/results/clustering_ablation_en.json",
        "artifacts/results/clustering_ablation_es.json",
    ]
    required_metrics = ["mva", "f1_macro", "balanced_accuracy", "disagreement_rate"]
    for f in local_gpt_results:
        exists = os.path.exists(f)
        all_passed &= check(f"{f}", exists)
        if not exists:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception as e:
            all_passed &= check(f"  {os.path.basename(f)}: well-formed JSON", False, str(e))
            continue
        missing = [k for k in required_metrics if k not in d]
        all_passed &= check(
            f"  {os.path.basename(f)}: mva/f1_macro/balanced_accuracy/disagreement_rate",
            not missing,
            f"missing {missing}" if missing else "",
        )
    for f in analysis_results:
        exists = os.path.exists(f)
        all_passed &= check(f"{f}", exists)
        if exists:
            try:
                with open(f) as fh:
                    json.load(fh)
            except Exception as e:
                all_passed &= check(f"  {os.path.basename(f)}: well-formed JSON", False, str(e))

    # 6e. Synthetic-data gate
    print("\n6e. Synthetic-data gate")
    tainted = []
    for name in sorted(os.listdir("artifacts/results")):
        path = os.path.join("artifacts/results", name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                content = fh.read()
        except Exception:
            continue
        if '"synthetic": true' in content:
            tainted.append(name)
    all_passed &= check(
        "no synthetic results in artifacts/results/",
        not tainted,
        f"tainted: {tainted}" if tainted else "",
    )
    if os.path.exists("paper/main.tex"):
        with open("paper/main.tex") as fh:
            tex = fh.read()
        all_passed &= check("paper/main.tex has no [P] placeholders", "[P]" not in tex)
        all_passed &= check(
            "paper/main.tex does not enable \\synthetictrue",
            "\\synthetictrue" not in tex,
        )

    # 7. LaTeX paper (only in a checkout that carries the paper)
    print("\n7. Paper")
    if os.path.isdir("paper"):
        all_passed &= check("paper/main.pdf exists", os.path.exists("paper/main.pdf"))
        all_passed &= check("paper/main.tex exists", os.path.exists("paper/main.tex"))
    else:
        warn("paper/ not present", "skipped (not distributed with this repository)")

    # 8. Sensitive files excluded from git
    print("\n8. Git safety")
    result = subprocess.run(
        ["git", "ls-files", "--cached", ".gitignore"],
        capture_output=True, text=True,
    )
    tracked = result.stdout.strip()
    all_passed &= check(
        ".gitignore not tracked",
        ".gitignore" not in tracked,
    )

    # 9. No API keys in git history
    print("\n9. API key safety")
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--format=%s%n%b"],
            capture_output=True, text=True, timeout=30,
        )
        no_keys = "sk-proj" not in result.stdout and "sk-" not in result.stdout[:5000]
        all_passed &= check("no API keys in commit messages", no_keys)
    except Exception as e:
        warn("API key check skipped", str(e))

    # 10. Git identity
    print("\n10. Git identity (last commit)")
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%an <%ae>"],
            capture_output=True, text=True,
        )
        author = result.stdout.strip()
        all_passed &= check(
            "author = mohammadi-hadi",
            "mohammadi-hadi" in author,
            author,
        )
    except Exception as e:
        warn("Author check skipped", str(e))

    # 11. Required Python imports
    print("\n11. Python modules")
    try:
        import config  # noqa
        all_passed &= check("config.py imports", True)
        from models.agent import Agent, MultiAgentSystem  # noqa
        all_passed &= check("models.agent imports", True)
        from evaluation.metrics import compute_all_metrics  # noqa
        all_passed &= check("evaluation.metrics imports", True)
    except Exception as e:
        all_passed &= check("module imports", False, str(e))

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{OK} ALL CHECKS PASSED — ready to commit and push")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"{FAIL} ONE OR MORE CHECKS FAILED — fix before committing")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
