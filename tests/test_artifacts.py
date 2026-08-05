"""Guards on the committed artifacts.

The EXIST 2024 data may not be redistributed: the per-text prediction files
must stay text IDs, votes and probabilities — nothing else. These tests keep
that invariant enforced forever, and check that the artifacts stay anonymous
and internally consistent.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = sorted((ROOT / "artifacts" / "predictions").glob("*.json"))
ARTIFACT_JSONS = sorted((ROOT / "artifacts").rglob("*.json"))

ALLOWED_KEYS = {"text_id", "agent_predictions", "agent_p_yes", "team_vote", "unanimous"}
CLUSTER_KEYS = {"cluster1", "cluster2", "cluster3",
                # legacy display names in the oldest runs, normalized at load time
                "Conservative", "Mainstream", "Sensitive"}
LABELS = {"YES", "NO"}


def test_predictions_exist():
    assert len(PREDICTIONS) >= 20


def test_predictions_contain_no_dataset_text():
    for path in PREDICTIONS:
        records = json.loads(path.read_text())
        assert isinstance(records, list) and records, path.name
        for rec in records:
            extra = set(rec) - ALLOWED_KEYS
            assert not extra, f"{path.name}: unexpected fields {extra}"
            tid = rec["text_id"]
            assert isinstance(tid, str) and len(tid) <= 32 and " " not in tid, (
                f"{path.name}: text_id does not look like an ID: {tid!r}")
            assert set(rec["agent_predictions"]) <= CLUSTER_KEYS, path.name
            for v in rec["agent_predictions"].values():
                assert v in LABELS, f"{path.name}: {v!r}"
            assert rec["team_vote"] in LABELS, path.name
            assert isinstance(rec["unanimous"], bool), path.name
            for v in rec.get("agent_p_yes", {}).values():
                assert 0.0 <= v <= 1.0, path.name


def test_artifacts_parse_and_stay_anonymous():
    assert ARTIFACT_JSONS
    for path in ARTIFACT_JSONS:
        text = path.read_text().lower()
        json.loads(text)  # every artifact is valid JSON
        for marker in ("utrecht", "mohammadi"):
            assert marker not in text, f"{path} leaks '{marker}'"


def test_cluster_summaries_ordered_by_yes_rate():
    for lang in ("en", "es"):
        summary = json.loads(
            (ROOT / "artifacts" / "clustering" / lang / "cluster_summary.json").read_text())
        rates = [summary["features"][c]["yes_rate"]["mean"]
                 for c in ("cluster1", "cluster2", "cluster3")]
        assert rates == sorted(rates), f"{lang}: clusters not ordered by YES rate {rates}"


def test_backfill_holds_aggregates_only():
    text = (ROOT / "artifacts" / "results" / "en_core_backfill.json").read_text()
    assert "text_id" not in text
    data = json.loads(text)
    for key, entry in data.items():
        core = entry["core"]
        assert core["n_gold"] > 0, key
        assert 0.0 <= core["f1_macro"] <= 1.0, key
