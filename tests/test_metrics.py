"""The homegrown metrics against scikit-learn on random labelings."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from evaluation.evaluate_local import binary_metrics, wilson_ci
from evaluation.metrics import (
    cluster_alignment_accuracy,
    disagreement_rate,
    f1_macro,
    majority_vote_accuracy,
)


def random_labels(seed, n=400, p_none=0.0):
    rng = np.random.RandomState(seed)
    golds = [None if rng.rand() < p_none else ("YES" if rng.rand() < 0.35 else "NO")
             for _ in range(n)]
    preds = ["YES" if rng.rand() < 0.4 else "NO" for _ in range(n)]
    # make sure both classes appear on both sides
    golds[:4] = ["YES", "NO", "YES", "NO"]
    preds[:4] = ["NO", "YES", "YES", "NO"]
    return golds, preds


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_f1_macro_matches_sklearn(seed):
    golds, preds = random_labels(seed, p_none=0.1)
    out = f1_macro(preds, golds)
    kept = [(g, p) for g, p in zip(golds, preds) if g is not None]
    ref = f1_score([g for g, _ in kept], [p for _, p in kept],
                   labels=["YES", "NO"], average="macro")
    assert out["f1_macro"] == pytest.approx(ref, abs=2e-4)
    ref_yes = f1_score([g for g, _ in kept], [p for _, p in kept], pos_label="YES")
    assert out["f1_yes"] == pytest.approx(ref_yes, abs=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_binary_metrics_match_sklearn(seed):
    golds, preds = random_labels(seed)
    out = binary_metrics(golds, preds)
    assert out["n"] == len(golds)
    assert out["accuracy"] == pytest.approx(accuracy_score(golds, preds), abs=1e-4)
    assert out["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(golds, preds), abs=1e-4)
    assert out["f1_macro"] == pytest.approx(
        f1_score(golds, preds, labels=["YES", "NO"], average="macro"), abs=2e-4)


def test_binary_metrics_empty_is_none():
    assert binary_metrics([], []) is None


@pytest.mark.parametrize("seed", [0, 1])
def test_majority_vote_accuracy_matches_sklearn(seed):
    golds, preds = random_labels(seed, p_none=0.2)
    out = majority_vote_accuracy(preds, golds)
    kept = [(g, p) for g, p in zip(golds, preds) if g is not None]
    ref = accuracy_score([g for g, _ in kept], [p for _, p in kept])
    assert out == pytest.approx(ref, abs=1e-9)


def test_disagreement_rate_counts_split_votes():
    preds = [
        {"cluster1": "YES", "cluster2": "YES", "cluster3": "YES"},
        {"cluster1": "YES", "cluster2": "NO", "cluster3": "YES"},
        {"cluster1": "NO", "cluster2": "NO", "cluster3": "NO"},
        {"cluster1": "NO", "cluster2": "YES", "cluster3": "YES"},
    ]
    assert disagreement_rate(preds) == pytest.approx(0.5)
    assert disagreement_rate([]) == 0.0


def test_cluster_alignment_skips_ties_and_missing():
    votes = [
        {"cluster1": {"majority": "YES"}},
        {"cluster1": {"majority": None}},   # tied cluster: skipped
        {},                                  # cluster absent: skipped
        {"cluster1": {"majority": "NO"}},
    ]
    preds = ["YES", "YES", "YES", "YES"]
    assert cluster_alignment_accuracy(preds, votes, "cluster1") == pytest.approx(0.5)
    assert cluster_alignment_accuracy([], [], "cluster1") is None


def test_wilson_ci_properties():
    lo, hi = wilson_ci(0.5, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # interval tightens with n
    lo2, hi2 = wilson_ci(0.5, 1000)
    assert hi2 - lo2 < hi - lo
    # never leaves [0, 1] at the extremes
    assert wilson_ci(1.0, 20)[1] <= 1.0
    assert wilson_ci(0.0, 20)[0] >= 0.0
    assert wilson_ci(0.5, 0) == (0.0, 0.0)


def test_wilson_ci_hand_formula():
    # independent transcription of the Wilson score interval, z = 1.96
    p, n, z = 0.3, 298, 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    lo, hi = wilson_ci(p, n)
    assert lo == pytest.approx(centre - half, abs=1e-12)
    assert hi == pytest.approx(centre + half, abs=1e-12)
