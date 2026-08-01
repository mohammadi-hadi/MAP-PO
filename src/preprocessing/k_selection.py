#!/usr/bin/env python3
"""
Cluster-count selection diagnostics
===================================
Silhouette alone picks k=2 on Spanish, but silhouette scores compactness in the
full standardized feature space, where agreement rate and label entropy dominate.
The multi-agent pipeline needs clusters that separate on the *YES-rate* axis,
because that is the behavior each agent has to reproduce.

For each k this reports, per language:
  silhouette, minimum cluster share,
  YES-rate MAD  -- mean |annotator YES rate - own cluster mean YES rate|,
                   i.e. how well one agent per cluster can represent its members,
  min pairwise gap between cluster mean YES rates (behavioral separation),
  per-cluster size, mean YES rate, and observed YES-rate span.

Run from the repo root:
    .venv-mlx/bin/python src/preprocessing/k_selection.py            # both languages
    .venv-mlx/bin/python src/preprocessing/k_selection.py --lang es

Writes artifacts/clustering/<lang>/k_selection.json (the honest scan in
kmeans_scan.json is left untouched).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FEATURES = ["yes_rate", "agreement_rate", "label_entropy"]
K_RANGE = range(2, 7)


def diagnostics(df, k):
    """Clustering diagnostics at one k, ordered by ascending mean YES rate."""
    X = StandardScaler().fit_transform(df[FEATURES].values)
    labels = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=20).fit_predict(X)
    order = np.argsort([df.yes_rate[labels == c].mean() for c in range(k)])

    clusters, abs_dev = [], 0.0
    for rank, c in enumerate(order, start=1):
        yes = df.yes_rate[labels == c]
        abs_dev += float(np.abs(yes - yes.mean()).sum())
        clusters.append({
            "cluster": f"cluster{rank}",
            "size": int(len(yes)),
            "share": round(len(yes) / len(df), 4),
            "mean_yes_rate": round(float(yes.mean()), 4),
            "yes_rate_min": round(float(yes.min()), 4),
            "yes_rate_max": round(float(yes.max()), 4),
            "yes_rate_span": round(float(yes.max() - yes.min()), 4),
        })

    means = [c["mean_yes_rate"] for c in clusters]
    gaps = [round(means[i + 1] - means[i], 4) for i in range(len(means) - 1)]
    return {
        "k": k,
        "silhouette": round(float(silhouette_score(X, labels)), 4),
        "min_cluster_share": round(min(c["share"] for c in clusters), 4),
        "yes_rate_mad": round(abs_dev / len(df), 4),
        "min_pairwise_yes_gap": min(gaps),
        "clusters": clusters,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=config.LANGUAGES, default=None)
    args = ap.parse_args()

    for lang in ([args.lang] if args.lang else config.LANGUAGES):
        df = pd.read_csv(config.DATA_DIR / f"behavioral_features_{lang}.csv")
        scan = [diagnostics(df, k) for k in K_RANGE]

        out_dir = config.ARTIFACTS_DIR / "clustering" / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "k_selection.json"
        with open(out_path, "w") as f:
            json.dump({"language": lang, "n_annotators": int(len(df)),
                       "features": FEATURES, "scan": scan}, f, indent=2)

        print(f"\n{lang.upper()}  (n={len(df)})  -> {out_path}")
        for row in scan:
            print(f"  k={row['k']}  silhouette={row['silhouette']:.3f}"
                  f"  min share={row['min_cluster_share']:.2f}"
                  f"  YES-rate MAD={row['yes_rate_mad']:.4f}"
                  f"  min gap={row['min_pairwise_yes_gap']:.3f}")


if __name__ == "__main__":
    main()
