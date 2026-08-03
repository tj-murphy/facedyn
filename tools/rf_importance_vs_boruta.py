"""Is Boruta's selection more stable than just ranking by RF importance?

The obvious objection to the whole feature-selection stage: a random
forest already reports feature importance, so why not rank by that and
take the top *k*? This measures the answer instead of asserting it.

The comparison is deliberately apples-to-apples. Both methods get the
**same** importance estimator -- `oob_permutation_importance`, the one R's
`Boruta` actually calls -- and the same 20 seeds. The only difference is
the decision rule:

- **Boruta** tests each feature against permuted shadow copies of itself
  and returns a confirmed/rejected verdict, so it can return *nothing*.
- **RF top-k** sorts the same importances and keeps the first *k*, so it
  always returns exactly *k* features whether or not any of them beat
  noise.

For each, we count how often each feature is picked across the 20 seeds.
That number is directly comparable: "fraction of seeded runs that chose
this feature".

    python tools/rf_importance_vs_boruta.py

Writes `tools/threshold_tuning/rf_vs_boruta.csv` (per-feature frequencies)
and `rf_vs_boruta_summary.json`. Reads Boruta's side from the notebook's
joblib cache, so it is a cache hit rather than another 15-minute fit.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.ensemble import RandomForestClassifier

from facedyn.evaluation import delong_roc_test, roc_auc_delong_ci
from facedyn.feature_selection import oob_permutation_importance

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
OUT = REPO / "tools" / "threshold_tuning"
METADATA = ["video_filename", "isfakeorreal", "emotion", "valence"]

# Matching `cmfts_replication.ipynb` exactly, so the Boruta side is a cache hit.
RANDOM_STATE = 12345
SELECTION_THRESHOLD = 0.5
N_SEEDS = 20

PAPER_8 = [
    "diff1x_pacf5_AU17_chin_raiser",
    "diff1x_pacf5_AU01_inner_brow_raiser",
    "diff2_acf1_AU01_inner_brow_raiser",
    "diff2_acf10_AU12_lip_corner_puller",
    "max_kl_shift_AU12_lip_corner_puller",
    "diff1_acf1_AU12_lip_corner_puller",
    "lumpiness_AU12_lip_corner_puller",
    "diff2_acf1_AU12_lip_corner_puller",
]


def main() -> None:
    sys.path.insert(0, str(NOTEBOOKS))
    from _cache_utils import fit_boruta_selector  # noqa: E402

    cmfts = pd.read_csv(REPO / "R Validation Data" / "r_cmfts_output_imputed_zerosd.csv")
    test = pd.read_csv(REPO / "R Validation Data" / "r_cmfts_output_imputed_zerosd_test.csv")
    features = [c for c in cmfts.columns if c not in METADATA]
    # 1 = real, matching the notebook this feeds.
    y = (cmfts["isfakeorreal"] == "real").astype(int).to_numpy()
    y_test = (test["isfakeorreal"] == "real").astype(int).to_numpy()

    memory = Memory(location=str(NOTEBOOKS / ".joblib_cache"), verbose=0)
    cached = memory.cache(fit_boruta_selector)
    selector = cached(
        cmfts, features, cmfts["isfakeorreal"], RANDOM_STATE, 20, SELECTION_THRESHOLD
    )
    boruta_freq = selector.selection_frequency_
    boruta_set = selector.selected_columns_
    print(f"Boruta at >= {SELECTION_THRESHOLD}: {len(boruta_set)} features")

    # --- the RF-importance side, same estimator, same number of seeds ----
    X = cmfts[features].to_numpy(dtype=float)
    started = time.time()
    rankings = []
    for seed in range(N_SEEDS):
        importance = oob_permutation_importance(X, y, random_state=seed, n_jobs=6)
        rankings.append(np.argsort(importance)[::-1])
        print(f"  seed {seed:2d}/{N_SEEDS} done ({time.time() - started:.0f}s)", flush=True)

    def top_k_frequency(k: int) -> dict[str, float]:
        counts = {f: 0 for f in features}
        for order in rankings:
            for idx in order[:k]:
                counts[features[idx]] += 1
        return {f: c / N_SEEDS for f, c in counts.items()}

    freq_top8 = top_k_frequency(8)
    # Size-matched to Boruta, so the stability comparison is not just a
    # comparison of how many features each method was asked for.
    freq_matched = top_k_frequency(max(1, len(boruta_set)))

    frame = pd.DataFrame({
        "feature": features,
        "boruta_freq": [boruta_freq[f] for f in features],
        "rf_top8_freq": [freq_top8[f] for f in features],
        f"rf_top{max(1, len(boruta_set))}_freq": [freq_matched[f] for f in features],
        "in_paper_8": [f in PAPER_8 for f in features],
    }).sort_values("boruta_freq", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "rf_vs_boruta.csv", index=False)

    # --- held-out scoring, averaged over model seeds --------------------
    def probs(cols, seeds=(0, 1, 2, 3, 4)):
        return np.mean([
            RandomForestClassifier(500, random_state=s, n_jobs=6)
            .fit(cmfts[cols], y).predict_proba(test[cols])[:, 1] for s in seeds
        ], axis=0)

    rf_top8_stable = [f for f, v in freq_top8.items() if v >= 0.5]
    rf_top8_seed0 = [features[i] for i in rankings[0][:8]]
    candidates = {
        "boruta >= 0.5": boruta_set,
        "RF top-8, majority of seeds": rf_top8_stable,
        "RF top-8, one seed": rf_top8_seed0,
        "Paper 1's published 8": PAPER_8,
        "all 108": features,
    }
    scored = {name: probs(cols) for name, cols in candidates.items() if cols}

    rows = []
    for name, cols in candidates.items():
        if not cols:
            continue
        auc, lo, hi = roc_auc_delong_ci(y_test, scored[name])
        rows.append({
            "feature set": name,
            "n": len(cols),
            "test ROC-AUC": round(auc, 3),
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
            "overlap with Paper 8": len(set(cols) & set(PAPER_8)),
        })
    held_out = pd.DataFrame(rows)
    held_out.to_csv(OUT / "rf_vs_boruta_heldout.csv", index=False)

    def stable_count(freqs, level):
        return sum(1 for v in freqs.values() if v >= level)

    summary = {
        "n_seeds": N_SEEDS,
        "boruta": {
            "n_selected": len(boruta_set),
            "features_at_1.0": stable_count(boruta_freq, 1.0),
            "features_at_0.8": stable_count(boruta_freq, 0.8),
            "features_at_0.5": stable_count(boruta_freq, 0.5),
            "features_ever_picked": stable_count(boruta_freq, 1 / N_SEEDS),
        },
        "rf_top8": {
            "features_at_1.0": stable_count(freq_top8, 1.0),
            "features_at_0.8": stable_count(freq_top8, 0.8),
            "features_at_0.5": stable_count(freq_top8, 0.5),
            "features_ever_picked": stable_count(freq_top8, 1 / N_SEEDS),
        },
        "elapsed_min": round((time.time() - started) / 60, 1),
    }
    if "boruta >= 0.5" in scored and "RF top-8, majority of seeds" in scored:
        t = delong_roc_test(
            y_test, scored["boruta >= 0.5"], scored["RF top-8, majority of seeds"]
        )
        summary["delong_boruta_vs_rf_top8"] = {
            "difference": round(float(t["difference"]), 3),
            "p_value": round(float(t["p_value"]), 3),
        }

    (OUT / "rf_vs_boruta_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- how many features are picked how often (of 108) ---")
    print(json.dumps({k: summary[k] for k in ("boruta", "rf_top8")}, indent=2))
    print("\n--- held out ---")
    print(held_out.to_string(index=False))
    if "delong_boruta_vs_rf_top8" in summary:
        print("\nboruta vs RF top-8 (paired DeLong):",
              summary["delong_boruta_vs_rf_top8"])
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
