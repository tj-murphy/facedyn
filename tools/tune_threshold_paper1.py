"""Tune ``selection_threshold`` on Paper 1's real training set.

Answers the question the shipped default has never had a clean answer to:
what does the threshold sweep look like when the choice is made on data
held out of *training*, rather than on the held-out test set?

Runs on R's own step-7 output (`R Validation Data/`, gitignored), so no
part of steps 1-7 is recomputed and the split is R's real one -- the same
370 training videos that produced the published numbers.

    python tools/tune_threshold_paper1.py [--fast] [--n-repeats 3] [--jobs -1]

At the defaults this is a long run: three 20-seed Boruta fits on ~296x108
plus one refit on 370x108, of the order of an hour. ``--fast`` swaps in
``importance="gini"`` and 8 runs for a first look at the shape of the
curve, and is *not* R's importance measure.

**The test set is loaded only after the threshold has been chosen**, and
only to report what the choice was worth. Nothing about the curve depends
on it -- that is the entire point of the exercise. See
`facedyn/tuning.py`'s module docstring.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from facedyn.classifiers import make_classifier
from facedyn.evaluation import classification_metrics
from facedyn.feature_selection import BorutaSelector
from facedyn.splitting import pair_group_report, pair_groups_from_filenames
from facedyn.tuning import plot_threshold_sweep, tune_selection_threshold

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "R Validation Data"
TRAIN_CSV = DATA / "r_cmfts_output_imputed_zerosd.csv"
TEST_CSV = DATA / "r_cmfts_output_imputed_zerosd_test.csv"
METADATA = ["video_filename", "isfakeorreal", "emotion", "valence"]
POSITIVE = "fake"


def load(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. This script needs R's real step-7 output, which "
            "is gitignored -- see EXTERNAL_FILE_REFERENCES.txt."
        )
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in METADATA]
    return df, df["isfakeorreal"].to_numpy(), features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="sub-train/validation splits, one Boruta fit each")
    parser.add_argument("--boruta-repeats", type=int, default=20,
                        help="seeds per Boruta fit")
    parser.add_argument("--jobs", type=int, default=-1, help="n_jobs inside each fit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true",
                        help="gini importance and 8 runs: a quick look, not R's measure")
    parser.add_argument("--out", type=Path, default=REPO / "tools" / "threshold_tuning")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    train, y_train, features = load(TRAIN_CSV)
    groups = pair_groups_from_filenames(train)
    report = pair_group_report(train, groups)
    print(f"train: {train.shape[0]} videos, {len(features)} features")
    print(report if isinstance(report, str) else report.to_string())

    selector = BorutaSelector(
        feature_columns=features,
        n_repeats=8 if args.fast else args.boruta_repeats,
        importance="gini" if args.fast else "permutation",
        random_state=args.seed,
    )
    # The scoring model is left to the tuner's own default (a forest at
    # scikit-learn's `max_features`), not Paper 1's `mtry=4`: a fixed mtry
    # is not a fair comparator across feature sets of size 1 to 108. The
    # held-out confirmations further down do use the paper's preset, since
    # those numbers are meant to be read against the published ones.
    started = time.time()
    result = tune_selection_threshold(
        train, y_train, groups,
        selector=selector,
        positive_label=POSITIVE,
        n_repeats=args.n_repeats,
        random_state=args.seed,
        n_jobs=args.jobs,
    )
    elapsed = time.time() - started

    print(f"\ntuning finished in {elapsed / 60:.1f} min")
    print(result.curve_.to_string(index=False))
    print(f"\nrule={result.rule_}  chosen threshold={result.best_threshold_:g}")
    print(f"validation {result.scoring_}={result.best_score_:.4f} "
          f"(selection score, NOT a generalisation estimate)")
    print(f"features kept on all {len(train)}: {len(result.selected_columns_)}")
    for column in result.selected_columns_[:25]:
        print(f"  {column}")
    if len(result.selected_columns_) > 25:
        print(f"  ... and {len(result.selected_columns_) - 25} more (see summary.json)")

    result.curve_.to_csv(args.out / "curve.csv", index=False)
    result.per_split_.drop(columns="features").to_csv(
        args.out / "per_split.csv", index=False
    )
    plot_threshold_sweep(
        result, show_splits=True, save_path="threshold_sweep.png", output_dir=args.out
    )

    # ---- everything below touches the held-out test set, once -----------
    # Confirmation only. It played no part in the choice above, and if it
    # had, the number it produces would mean nothing.
    test, y_test, _ = load(TEST_CSV)
    chosen = result.selected_columns_
    summary = {
        "n_repeats": args.n_repeats,
        "boruta_repeats": selector.n_repeats,
        "importance": selector.importance,
        "seed": args.seed,
        "elapsed_min": round(elapsed / 60, 1),
        "best_threshold": result.best_threshold_,
        "validation_score": result.best_score_,
        "n_features": len(chosen),
        "features": chosen,
        "held_out": {},
    }

    print("\n--- held-out test set (n=%d), confirmation only ---" % len(test))
    fitted = make_classifier("random_forest", random_state=args.seed)
    fitted.fit(train[chosen], y_train)
    scores = fitted.predict_proba(test[chosen])[:, list(fitted.classes_).index(POSITIVE)]
    metrics = classification_metrics(
        y_test, fitted.predict(test[chosen]), scores, positive_label=POSITIVE
    )
    summary["held_out"]["tuned"] = {
        "threshold": result.best_threshold_,
        "n_features": len(chosen),
        "roc_auc": float(metrics["roc_auc"]),
        "accuracy": float(metrics["accuracy"]),
    }
    print(f"tuned threshold {result.best_threshold_:g}: "
          f"{len(chosen)} features, ROC-AUC {metrics['roc_auc']:.4f}, "
          f"accuracy {metrics['accuracy']:.4f}")

    # The same comparison for the thresholds already on record, so the
    # tuned choice can be read against them rather than in isolation.
    for reference in (0.8, 0.5, 0.25):
        columns = [
            f for f in result.selector_.feature_columns_
            if result.selector_.selection_frequency_[f] >= reference
        ]
        if not columns:
            print(f"reference threshold {reference:g}: selects nothing")
            summary["held_out"][str(reference)] = {"n_features": 0}
            continue
        other = make_classifier("random_forest", random_state=args.seed)
        other.fit(train[columns], y_train)
        other_scores = other.predict_proba(test[columns])[
            :, list(other.classes_).index(POSITIVE)
        ]
        other_metrics = classification_metrics(
            y_test, other.predict(test[columns]), other_scores, positive_label=POSITIVE
        )
        summary["held_out"][str(reference)] = {
            "n_features": len(columns),
            "roc_auc": float(other_metrics["roc_auc"]),
            "accuracy": float(other_metrics["accuracy"]),
        }
        print(f"reference threshold {reference:g}: {len(columns)} features, "
              f"ROC-AUC {other_metrics['roc_auc']:.4f}, "
              f"accuracy {other_metrics['accuracy']:.4f}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
