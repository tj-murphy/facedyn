"""Unit tests for the Boruta selection-threshold tuner.

The tuner has no R counterpart to validate against -- R ran Boruta once at
a fixed seed and never tuned a threshold -- so what is tested here is its
structure: that the candidate grid is the one the frequencies can actually
take, that tighter thresholds keep a subset of what looser ones keep, that
the selection rules do what they claim on a curve whose answer is known by
hand, and that the sub-train/validation splits never break a matched pair.

Everything runs on small synthetic data with `importance="gini"` and few
trees, so the file takes seconds rather than the tens of minutes a real
tuning run costs.
"""

import numpy as np
import pandas as pd
import pytest

from facedyn.classifiers import make_classifier
from facedyn.feature_selection import BorutaSelector
from facedyn.splitting import pair_groups
from facedyn.tuning import (
    ThresholdTuningResult,
    _inner_splits,
    _select_threshold,
    threshold_grid,
    threshold_sweep,
    tune_selection_threshold,
)

matplotlib = pytest.importorskip("matplotlib", reason="plots need the viz extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from facedyn.tuning import plot_threshold_sweep  # noqa: E402

#: 28 s: the module fixtures fit Boruta once per inner split. Skipped unless `pytest --runslow`; always run in CI.
pytestmark = pytest.mark.slow


FEATURES = ["f_strong", "f_weak", "f_pair", "noise_0", "noise_1", "noise_2"]


def make_paired_dataset(n_pairs: int = 60, seed: int = 0):
    """Matched-pair video-level data with features of graded relevance.

    The point of a *threshold* sweep is that features differ in how often
    Boruta confirms them, so the columns here are built to spread out:

    - ``f_strong`` separates the classes cleanly and should be confirmed in
      every run;
    - ``f_weak`` carries a signal small enough that runs disagree, which is
      what makes an intermediate threshold behave differently from the
      extremes;
    - ``f_pair`` is a high-variance value **shared by both members of a
      pair** with no class information -- the stand-in for what a deepfake
      inherits from its source video, and the reason the splits have to be
      group-aware;
    - three pure-noise columns that nothing should confirm.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        identity = rng.normal(scale=4)
        for label in ("real", "fake"):
            sign = 1.0 if label == "fake" else -1.0
            rows.append({
                "video_filename": f"{label}_{i}",
                "corresponding_video": f"{'fake' if label == 'real' else 'real'}_{i}",
                "isfakeorreal": label,
                "f_strong": sign * 0.8 + rng.normal(scale=1.0),
                "f_weak": sign * 0.15 + rng.normal(scale=1.0),
                "f_pair": identity + rng.normal(scale=0.05),
                "noise_0": rng.normal(),
                "noise_1": rng.normal(),
                "noise_2": rng.normal(),
            })
    df = pd.DataFrame(rows)
    return df, df["isfakeorreal"].to_numpy(), pair_groups(df)


def fast_selector(n_repeats: int = 8, random_state: int = 0) -> BorutaSelector:
    """A Boruta configuration cheap enough for a unit test."""
    return BorutaSelector(
        feature_columns=FEATURES,
        n_repeats=n_repeats,
        importance="gini",
        n_estimators=30,
        max_iter=10,
        correlation_threshold=None,
        random_state=random_state,
    )


def fast_model():
    return make_classifier("random_forest", n_estimators=30, random_state=0)


# --- candidate grid -------------------------------------------------------

def test_threshold_grid_is_the_values_frequencies_can_take():
    # With 4 runs a feature is confirmed in 0, 1, 2, 3 or 4 of them, so
    # quarters are the only thresholds that change anything.
    assert threshold_grid(4).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(threshold_grid(20)) == 21
    assert threshold_grid(20)[0] == 0.0
    assert threshold_grid(20)[-1] == 1.0


def test_threshold_grid_rejects_zero_runs():
    with pytest.raises(ValueError, match="n_boruta_repeats"):
        threshold_grid(0)


# --- the selection rules --------------------------------------------------

def hand_built_curve(scores, ses, thresholds=None) -> pd.DataFrame:
    """An aggregated curve with known contents, no Boruta involved."""
    thresholds = thresholds if thresholds is not None else np.linspace(0, 1, len(scores))
    return pd.DataFrame({
        "threshold": thresholds,
        "n_splits_scored": 3,
        "n_features_mean": np.arange(len(scores))[::-1].astype(float),
        "n_features_min": np.arange(len(scores))[::-1],
        "n_features_max": np.arange(len(scores))[::-1],
        "score_mean": scores,
        "score_se": ses,
    })


def test_best_rule_takes_the_argmax():
    curve = hand_built_curve([0.60, 0.72, 0.68, 0.55, 0.50], [0.05] * 5)
    assert _select_threshold(curve, "best") == pytest.approx(0.25)


def test_one_se_rule_takes_the_largest_threshold_within_one_se():
    # Peak 0.72 at threshold 0.25, SE 0.05 -> anything >= 0.67 qualifies,
    # which includes 0.68 at threshold 0.50 but not 0.55 at 0.75. The rule
    # takes the *largest* qualifying threshold, i.e. the fewest features.
    curve = hand_built_curve([0.60, 0.72, 0.68, 0.55, 0.50], [0.05] * 5)
    assert _select_threshold(curve, "one_se") == pytest.approx(0.50)


def test_one_se_rule_reduces_to_the_argmax_when_the_curve_is_sharp():
    curve = hand_built_curve([0.50, 0.51, 0.80, 0.52, 0.50], [0.01] * 5)
    assert _select_threshold(curve, "one_se") == pytest.approx(0.50)
    assert _select_threshold(curve, "best") == pytest.approx(0.50)


def test_rules_ignore_thresholds_that_scored_nothing():
    # The strictest thresholds selected no features at all.
    curve = hand_built_curve(
        [0.60, 0.72, 0.68, np.nan, np.nan], [0.05, 0.05, 0.05, np.nan, np.nan]
    )
    assert _select_threshold(curve, "one_se") == pytest.approx(0.50)


def test_one_se_falls_back_to_argmax_without_a_standard_error():
    curve = hand_built_curve([0.60, 0.72, 0.68, 0.55, 0.50], [np.nan] * 5)
    with pytest.warns(UserWarning, match="one-standard-error"):
        assert _select_threshold(curve, "one_se") == pytest.approx(0.25)


def test_an_all_nan_curve_is_an_error_not_a_choice():
    curve = hand_built_curve([np.nan] * 5, [np.nan] * 5)
    with pytest.raises(ValueError, match="selected zero features"):
        _select_threshold(curve, "one_se")


def test_unknown_rule_rejected():
    with pytest.raises(ValueError, match="rule must be"):
        _select_threshold(hand_built_curve([0.6] * 5, [0.05] * 5), "oneSE")


# --- the inner splits -----------------------------------------------------

def test_inner_splits_never_break_a_pair():
    df, y, groups = make_paired_dataset()
    splits = _inner_splits(df, y, groups, n_splits=5, n_repeats=3, random_state=0)

    assert len(splits) == 3
    for sub_idx, val_idx in splits:
        sub_groups = set(groups[sub_idx])
        val_groups = set(groups[val_idx])
        assert not (sub_groups & val_groups)
        # Both members of every validation pair are in the validation fold.
        assert len(val_idx) == 2 * len(val_groups)


def test_inner_splits_are_one_fifth_and_stratified():
    df, y, groups = make_paired_dataset(n_pairs=60)
    for sub_idx, val_idx in _inner_splits(df, y, groups, 5, 3, random_state=0):
        assert len(sub_idx) + len(val_idx) == len(df)
        assert len(val_idx) == pytest.approx(len(df) / 5, abs=4)
        assert set(y[val_idx]) == {"real", "fake"}


def test_inner_splits_differ_between_repeats():
    df, y, groups = make_paired_dataset()
    validations = [
        frozenset(val_idx)
        for _, val_idx in _inner_splits(df, y, groups, 5, 3, random_state=0)
    ]
    assert len(set(validations)) == 3


# --- threshold_sweep ------------------------------------------------------

@pytest.fixture(scope="module")
def swept():
    """One fitted selector and its sweep, shared across the sweep tests."""
    df, y, groups = make_paired_dataset()
    (sub_idx, val_idx), = _inner_splits(df, y, groups, 5, 1, random_state=0)
    X_sub, X_val = df.iloc[sub_idx], df.iloc[val_idx]
    y_sub, y_val = y[sub_idx], y[val_idx]

    selector = fast_selector().fit(X_sub, y_sub)
    sweep = threshold_sweep(
        selector, X_sub, y_sub, X_val, y_val,
        model=fast_model(), positive_label="fake",
    )
    return selector, sweep, (X_sub, y_sub, X_val, y_val)


def test_sweep_has_one_row_per_candidate_threshold(swept):
    selector, sweep, _ = swept
    assert list(sweep["threshold"]) == threshold_grid(selector.n_repeats).tolist()
    assert list(sweep.columns) == [
        "threshold", "n_features", "features", "score", "score_se"
    ]


def test_sweep_keeps_everything_at_threshold_zero(swept):
    selector, sweep, _ = swept
    row = sweep.loc[sweep["threshold"] == 0.0].iloc[0]
    assert row["n_features"] == len(FEATURES)
    assert set(row["features"]) == set(FEATURES)


def test_sweep_feature_sets_are_nested_and_shrink(swept):
    _, sweep, _ = swept
    counts = sweep["n_features"].to_numpy()
    assert (np.diff(counts) <= 0).all()
    for tighter, looser in zip(sweep["features"][1:], sweep["features"][:-1]):
        assert set(tighter) <= set(looser)


def test_sweep_scores_are_probabilities_where_features_exist(swept):
    _, sweep, _ = swept
    scored = sweep[sweep["n_features"] > 0]
    assert len(scored) > 1
    assert scored["score"].between(0, 1).all()
    # Finite and non-negative rather than strictly positive: DeLong's
    # variance is genuinely zero under perfect separation, which a small
    # validation fold can produce, and pinning it above zero would be
    # asserting that the fixture stays hard rather than that the code works.
    assert np.isfinite(scored["score_se"]).all()
    assert (scored["score_se"] >= 0).all()


def test_sweep_returns_nan_rather_than_raising_on_an_empty_selection(swept):
    selector, sweep, (X_sub, y_sub, X_val, y_val) = swept
    # Force the empty case regardless of what this fixture's Boruta run
    # confirmed: a threshold above 1.0 can never be met.
    forced = threshold_sweep(
        selector, X_sub, y_sub, X_val, y_val,
        model=fast_model(), positive_label="fake", thresholds=[1.5],
    )
    assert forced.loc[0, "n_features"] == 0
    assert np.isnan(forced.loc[0, "score"])
    assert np.isnan(forced.loc[0, "score_se"])


def test_sweep_finds_the_planted_signal(swept):
    selector, _, _ = swept
    # The strong feature should be confirmed more often than the noise and
    # more often than the pair-shared nuisance. Only the ordering is
    # asserted, not the level: an 8-run fixture cannot pin a frequency, and
    # a threshold that had never had backing is exactly the trap
    # `test_boruta_r_validation.py` fell into on a scikit-learn bump.
    frequency = selector.selection_frequency_
    assert frequency["f_strong"] == max(frequency.values())
    assert all(frequency[f"noise_{i}"] < frequency["f_strong"] for i in range(3))
    assert frequency["f_pair"] < frequency["f_strong"]


def test_sweep_is_deterministic(swept):
    selector, sweep, (X_sub, y_sub, X_val, y_val) = swept
    again = threshold_sweep(
        selector, X_sub, y_sub, X_val, y_val,
        model=fast_model(), positive_label="fake",
    )
    pd.testing.assert_frame_equal(sweep, again)


def test_sweep_accepts_an_sklearn_scorer_without_a_standard_error(swept):
    selector, _, (X_sub, y_sub, X_val, y_val) = swept
    sweep = threshold_sweep(
        selector, X_sub, y_sub, X_val, y_val,
        model=fast_model(), scoring="accuracy", positive_label="fake",
    )
    scored = sweep[sweep["n_features"] > 0]
    assert scored["score"].between(0, 1).all()
    assert scored["score_se"].isna().all()


# --- tune_selection_threshold --------------------------------------------

@pytest.fixture(scope="module")
def tuned():
    df, y, groups = make_paired_dataset()
    result = tune_selection_threshold(
        df, y, groups,
        selector=fast_selector(),
        model=fast_model(),
        positive_label="fake",
        n_repeats=2,
        random_state=0,
    )
    return df, y, groups, result


def test_tuning_returns_a_choice_on_the_grid(tuned):
    *_, result = tuned
    assert isinstance(result, ThresholdTuningResult)
    assert result.best_threshold_ in threshold_grid(8).tolist()
    assert 0 <= result.best_score_ <= 1
    assert result.best_n_features_ >= 1
    assert result.rule_ == "one_se"
    assert result.scoring_ == "roc_auc"


def test_tuning_curve_has_one_row_per_threshold_and_a_spread(tuned):
    *_, result = tuned
    assert len(result.curve_) == len(threshold_grid(8))
    assert (result.curve_["n_splits_scored"] <= 2).all()
    scored = result.curve_[result.curve_["n_splits_scored"] == 2]
    assert (scored["score_se"] >= 0).all()
    assert len(result.per_split_) == 2 * len(threshold_grid(8))
    assert sorted(result.per_split_["split"].unique()) == [0, 1]


def test_tuning_refits_on_everything_at_the_chosen_threshold(tuned):
    df, _, _, result = tuned
    assert result.selector_ is not None
    assert result.selector_.selection_threshold == result.best_threshold_
    assert result.selector_.feature_columns_ == FEATURES
    # Refitted on all the data, not on a sub-train fold.
    assert len(result.selector_.runs_[0].importance_history) >= 1
    assert set(result.selected_columns_) <= set(FEATURES)
    assert result.selector_.transform(df).shape[0] == len(df)


def test_tuning_is_deterministic(tuned):
    df, y, groups, result = tuned
    again = tune_selection_threshold(
        df, y, groups,
        selector=fast_selector(),
        model=fast_model(),
        positive_label="fake",
        n_repeats=2,
        random_state=0,
        refit=False,
    )
    assert again.best_threshold_ == result.best_threshold_
    pd.testing.assert_frame_equal(again.curve_, result.curve_)


def test_tuning_without_refit_has_no_selector():
    df, y, groups = make_paired_dataset()
    result = tune_selection_threshold(
        df, y, groups,
        selector=fast_selector(),
        model=fast_model(),
        positive_label="fake",
        n_repeats=1,
        random_state=0,
        refit=False,
    )
    assert result.selector_ is None
    with pytest.raises(AttributeError, match="refitted selector"):
        result.selected_columns_


def test_a_single_split_falls_back_to_the_delong_standard_error():
    df, y, groups = make_paired_dataset()
    result = tune_selection_threshold(
        df, y, groups,
        selector=fast_selector(),
        model=fast_model(),
        positive_label="fake",
        n_repeats=1,
        random_state=0,
        refit=False,
    )
    scored = result.curve_[result.curve_["n_splits_scored"] == 1]
    assert np.isfinite(scored["score_se"]).all()
    # With one split the aggregate error is exactly that split's own.
    merged = scored.merge(result.per_split_, on="threshold", suffixes=("", "_split"))
    assert np.allclose(merged["score_se"], merged["score_se_split"])


def test_tuning_requires_groups():
    df, y, _ = make_paired_dataset()
    with pytest.raises(ValueError, match="requires groups"):
        tune_selection_threshold(df, y, None, selector=fast_selector())


def test_tuning_rejects_zero_repeats():
    df, y, groups = make_paired_dataset()
    with pytest.raises(ValueError, match="n_repeats must be >= 1"):
        tune_selection_threshold(df, y, groups, n_repeats=0)


def test_the_best_rule_never_scores_below_the_one_se_rule(tuned):
    df, y, groups, one_se = tuned
    best = tune_selection_threshold(
        df, y, groups,
        selector=fast_selector(),
        model=fast_model(),
        positive_label="fake",
        n_repeats=2,
        rule="best",
        random_state=0,
        refit=False,
    )
    # Same curve, different reading of it: "best" is the argmax by
    # construction, and "one_se" trades score for parsimony.
    pd.testing.assert_frame_equal(best.curve_, one_se.curve_)
    assert best.best_score_ >= one_se.best_score_
    assert best.best_threshold_ <= one_se.best_threshold_


# --- the figure -----------------------------------------------------------

def test_plot_threshold_sweep_draws(tuned):
    *_, result = tuned
    ax = plot_threshold_sweep(result)
    assert ax.get_xlabel() == "selection_threshold"
    assert "roc_auc" in ax.get_ylabel()
    plt.close(ax.figure)


def test_plot_threshold_sweep_can_show_individual_splits(tuned):
    *_, result = tuned
    ax = plot_threshold_sweep(result, show_splits=True)
    assert len(ax.lines) > 2
    plt.close(ax.figure)


def test_plot_threshold_sweep_saves(tuned, tmp_path):
    *_, result = tuned
    ax = plot_threshold_sweep(result, save_path="sweep.png", output_dir=tmp_path)
    assert (tmp_path / "sweep.png").exists()
    plt.close(ax.figure)
