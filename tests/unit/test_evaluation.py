"""Unit tests for the evaluation metrics, DeLong tests, CV harness and plots.

The metric formulas are pinned against R's own printed output in
`tests/validation/test_classifier_metrics_validation.py`; what is tested
here is everything that does not need R -- hand-computable cases, the
DeLong estimator's agreement with scikit-learn's AUC, degenerate inputs,
and the grouped-CV guard rail.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from facedyn.classifiers import make_classifier, make_classifiers
from facedyn.evaluation import (
    classification_metrics,
    confusion_matrix_frame,
    cross_validate_grouped,
    delong_roc_test,
    evaluate_models,
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_probability_distributions,
    plot_roc_curves,
    roc_auc_delong_ci,
)
from facedyn.splitting import pair_groups

matplotlib = pytest.importorskip("matplotlib", reason="plots need the viz extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_labels(tn: int, fp: int, fn: int, tp: int):
    """Expand a confusion matrix into label arrays. Positive class = 1."""
    y_true = np.array([0] * tn + [0] * fp + [1] * fn + [1] * tp)
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    return y_true, y_pred


def make_paired_dataset(n_pairs: int = 30, seed: int = 0):
    """Video-level data with matched pairs, built to mimic the real design.

    Two features, doing two different jobs:

    - ``f0`` carries a modest real/fake signal and nothing else. It is the
      only feature that generalises to a video the model has not seen.
    - ``f1`` is a high-variance value **shared by both members of a pair**
      and carrying no class information at all -- the stand-in for
      everything a deepfake inherits from its source video (scene,
      lighting, driving performance).

    That combination is what makes ungrouped folds harmful rather than
    merely wasteful: with a pair's other member in the training set, ``f1``
    identifies the twin almost exactly, and the twin's label is the
    opposite one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        identity = rng.normal(scale=4)
        for label in ("real", "fake"):
            signal = 0.7 if label == "fake" else -0.7
            rows.append({
                "video_filename": f"{label}_{i}",
                "corresponding_video": f"{'fake' if label == 'real' else 'real'}_{i}",
                "isfakeorreal": label,
                "f0": signal + rng.normal(scale=1.0),
                "f1": identity + rng.normal(scale=0.05),
            })
    df = pd.DataFrame(rows)
    return df, df[["f0", "f1"]], df["isfakeorreal"].to_numpy()


# --- confusion-matrix statistics -----------------------------------------


def test_metrics_match_hand_computed_values():
    y_true, y_pred = make_labels(tn=30, fp=10, fn=5, tp=55)
    m = classification_metrics(y_true, y_pred, positive_label=1)

    assert m["n"] == 100
    assert m["accuracy"] == pytest.approx(0.85)
    assert m["sensitivity"] == pytest.approx(55 / 60)
    assert m["specificity"] == pytest.approx(30 / 40)
    assert m["ppv"] == pytest.approx(55 / 65)
    assert m["npv"] == pytest.approx(30 / 35)
    assert m["prevalence"] == pytest.approx(0.60)
    assert m["detection_rate"] == pytest.approx(0.55)
    assert m["detection_prevalence"] == pytest.approx(0.65)
    assert m["balanced_accuracy"] == pytest.approx((55 / 60 + 30 / 40) / 2)
    assert m["no_information_rate"] == pytest.approx(0.60)


def test_accuracy_interval_is_clopper_pearson_not_wilson():
    # 64 of 94 correct, the paper's own random forest. R's confusionMatrix
    # calls binom.test, giving a wider interval than Wilson's would.
    y_true, y_pred = make_labels(tn=26, fp=21, fn=9, tp=38)
    m = classification_metrics(y_true, y_pred, positive_label=1)

    assert m["accuracy"] == pytest.approx(64 / 94)
    assert m["accuracy_ci_lower"] == pytest.approx(0.5767, abs=1e-4)
    assert m["accuracy_ci_upper"] == pytest.approx(0.7733, abs=1e-4)
    # Wilson's interval for the same counts is (0.5811, 0.7664) -- close,
    # but not what R reported, so the distinction is worth pinning.
    assert m["accuracy_ci_lower"] < 0.5811
    assert m["accuracy_ci_upper"] > 0.7664


def test_no_information_rate_uses_the_majority_class_of_the_reference():
    y_true = np.array([1] * 80 + [0] * 20)
    y_pred = np.ones(100, dtype=int)
    m = classification_metrics(y_true, y_pred, positive_label=1)

    assert m["no_information_rate"] == pytest.approx(0.80)
    # Predicting the majority class throughout cannot beat the NIR.
    assert m["p_acc_gt_nir"] > 0.4


def test_mcnemar_skips_the_continuity_correction_when_counts_are_equal():
    equal = classification_metrics(*make_labels(29, 18, 18, 29), positive_label=1)
    unequal = classification_metrics(*make_labels(26, 21, 9, 38), positive_label=1)

    assert equal["mcnemar_p"] == pytest.approx(1.0)
    assert unequal["mcnemar_p"] == pytest.approx(0.04461, abs=1e-5)


def test_mcnemar_is_nan_when_there_is_nothing_discordant():
    m = classification_metrics(*make_labels(40, 0, 0, 60), positive_label=1)
    assert np.isnan(m["mcnemar_p"])
    assert m["accuracy"] == 1.0


def test_positive_label_is_required_for_non_binary_coded_labels():
    y = np.array(["real", "fake", "real", "fake"])
    with pytest.raises(ValueError, match="positive_label is required"):
        classification_metrics(y, y)


def test_positive_label_choice_swaps_sensitivity_and_specificity():
    y_true = np.array(["fake"] * 6 + ["real"] * 4)
    y_pred = np.array(["fake"] * 5 + ["real"] * 5)

    as_fake = classification_metrics(y_true, y_pred, positive_label="fake")
    as_real = classification_metrics(y_true, y_pred, positive_label="real")

    assert as_fake["sensitivity"] == pytest.approx(as_real["specificity"])
    assert as_fake["specificity"] == pytest.approx(as_real["sensitivity"])
    assert as_fake["accuracy"] == pytest.approx(as_real["accuracy"])


def test_metrics_without_scores_leave_the_auc_fields_missing():
    m = classification_metrics(*make_labels(30, 10, 5, 55), positive_label=1)
    assert np.isnan(m["roc_auc"])
    assert not np.isnan(m["accuracy"])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="y_true has"):
        classification_metrics(np.array([0, 1]), np.array([0, 1, 1]), positive_label=1)


def test_confusion_matrix_frame_is_prediction_by_reference():
    y_true, y_pred = make_labels(tn=30, fp=10, fn=5, tp=55)
    table = confusion_matrix_frame(y_true, y_pred)

    assert table.index.name == "Prediction"
    assert table.columns.name == "Reference"
    assert table.loc[1, 1] == 55  # predicted 1, truly 1
    assert table.loc[1, 0] == 10  # predicted 1, truly 0


# --- DeLong --------------------------------------------------------------


def test_delong_auc_equals_sklearns():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    score = rng.normal(size=200) + y_true

    auc, lower, upper = roc_auc_delong_ci(y_true, score, positive_label=1)
    assert auc == pytest.approx(roc_auc_score(y_true, score))
    assert lower < auc < upper


def test_delong_interval_narrows_with_more_data():
    rng = np.random.default_rng(2)
    widths = []
    for n in (60, 600):
        y_true = rng.integers(0, 2, size=n)
        score = rng.normal(size=n) + y_true
        _, lower, upper = roc_auc_delong_ci(y_true, score, positive_label=1)
        widths.append(upper - lower)

    assert widths[1] < widths[0] / 2


def test_delong_interval_is_clipped_to_zero_one():
    y_true = np.array([0] * 8 + [1] * 8)
    score = np.concatenate([np.zeros(8), np.ones(8)])  # perfect separation
    auc, lower, upper = roc_auc_delong_ci(y_true, score, positive_label=1)

    assert auc == 1.0
    assert 0.0 <= lower <= upper <= 1.0


def test_delong_needs_both_classes():
    with pytest.raises(ValueError, match="at least one sample of each class"):
        roc_auc_delong_ci(np.ones(10, dtype=int), np.arange(10.0), positive_label=1)


def test_paired_delong_finds_no_difference_between_identical_curves():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=150)
    score = rng.normal(size=150) + y_true

    result = delong_roc_test(y_true, score, score, positive_label=1)
    assert result["paired"] is True
    assert result["difference"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_paired_delong_detects_a_real_difference():
    rng = np.random.default_rng(4)
    y_true = np.repeat([0, 1], 150)
    good = rng.normal(size=300) + 2.0 * y_true
    weak = rng.normal(size=300) + 0.1 * y_true

    result = delong_roc_test(y_true, good, weak, positive_label=1)
    assert result["auc_a"] > result["auc_b"]
    assert result["p_value"] < 0.001


def test_unpaired_delong_compares_two_different_samples():
    rng = np.random.default_rng(5)
    y_a = np.repeat([0, 1], 100)
    y_b = np.repeat([0, 1], 60)
    score_a = rng.normal(size=200) + 2.0 * y_a
    score_b = rng.normal(size=120) + 0.1 * y_b

    result = delong_roc_test(y_a, score_a, score_b, positive_label=1, y_true_b=y_b)
    assert result["paired"] is False
    assert result["p_value"] < 0.01

    # The unpaired form ignores correlation, so on the same sample it is
    # more conservative than the paired one.
    same_sample_paired = delong_roc_test(y_a, score_a, score_a * 0.5, positive_label=1)
    assert same_sample_paired["difference"] == pytest.approx(0.0)


# --- evaluate_models -----------------------------------------------------


@pytest.mark.filterwarnings("ignore:Setting penalty=None")
def test_evaluate_models_returns_a_row_per_model():
    df, X, y = make_paired_dataset()
    models = {
        name: make_classifier(name, random_state=0, n_estimators=20)
        for name in ("random_forest", "c50_substitute")
    }
    for model in models.values():
        model.fit(X, y)

    results = evaluate_models(models, X, y, positive_label="fake")

    assert list(results.index) == ["random_forest", "c50_substitute"]
    assert results["roc_auc"].between(0, 1).all()
    assert (results["accuracy_ci_lower"] <= results["accuracy"]).all()
    assert (results["accuracy"] <= results["accuracy_ci_upper"]).all()


def test_evaluate_models_requires_a_positive_label_for_string_classes():
    df, X, y = make_paired_dataset(n_pairs=10)
    models = {"random_forest": make_classifier("random_forest", random_state=0,
                                               n_estimators=10).fit(X, y)}
    with pytest.raises(ValueError, match="positive_label is required"):
        evaluate_models(models, X, y)


# --- cross-validation ----------------------------------------------------


def test_cross_validate_grouped_returns_one_row_per_fold():
    df, X, y = make_paired_dataset()
    groups = pair_groups(df)
    model = make_classifier("random_forest", random_state=0, n_estimators=25)

    scores = cross_validate_grouped(
        model, X, y, groups=groups, n_splits=5, n_repeats=2, random_state=0
    )

    assert scores.shape == (10, 1)
    assert list(scores.columns) == ["roc_auc"]
    assert scores["roc_auc"].between(0, 1).all()


def test_cross_validate_grouped_accepts_several_metrics():
    df, X, y = make_paired_dataset()
    model = make_classifier("random_forest", random_state=0, n_estimators=25)

    scores = cross_validate_grouped(
        model, X, y, groups=pair_groups(df), scoring=["roc_auc", "accuracy"],
        n_splits=3, n_repeats=1, random_state=0,
    )
    assert list(scores.columns) == ["roc_auc", "accuracy"]


def test_cross_validate_without_groups_warns_and_still_runs():
    df, X, y = make_paired_dataset()
    model = make_classifier("random_forest", random_state=0, n_estimators=25)

    with pytest.warns(UserWarning, match="not group-aware"):
        scores = cross_validate_grouped(model, X, y, n_splits=3, n_repeats=1, random_state=0)

    assert len(scores) == 3


def test_grouped_cv_beats_ungrouped_on_paired_data():
    """The regression test for the bug this module exists to prevent.

    Ungrouped folds leave a pair's other member in the training set, where
    `f1` identifies it almost exactly under the opposite label. Measured
    margin is ~0.07 ROC-AUC and is stable across data seeds and across
    scikit-learn versions; the threshold below is deliberately well inside
    that so the test pins the direction rather than the exact number.
    """
    df, X, y = make_paired_dataset(n_pairs=60, seed=7)
    model = make_classifier("random_forest", random_state=0, n_estimators=100)

    grouped = cross_validate_grouped(
        model, X, y, groups=pair_groups(df), n_splits=5, n_repeats=2, random_state=0
    )["roc_auc"].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ungrouped = cross_validate_grouped(
            model, X, y, n_splits=5, n_repeats=2, random_state=0
        )["roc_auc"].mean()

    assert grouped - ungrouped > 0.03


# --- figures -------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Setting penalty=None")
def test_plot_roc_curves_draws_a_line_per_model():
    df, X, y = make_paired_dataset()
    models = make_classifiers(["random_forest", "svm_rbf"], random_state=0)
    for model in models.values():
        model.fit(X, y)

    ax = plot_roc_curves(models, X, y, positive_label="fake")
    # One line per model, plus the diagonal reference.
    assert len(ax.lines) == 3
    assert "AUC" in ax.get_legend().get_texts()[0].get_text()
    plt.close(ax.figure)


def test_plot_confusion_matrix_annotates_counts():
    y_true, y_pred = make_labels(tn=30, fp=10, fn=5, tp=55)
    ax = plot_confusion_matrix(y_true, y_pred)

    annotated = {text.get_text() for text in ax.texts}
    assert {"30", "10", "5", "55"} <= annotated
    plt.close(ax.figure)


def test_plot_probability_distributions_marks_the_threshold():
    rng = np.random.default_rng(0)
    y_true = np.array(["fake"] * 50 + ["real"] * 50)
    score = np.concatenate([rng.beta(5, 2, 50), rng.beta(2, 5, 50)])

    ax = plot_probability_distributions(y_true, score, positive_label="fake")
    assert any(line.get_linestyle() == ":" for line in ax.lines)
    plt.close(ax.figure)


def test_plot_probability_distributions_survives_a_constant_score():
    y_true = np.array(["fake"] * 10 + ["real"] * 10)
    score = np.full(20, 0.5)  # kde would fail on zero variance
    ax = plot_probability_distributions(y_true, score, positive_label="fake")
    plt.close(ax.figure)


def test_plot_decision_boundary_holds_other_features_at_the_prototype():
    df, X, y = make_paired_dataset()
    model = make_classifier("random_forest", random_state=0, n_estimators=20).fit(X, y)

    ax = plot_decision_boundary(
        model, X, y, features=("f0", "f1"), positive_label="fake", resolution=25
    )
    assert ax.get_xlabel() == "f0"
    assert len(ax.collections) > 0  # contours plus the scattered points
    plt.close(ax.figure)


def test_plot_decision_boundary_rejects_unknown_features():
    df, X, y = make_paired_dataset(n_pairs=10)
    model = make_classifier("random_forest", random_state=0, n_estimators=10).fit(X, y)

    with pytest.raises(KeyError, match="nope"):
        plot_decision_boundary(model, X, y, features=("f0", "nope"), positive_label="fake")


def test_plot_accuracy_comparison_draws_a_bar_per_model():
    df, X, y = make_paired_dataset()
    models = make_classifiers(["random_forest", "svm_rbf"], random_state=0)
    for model in models.values():
        model.fit(X, y)
    results = evaluate_models(models, X, y, positive_label="fake")

    ax = plot_accuracy_comparison(results)
    assert len(ax.patches) == 2
    plt.close(ax.figure)


def test_plot_accuracy_comparison_rejects_the_wrong_table():
    with pytest.raises(ValueError, match="missing"):
        plot_accuracy_comparison(pd.DataFrame({"roc_auc": [0.7]}))
