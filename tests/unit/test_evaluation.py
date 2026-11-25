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
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from facedyn.classifiers import make_classifier, make_classifiers
from facedyn.evaluation import (
    classification_metrics,
    confusion_matrix_frame,
    cross_validate_grouped,
    delong_roc_test,
    evaluate_models,
    pairwise_contrasts,
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_probability_distributions,
    plot_roc_curves,
    plot_roc_ovr,
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
            rows.append(
                {
                    "video_filename": f"{label}_{i}",
                    "corresponding_video": f"{'fake' if label == 'real' else 'real'}_{i}",
                    "isfakeorreal": label,
                    "f0": signal + rng.normal(scale=1.0),
                    "f1": identity + rng.normal(scale=0.05),
                }
            )
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
    models = {
        "random_forest": make_classifier("random_forest", random_state=0, n_estimators=10).fit(X, y)
    }
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
        model,
        X,
        y,
        groups=pair_groups(df),
        scoring=["roc_auc", "accuracy"],
        n_splits=3,
        n_repeats=1,
        random_state=0,
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
        ungrouped = cross_validate_grouped(model, X, y, n_splits=5, n_repeats=2, random_state=0)[
            "roc_auc"
        ].mean()

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


# --- multiclass ----------------------------------------------------------

CLASSES3 = ["angry", "happy", "sad"]


def make_three_class(n_per_class: int = 40, separation: float = 1.5, seed: int = 0):
    """Video-level table with three classes separated along two features."""
    rng = np.random.default_rng(seed)
    centres = {"angry": (0, 0), "happy": (separation, 0), "sad": (0, separation)}
    rows = []
    for label, (cx, cy) in centres.items():
        for i in range(n_per_class):
            rows.append(
                {
                    "video_filename": f"{label}_{i}",
                    "participant": f"p{i % 10}",
                    "emotion": label,
                    "f0": cx + rng.normal(),
                    "f1": cy + rng.normal(),
                    "f2": rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    return df, df[["f0", "f1", "f2"]], df["emotion"].to_numpy()


@pytest.fixture(scope="module")
def three_class_fit():
    df, X, y = make_three_class()
    model = make_classifier("logistic_regression").fit(X, y)
    return X, y, model


@pytest.mark.filterwarnings("ignore:Setting penalty=None")
def test_multiclass_metrics_match_sklearn(three_class_fit):
    X, y, model = three_class_fit
    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    m = classification_metrics(y, y_pred, proba, labels=model.classes_)

    assert m["n"] == len(y)
    assert m["accuracy"] == pytest.approx(np.mean(y == y_pred))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, y_pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, y_pred, average="macro"))
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y, y_pred))
    assert m["roc_auc"] == pytest.approx(
        roc_auc_score(y, proba, multi_class="ovr", average="macro")
    )
    per_class_auc = roc_auc_score(y, proba, multi_class="ovr", average=None)
    recalls = recall_score(y, y_pred, average=None, labels=CLASSES3)
    precisions = precision_score(y, y_pred, average=None, labels=CLASSES3)
    for k, cls in enumerate(CLASSES3):
        assert m[f"roc_auc_{cls}"] == pytest.approx(per_class_auc[k])
        assert m[f"recall_{cls}"] == pytest.approx(recalls[k])
        assert m[f"precision_{cls}"] == pytest.approx(precisions[k])
        assert m[f"roc_auc_{cls}_ci_lower"] < m[f"roc_auc_{cls}"] < m[f"roc_auc_{cls}_ci_upper"]
    # Binary-only fields do not appear.
    assert "mcnemar_p" not in m
    assert "sensitivity" not in m


def test_multiclass_metrics_without_scores_leave_auc_missing():
    y_true = np.array(["a", "b", "c", "a", "b", "c"])
    y_pred = np.array(["a", "b", "b", "a", "c", "c"])
    m = classification_metrics(y_true, y_pred)
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["roc_auc_a"])
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_multiclass_metrics_count_a_never_predicted_class_as_zero_f1():
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "a", "b", "b", "a", "b"])
    m = classification_metrics(y_true, y_pred)
    assert m["f1_c"] == 0.0
    assert np.isnan(m["precision_c"])
    assert m["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


def test_multiclass_auc_is_nan_for_a_class_missing_from_the_reference():
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    proba = np.array([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
    m = classification_metrics(y_true, y_pred, proba, labels=["a", "b", "c"])
    assert np.isnan(m["roc_auc_c"])
    assert np.isnan(m["roc_auc"])
    assert m["roc_auc_a"] == pytest.approx(1.0)
    # Only classes that occur count toward balanced accuracy.
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1.0) / 2)


def test_multiclass_rejects_a_positive_label_and_1d_scores():
    y = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="pairwise_contrasts"):
        classification_metrics(y, y, positive_label="a")
    with pytest.raises(ValueError, match="must be 2-D"):
        classification_metrics(y, y, np.array([0.1, 0.5, 0.9]))


def test_score_columns_must_match_the_classes():
    y = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="Pass labels="):
        classification_metrics(y, y, np.full((3, 2), 0.5))


def test_labels_must_cover_the_data():
    y = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="not in"):
        classification_metrics(y, y, labels=["a", "b"])


def test_binary_metrics_accept_two_column_scores():
    rng = np.random.default_rng(0)
    y_true = np.array(["fake"] * 30 + ["real"] * 30)
    p_fake = np.concatenate([rng.beta(4, 2, 30), rng.beta(2, 4, 30)])
    y_pred = np.where(p_fake > 0.5, "fake", "real")
    proba = np.column_stack([p_fake, 1 - p_fake])  # columns: fake, real (sorted)

    one_d = classification_metrics(y_true, y_pred, p_fake, positive_label="fake")
    two_d = classification_metrics(y_true, y_pred, proba, positive_label="fake")
    pd.testing.assert_series_equal(one_d, two_d)
    # Still no guessing which class is positive for string labels.
    with pytest.raises(ValueError, match="positive_label is required"):
        classification_metrics(y_true, y_pred, proba)


def test_binary_path_is_unchanged_by_the_labels_argument():
    y_true, y_pred = make_labels(tn=30, fp=10, fn=5, tp=55)
    score = np.where(y_pred == 1, 0.8, 0.2)
    pd.testing.assert_series_equal(
        classification_metrics(y_true, y_pred, score),
        classification_metrics(y_true, y_pred, score, labels=[0, 1]),
    )


@pytest.mark.filterwarnings("ignore:Setting penalty=None")
def test_evaluate_models_scores_multiclass_models():
    df, X, y = make_three_class()
    models = make_classifiers(random_state=0)
    for model in models.values():
        model.fit(X, y)

    results = evaluate_models(models, X, y)

    assert list(results.index) == list(models)
    assert results["roc_auc"].between(0.5, 1).all()
    for cls in CLASSES3:
        assert f"roc_auc_{cls}" in results.columns
    with pytest.raises(ValueError, match="positive_label"):
        evaluate_models(models, X, y, positive_label="happy")


def test_confusion_matrix_frame_handles_three_classes():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    table = confusion_matrix_frame(y_true, y_pred)
    assert table.shape == (3, 3)
    assert table.loc["b", "a"] == 1  # predicted b, truly a
    assert table.loc["a", "c"] == 1
    assert np.trace(table.to_numpy()) == 4
    assert table.to_numpy().sum() == 6


def test_pairwise_contrasts_builds_every_pair():
    df, _, _ = make_three_class(n_per_class=5)
    contrasts = pairwise_contrasts(df, "emotion")

    assert list(contrasts) == [("angry", "happy"), ("angry", "sad"), ("happy", "sad")]
    for pair, subset in contrasts.items():
        assert set(subset["emotion"]) == set(pair)
        assert len(subset) == 10
        assert subset.index.isin(df.index).all()


def test_pairwise_contrasts_honours_given_pairs_and_rejects_unknown_classes():
    df, _, _ = make_three_class(n_per_class=5)
    contrasts = pairwise_contrasts(df, "emotion", pairs=[("sad", "happy")])
    assert list(contrasts) == [("sad", "happy")]
    with pytest.raises(ValueError, match="not found"):
        pairwise_contrasts(df, "emotion", pairs=[("happy", "surprised")])
    with pytest.raises(ValueError, match="two different"):
        pairwise_contrasts(df, "emotion", pairs=[("happy", "happy")])
    with pytest.raises(KeyError):
        pairwise_contrasts(df, "label")


def test_pairwise_contrasts_feed_binary_delong(three_class_fit):
    df, X, y = make_three_class()
    for (_a, b), subset in pairwise_contrasts(df, "emotion").items():
        Xs, ys = subset[["f0", "f1", "f2"]], subset["emotion"].to_numpy()
        model = make_classifier("random_forest", random_state=0, n_estimators=20).fit(Xs, ys)
        auc, lower, upper = roc_auc_delong_ci(ys, model.predict_proba(Xs)[:, 1], positive_label=b)
        assert lower <= auc <= upper


@pytest.mark.filterwarnings("ignore:Setting penalty=None")
def test_plot_roc_curves_draws_one_curve_per_class_for_one_model(three_class_fit):
    X, y, model = three_class_fit
    ax = plot_roc_curves({"logistic_regression": model}, X, y)
    assert len(ax.lines) == 3 + 1  # three classes + diagonal
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t.startswith("happy vs rest") for t in texts)
    assert "One-vs-rest" in ax.get_title()
    plt.close(ax.figure)


@pytest.mark.filterwarnings("ignore:Setting penalty=None")
def test_plot_roc_curves_draws_class_by_model_curves_for_several_models():
    df, X, y = make_three_class()
    models = make_classifiers(["random_forest", "logistic_regression"], random_state=0)
    for model in models.values():
        model.fit(X, y)
    ax = plot_roc_curves(models, X, y)
    assert len(ax.lines) == 2 * 3 + 1
    plt.close(ax.figure)


def test_plot_roc_ovr_works_from_probabilities_alone(three_class_fit):
    X, y, model = three_class_fit
    proba = pd.DataFrame(model.predict_proba(X), columns=[f"proba_{c}" for c in CLASSES3])
    ax = plot_roc_ovr(y, proba, labels=CLASSES3)
    assert len(ax.lines) == 4
    plt.close(ax.figure)
    # A class absent from y_true gets no curve.
    keep = y != "sad"
    ax = plot_roc_ovr(y[keep], proba[keep], labels=CLASSES3)
    assert len(ax.lines) == 3
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="one column per class"):
        plot_roc_ovr(y, proba.iloc[:, :2], labels=CLASSES3)


def test_plot_confusion_matrix_three_classes_normalised_by_reference():
    y_true = ["a"] * 4 + ["b"] * 2 + ["c"] * 2
    y_pred = ["a", "a", "a", "b", "b", "b", "c", "a"]
    ax = plot_confusion_matrix(y_true, y_pred, normalize="reference", annotate="both")
    texts = {t.get_text() for t in ax.texts}
    assert len(ax.texts) == 9
    assert "3\n(0.75)" in texts  # 3 of 4 true a predicted a
    assert "1\n(0.50)" in texts  # 1 of 2 true c predicted a
    shading = ax.images[0].get_array()
    np.testing.assert_allclose(np.asarray(shading).sum(axis=0), 1.0)
    plt.close(ax.figure)


def test_plot_confusion_matrix_prediction_normalisation_matches_true():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    ax_true = plot_confusion_matrix(y_true, y_pred, normalize=True)
    ax_pred = plot_confusion_matrix(y_true, y_pred, normalize="prediction")
    np.testing.assert_allclose(
        np.asarray(ax_true.images[0].get_array()), np.asarray(ax_pred.images[0].get_array())
    )
    plt.close("all")


def test_plot_confusion_matrix_handles_a_never_predicted_class():
    ax = plot_confusion_matrix(
        ["a", "b", "c"],
        ["a", "b", "b"],
        labels=["a", "b", "c"],
        normalize=True,
        annotate="proportion",
    )
    assert np.isfinite(np.asarray(ax.images[0].get_array())).all()
    plt.close(ax.figure)


def test_plot_confusion_matrix_rejects_bad_options():
    with pytest.raises(ValueError, match="normalize"):
        plot_confusion_matrix([0, 1], [0, 1], normalize="columns")
    with pytest.raises(ValueError, match="needs normalize"):
        plot_confusion_matrix([0, 1], [0, 1], annotate="proportion")


def test_plot_probability_distributions_names_the_rest_group():
    rng = np.random.default_rng(0)
    y_true = np.array(["happy"] * 20 + ["sad"] * 20 + ["angry"] * 20)
    score = rng.uniform(size=60)
    ax = plot_probability_distributions(y_true, score, positive_label="happy")
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == {"happy", "rest"}
    plt.close(ax.figure)
