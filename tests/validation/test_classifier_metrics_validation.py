"""Validation of the evaluation metrics against R's own real output.

Unusually for this directory, these tests run **everywhere** -- they need
neither the private dataset nor an R installation.
`caret::confusionMatrix` is a deterministic function of a confusion
matrix, so the confusion matrices printed in
`final_analysis_NMF_check.Rmd` -- together with the statistics R printed
beside them -- are a complete reference fixture. Reconstructing the label
arrays from the counts and re-deriving the statistics reproduces R's
numbers to every digit R printed.

That makes this the tightest validation in the project: no seeds, no
floating-point drift, no "statistical equivalence" caveat. It pins the
four choices where R's estimator is not the obvious one and a plausible
substitute would pass casual inspection:

- the accuracy interval is **Clopper-Pearson** (R's `binom.test`), not
  Wilson -- for the random forest, Wilson gives (0.5811, 0.7664) where R
  reports (0.5767, 0.7733)
- **P[Acc > NIR]** is a one-sided **exact binomial** test, not a normal
  approximation
- **McNemar's** test carries R's continuity correction, but R drops the
  correction when the two discordant counts are equal, giving p = 1 for
  the SVM rather than 0.87
- R reports **Cohen's kappa**; MCC is an addition of this package's, not
  a replacement

The DeLong ROC interval needs per-video probabilities rather than a
confusion matrix, so it is pinned separately against a real `pROC` run,
using the 94 predicted probabilities committed in
``fixtures/r_test_set_predictions.csv``.

Sources: `final_analysis_NMF_check.Rmd` ~L1390 (random forest), ~L1473
(C5.0), ~L3090 (SVM), each printed under `trainControl(method = "none")`
with `positive = "fake"`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from facedyn.evaluation import (
    classification_metrics,
    confusion_matrix_frame,
    delong_roc_test,
    roc_auc_delong_ci,
)


def labels_from_counts(
    predicted_real_truly_real: int,
    predicted_real_truly_fake: int,
    predicted_fake_truly_real: int,
    predicted_fake_truly_fake: int,
):
    """Rebuild label arrays from a caret-style printed confusion matrix.

    caret prints predictions down the rows and the reference across the
    columns, so the four counts are read off the printed table directly.
    """
    y_true = (
        ["real"] * predicted_real_truly_real
        + ["fake"] * predicted_real_truly_fake
        + ["real"] * predicted_fake_truly_real
        + ["fake"] * predicted_fake_truly_fake
    )
    y_pred = (
        ["real"] * (predicted_real_truly_real + predicted_real_truly_fake)
        + ["fake"] * (predicted_fake_truly_real + predicted_fake_truly_fake)
    )
    return np.array(y_true), np.array(y_pred)


# Each entry: R's printed confusion matrix, then the statistics R printed
# beside it. `None` marks a statistic R's output did not include.
R_PUBLISHED_RESULTS = {
    "random_forest": {
        "counts": (26, 9, 21, 38),
        "accuracy": 0.6809,
        "accuracy_ci": (0.5767, 0.7733),
        "no_information_rate": 0.5,
        "p_acc_gt_nir": 0.000294,
        "kappa": 0.3617,
        "mcnemar_p": 0.04461,
        "sensitivity": 0.8085,
        "specificity": 0.5532,
        "ppv": 0.6441,
        "npv": 0.7429,
        "detection_rate": 0.4043,
        "detection_prevalence": 0.6277,
        "balanced_accuracy": 0.6809,
        "f1": 0.7170,
    },
    "c50": {
        "counts": (24, 10, 23, 37),
        "accuracy": 0.6489,
        "accuracy_ci": (0.5436, 0.7446),
        "no_information_rate": 0.5,
        "p_acc_gt_nir": 0.002539,
        "kappa": 0.2979,
        "mcnemar_p": 0.036714,
        "sensitivity": 0.7872,
        "specificity": 0.5106,
        "ppv": 0.6167,
        "npv": 0.7059,
        "detection_rate": 0.3936,
        "detection_prevalence": 0.6383,
        "balanced_accuracy": 0.6489,
        "f1": 0.6916,
    },
    "svm_rbf": {
        "counts": (29, 18, 18, 29),
        "accuracy": 0.617,
        "accuracy_ci": (0.511, 0.7154),
        "no_information_rate": 0.5,
        "p_acc_gt_nir": 0.01488,
        "kappa": 0.234,
        "mcnemar_p": 1.0,
        "sensitivity": 0.6170,
        "specificity": 0.6170,
        "ppv": 0.6170,
        "npv": 0.6170,
        "detection_rate": 0.3085,
        "detection_prevalence": 0.5000,
        "balanced_accuracy": 0.6170,
        "f1": None,  # not printed for this model
    },
}


@pytest.fixture(params=sorted(R_PUBLISHED_RESULTS), ids=sorted(R_PUBLISHED_RESULTS))
def published(request):
    expected = R_PUBLISHED_RESULTS[request.param]
    y_true, y_pred = labels_from_counts(*expected["counts"])
    metrics = classification_metrics(y_true, y_pred, positive_label="fake")
    return expected, metrics


def test_reconstructed_labels_reproduce_rs_confusion_matrix(published):
    expected, _ = published
    real_real, real_fake, fake_real, fake_fake = expected["counts"]
    y_true, y_pred = labels_from_counts(*expected["counts"])

    table = confusion_matrix_frame(y_true, y_pred)
    assert table.loc["real", "real"] == real_real
    assert table.loc["real", "fake"] == real_fake
    assert table.loc["fake", "real"] == fake_real
    assert table.loc["fake", "fake"] == fake_fake
    assert table.to_numpy().sum() == 94  # the held-out test set


def test_accuracy_and_its_exact_interval_match_r(published):
    expected, metrics = published
    lower, upper = expected["accuracy_ci"]

    assert metrics["accuracy"] == pytest.approx(expected["accuracy"], abs=5e-5)
    assert metrics["accuracy_ci_lower"] == pytest.approx(lower, abs=5e-5)
    assert metrics["accuracy_ci_upper"] == pytest.approx(upper, abs=5e-5)


def test_no_information_rate_test_matches_r(published):
    expected, metrics = published
    assert metrics["no_information_rate"] == pytest.approx(expected["no_information_rate"])
    assert metrics["p_acc_gt_nir"] == pytest.approx(expected["p_acc_gt_nir"], rel=1e-3)


def test_agreement_and_mcnemar_statistics_match_r(published):
    expected, metrics = published
    assert metrics["kappa"] == pytest.approx(expected["kappa"], abs=5e-5)
    assert metrics["mcnemar_p"] == pytest.approx(expected["mcnemar_p"], rel=1e-3)


@pytest.mark.parametrize(
    "statistic",
    ["sensitivity", "specificity", "ppv", "npv",
     "detection_rate", "detection_prevalence", "balanced_accuracy", "f1"],
)
def test_class_statistics_match_r(published, statistic):
    expected, metrics = published
    if expected[statistic] is None:
        pytest.skip(f"R's output did not print {statistic} for this model")
    assert metrics[statistic] == pytest.approx(expected[statistic], abs=5e-5)


def test_prevalence_is_balanced_in_every_published_table(published):
    _, metrics = published
    assert metrics["prevalence"] == pytest.approx(0.5)


# --- DeLong, against a real pROC run -------------------------------------
#
# `fixtures/r_test_set_predictions.csv` holds R's own predicted
# probabilities for the 94 held-out videos, from `caret::train(method="rf")`
# and `method="C5.0"` fitted on the 370 training videos with Paper 1's
# eight published features (`trainControl(method="none")`, seed 12345).
# The C5.0 predictions reproduce the paper's printed confusion matrix
# exactly; the random forest's differ on a single video, which is what a
# stochastic forest refitted years later is expected to do.
#
# The reference values below come from running `pROC` on these same
# probabilities: `ci.auc` (DeLong) and `roc.test(method="delong")`.

PREDICTIONS_PATH = Path(__file__).parent / "fixtures" / "r_test_set_predictions.csv"

R_PROC_REFERENCE = {
    "p_fake_rf": {"auc": 0.692621, "ci": (0.583576, 0.801666)},
    "p_fake_c50": {"auc": 0.651200, "ci": (0.558039, 0.744360)},
}
R_PROC_PAIRED_TEST_P = 0.446625


@pytest.fixture(scope="module")
def r_predictions():
    return pd.read_csv(PREDICTIONS_PATH)


def test_prediction_fixture_is_the_real_held_out_set(r_predictions):
    assert len(r_predictions) == 94
    assert r_predictions["truth"].value_counts().to_dict() == {"fake": 47, "real": 47}


@pytest.mark.parametrize("column", sorted(R_PROC_REFERENCE))
def test_delong_interval_matches_proc(r_predictions, column):
    expected = R_PROC_REFERENCE[column]
    auc, lower, upper = roc_auc_delong_ci(
        r_predictions["truth"], r_predictions[column], positive_label="fake"
    )

    assert auc == pytest.approx(expected["auc"], abs=1e-6)
    assert lower == pytest.approx(expected["ci"][0], abs=1e-6)
    assert upper == pytest.approx(expected["ci"][1], abs=1e-6)


def test_paired_delong_test_matches_procs_roc_test(r_predictions):
    result = delong_roc_test(
        r_predictions["truth"],
        r_predictions["p_fake_rf"],
        r_predictions["p_fake_c50"],
        positive_label="fake",
    )
    assert result["p_value"] == pytest.approx(R_PROC_PAIRED_TEST_P, abs=1e-6)


def test_r_c50_predictions_reproduce_the_published_confusion_matrix(r_predictions):
    """Ties the probability fixture back to the table printed in the Rmd."""
    predicted = np.where(r_predictions["p_fake_c50"] >= 0.5, "fake", "real")
    table = confusion_matrix_frame(r_predictions["truth"], predicted)

    published = R_PUBLISHED_RESULTS["c50"]["counts"]
    assert table.loc["real", "real"] == published[0]
    assert table.loc["real", "fake"] == published[1]
    assert table.loc["fake", "real"] == published[2]
    assert table.loc["fake", "fake"] == published[3]


def test_wilson_interval_would_not_reproduce_rs_numbers():
    """The distinction this file exists to pin, stated as a test.

    Wilson is the more common default (and what an earlier plan for this
    module assumed R used), so a substitution would be easy to make and
    hard to notice: it agrees to two decimal places.
    """
    from scipy.stats import binomtest

    expected = R_PUBLISHED_RESULTS["random_forest"]
    correct = expected["counts"][0] + expected["counts"][3]
    wilson = binomtest(correct, 94).proportion_ci(0.95, "wilson")

    assert wilson.low == pytest.approx(0.5811, abs=5e-5)
    assert wilson.high == pytest.approx(0.7664, abs=5e-5)
    assert wilson.low != pytest.approx(expected["accuracy_ci"][0], abs=5e-5)
