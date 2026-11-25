"""Classifier evaluation: metrics, DeLong tests, grouped CV, figures.

Binary and multiclass labels are both supported.

For binary labels the metrics replicate, function for function, what the R
analysis of the deepfake study (Murphy, Cook & Cuve, in prep) reported:
``caret::confusionMatrix`` for the confusion-matrix statistics and
``pROC::auc``/``ci.auc``/``roc.test`` for the ROC ones. Where R's choice of
estimator is not the obvious one, this module follows R rather than the
obvious one -- the accuracy interval is Clopper-Pearson (not Wilson), the
accuracy-vs-no-information-rate test is an exact one-sided binomial test
(not a normal approximation), McNemar's test carries the continuity
correction, and the ROC interval is DeLong's. Each is verified against the
numbers printed in that study's R output; see
``tests/validation/test_classifier_metrics_validation.py``.

Cross-validation on repeated-measures data (several videos per participant)
or matched pairs **must** be grouped by participant or pair, so that no
group is split across training and test folds.
:func:`facedyn.crossval.cross_validate_pipeline` (from frame-level data) and
:func:`cross_validate_grouped` (from a finished feature table) make that the
default rather than something each caller has to remember.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate

from facedyn._plot_utils import save_figure
from facedyn.classifiers import CLASSIFIER_LABELS
from facedyn.splitting import RepeatedStratifiedGroupKFold

__all__ = [
    "MODEL_COLOURS",
    "classification_metrics",
    "confusion_matrix_frame",
    "cross_validate_grouped",
    "delong_roc_test",
    "evaluate_models",
    "pairwise_contrasts",
    "plot_accuracy_comparison",
    "plot_confusion_matrix",
    "plot_decision_boundary",
    "plot_probability_distributions",
    "plot_roc_curves",
    "plot_roc_ovr",
    "roc_auc_delong_ci",
]

# Okabe-Ito colour-blind-safe palette, keyed by classifier name.
MODEL_COLOURS = {
    "random_forest": "#E69F00",
    "c50_substitute": "#009E73",
    "svm_rbf": "#CC79A7",
    "logistic_regression": "#0072B2",
}
_CLASS_COLOURS = {"positive": "#E69F00", "negative": "#009E73"}
_FALLBACK_COLOURS = ["#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#56B4E9"]


# --------------------------------------------------------------------------
# DeLong's method
# --------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, as DeLong's estimator requires."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_auc_cov(scores: np.ndarray, positive: np.ndarray):
    """AUCs and their covariance matrix for one or more score vectors.

    Implements the O(n log n) formulation of DeLong's estimator given by
    Sun & Xu (2014), "Fast implementation of DeLong's algorithm for
    comparing the areas under correlated receiver operating characteristic
    curves", IEEE Signal Processing Letters 21(11).

    Parameters
    ----------
    scores : np.ndarray of shape (n_models, n_samples)
        Predicted score for the positive class.
    positive : np.ndarray of bool, shape (n_samples,)
        True class membership.

    Returns
    -------
    aucs : np.ndarray of shape (n_models,)
    cov : np.ndarray of shape (n_models, n_models)
    """
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    positive = np.asarray(positive, dtype=bool)

    pos = scores[:, positive]
    neg = scores[:, ~positive]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        raise ValueError("DeLong's method needs at least one sample of each class.")

    k = scores.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    # np.cov needs >1 observation per variable; with a single model it
    # returns a 0-d array, hence the reshape.
    sx = np.cov(v01).reshape(k, k)
    sy = np.cov(v10).reshape(k, k)
    return aucs, sx / m + sy / n


def roc_auc_delong_ci(y_true, y_score, positive_label=None, alpha: float = 0.05):
    """ROC-AUC with DeLong's confidence interval.

    The Python equivalent of ``pROC::auc`` + ``pROC::ci.auc``, which is
    what the deepfake study's R analysis reported. Agrees with `pROC` to
    ~1e-9 on that study's test-set predictions.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted score (usually probability) for the positive class.
    positive_label : optional
        Which label is the positive class. Inferred for 0/1 or boolean
        labels; required otherwise. See :func:`classification_metrics`.
    alpha : float, default 0.05
        1 - confidence level. The default gives a 95% interval.

    Returns
    -------
    auc, lower, upper : float
        Bounds are clipped to ``[0, 1]``, as `pROC` does.
    """
    positive = _positive_mask(y_true, positive_label)
    aucs, cov = _delong_auc_cov(np.asarray(y_score, dtype=float), positive)
    auc = float(aucs[0])
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, float(np.clip(auc - z * se, 0.0, 1.0)), float(np.clip(auc + z * se, 0.0, 1.0))


def delong_roc_test(
    y_true,
    y_score_a,
    y_score_b,
    positive_label=None,
    y_true_b=None,
) -> pd.Series:
    """DeLong's test for a difference between two ROC curves.

    Replicates ``pROC::roc.test(..., method = "delong")`` in both of the
    forms: paired (two models scored on the *same* samples, e.g. random
    forest vs SVM on the test set) and unpaired (the same model on two
    disjoint samples, e.g. posed vs spontaneous expressions).

    Parameters
    ----------
    y_true : array-like
        True labels for ``y_score_a`` -- and for ``y_score_b`` too, unless
        ``y_true_b`` is given.
    y_score_a, y_score_b : array-like
        Positive-class scores of the two curves being compared.
    positive_label : optional
        Which label is the positive class. See
        :func:`classification_metrics`.
    y_true_b : array-like, optional
        True labels for ``y_score_b``. Supplying this switches the test to
        the **unpaired** form, which ignores any correlation between the
        curves because the two samples are different. Leave it out for the
        paired form.

    Returns
    -------
    pd.Series
        ``auc_a``, ``auc_b``, ``difference`` (a - b), ``z``, ``p_value``,
        and ``paired``.
    """
    if y_true_b is None:
        positive = _positive_mask(y_true, positive_label)
        scores = np.vstack([
            np.asarray(y_score_a, dtype=float),
            np.asarray(y_score_b, dtype=float),
        ])
        aucs, cov = _delong_auc_cov(scores, positive)
        variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        paired = True
    else:
        auc_a, var_a = _auc_and_variance(y_true, y_score_a, positive_label)
        auc_b, var_b = _auc_and_variance(y_true_b, y_score_b, positive_label)
        aucs = np.array([auc_a, auc_b])
        variance = var_a + var_b
        paired = False

    difference = float(aucs[0] - aucs[1])
    if variance <= 0:
        z, p_value = (0.0, 1.0) if difference == 0 else (np.inf * np.sign(difference), 0.0)
    else:
        z = difference / float(np.sqrt(variance))
        p_value = float(2 * stats.norm.sf(abs(z)))

    return pd.Series({
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "difference": difference,
        "z": float(z),
        "p_value": float(p_value),
        "paired": paired,
    })


def _auc_and_variance(y_true, y_score, positive_label):
    positive = _positive_mask(y_true, positive_label)
    aucs, cov = _delong_auc_cov(np.asarray(y_score, dtype=float), positive)
    return float(aucs[0]), float(cov[0, 0])


# --------------------------------------------------------------------------
# Confusion-matrix statistics
# --------------------------------------------------------------------------

def _positive_mask(y_true, positive_label) -> np.ndarray:
    y_true = np.asarray(y_true)
    if positive_label is None:
        classes = set(pd.unique(y_true).tolist())
        if classes <= {0, 1} or classes <= {False, True}:
            positive_label = True if classes <= {False, True} else 1
        else:
            raise ValueError(
                f"positive_label is required for labels {sorted(map(str, classes))}. "
                "Sensitivity, specificity and ROC-AUC are all defined relative "
                "to one class, so there is no safe default here -- pass the "
                "class you treat as the target (e.g. positive_label='fake' "
                "for deepfake detection)."
            )
    return y_true == positive_label


def confusion_matrix_frame(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Confusion matrix laid out the way ``caret::confusionMatrix`` prints it.

    Rows are predictions, columns are the reference (true) labels.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    labels : list, optional
        Label order. Defaults to the sorted union of both inputs.

    Returns
    -------
    pd.DataFrame
        Counts, with index named ``"Prediction"`` and columns named
        ``"Reference"``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)

    table = pd.crosstab(
        pd.Series(y_pred, name="Prediction"),
        pd.Series(y_true, name="Reference"),
        dropna=False,
    )
    return table.reindex(index=labels, columns=labels, fill_value=0)


def classification_metrics(
    y_true,
    y_pred,
    y_score=None,
    positive_label=None,
    alpha: float = 0.05,
    labels=None,
) -> pd.Series:
    """Classification statistics for a fitted classifier's predictions.

    Handles two classes and more. With more than two classes (or a 2-D
    ``y_score`` with more than two columns) the multiclass statistics
    described under *Returns* are reported instead; the binary path is
    unchanged.

    Reproduces ``caret::confusionMatrix(..., positive=...)`` plus
    ``pROC::auc``/``ci.auc``. Verified against the deepfake study's printed
    R output: fed the random forest's published confusion matrix it returns
    accuracy 0.6809, CI (0.5767, 0.7733), P[Acc > NIR] 0.000294, kappa
    0.3617, sensitivity 0.8085 and specificity 0.5532 -- R's numbers to
    every digit R printed.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    y_score : array-like, optional
        Binary: predicted probability of the positive class (1-D), or a
        2-D ``(n_samples, 2)`` array of class probabilities in ``labels``
        order. Multiclass: a 2-D ``(n_samples, n_classes)`` array of class
        probabilities in ``labels`` order, e.g. ``model.predict_proba(X)``.
        Without it the ROC-AUC fields are ``NaN``; everything else is
        unaffected.
    positive_label : optional
        Binary only. Which label counts as positive. Inferred for 0/1 and
        boolean labels; **required** for any other labelling, because
        sensitivity and specificity are meaningless without it and guessing
        would silently swap them. Must be left out with more than two
        classes, where the per-class columns give each class's one-vs-rest
        figures.
    alpha : float, default 0.05
        1 - confidence level for the intervals.
    labels : list, optional
        The full set of classes, in the column order of a 2-D ``y_score``.
        Defaults to the sorted classes found in ``y_true`` and ``y_pred``,
        which matches a scikit-learn model's ``classes_`` only when every
        class appears; pass ``model.classes_`` to be safe.

    Returns
    -------
    pd.Series
        **Binary:** ``n``, ``accuracy``, ``accuracy_ci_lower``/``_upper``
        (Clopper-Pearson exact, matching R's ``binom.test``),
        ``no_information_rate``, ``p_acc_gt_nir`` (one-sided exact
        binomial), ``kappa``, ``mcnemar_p`` (with continuity correction),
        ``sensitivity``, ``specificity``, ``ppv``, ``npv``,
        ``prevalence``, ``detection_rate``, ``detection_prevalence``,
        ``balanced_accuracy``, ``f1``, ``mcc``, ``roc_auc``,
        ``roc_auc_ci_lower``/``_upper`` (DeLong).

        **Multiclass:** ``n``, ``accuracy`` and its interval,
        ``no_information_rate``, ``p_acc_gt_nir``, ``kappa``, ``mcc``,
        ``balanced_accuracy`` (mean per-class recall), ``macro_f1``,
        ``roc_auc`` (macro one-vs-rest: the unweighted mean of the
        per-class AUCs, as ``roc_auc_score(multi_class="ovr")``), and for
        each class ``c``: ``precision_<c>``, ``recall_<c>``, ``f1_<c>``,
        ``roc_auc_<c>`` and ``roc_auc_<c>_ci_lower``/``_upper``. Each
        per-class AUC is an ordinary binary AUC (class ``c`` against the
        rest), so its DeLong interval is valid; the macro AUC has no
        interval. A class absent from ``y_true`` has ``NaN`` AUCs, and
        then so does the macro AUC. McNemar's test is binary and omitted.

    Notes
    -----
    ``mcc`` is an addition, not a replication: R reported Cohen's kappa,
    which is also here. The two answer the same question and usually agree
    closely on balanced data like this.

    For pairwise contrasts between classes with DeLong tests, fit one
    binary model per pair on the data from :func:`pairwise_contrasts`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true has {len(y_true)} entries but y_pred has {len(y_pred)}."
        )

    classes = _class_list(y_true, y_pred, labels)
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if y_score.ndim == 2:
            if y_score.shape[1] != len(classes):
                raise ValueError(
                    f"y_score has {y_score.shape[1]} columns but there are "
                    f"{len(classes)} classes {classes}. Pass labels= in the "
                    "column order of y_score (e.g. model.classes_)."
                )
        elif y_score.ndim != 1:
            raise ValueError(f"y_score must be 1-D or 2-D, got {y_score.ndim}-D.")

    if len(classes) > 2:
        if positive_label is not None:
            raise ValueError(
                f"positive_label={positive_label!r} was given, but there are "
                f"{len(classes)} classes. Leave it out: the multiclass result "
                "has one-vs-rest recall, precision and AUC for every class. "
                "For a binary contrast, use pairwise_contrasts."
            )
        if y_score is not None and y_score.ndim == 1:
            raise ValueError(
                "With more than two classes y_score must be 2-D, one column "
                "per class (e.g. model.predict_proba(X))."
            )
        return _multiclass_metrics(y_true, y_pred, y_score, classes, alpha)

    if y_score is not None and y_score.ndim == 2:
        if positive_label is None:
            # Raises for non-0/1 labels, exactly as the 1-D path does.
            _positive_mask(y_true, positive_label)
            positive_label = classes[-1]
        if positive_label not in classes:
            raise ValueError(f"positive_label={positive_label!r} is not one of {classes}.")
        y_score = y_score[:, classes.index(positive_label)]

    positive_true = _positive_mask(y_true, positive_label)
    positive_pred = _positive_mask(y_pred, positive_label)

    n = len(y_true)
    tp = int(np.sum(positive_true & positive_pred))
    tn = int(np.sum(~positive_true & ~positive_pred))
    fp = int(np.sum(~positive_true & positive_pred))
    fn = int(np.sum(positive_true & ~positive_pred))
    correct = tp + tn

    accuracy = correct / n
    # R: binom.test(correct, n)$conf.int -- Clopper-Pearson, not Wilson.
    ci_lower, ci_upper = stats.binomtest(correct, n).proportion_ci(1 - alpha, "exact")
    # R: the largest class share of the *reference*, tested one-sided and
    # exactly, not with a normal approximation.
    _, counts = np.unique(y_true, return_counts=True)
    nir = float(counts.max() / n)
    p_acc_gt_nir = float(stats.binomtest(correct, n, nir, alternative="greater").pvalue)

    # R's mcnemar.test defaults to `correct = TRUE`, but applies the
    # continuity correction only when the two discordant counts differ --
    # equal counts give chi2 = 0 and p = 1, not the 0.87 that subtracting
    # 1 from a zero difference would produce.
    discordant = fp + fn
    if discordant == 0:
        mcnemar_p = float("nan")
    else:
        correction = 1 if fp != fn else 0
        chi2 = (abs(fp - fn) - correction) ** 2 / discordant
        mcnemar_p = float(stats.chi2.sf(chi2, df=1))

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = (
        2 * ppv * sensitivity / (ppv + sensitivity)
        if np.isfinite([ppv, sensitivity]).all() and (ppv + sensitivity) > 0
        else float("nan")
    )

    if y_score is None:
        auc = auc_lower = auc_upper = float("nan")
    else:
        auc, auc_lower, auc_upper = roc_auc_delong_ci(
            y_true, y_score, positive_label=positive_label, alpha=alpha
        )

    return pd.Series({
        "n": n,
        "accuracy": accuracy,
        "accuracy_ci_lower": float(ci_lower),
        "accuracy_ci_upper": float(ci_upper),
        "no_information_rate": nir,
        "p_acc_gt_nir": p_acc_gt_nir,
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcnemar_p": mcnemar_p,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "prevalence": (tp + fn) / n,
        "detection_rate": tp / n,
        "detection_prevalence": (tp + fp) / n,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": f1,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": auc,
        "roc_auc_ci_lower": auc_lower,
        "roc_auc_ci_upper": auc_upper,
    })


def _class_list(y_true, y_pred, labels) -> list:
    """Return the classes, in ``labels`` order or sorted as scikit-learn sorts them."""
    if labels is not None:
        classes = list(labels)
        unknown = (set(y_true.tolist()) | set(y_pred.tolist())) - set(classes)
        if unknown:
            raise ValueError(
                f"Labels {sorted(map(str, unknown))} appear in the data but not in "
                f"labels={classes}."
            )
        return classes
    return np.unique(np.concatenate([y_true, y_pred])).tolist()


def _multiclass_metrics(y_true, y_pred, y_score, classes, alpha) -> pd.Series:
    n = len(y_true)
    correct = int(np.sum(y_true == y_pred))
    ci_lower, ci_upper = stats.binomtest(correct, n).proportion_ci(1 - alpha, "exact")
    _, counts = np.unique(y_true, return_counts=True)
    nir = float(counts.max() / n)

    out = {
        "n": n,
        "accuracy": correct / n,
        "accuracy_ci_lower": float(ci_lower),
        "accuracy_ci_upper": float(ci_upper),
        "no_information_rate": nir,
        "p_acc_gt_nir": float(stats.binomtest(correct, n, nir, alternative="greater").pvalue),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=classes)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    per_class = {}
    recalls, f1s, aucs = [], [], []
    for k, cls in enumerate(classes):
        is_true, is_pred = y_true == cls, y_pred == cls
        tp = int(np.sum(is_true & is_pred))
        precision = tp / is_pred.sum() if is_pred.any() else float("nan")
        recall = tp / is_true.sum() if is_true.any() else float("nan")
        # 2TP / (2TP + FP + FN): the harmonic mean of precision and recall,
        # and 0 (not NaN) for a class that occurs but is never predicted.
        denominator = is_true.sum() + is_pred.sum()
        f1 = 2 * tp / denominator if denominator else float("nan")
        if y_score is not None and is_true.any() and (~is_true).any():
            auc, lower, upper = roc_auc_delong_ci(
                is_true, y_score[:, k], positive_label=True, alpha=alpha
            )
        else:
            auc = lower = upper = float("nan")
        per_class.update({
            f"precision_{cls}": precision,
            f"recall_{cls}": recall,
            f"f1_{cls}": f1,
            f"roc_auc_{cls}": auc,
            f"roc_auc_{cls}_ci_lower": lower,
            f"roc_auc_{cls}_ci_upper": upper,
        })
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc)

    # Means over the classes that occur; a class only named in `labels`
    # has nothing to average.
    out["balanced_accuracy"] = float(np.nanmean(recalls))
    out["macro_f1"] = float(np.nanmean(f1s))
    out["roc_auc"] = float(np.mean(aucs)) if y_score is not None else float("nan")
    out.update(per_class)
    return pd.Series(out)


def pairwise_contrasts(data: pd.DataFrame, label_col: str, pairs=None) -> dict:
    """Binary subsets of a multiclass dataset, one per pair of classes.

    DeLong's tests are binary, so a multiclass study reports its
    pairwise contrasts (e.g. happy-sad, happy-angry, sad-angry) by fitting
    and evaluating a binary model on each subset. Works on frame-level
    data (labels constant within each video) or a video-level table alike;
    group ids carried in ``data`` stay with their rows.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain ``label_col``.
    label_col : str
        Column holding the class labels.
    pairs : list of (label, label), optional
        Which contrasts to build, in that order. Defaults to every pair of
        classes, in sorted order.

    Returns
    -------
    dict
        ``(class_a, class_b) -> DataFrame`` holding only the rows of those
        two classes (a copy, with the original index).

    Examples
    --------
    >>> for (a, b), subset in pairwise_contrasts(frames, "emotion").items():
    ...     result = cross_validate_pipeline(subset, "emotion", "participant")
    ...     print(a, b, result.oof_metrics())  # doctest: +SKIP
    """
    if label_col not in data:
        raise KeyError(f"data has no column {label_col!r}")
    classes = np.unique(data[label_col].dropna()).tolist()
    if pairs is None:
        from itertools import combinations

        pairs = list(combinations(classes, 2))
    out = {}
    for pair in pairs:
        pair = tuple(pair)
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"Each pair must name two different classes, got {pair!r}.")
        missing = [c for c in pair if c not in classes]
        if missing:
            raise ValueError(f"{missing} not found in {label_col!r}; classes are {classes}.")
        out[pair] = data.loc[data[label_col].isin(pair)].copy()
    return out


def _positive_proba(model, X, positive_label) -> np.ndarray:
    """Predicted probability of the positive class from a fitted model."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{type(model).__name__} has no predict_proba, so ROC statistics "
            "cannot be computed. Fit an SVM with probability=True, or use "
            "facedyn.classifiers.make_classifier."
        )
    classes = list(model.classes_)
    if positive_label is None:
        if set(classes) <= {0, 1} or set(classes) <= {False, True}:
            positive_label = classes[-1]
        else:
            raise ValueError(
                f"positive_label is required for classes {classes}: pass the "
                "class you treat as the target (e.g. positive_label='fake' "
                "for deepfake detection)."
            )
    if positive_label not in classes:
        raise ValueError(
            f"positive_label={positive_label!r} is not one of the model's "
            f"classes {classes}."
        )
    return model.predict_proba(X)[:, classes.index(positive_label)]


def evaluate_models(
    models: dict,
    X,
    y,
    positive_label=None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Score several fitted models on the same held-out data.

    Parameters
    ----------
    models : dict of str to fitted estimator
        As returned by :func:`facedyn.classifiers.fit_classifiers`.
    X : pd.DataFrame or array-like
        Held-out features, transformed by the *fitted* pipeline stages --
        never refitted on the test set.
    y : array-like
        Held-out labels.
    positive_label : optional
        Which label counts as positive. See :func:`classification_metrics`.
        Leave it out for models with more than two classes, which are
        scored with the multiclass statistics.
    alpha : float, default 0.05
        1 - confidence level for the intervals.

    Returns
    -------
    pd.DataFrame
        One row per model, indexed by model name, with the columns of
        :func:`classification_metrics`.
    """
    rows = {}
    for name, model in models.items():
        y_pred = model.predict(X)
        classes = list(getattr(model, "classes_", []))
        if len(classes) > 2:
            if not hasattr(model, "predict_proba"):
                raise TypeError(f"{type(model).__name__} has no predict_proba.")
            rows[name] = classification_metrics(
                y, y_pred, model.predict_proba(X), positive_label=positive_label,
                alpha=alpha, labels=classes,
            )
            continue
        y_score = _positive_proba(model, X, positive_label)
        label = positive_label
        if label is None:
            label = list(model.classes_)[-1]
        rows[name] = classification_metrics(
            y, y_pred, y_score, positive_label=label, alpha=alpha
        )
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def cross_validate_grouped(
    model,
    X,
    y,
    groups=None,
    cv=None,
    scoring="roc_auc",
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Cross-validate a model, keeping grouped observations in one fold.

    Defaults to :class:`facedyn.splitting.RepeatedStratifiedGroupKFold` at
    5 folds x 3 repeats, matching R's ``trainControl(method="repeatedcv",
    number=5, repeats=3)``.

    **Use this only on a feature table nothing has been fitted to.** If
    ``X`` came out of fitted steps (normalisation, NMF, imputation and,
    especially, supervised feature selection such as Boruta) run on all
    videos, those steps have seen every test fold and the score is
    leaky. Start from frame-level data with
    :func:`facedyn.crossval.cross_validate_pipeline` instead, which refits
    every step inside each fold and returns out-of-fold predictions.

    **Pass ``groups``.** When one participant contributes several videos,
    ungrouped folds let the model learn that person's face and test on it,
    which usually inflates the score. On matched-pairs data the error can
    run the other way: each test video's near-identical twin, carrying the
    *opposite* label, sits in training and inverts predictions. In the
    deepfake study's training set (real videos paired with fakes made from
    them), a random forest on all 108 features scored 0.394 ROC-AUC
    ungrouped against 0.640 grouped; the problem was noticed because eight
    *random* features also landed below 0.5, and nothing does that by
    accident. Omitting ``groups`` here warns and falls back to ungrouped
    folds rather than silently producing such a number.

    Parameters
    ----------
    model : estimator
        Anything with `fit`/`predict`, typically from
        :func:`facedyn.classifiers.make_classifier`.
    X : pd.DataFrame or array-like
        Feature columns only.
    y : array-like
        Labels.
    groups : array-like, optional
        Group ids from :func:`facedyn.splitting.groups_from_column`
        (repeated measures: one group per participant),
        :func:`facedyn.splitting.pair_groups` or
        :func:`facedyn.splitting.pair_groups_from_filenames` (matched pairs).
    cv : cross-validator, optional
        Overrides the default splitter, and then ``n_splits``,
        ``n_repeats`` and ``random_state`` are ignored.
    scoring : str or list of str, default "roc_auc"
        Any scikit-learn scoring name(s).
    n_splits, n_repeats : int
        Folds per repeat and number of repeats for the default splitter.
    random_state : int, optional
        Seeds the default splitter's shuffling.
    n_jobs : int, optional
        Parallelism across folds.

    Returns
    -------
    pd.DataFrame
        One row per fold, one column per metric (named as passed in
        ``scoring``). ``.mean()`` gives the summary; the spread across
        folds is worth looking at, since it is wide on data this size.
    """
    if groups is None:
        warnings.warn(
            "cross_validate_grouped was called without groups, so folds are "
            "not group-aware. If your data has repeated measures (several "
            "videos per participant) or matched pairs, splitting them across "
            "folds can inflate the score or deflate it -- in one matched-pairs "
            "dataset it pushed ROC-AUC below chance. "
            "Build groups with facedyn.splitting.groups_from_column (one "
            "group per participant) or facedyn.splitting.pair_groups.",
            UserWarning,
            stacklevel=2,
        )
        from sklearn.model_selection import RepeatedStratifiedKFold

        cv = cv or RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    elif cv is None:
        cv = RepeatedStratifiedGroupKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    scoring_list = [scoring] if isinstance(scoring, str) else list(scoring)
    results = cross_validate(
        model, X, y, groups=groups, cv=cv, scoring=scoring_list, n_jobs=n_jobs
    )
    return pd.DataFrame(
        {name: results[f"test_{name}"] for name in scoring_list}
    ).rename_axis("fold")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _group_name(values) -> str:
    """Legend name for a group of labels: the label, or "rest" if mixed."""
    present = pd.unique(values)
    if len(present) == 0:
        return ""
    return str(present[0]) if len(present) == 1 else "rest"


def _require_matplotlib(function_name: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            f"{function_name} requires matplotlib. Install with: pip install facedyn[viz]"
        ) from e
    return plt


def _model_colour(name: str, index: int) -> str:
    return MODEL_COLOURS.get(name, _FALLBACK_COLOURS[index % len(_FALLBACK_COLOURS)])


def _model_label(name: str) -> str:
    return CLASSIFIER_LABELS.get(name, name)


_LINESTYLES = ["-", "--", ":", "-."]


def _finish_roc_axes(ax, title):
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, frameon=False)


def _draw_ovr_curves(ax, y, proba, classes, colours, linestyles, prefix=""):
    """One one-vs-rest ROC curve per class present in ``y``."""
    y = np.asarray(y)
    for k, cls in enumerate(classes):
        is_class = y == cls
        if not is_class.any() or is_class.all():
            continue  # no curve without both a class and a rest
        fpr, tpr, _ = roc_curve(is_class, proba[:, k])
        auc = roc_auc_score(is_class, proba[:, k])
        ax.plot(
            fpr, tpr, linewidth=2, color=colours[k], linestyle=linestyles[k],
            label=f"{prefix}{cls} vs rest (AUC = {auc:.3f})",
        )


def plot_roc_curves(
    models: dict,
    X,
    y,
    positive_label=None,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """ROC curves for several fitted models on one test set.

    Binary models get one curve each. A model with more than two classes
    gets one one-vs-rest curve per class: coloured by class when it is the
    only model, otherwise coloured by model with a line style per class.
    For out-of-fold probabilities rather than fitted models, use
    :func:`plot_roc_ovr`.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    models : dict of str to fitted estimator
        Keys are used for the legend; the four names in
        :data:`facedyn.classifiers.CLASSIFIER_NAMES` get fixed colours
        (:data:`MODEL_COLOURS`).
    X, y : array-like
        Held-out features and labels.
    positive_label : optional
        Binary only. Which label counts as positive. See
        :func:`classification_metrics`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    save_path : str or pathlib.Path, optional
        Filename to save the figure to. Not saved if ``None``.
    output_dir : str or pathlib.Path, default "."
        Directory `save_path` is written into, created if needed.
    dpi : int, default 300
        Resolution for raster formats.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_roc_curves")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    multiclass = False
    for index, (name, model) in enumerate(models.items()):
        classes = list(getattr(model, "classes_", []))
        if len(classes) > 2:
            multiclass = True
            proba = model.predict_proba(X)
            if len(models) == 1:
                colours = [_FALLBACK_COLOURS[k % len(_FALLBACK_COLOURS)]
                           for k in range(len(classes))]
                styles, prefix = ["-"] * len(classes), ""
            else:
                colours = [_model_colour(name, index)] * len(classes)
                styles = [_LINESTYLES[k % len(_LINESTYLES)] for k in range(len(classes))]
                prefix = f"{_model_label(name)}: "
            _draw_ovr_curves(ax, y, proba, classes, colours, styles, prefix)
            continue
        score = _positive_proba(model, X, positive_label)
        label = positive_label if positive_label is not None else classes[-1]
        fpr, tpr, _ = roc_curve(np.asarray(y) == label, score)
        auc = roc_auc_score(np.asarray(y) == label, score)
        ax.plot(
            fpr, tpr, linewidth=2, color=_model_colour(name, index),
            label=f"{_model_label(name)} (AUC = {auc:.3f})",
        )

    _finish_roc_axes(
        ax,
        "One-vs-rest ROC curves on the held-out test set" if multiclass
        else "ROC curves on the held-out test set",
    )
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_roc_ovr(
    y_true,
    y_score,
    labels=None,
    ax=None,
    title: str = "One-vs-rest ROC curves",
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """One-vs-rest ROC curves, one per class, from predicted probabilities.

    The multiclass counterpart of :func:`plot_roc_curves` for when there is
    no single fitted model, e.g. the pooled out-of-fold predictions of
    :func:`facedyn.crossval.cross_validate_pipeline`. A class missing from
    ``y_true`` (or making up all of it) has no curve.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_score : array-like of shape (n_samples, n_classes) or pd.DataFrame
        Class probabilities, columns in ``labels`` order.
    labels : list, optional
        Class of each column of ``y_score``. Defaults to the sorted classes
        of ``y_true``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    title : str
        Axes title.
    save_path, output_dir, dpi
        As :func:`plot_roc_curves`.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> oof = result.oof[result.oof["repeat"] == 0]  # doctest: +SKIP
    >>> plot_roc_ovr(oof["y_true"], oof[[f"proba_{c}" for c in result.classes]],
    ...              labels=result.classes)  # doctest: +SKIP
    """
    plt = _require_matplotlib("plot_roc_ovr")
    y_true = np.asarray(y_true)
    proba = np.asarray(y_score, dtype=float)
    classes = list(labels) if labels is not None else np.unique(y_true).tolist()
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(
            f"y_score must be 2-D with one column per class ({len(classes)}: "
            f"{classes}); got shape {proba.shape}."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    colours = [_FALLBACK_COLOURS[k % len(_FALLBACK_COLOURS)] for k in range(len(classes))]
    _draw_ovr_curves(ax, y_true, proba, classes, colours, ["-"] * len(classes))
    _finish_roc_axes(ax, title)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    normalize: bool | str = False,
    annotate: str = "count",
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Heatmap of :func:`confusion_matrix_frame`, annotated with counts.

    Works for any number of classes. Rows are predictions and columns the
    true (reference) class, as ``caret`` prints it.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    labels : list, optional
        Label order. Defaults to sorted.
    normalize : {False, True, "prediction", "reference"}, default False
        Shade cells by proportion instead of raw count. ``True`` or
        ``"prediction"`` divides each row (prediction) by its total, giving
        how often each prediction is right (precision on the diagonal).
        ``"reference"`` divides each column (true class) by its total,
        giving where each true class ends up (recall on the diagonal), the
        usual view for showing which classes are confused.
    annotate : {"count", "proportion", "both"}, default "count"
        What to print in each cell. ``"proportion"`` and ``"both"`` need
        ``normalize``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    save_path, output_dir, dpi
        As :func:`plot_roc_curves`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_confusion_matrix")
    if normalize not in (False, True, "prediction", "reference"):
        raise ValueError(
            f"normalize must be False, True, 'prediction' or 'reference', got {normalize!r}."
        )
    if annotate not in ("count", "proportion", "both"):
        raise ValueError(f"annotate must be 'count', 'proportion' or 'both', got {annotate!r}.")
    if annotate != "count" and normalize is False:
        raise ValueError(f"annotate={annotate!r} needs normalize to be set.")

    table = confusion_matrix_frame(y_true, y_pred, labels=labels)
    counts = table.to_numpy()
    if normalize is False:
        shading = counts.astype(float)
    else:
        axis = 0 if normalize == "reference" else 1
        totals = counts.sum(axis=axis, keepdims=True)
        # An empty row/column (a class never predicted, or absent) shades as 0.
        shading = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    image = ax.imshow(shading, cmap="Blues")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")
    threshold = 0.5 if normalize is not False else (shading.max() / 2 if shading.size else 0)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            text = {
                "count": f"{counts[i, j]:d}",
                "proportion": f"{shading[i, j]:.2f}",
                "both": f"{counts[i, j]:d}\n({shading[i, j]:.2f})",
            }[annotate]
            ax.text(
                j, i, text, ha="center", va="center",
                color="white" if shading[i, j] > threshold else "black",
            )
    if normalize is not False:
        image.set_clim(0, 1)
    ax.figure.colorbar(image, ax=ax, shrink=0.8)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_probability_distributions(
    y_true,
    y_score,
    positive_label=None,
    threshold: float = 0.5,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Predicted-probability densities by true class.

    The class-separability view: two well-separated humps mean the model
    is confident and right, one overlapping blob means it is guessing --
    information an AUC alone hides.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted probability of the positive class.
    positive_label : optional
        Which label counts as positive. See :func:`classification_metrics`.
    threshold : float, default 0.5
        Where to draw the decision-threshold line.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    save_path, output_dir, dpi
        As :func:`plot_roc_curves`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_probability_distributions")
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    positive = _positive_mask(y_true, positive_label)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))

    grid = np.linspace(0, 1, 512)
    for mask, colour, style in [
        (positive, _CLASS_COLOURS["positive"], "--"),
        (~positive, _CLASS_COLOURS["negative"], "-"),
    ]:
        values = y_score[mask]
        name = _group_name(y_true[mask])
        if len(values) < 2 or np.allclose(values, values[0]):
            # gaussian_kde needs spread; fall back to a histogram.
            ax.hist(values, bins=20, range=(0, 1), density=True, alpha=0.4,
                    color=colour, label=name)
            continue
        density = stats.gaussian_kde(values)(grid)
        ax.fill_between(grid, density, color=colour, alpha=0.4)
        ax.plot(grid, density, color="black", linewidth=0.8, linestyle=style, label=name)

    ax.axvline(threshold, linestyle=":", color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability of the positive class")
    ax.set_ylabel("Density")
    ax.set_title("Predicted probabilities by true class")
    ax.legend(frameon=False)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_decision_boundary(
    model,
    X: pd.DataFrame,
    y,
    features: tuple[str, str],
    positive_label=None,
    prototype=None,
    resolution: int = 200,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Decision regions over two features, with the data on top.

    Everything except the two plotted features is held fixed, so this is a
    two-dimensional slice through the model, not a picture of the whole
    decision function. Points can and do fall on the "wrong" side of the
    drawn boundary because their other features differ from the slice.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    model : fitted estimator
        Must expose `predict_proba` and accept a DataFrame with the same
        columns it was fitted on.
    X : pd.DataFrame
        Feature table the model was fitted on (or its test-set
        equivalent). Column names must match.
    y : array-like
        Labels for the scattered points.
    features : (str, str)
        The two columns to vary, x-axis first.
    positive_label : optional
        Which label counts as positive. See :func:`classification_metrics`.
    prototype : pd.Series, optional
        Values to hold the remaining features at. Defaults to the
        **column medians**. The R code used the first row of its test set
        instead; pass ``X.iloc[0]`` to reproduce that exactly. The median
        is the default here because a row-order-dependent slice changes
        whenever the data is re-sorted.
    resolution : int, default 200
        Grid points per axis.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    save_path, output_dir, dpi
        As :func:`plot_roc_curves`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_decision_boundary")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("plot_decision_boundary needs a DataFrame so it can hold "
                        "the unplotted features fixed by name.")
    x_name, y_name = features
    for name in features:
        if name not in X.columns:
            raise KeyError(f"{name!r} is not a column of X.")

    if prototype is None:
        prototype = X.median(numeric_only=True)

    x_grid = np.linspace(X[x_name].min(), X[x_name].max(), resolution)
    y_grid = np.linspace(X[y_name].min(), X[y_name].max(), resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)

    grid = pd.DataFrame(
        np.tile(prototype[X.columns].to_numpy(dtype=float), (xx.size, 1)),
        columns=X.columns,
    )
    grid[x_name] = xx.ravel()
    grid[y_name] = yy.ravel()

    probability = _positive_proba(model, grid, positive_label).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    ax.contourf(
        xx, yy, probability, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.45,
    )
    ax.contour(xx, yy, probability, levels=[0.5], colors="black", linewidths=1.2)

    positive = _positive_mask(np.asarray(y), positive_label)
    for mask, colour, marker in [
        (positive, _CLASS_COLOURS["positive"], "^"),
        (~positive, _CLASS_COLOURS["negative"], "o"),
    ]:
        if not mask.any():
            continue
        ax.scatter(
            X.loc[mask.tolist(), x_name], X.loc[mask.tolist(), y_name],
            c=colour, marker=marker, edgecolor="black", linewidth=0.4, s=32,
            label=_group_name(np.asarray(y)[mask]),
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title("Decision regions (other features held fixed)")
    ax.legend(frameon=False, loc="best", fontsize=9)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_accuracy_comparison(
    results: pd.DataFrame,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Accuracy per model with exact CIs and the no-information-rate line.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    results : pd.DataFrame
        Output of :func:`evaluate_models`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    save_path, output_dir, dpi
        As :func:`plot_roc_curves`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_accuracy_comparison")
    required = {"accuracy", "accuracy_ci_lower", "accuracy_ci_upper"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing {sorted(missing)}; pass evaluate_models output.")

    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 * len(results) + 2, 4.5))

    names = list(results.index)
    accuracy = results["accuracy"].to_numpy(dtype=float)
    lower = accuracy - results["accuracy_ci_lower"].to_numpy(dtype=float)
    upper = results["accuracy_ci_upper"].to_numpy(dtype=float) - accuracy
    colours = [_model_colour(name, i) for i, name in enumerate(names)]

    ax.bar(range(len(names)), accuracy, color=colours, alpha=0.85, width=0.6)
    ax.errorbar(
        range(len(names)), accuracy, yerr=[lower, upper],
        fmt="none", ecolor="black", capsize=4, linewidth=1,
    )
    if "no_information_rate" in results:
        nir = float(results["no_information_rate"].iloc[0])
        ax.axhline(nir, linestyle="--", color="red", alpha=0.6)
        ax.annotate(f"NIR ({nir:.2f})", xy=(len(names) - 0.5, nir),
                    xytext=(0, 4), textcoords="offset points",
                    ha="right", color="red", fontsize=9)

    ax.set_xticks(range(len(names)), [_model_label(name) for name in names],
                  rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Test-set accuracy (95% exact CIs)")
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax
