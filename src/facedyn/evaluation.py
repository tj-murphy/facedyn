"""Classifier evaluation: metrics, DeLong tests, grouped CV, figures.

The metrics replicate what the R analysis reported, function for function:
``caret::confusionMatrix`` for the confusion-matrix statistics and
``pROC::auc``/``ci.auc``/``roc.test`` for the ROC ones. Where R's choice of
estimator is not the obvious one, this module follows R rather than the
obvious one -- the accuracy interval is Clopper-Pearson (not Wilson), the
accuracy-vs-no-information-rate test is an exact one-sided binomial test
(not a normal approximation), McNemar's test carries the continuity
correction, and the ROC interval is DeLong's. Each is verified against the
numbers printed in the paper's own output; see
``tests/validation/test_classifier_metrics_validation.py``.

Cross-validation on matched-pairs data **must** be grouped by pair --
:func:`cross_validate_grouped` is the entry point that makes that the
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

# The paper's Figure 2 palette (Okabe-Ito), keyed by classifier name.
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
    what the R analysis reported. Agrees with `pROC` to ~1e-9 on the
    paper's own test-set predictions.

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
    forms the R analysis uses: paired (two models scored on the *same*
    samples, e.g. random forest vs SVM on the test set) and unpaired (the
    same model on two disjoint samples, e.g. the emotion vs no-emotion
    subsets).

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
                "to one class, so there is no safe default here -- the R "
                "analysis used positive='fake'."
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
) -> pd.Series:
    """Every statistic the R analysis reported for a fitted classifier.

    Reproduces ``caret::confusionMatrix(..., positive=...)`` plus
    ``pROC::auc``/``ci.auc``. Verified against the paper's own printed
    output: fed the random forest's published confusion matrix it returns
    accuracy 0.6809, CI (0.5767, 0.7733), P[Acc > NIR] 0.000294, kappa
    0.3617, sensitivity 0.8085 and specificity 0.5532 -- R's numbers to
    every digit R printed.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    y_score : array-like, optional
        Predicted probability of the positive class. Without it the
        ROC-AUC fields are ``NaN``; everything else is unaffected.
    positive_label : optional
        Which label counts as positive. Inferred for 0/1 and boolean
        labels; **required** for any other labelling, because sensitivity
        and specificity are meaningless without it and guessing would
        silently swap them. Paper 1 used ``"fake"``.
    alpha : float, default 0.05
        1 - confidence level for both intervals.

    Returns
    -------
    pd.Series
        ``n``, ``accuracy``, ``accuracy_ci_lower``/``_upper``
        (Clopper-Pearson exact, matching R's ``binom.test``),
        ``no_information_rate``, ``p_acc_gt_nir`` (one-sided exact
        binomial), ``kappa``, ``mcnemar_p`` (with continuity correction),
        ``sensitivity``, ``specificity``, ``ppv``, ``npv``,
        ``prevalence``, ``detection_rate``, ``detection_prevalence``,
        ``balanced_accuracy``, ``f1``, ``mcc``, ``roc_auc``,
        ``roc_auc_ci_lower``/``_upper`` (DeLong).

    Notes
    -----
    ``mcc`` is an addition, not a replication: R reported Cohen's kappa,
    which is also here. The two answer the same question and usually agree
    closely on balanced data like this.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true has {len(y_true)} entries but y_pred has {len(y_pred)}."
        )

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
                f"positive_label is required for classes {classes}. "
                "Paper 1 used positive_label='fake'."
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

    **Pass ``groups``.** On matched-pairs data, ungrouped folds put a fake
    video in the test fold while its near-identical real twin -- carrying
    the *opposite* label -- trains the model, which inverts predictions
    rather than merely weakening them. Measured on Paper 1's training set,
    a random forest on all 108 features scores 0.394 ROC-AUC ungrouped
    against 0.640 grouped. The problem was noticed because eight *random*
    features also landed below 0.5, and nothing does that by accident.
    Omitting ``groups`` here warns and falls back to ungrouped folds rather
    than silently producing that number.

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
        Group ids from :func:`facedyn.splitting.pair_groups` or
        :func:`facedyn.splitting.pair_groups_from_filenames`.
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
            "not group-aware. If your data has matched pairs (or repeated "
            "measures from one subject), splitting them across folds can push "
            "the score below chance -- on Paper 1's data it cost 0.18 ROC-AUC. "
            "Build groups with facedyn.splitting.pair_groups.",
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
    """ROC curves for several fitted models on one test set (Figure 2B).

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    models : dict of str to fitted estimator
        Keys are used for the legend; the four names in
        :data:`facedyn.classifiers.CLASSIFIER_NAMES` get the paper's own
        colours.
    X, y : array-like
        Held-out features and labels.
    positive_label : optional
        Which label counts as positive. See :func:`classification_metrics`.
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

    for index, (name, model) in enumerate(models.items()):
        score = _positive_proba(model, X, positive_label)
        label = positive_label if positive_label is not None else list(model.classes_)[-1]
        fpr, tpr, _ = roc_curve(np.asarray(y) == label, score)
        auc = roc_auc_score(np.asarray(y) == label, score)
        ax.plot(
            fpr, tpr, linewidth=2, color=_model_colour(name, index),
            label=f"{_model_label(name)} (AUC = {auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    normalize: bool = False,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Heatmap of :func:`confusion_matrix_frame`, annotated with counts.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    labels : list, optional
        Label order. Defaults to sorted.
    normalize : bool, default False
        Shade cells by row proportion instead of raw count. Counts are
        always the printed annotation.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    save_path, output_dir, dpi
        As :func:`plot_roc_curves`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_confusion_matrix")
    table = confusion_matrix_frame(y_true, y_pred, labels=labels)
    counts = table.to_numpy()
    shading = counts / counts.sum(axis=1, keepdims=True) if normalize else counts

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    image = ax.imshow(shading, cmap="Blues")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")
    threshold = shading.max() / 2 if shading.size else 0
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(
                j, i, f"{counts[i, j]:d}", ha="center", va="center",
                color="white" if shading[i, j] > threshold else "black",
            )
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
    """Predicted-probability densities by true class (Figure 2C).

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
        name = str(pd.unique(y_true[mask])[0]) if mask.any() else ""
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
    """Decision regions over two features, with the data on top (Figure 2D).

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
            label=str(pd.unique(np.asarray(y)[mask])[0]),
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
