"""Feature effects: first-order accumulated local effects (ALE).

Importance scores (Gini, permutation, Boruta) say *which* features a model
uses. They don't say *how* the prediction changes as a feature changes,
e.g. whether happier videos have a higher or lower AU12 kurtosis. Effect
curves answer that.

Partial dependence (PDP) is the familiar effect curve, but it averages the
model over the whole marginal distribution of the other features. When
features are correlated, it therefore evaluates the model on combinations
that never occur in the data. Time-series features of the same AU are
strongly correlated (``sd``, ``range`` and ``iqr``; ``acf1`` and
``diff1_sd``), so facedyn uses accumulated local effects instead (Apley &
Zhu, 2020). ALE only compares predictions *within* small intervals of the
feature, over the observations that actually fall in each interval, so the
other features keep realistic values.

The estimator, from Apley & Zhu (2020, eq. 15-16), for feature ``j``:

1. Split the observed range of ``x_j`` into ``K`` intervals at its
   quantiles, ``z_0 < z_1 < ... < z_K``.
2. For each observation in interval ``k``, the local effect is
   ``f(x with x_j = z_k) - f(x with x_j = z_{k-1})``.
3. Average the local effects within each interval and accumulate them:
   ``g(z_k) = sum over k' <= k`` of those averages, with ``g(z_0) = 0``.
4. Centre, so the effect has mean zero over the data: subtract
   ``sum_k n_k * (g(z_{k-1}) + g(z_k)) / 2 / n``.

The result reads as "how much the prediction moves, relative to its
average, as the feature moves from one value to another, holding the
other features at the values they really take alongside it". For a
classifier it is in units of predicted probability, one curve per class.

This module implements that definition directly. It doesn't transcribe or
depend on ``ALEPlot``, ``alibi`` or ``PyALE``.

References
----------
Apley, D. W., & Zhu, J. (2020). Visualizing the effects of predictor
variables in black box supervised learning models. *Journal of the Royal
Statistical Society Series B*, 82(4), 1059-1086.
https://doi.org/10.1111/rssb.12377
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from facedyn._plot_utils import save_figure
from facedyn.au_labels import humanise_feature_label

__all__ = [
    "ale_grid",
    "feature_effect_ale",
    "feature_effect_ale_cv",
    "plot_feature_effect",
    "summarise_feature_effects",
]

_CURVE_COLUMNS = ["feature", "class", "value", "effect", "n"]
_FALLBACK_COLOURS = ["#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#56B4E9"]


# --------------------------------------------------------------------------
# Core estimator
# --------------------------------------------------------------------------


def ale_grid(values, grid_size: int = 20) -> np.ndarray:
    """Interval edges for ALE: the unique empirical quantiles of ``values``.

    Parameters
    ----------
    values : array-like
        Observed values of one feature. Missing values are ignored.
    grid_size : int, default 20
        Number of intervals ``K`` before ties are merged. Features with
        few distinct values get fewer intervals.

    Returns
    -------
    np.ndarray
        Sorted, unique edges ``z_0 < ... < z_K``.

    Raises
    ------
    ValueError
        If ``values`` has fewer than two distinct finite values.
    """
    if int(grid_size) < 1:
        raise ValueError(f"grid_size must be at least 1, got {grid_size}.")
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    # Empirical quantiles (R's type 1), so every edge is an observed value
    # and a discrete feature isn't given edges between its levels.
    probs = np.linspace(0, 1, int(grid_size) + 1)
    edges = np.unique(np.quantile(x, probs, method="inverted_cdf")) if len(x) else x
    if len(edges) < 2:
        raise ValueError(
            f"An ALE curve needs a feature with at least two distinct values; got {len(edges)}."
        )
    return edges


def _ale_curve(predict, X: pd.DataFrame, feature, edges: np.ndarray, as_array: bool):
    """Centred ALE of one feature at ``edges``.

    Returns ``(effect, counts)``: ``effect`` has shape ``(len(edges),
    n_outputs)`` and ``counts`` shape ``(len(edges),)``, where
    ``counts[k]`` is the number of observations in the interval ending at
    ``edges[k]`` (so ``counts[0] == 0``). Observations outside
    ``[edges[0], edges[-1]]`` or missing the feature are not used. An
    interval with no observations contributes no change (a flat segment).
    """
    n_bins = len(edges) - 1
    x = X[feature].to_numpy(dtype=float)
    inside = np.isfinite(x) & (x >= edges[0]) & (x <= edges[-1])
    if not inside.any():
        raise ValueError(f"No observations of {feature!r} fall within the ALE grid.")
    rows = X.loc[inside]
    x = x[inside]

    # Interval k covers (z_{k-1}, z_k]; the first also includes z_0.
    bin_idx = np.clip(np.searchsorted(edges, x, side="left"), 1, n_bins) - 1

    lower = rows.copy()
    upper = rows.copy()
    lower[feature] = edges[bin_idx]
    upper[feature] = edges[bin_idx + 1]
    if as_array:
        lower, upper = lower.to_numpy(), upper.to_numpy()
    local = np.asarray(predict(upper), dtype=float) - np.asarray(predict(lower), dtype=float)
    if local.ndim == 1:
        local = local[:, None]

    counts = np.bincount(bin_idx, minlength=n_bins)
    sums = np.zeros((n_bins, local.shape[1]))
    np.add.at(sums, bin_idx, local)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], 0.0)

    accumulated = np.vstack([np.zeros((1, local.shape[1])), np.cumsum(means, axis=0)])
    # Mean of the piecewise-linear curve over the data: each interval's
    # observations sit, on average, at the interval's midpoint height.
    midpoints = (accumulated[:-1] + accumulated[1:]) / 2
    centre = (counts[:, None] * midpoints).sum(axis=0) / counts.sum()
    return accumulated - centre, np.concatenate([[0], counts])


def _output_spec(model, classes, all_classes=None):
    """Which model outputs to trace, and how to compute them.

    Returns ``(labels, predict)``: ``predict(X)`` gives an
    ``(n, len(labels))`` array. Classifiers give probabilities, aligned to
    ``all_classes`` (or ``model.classes_``) so a class the model never saw
    has probability 0. Models without ``predict_proba`` give ``predict``
    under the single label ``None``.
    """
    if not hasattr(model, "predict_proba"):
        if classes is not None:
            raise ValueError("classes= needs a classifier with predict_proba.")
        return [None], lambda X: np.asarray(model.predict(X), dtype=float)

    model_classes = list(model.classes_)
    known = list(all_classes) if all_classes is not None else model_classes
    if classes is None:
        labels = [known[-1]] if len(known) == 2 else known
    else:
        labels = [classes] if np.isscalar(classes) else list(classes)
        missing = [c for c in labels if c not in known]
        if missing:
            raise ValueError(f"classes {missing} are not among the model's classes {known}.")

    columns = [model_classes.index(c) if c in model_classes else None for c in labels]

    def predict(X):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        out = np.zeros((proba.shape[0], len(labels)))
        for j, col in enumerate(columns):
            if col is not None:
                out[:, j] = proba[:, col]
        return out

    return labels, predict


def _as_frame(X):
    if isinstance(X, pd.DataFrame):
        return X, False
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {array.shape}.")
    return pd.DataFrame(array), True


def _as_list(features) -> list:
    if isinstance(features, str | int | np.integer):
        return [features]
    return list(features)


def _curve_frame(feature, labels, edges, effect, counts) -> pd.DataFrame:
    parts = [
        pd.DataFrame(
            {
                "feature": feature,
                "class": label,
                "value": edges,
                "effect": effect[:, j],
                "n": counts,
            }
        )
        for j, label in enumerate(labels)
    ]
    return pd.concat(parts, ignore_index=True)[_CURVE_COLUMNS]


def feature_effect_ale(
    model,
    X,
    features,
    grid_size: int = 20,
    classes=None,
    grid: dict | None = None,
) -> pd.DataFrame:
    """First-order accumulated local effects (ALE) of one or more features.

    See the module docstring for the definition and why facedyn uses ALE
    rather than partial dependence. The curves are centred, so each has
    mean zero over the data: read them as how far the prediction sits
    above or below its average at each value of the feature.

    Parameters
    ----------
    model : fitted estimator
        A classifier with ``predict_proba`` (curves are in probability
        units) or any model with ``predict``.
    X : pd.DataFrame or array-like
        The data the effect is estimated over, in the columns and order
        the model was fitted on. Held-out data shows how the model
        behaves on videos it didn't see; training data is also valid. A
        NumPy array is passed to the model as an array, and its features
        are named by column position.
    features : str, int or list
        The feature(s) to trace.
    grid_size : int, default 20
        Number of quantile intervals per feature (fewer if the feature
        has ties). More intervals give a finer but noisier curve.
    classes : label or list, optional
        Classifiers only: whose probability to trace. Defaults to the
        positive (last) class for a binary model, since the other class's
        curve is its mirror image, and to every class otherwise.
    grid : dict, optional
        ``feature -> edges`` to use instead of the quantile grid, e.g. to
        put several models on a common grid. Observations outside the
        edges are not used.

    Returns
    -------
    pd.DataFrame
        Long format, one row per feature, class and grid value: columns
        ``feature``, ``class`` (``None`` for models without
        ``predict_proba``), ``value`` (the grid edge), ``effect`` (centred
        ALE) and ``n`` (observations in the interval ending at ``value``;
        0 for the first edge).

    Notes
    -----
    For a multiclass model the per-class curves sum to zero at every
    value, because the probabilities sum to one. An increase in one class
    is always an equal decrease shared among the others.

    References
    ----------
    Apley, D. W., & Zhu, J. (2020). Visualizing the effects of predictor
    variables in black box supervised learning models. *JRSS-B*, 82(4),
    1059-1086.

    Examples
    --------
    >>> curves = feature_effect_ale(model, X_test, ["kurtosis_smth_AU12_r"])  # doctest: +SKIP
    >>> curves.pivot(index="value", columns="class", values="effect")  # doctest: +SKIP
    """
    X, as_array = _as_frame(X)
    labels, predict = _output_spec(model, classes)
    grid = grid or {}
    parts = []
    for feature in _as_list(features):
        if feature not in X.columns:
            raise KeyError(f"X has no feature {feature!r}.")
        edges = (
            np.unique(np.asarray(grid[feature], dtype=float))
            if feature in grid
            else ale_grid(X[feature], grid_size)
        )
        if len(edges) < 2:
            raise ValueError(f"The grid for {feature!r} needs at least two distinct edges.")
        effect, counts = _ale_curve(predict, X, feature, edges, as_array)
        parts.append(_curve_frame(feature, labels, edges, effect, counts))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Across cross-validation folds
# --------------------------------------------------------------------------


def _fold_items(result, repeat):
    if not result.fold_models:
        raise ValueError(
            "The result has no fitted fold models. Rerun cross_validate_pipeline "
            "with keep_models=True."
        )
    items = sorted(result.fold_models.items())
    if repeat is not None:
        repeats = [repeat] if np.isscalar(repeat) else list(repeat)
        items = [(key, fold) for key, fold in items if key[0] in repeats]
        if not items:
            raise ValueError(f"No folds for repeat {repeat!r}.")
    return items


def _fold_data(fold, X):
    """Return the table one fold's effect is estimated over, or None if unusable."""
    if X is None:
        return fold["X_test"]
    if any(col not in X.columns for col in fold["features"]):
        return None
    return X[fold["features"]]


def feature_effect_ale_cv(
    result,
    features,
    X: pd.DataFrame | None = None,
    grid_size: int = 20,
    classes=None,
    repeat=None,
) -> pd.DataFrame:
    """ALE curves for every fold model of a cross-validation run.

    Each fold's model is traced on a grid shared by all folds, so the
    curves can be averaged and their spread shows how stable the effect
    is. This ties interpretation to the same folds that produced the
    out-of-fold score.

    Parameters
    ----------
    result : CVPipelineResult
        From :func:`facedyn.crossval.cross_validate_pipeline` with
        ``keep_models=True``.
    features : str or list of str
        Feature column name(s), as in ``result.selected_features``.
    X : pd.DataFrame, optional
        A video-level feature table to estimate every fold's effect over.
        It must contain each fold model's features; folds it doesn't cover
        are skipped. By default each fold uses its own test videos, which
        the model never saw. The pooled test values across folds set the
        shared grid.
    grid_size : int, default 20
        Number of quantile intervals of the shared grid.
    classes : label or list, optional
        As :func:`feature_effect_ale`, over ``result.classes``.
    repeat : int or list of int, optional
        Restrict to these CV repeats. Defaults to all.

    Returns
    -------
    pd.DataFrame
        The columns of :func:`feature_effect_ale` plus ``repeat`` and
        ``fold``. ``n`` counts the fold's own observations per interval;
        an interval with none is flat in that fold's curve. A fold whose
        selector dropped a feature has no rows for it: the model ignores
        it, so its effect is exactly zero there. Compare the number of
        folds per feature with ``result.feature_frequency()``.

    Raises
    ------
    ValueError
        If the result has no fold models, or no fold used a feature.
    """
    items = _fold_items(result, repeat)
    feature_list = _as_list(features)
    parts = []
    for feature in feature_list:
        using = []
        for key, fold in items:
            if feature not in fold["features"]:
                continue
            data = _fold_data(fold, X)
            if data is not None and len(data):
                using.append((key, fold, data))
        if not using:
            raise ValueError(
                f"No fold model used {feature!r} (or X lacks its features), so there "
                "is no effect to trace."
            )
        values = X[feature] if X is not None else pd.concat([data[feature] for _, _, data in using])
        edges = ale_grid(values, grid_size)
        for (rep, fold_id), fold, data in using:
            labels, predict = _output_spec(fold["model"], classes, result.classes)
            effect, counts = _ale_curve(predict, data, feature, edges, as_array=False)
            parts.append(
                _curve_frame(feature, labels, edges, effect, counts).assign(
                    repeat=rep, fold=fold_id
                )
            )
    curves = pd.concat(parts, ignore_index=True)
    return curves[["repeat", "fold", *_CURVE_COLUMNS]]


def summarise_feature_effects(curves: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of per-fold ALE curves at each grid value.

    Parameters
    ----------
    curves : pd.DataFrame
        Output of :func:`feature_effect_ale_cv`.

    Returns
    -------
    pd.DataFrame
        One row per ``feature``, ``class`` and ``value``, with ``mean``,
        ``sd``, ``min``, ``max`` of ``effect`` and ``n_folds`` (the folds
        whose model used the feature).
    """
    keys = ["feature", "class", "value"]
    summary = (
        curves.groupby(keys, sort=False, dropna=False)["effect"]
        .agg(["mean", "std", "min", "max", "size"])
        .rename(columns={"std": "sd", "size": "n_folds"})
        .reset_index()
    )
    summary["sd"] = summary["sd"].fillna(0.0)
    return summary


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------


def _require_matplotlib(function_name: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            f"{function_name} requires matplotlib. Install with: pip install facedyn[viz]"
        ) from e
    return plt


def _is_cv_result(obj) -> bool:
    return hasattr(obj, "fold_models") and hasattr(obj, "oof")


def _rug_values(model_or_result, X, feature, is_cv, repeat) -> np.ndarray:
    """Return the observed values of ``feature``, for the rug under each panel."""
    if X is not None:
        frame = _as_frame(X)[0]
        return frame[feature].to_numpy(dtype=float) if feature in frame.columns else np.array([])
    if not is_cv:
        return np.array([])
    # Every repeat tests every video once, so one repeat's test values are the data.
    items = _fold_items(model_or_result, repeat)
    first = items[0][0][0]
    parts = [
        fold["X_test"][feature]
        for key, fold in items
        if key[0] == first and feature in fold["X_test"].columns
    ]
    return pd.concat(parts).to_numpy(dtype=float) if parts else np.array([])


def plot_feature_effect(
    model_or_result,
    X=None,
    features=None,
    grid_size: int = 20,
    classes=None,
    repeat=None,
    band: str | None = "sd",
    show_folds: bool = False,
    humanise: bool = True,
    ncols: int = 3,
    axes=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot ALE curves, one panel per feature, with a rug of the data.

    Pass a fitted model with ``X`` to plot one model's effects, or a
    :class:`~facedyn.crossval.CVPipelineResult` (``keep_models=True``) to
    plot the mean curve across fold models, with a band showing how much
    the folds agree. Each panel's title says in how many folds the
    feature was used, since a fold that dropped it contributes no curve.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    model_or_result : fitted estimator or CVPipelineResult
        What to explain.
    X : pd.DataFrame or array-like, optional
        Required for a single model (see :func:`feature_effect_ale`).
        Optional for a CV result (see :func:`feature_effect_ale_cv`).
    features : str or list
        The feature(s) to plot. For a CV result it defaults to the six
        most often selected features.
    grid_size, classes
        As :func:`feature_effect_ale`.
    repeat : int or list of int, optional
        CV result only: restrict to these repeats.
    band : {"sd", "range", None}, default "sd"
        CV result only: shade mean ± 1 SD across folds, the min-max range,
        or nothing.
    show_folds : bool, default False
        CV result only: also draw each fold's curve as a thin line.
    humanise : bool, default True
        Title panels with :func:`~facedyn.au_labels.humanise_feature_label`,
        e.g. ``"AU12 - Lip Corner Puller — kurtosis"``.
    ncols : int, default 3
        Panels per row.
    axes : array of matplotlib.axes.Axes, optional
        Axes to draw on, at least one per feature.
    save_path, output_dir, dpi
        Filename, directory and resolution for saving. Nothing is saved
        unless ``save_path`` is given.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> result = cross_validate_pipeline(frames, "emotion", "participant",
    ...                                  keep_models=True)  # doctest: +SKIP
    >>> plot_feature_effect(result, features=["kurtosis_smth_AU12_r"])  # doctest: +SKIP
    """
    plt = _require_matplotlib("plot_feature_effect")
    if band not in ("sd", "range", None):
        raise ValueError(f"band must be 'sd', 'range' or None, got {band!r}.")
    is_cv = _is_cv_result(model_or_result)

    if features is None:
        if not is_cv:
            raise ValueError("Pass features= to say which features to plot.")
        features = model_or_result.feature_frequency().index[:6].tolist()
    feature_list = _as_list(features)

    if is_cv:
        curves = feature_effect_ale_cv(
            model_or_result, feature_list, X=X, grid_size=grid_size, classes=classes, repeat=repeat
        )
        summary = summarise_feature_effects(curves)
        n_total = len(_fold_items(model_or_result, repeat))
    else:
        if X is None:
            raise ValueError("X is required to plot the effects of a single model.")
        curves = feature_effect_ale(
            model_or_result, X, feature_list, grid_size=grid_size, classes=classes
        )
        summary = curves.rename(columns={"effect": "mean"})

    labels = list(pd.unique(curves["class"]))
    colours = {
        label: _FALLBACK_COLOURS[k % len(_FALLBACK_COLOURS)] for k, label in enumerate(labels)
    }

    if axes is None:
        ncols = max(1, min(int(ncols), len(feature_list)))
        nrows = int(np.ceil(len(feature_list) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = np.ravel(np.asarray(axes, dtype=object))
    if len(axes_flat) < len(feature_list):
        raise ValueError(f"Need {len(feature_list)} axes, got {len(axes_flat)}.")
    fig = axes_flat[0].figure

    # A grid can have more axes than features; the spare axes are hidden below.
    for ax, feature in zip(axes_flat, feature_list, strict=False):
        for label in labels:
            part = summary[(summary["feature"] == feature) & _matches(summary["class"], label)]
            name = "effect" if label is None else str(label)
            colour = colours[label]
            if is_cv and show_folds:
                fold_part = curves[
                    (curves["feature"] == feature) & _matches(curves["class"], label)
                ]
                for _, one in fold_part.groupby(["repeat", "fold"]):
                    ax.plot(one["value"], one["effect"], color=colour, alpha=0.25, lw=0.7)
            if is_cv and band is not None:
                low, high = (
                    (part["mean"] - part["sd"], part["mean"] + part["sd"])
                    if band == "sd"
                    else (part["min"], part["max"])
                )
                ax.fill_between(part["value"], low, high, color=colour, alpha=0.2, lw=0)
            ax.plot(part["value"], part["mean"], color=colour, lw=2, label=name)

        rug = _rug_values(model_or_result, X, feature, is_cv, repeat)
        rug = rug[np.isfinite(rug)]
        if len(rug):
            ax.plot(
                rug,
                np.zeros_like(rug),
                "|",
                color="0.3",
                alpha=0.5,
                ms=8,
                transform=ax.get_xaxis_transform(),
            )
        ax.axhline(0, color="0.6", lw=0.8, ls="--")

        title = humanise_feature_label(str(feature)) if humanise else str(feature)
        if is_cv:
            n_used = int(
                curves.loc[curves["feature"] == feature, ["repeat", "fold"]]
                .drop_duplicates()
                .shape[0]
            )
            title += f"\n(used in {n_used} of {n_total} folds)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("ALE (Δ predicted probability)" if labels != [None] else "ALE")
        if ax is axes_flat[0] and (len(labels) > 1 or labels[0] is not None):
            ax.legend(fontsize=8, frameon=False, title="Class" if labels != [None] else None)

    for ax in axes_flat[len(feature_list) :]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, save_path, output_dir, dpi)
    return fig


def _matches(column: pd.Series, label) -> pd.Series:
    return column.isna() if label is None else column == label
