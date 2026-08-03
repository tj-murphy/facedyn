"""Choosing :class:`~facedyn.feature_selection.BorutaSelector`'s
``selection_threshold`` from data held out of training.

``selection_threshold`` is the most consequential setting in the package
and, until this module, the one with the least backing. Every number
arguing about it had been read off the *held-out test set* -- which is not
information a user starting out has, and spending it on a tuning decision
costs the study its one honest estimate of generalisation.

This module is what let the shipped default move from ``0.8`` to ``0.5``
on evidence that never touched that test set. It also produced a result
worth knowing before trusting it: on Paper 1's data the tuned answer was
lower still (``0.05``), because the selector inside the loop sees 296
videos where the final fit sees 370 and therefore confirms fewer features
at any given threshold. **A threshold tuned on inner splits is biased low
when transplanted into the full fit**, by roughly the amount that
shrinkage costs. See ``PIPELINE.md`` "Step 8b".

This module picks the threshold the way it should be picked: on a
validation fold carved out of the **training** set.

::

    464 videos
     |-- 370 train --+-- ~296 sub-train  -> BorutaSelector (20 seeds)
     |               +--  ~74 validation -> sweep thresholds, pick one
     |                                      -> refit on all 370
     +--  94 test  --------------------- touched once, at the end

**Never pass the test set to anything here.** A threshold chosen against
it is a threshold fitted to it, and every number reported from that set
afterwards is optimistic by an unknown amount.

Two entry points:

- :func:`threshold_sweep` is the cheap core. It takes an **already
  fitted** selector and re-thresholds its ``selection_frequency_``, so the
  expensive part (the 20 Boruta runs) happens once no matter how many
  candidate thresholds are scored.
- :func:`tune_selection_threshold` is the orchestrator: it makes the inner
  splits, fits a selector *per split* (a selector fitted on everything has
  already seen every validation fold, which is the bias this exists to
  avoid), aggregates the curves and applies a selection rule.

None of this replicates the R analysis -- R ran Boruta once at
``set.seed(12345)`` and never tuned a threshold, because with one run
there is no selection frequency to threshold. It is validated
structurally and by measurement instead of against R output.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.utils.validation import check_is_fitted

from facedyn._plot_utils import save_figure
from facedyn.classifiers import make_classifier
from facedyn.evaluation import _delong_auc_cov, _positive_mask, _positive_proba
from facedyn.feature_selection import BorutaSelector
from facedyn.splitting import RepeatedStratifiedGroupKFold

__all__ = [
    "ThresholdTuningResult",
    "threshold_grid",
    "threshold_sweep",
    "tune_selection_threshold",
    "plot_threshold_sweep",
]


def threshold_grid(n_boruta_repeats: int) -> np.ndarray:
    """Candidate thresholds worth scoring for a given number of Boruta runs.

    A selection frequency is a count of confirming runs over the number of
    runs, so with ``n_boruta_repeats`` runs it can only take the values
    ``0, 1/n, ..., 1``. Anything between two of those keeps exactly the
    same features as the lower one, so a finer grid costs fits and buys
    nothing.

    The grid starts at ``0.0``, which keeps every feature Boruta was given
    regardless of whether it was ever confirmed. That point is the "no
    selection at all" baseline, and having it on the curve is what makes
    the curve answer *whether* selection helps, not just how much of it to
    do.

    Parameters
    ----------
    n_boruta_repeats : int
        ``n_repeats`` of the :class:`~facedyn.feature_selection.BorutaSelector`
        whose frequencies are being thresholded.

    Returns
    -------
    np.ndarray
        ``n_boruta_repeats + 1`` values from 0.0 to 1.0 inclusive.
    """
    if n_boruta_repeats < 1:
        raise ValueError(f"n_boruta_repeats must be >= 1, got {n_boruta_repeats}")
    return np.linspace(0.0, 1.0, n_boruta_repeats + 1)


def _default_model(random_state):
    """The classifier used to *measure* thresholds, not to report results.

    Deliberately **not** the ``"paper1"`` preset every other entry point
    defaults to. That preset fixes ``max_features=4``, a value chosen for
    an eight-feature table, and a fixed ``mtry`` is not a fair comparator
    along the one axis this module varies: at a loose threshold the forest
    would draw 4 of 108 features per split, while at a strict one it draws
    4 of 4 -- no feature subsampling at all. The comparison would then be
    partly between two different models rather than between two feature
    sets. scikit-learn's ``max_features="sqrt"`` scales with the set size
    and keeps the sweep like-for-like; the tree count is raised to Paper
    1's 500 because a noisy curve is the thing most likely to mislead here.
    """
    return make_classifier(
        "random_forest", preset="default", n_estimators=500, random_state=random_state
    )


def _features_at(selector: BorutaSelector, threshold: float) -> list[str]:
    """Columns `selector` would keep at `threshold`, in its own column order."""
    frequency = selector.selection_frequency_
    return [f for f in selector.feature_columns_ if frequency[f] >= threshold]


def _score_features(
    model,
    features: list[str],
    X_sub,
    y_sub,
    X_val,
    y_val,
    scoring: str,
    positive_label,
) -> tuple[float, float]:
    """Fit `model` on the selected features and score it on the validation fold.

    Returns ``(score, standard_error)``. The standard error is DeLong's
    analytic one for ``scoring="roc_auc"`` and ``nan`` otherwise -- an
    arbitrary sklearn scorer has no variance estimate attached to it.
    """
    fitted = clone(model).fit(X_sub[features], y_sub)

    if scoring != "roc_auc":
        return float(get_scorer(scoring)(fitted, X_val[features], y_val)), float("nan")

    y_score = _positive_proba(fitted, X_val[features], positive_label)
    label = positive_label
    if label is None:
        label = list(fitted.classes_)[-1]
    aucs, cov = _delong_auc_cov(
        np.asarray(y_score, dtype=float), _positive_mask(y_val, label)
    )
    return float(aucs[0]), float(np.sqrt(cov[0, 0]))


def threshold_sweep(
    selector: BorutaSelector,
    X_sub,
    y_sub,
    X_val,
    y_val,
    model=None,
    scoring: str = "roc_auc",
    positive_label=None,
    thresholds=None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Score every candidate ``selection_threshold`` on a validation fold.

    Re-thresholding a fitted selector's ``selection_frequency_`` is free,
    so the cost here is one classifier fit per *distinct feature set*, not
    per threshold -- adjacent thresholds routinely select the same columns
    and the fit is reused.

    **`X_val` must be data the selector never saw.** `selector` is
    expected to have been fitted on `X_sub` alone; if it was fitted on
    both, every score here is optimistic in a way that grows with the
    number of features kept -- that is, precisely along the axis being
    swept -- and the sweep will lean toward thresholds that are too loose.
    :func:`tune_selection_threshold` arranges this correctly.

    Parameters
    ----------
    selector : BorutaSelector
        Already fitted, on `X_sub` only.
    X_sub, y_sub : pd.DataFrame, array-like
        The sub-training fold the selector was fitted on. Classifiers are
        fitted here.
    X_val, y_val : pd.DataFrame, array-like
        The held-out validation fold. Classifiers are scored here.
    model : estimator, optional
        The classifier used to *measure* a threshold; it is discarded
        afterwards and is not the model you report. Defaults to a
        500-tree random forest at scikit-learn's own ``max_features``,
        **not** the ``"paper1"`` preset -- see :func:`_default_model` for
        why a fixed ``mtry`` would bias the sweep toward strict thresholds.
    scoring : str, default "roc_auc"
        ``"roc_auc"`` is scored through DeLong, which also supplies a
        standard error. Any other value is looked up with
        :func:`sklearn.metrics.get_scorer` and gets ``nan`` for its error.
    positive_label : optional
        Which label counts as positive. Inferred for 0/1 and boolean
        labels; required otherwise. Paper 1 used ``"fake"``.
    thresholds : array-like, optional
        Candidates to score. Defaults to :func:`threshold_grid` for the
        selector's number of runs.
    random_state : int, optional
        Seeds the default classifier. Ignored when `model` is given.

    Returns
    -------
    pd.DataFrame
        One row per threshold: ``threshold``, ``n_features``, ``features``
        (a tuple), ``score``, ``score_se``. A threshold that selects
        nothing scores ``nan`` with ``n_features = 0`` rather than raising
        -- an empty selection is a real, informative outcome at the strict
        end of the grid.
    """
    check_is_fitted(selector, "selection_frequency_")
    if model is None:
        model = _default_model(random_state)
    if thresholds is None:
        thresholds = threshold_grid(len(selector.runs_))

    scored: dict[frozenset, tuple[float, float]] = {}
    rows = []
    for threshold in np.asarray(thresholds, dtype=float):
        features = _features_at(selector, threshold)
        if not features:
            score, score_se = float("nan"), float("nan")
        else:
            key = frozenset(features)
            if key not in scored:
                scored[key] = _score_features(
                    model, features, X_sub, y_sub, X_val, y_val, scoring, positive_label
                )
            score, score_se = scored[key]
        rows.append({
            "threshold": float(threshold),
            "n_features": len(features),
            "features": tuple(features),
            "score": score,
            "score_se": score_se,
        })
    return pd.DataFrame(rows)


def _inner_splits(X, y, groups, n_splits: int, n_repeats: int, random_state):
    """One sub-train/validation split per repeat, grouped and stratified.

    Takes the first fold of each repeat of
    :class:`~facedyn.splitting.RepeatedStratifiedGroupKFold`. Scoring all
    `n_splits` folds of a repeat would multiply the Boruta fits by
    `n_splits` for far less than that in extra information, and taking one
    fold per repeat keeps the splits independently randomised rather than
    being complementary slices of a single partition.
    """
    cv = RepeatedStratifiedGroupKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    return [
        split
        for index, split in enumerate(cv.split(X, y, groups))
        if index % n_splits == 0
    ]


def _aggregate_curve(per_split: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-split sweeps into one curve with a spread."""
    rows = []
    for threshold, group in per_split.groupby("threshold", sort=True):
        scores = group["score"].to_numpy(dtype=float)
        finite = scores[np.isfinite(scores)]
        if len(finite) > 1:
            # Spread across splits: the honest error bar, since it carries
            # both the classifier's variance and the selector's.
            score_se = float(finite.std(ddof=1) / np.sqrt(len(finite)))
        elif len(finite) == 1:
            # A single split has no across-split spread, so fall back to
            # DeLong's analytic error for that one validation fold. It is
            # the narrower of the two and does not see selection variance.
            score_se = float(group.loc[np.isfinite(scores), "score_se"].iloc[0])
        else:
            score_se = float("nan")
        rows.append({
            "threshold": float(threshold),
            "n_splits_scored": int(len(finite)),
            "n_features_mean": float(group["n_features"].mean()),
            "n_features_min": int(group["n_features"].min()),
            "n_features_max": int(group["n_features"].max()),
            "score_mean": float(finite.mean()) if len(finite) else float("nan"),
            "score_se": score_se,
        })
    return pd.DataFrame(rows)


def _select_threshold(curve: pd.DataFrame, rule: str) -> float:
    """Apply `rule` to an aggregated curve and return the chosen threshold."""
    if rule not in ("one_se", "best"):
        raise ValueError(f"rule must be 'one_se' or 'best', got {rule!r}")

    usable = curve[np.isfinite(curve["score_mean"])]
    if usable.empty:
        raise ValueError(
            "No threshold could be scored -- every candidate selected zero "
            "features on every split. Boruta confirmed nothing here, so there "
            "is no threshold to choose; check that the features carry signal "
            "for this target before tuning."
        )

    best_row = usable.loc[usable["score_mean"].idxmax()]
    if rule == "best":
        return float(best_row["threshold"])

    se = float(best_row["score_se"])
    if not np.isfinite(se):
        warnings.warn(
            "The one-standard-error rule needs a standard error, and none "
            "could be computed (a single split with a non-ROC metric has no "
            "variance estimate). Falling back to the best-scoring threshold, "
            "which on a small validation fold will partly be fitting noise.",
            UserWarning,
            stacklevel=3,
        )
        return float(best_row["threshold"])

    # The largest threshold -- i.e. the fewest features -- whose mean score
    # is still within one standard error of the best.
    within = usable[usable["score_mean"] >= best_row["score_mean"] - se]
    return float(within["threshold"].max())


@dataclass
class ThresholdTuningResult:
    """What :func:`tune_selection_threshold` measured and chose.

    Attributes
    ----------
    best_threshold_ : float
        The chosen ``selection_threshold``.
    best_score_, best_n_features_ : float
        Mean validation score and mean feature count at that threshold.
        The score is **not** an estimate of held-out performance -- it was
        used to make the choice. Score the refitted pipeline on the test
        set for that.
    curve_ : pd.DataFrame
        One row per threshold, averaged across splits: ``threshold``,
        ``n_splits_scored``, ``n_features_mean``/``_min``/``_max``,
        ``score_mean``, ``score_se``.
    per_split_ : pd.DataFrame
        The unaggregated sweeps, with a ``split`` column -- worth looking
        at, since splits disagreeing about the shape of the curve is the
        signal that the validation folds are too small to be decisive.
    selector_ : BorutaSelector or None
        Fitted on **all** the training data at ``best_threshold_``, ready
        to ``transform``. ``None`` when ``refit=False``.
    rule_, scoring_ : str
        How the choice was made, carried for reporting.
    """

    best_threshold_: float
    best_score_: float
    best_n_features_: float
    curve_: pd.DataFrame = field(repr=False)
    per_split_: pd.DataFrame = field(repr=False)
    selector_: BorutaSelector | None = field(default=None, repr=False)
    rule_: str = "one_se"
    scoring_: str = "roc_auc"

    @property
    def selected_columns_(self) -> list[str]:
        """Columns the refitted selector keeps. Needs ``refit=True``."""
        if self.selector_ is None:
            raise AttributeError(
                "selected_columns_ needs the refitted selector, which "
                "tune_selection_threshold(refit=False) did not build. Either "
                "refit, or fit a BorutaSelector yourself with "
                "selection_threshold=result.best_threshold_."
            )
        return self.selector_.selected_columns_


def tune_selection_threshold(
    X: pd.DataFrame,
    y,
    groups,
    selector: BorutaSelector | None = None,
    model=None,
    scoring: str = "roc_auc",
    positive_label=None,
    thresholds=None,
    n_splits: int = 5,
    n_repeats: int = 3,
    rule: str = "one_se",
    refit: bool = True,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> ThresholdTuningResult:
    """Choose ``selection_threshold`` on validation folds held out of training.

    Carves `n_repeats` sub-train/validation splits out of the training set
    with :class:`~facedyn.splitting.RepeatedStratifiedGroupKFold`, taking
    the first fold of each repeat so the splits are grouped *and*
    stratified at an ``n_splits``-to-1 ratio (5 folds -> ~80/20, matching
    the outer split). A selector is fitted on each sub-train fold and
    swept with :func:`threshold_sweep` on the matching validation fold.

    **Pass the training set only.** See this module's docstring.

    **This is the expensive function in facedyn.** It fits Boruta
    ``n_repeats`` times, plus once more when ``refit=True``; a default
    20-run fit takes roughly 12-19 minutes on Paper 1's 370x108 table, so
    the defaults here are of the order of an hour. Everything else in the
    tuner is negligible beside that. ``BorutaSelector(importance="gini")``
    is the fast path if you want the shape of the curve before committing
    to the real thing.

    Parameters
    ----------
    X : pd.DataFrame
        One row per group (per video), features already pivoted wide --
        the same shape :class:`~facedyn.feature_selection.BorutaSelector`
        expects. Metadata columns may be present.
    y : array-like
        Training labels.
    groups : array-like
        Group ids from :func:`facedyn.splitting.pair_groups` or
        :func:`facedyn.splitting.pair_groups_from_filenames`.
        **Required.** A matched pair straddling the sub-train/validation
        line puts a near-identical twin carrying the opposite label in
        training, which pushes validation scores below chance -- and every
        point on the curve would be built from those scores.
    selector : BorutaSelector, optional
        An **unfitted** template, cloned and fitted once per split.
        Defaults to ``BorutaSelector(random_state=random_state)``. Set its
        ``importance``, ``n_estimators``, ``n_repeats`` or
        ``feature_columns`` here; its ``selection_threshold`` is ignored,
        being the thing under test.
    model, scoring, positive_label, thresholds
        As :func:`threshold_sweep`.
    n_splits : int, default 5
        Folds the training set is divided into. The validation fold is one
        of them, so this sets the split ratio rather than the number of
        splits scored.
    n_repeats : int, default 3
        How many sub-train/validation splits to score, one per repeat.
        Each costs a full Boruta fit. More than one matters here: a ~74
        video validation fold carries a ROC-AUC interval of roughly +-0.13,
        wide enough that a single split's argmax is largely noise.
    rule : {"one_se", "best"}, default "one_se"
        ``"one_se"`` takes the **largest** threshold -- the fewest
        features -- whose mean score is within one standard error of the
        best, which is `caret`'s ``oneSE`` logic and the right default on a
        curve this noisy. ``"best"`` takes the plain argmax.
    refit : bool, default True
        Refit a selector on all of `X` at the chosen threshold, as
        :class:`~sklearn.model_selection.GridSearchCV` does. Costs one
        more Boruta fit; set ``False`` if you only want the curve.
    random_state : int, optional
        Seeds the splits and the default selector and classifier.
    n_jobs : int, optional
        Parallelism *inside* each Boruta fit (across its runs). Splits are
        fitted one after another, so there is no nested parallelism -- the
        configuration that deadlocked a Jupyter kernel outright, see
        :meth:`~facedyn.feature_selection.BorutaSelector.fit`.

    Returns
    -------
    ThresholdTuningResult

    Examples
    --------
    >>> groups = pair_groups_from_filenames(train_wide)  # doctest: +SKIP
    >>> result = tune_selection_threshold(  # doctest: +SKIP
    ...     train_wide, train_wide["isfakeorreal"], groups,
    ...     positive_label="fake", random_state=0, n_jobs=-1,
    ... )
    >>> result.best_threshold_, result.selected_columns_  # doctest: +SKIP
    """
    if groups is None:
        raise ValueError(
            "tune_selection_threshold requires groups so a matched pair is "
            "never split between the sub-train and validation folds. Build "
            "them with facedyn.splitting.pair_groups (or "
            "pair_groups_from_filenames). Without them the validation scores "
            "the whole curve is made of are not meaningful."
        )
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    template = selector if selector is not None else BorutaSelector(
        random_state=random_state
    )
    if model is None:
        model = _default_model(random_state)

    y = np.asarray(y)
    splits = _inner_splits(X, y, groups, n_splits, n_repeats, random_state)

    frames = []
    for split_index, (sub_idx, val_idx) in enumerate(splits):
        X_sub, X_val = X.iloc[sub_idx], X.iloc[val_idx]
        y_sub, y_val = y[sub_idx], y[val_idx]

        fitted = clone(template)
        if n_jobs is not None:
            fitted.set_params(n_jobs=n_jobs)
        fitted.fit(X_sub, y_sub)

        sweep = threshold_sweep(
            fitted, X_sub, y_sub, X_val, y_val,
            model=model, scoring=scoring, positive_label=positive_label,
            thresholds=thresholds, random_state=random_state,
        )
        sweep.insert(0, "split", split_index)
        frames.append(sweep)

    per_split = pd.concat(frames, ignore_index=True)
    curve = _aggregate_curve(per_split)
    best_threshold = _select_threshold(curve, rule)
    chosen = curve.loc[curve["threshold"] == best_threshold].iloc[0]

    final_selector = None
    if refit:
        final_selector = clone(template)
        final_selector.set_params(selection_threshold=best_threshold)
        if n_jobs is not None:
            final_selector.set_params(n_jobs=n_jobs)
        final_selector.fit(X, y)

    return ThresholdTuningResult(
        best_threshold_=best_threshold,
        best_score_=float(chosen["score_mean"]),
        best_n_features_=float(chosen["n_features_mean"]),
        curve_=curve,
        per_split_=per_split,
        selector_=final_selector,
        rule_=rule,
        scoring_=scoring,
    )


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def _require_matplotlib(function_name: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            f"{function_name} requires matplotlib. Install with: pip install facedyn[viz]"
        ) from e
    return plt


def plot_threshold_sweep(
    result: ThresholdTuningResult,
    show_splits: bool = False,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Validation score against ``selection_threshold``, with the choice marked.

    Requires matplotlib (``pip install facedyn[viz]``).

    The shaded band is one standard error either side of the mean, which
    is also the band the ``"one_se"`` rule selects within: the chosen
    threshold is the rightmost point whose mean sits above the band's
    lower edge at the peak.

    Parameters
    ----------
    result : ThresholdTuningResult
        From :func:`tune_selection_threshold`.
    show_splits : bool, default False
        Also draw each split's own curve. Worth turning on when the splits
        disagree -- a wide fan is the honest picture of how much a
        validation fold this size can settle.
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
    plt = _require_matplotlib("plot_threshold_sweep")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    curve = result.curve_
    threshold = curve["threshold"].to_numpy()
    mean = curve["score_mean"].to_numpy()
    se = curve["score_se"].to_numpy()

    if show_splits:
        for split, group in result.per_split_.groupby("split"):
            ax.plot(
                group["threshold"], group["score"], linewidth=1,
                color="#BBBBBB", zorder=1,
                label="individual splits" if split == 0 else None,
            )

    ax.fill_between(
        threshold, mean - se, mean + se, color="#4C72B0", alpha=0.18,
        linewidth=0, zorder=2, label="±1 standard error",
    )
    ax.plot(
        threshold, mean, marker="o", markersize=4, linewidth=2,
        color="#4C72B0", zorder=3, label=f"mean {result.scoring_}",
    )
    ax.axvline(
        result.best_threshold_, linestyle="--", linewidth=1.2, color="#C44E52",
        zorder=4,
        label=f"chosen: {result.best_threshold_:g} "
              f"({result.best_n_features_:g} features, {result.rule_})",
    )
    if result.scoring_ == "roc_auc":
        ax.axhline(0.5, linestyle=":", linewidth=1, color="#999999", zorder=0)

    counts = ax.twinx()
    counts.plot(
        threshold, curve["n_features_mean"], linewidth=1.2, linestyle="-.",
        color="#55A868", zorder=1,
    )
    counts.set_ylabel("features kept (mean)", color="#55A868")
    counts.tick_params(axis="y", labelcolor="#55A868")
    counts.set_ylim(bottom=0)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("selection_threshold")
    ax.set_ylabel(f"validation {result.scoring_}")
    ax.set_title("Selection threshold against held-out validation score")
    ax.set_zorder(counts.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax
