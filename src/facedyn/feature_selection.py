"""Boruta feature selection: the final method R's pipeline actually used.

`final_analysis_NMF_check.Rmd` also runs `caret::findCorrelation` and
`caret::rfe`, but neither feeds the classifier's feature set -- both are
diagnostic-only, printed and never referenced again. Boruta is the one
method whose output the real analysis actually uses downstream.

**Why this module reports stability across seeds by default.** Paper 1
ran Boruta once, at `set.seed(12345)`, and read 8 confirmed features off
the resulting plot. That run reproduces exactly (3 Confirmed + 6
Tentative, then `TentativeRoughFix` promotes 5 of the 6). But re-running
the identical R code at other seeds returns only 3-5 of those same 8
features, because the candidate features are near-duplicates of one
another -- the AU12 autocorrelation family correlates at up to
``|r| = 0.88`` -- and permutation importance splits credit among
correlated predictors close to arbitrarily. The *family* signal is
stable; which member of the family wins is close to a coin flip.

A single Boruta run therefore produces a confident-looking list that is
substantially noise, with nothing in the output to say so. That is the
trap this module is built to close: :class:`BorutaSelector` repeats the
run across seeds and reports how often each feature survives, and
:func:`correlated_feature_clusters` surfaces the near-duplicate groups
that cause the churn in the first place. See ``PIPELINE.md`` step 8 for
the full investigation and the measured numbers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import binom
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap
from sklearn.utils.validation import check_is_fitted, check_random_state

from facedyn._plot_utils import save_figure

CONFIRMED = "Confirmed"
TENTATIVE = "Tentative"
REJECTED = "Rejected"

# R's Boruta duplicates the shadow block until it holds at least 5
# columns, so that `max(shadowImp)` is never taken over a degenerate
# one- or two-column sample once most features have been rejected.
_MIN_SHADOWS = 5

# Okabe-Ito, consistent with the rest of facedyn's plots.
_DECISION_COLOURS = {
    CONFIRMED: "#009E73",
    TENTATIVE: "#E69F00",
    REJECTED: "#999999",
}
_SHADOW_COLOUR = "#0072B2"


def _resolve_feature_columns(X: pd.DataFrame, feature_columns: list[str] | None) -> list[str]:
    if feature_columns is not None:
        return list(feature_columns)
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


# --------------------------------------------------------------------------
# Importance backends
# --------------------------------------------------------------------------


def oob_permutation_importance(
    X,
    y,
    n_estimators: int = 500,
    max_features="sqrt",
    random_state=None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Out-of-bag permutation importance, scaled by its standard error.

    The Python equivalent of what R's `Boruta` actually calls. Confirmed
    from the R package source (``Boruta:::getImpRfZ``): R's real default
    is ``ranger::ranger(importance="permutation",
    scale.permutation.importance=TRUE)`` -- for each tree, the drop in
    accuracy on *that tree's own* out-of-bag samples when a feature is
    shuffled, averaged over trees and divided by the standard error of
    that average.

    **The per-tree OOB part is the part that matters**, and it is why
    this is implemented directly rather than via
    `sklearn.inspection.permutation_importance`. An earlier version of
    this module approximated R with K-fold cross-validated permutation
    importance on a *refit* forest, scaled by the standard deviation
    across 5 folds. That is a different estimator, not an approximation
    of this one: 5 folds' worth of noise instead of 500 trees', and a
    scale roughly ten times larger. It disagreed with R badly enough to
    look like evidence that permutation importance was the wrong choice.
    It was evidence that the implementation was. Checked against
    `ranger` on the real Paper 1 data, this function's top feature
    scores 3.796 against `ranger`'s 3.819.

    Uses `sklearn.ensemble._forest`'s bootstrap-index helpers, which are
    private but are the only route to per-tree OOB membership; the
    alternative is reimplementing sklearn's bootstrap sampling, which
    would be far more fragile than depending on these two functions.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    n_estimators : int, default 500
        Matches R's real default (`ranger`'s ``num.trees=500`` via
        `Boruta`).
    max_features : default "sqrt"
        sklearn's ``"sqrt"`` and `ranger`'s default ``mtry`` agree --
        both floor the square root of the column count.
    random_state : int or RandomState, optional
    n_jobs : int, optional
        Passed to the forest. Left at ``None`` (single-threaded) when
        :class:`BorutaSelector` is already parallelising over repeats, to
        avoid nested parallelism.

    Returns
    -------
    np.ndarray of shape (n_features,)
        Importance per column. Zero where a feature's accuracy drop has
        no variance across trees (it never changed anything).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        bootstrap=True,
        random_state=random_state,
        n_jobs=n_jobs,
    ).fit(X, y)

    n_samples, n_features = X.shape
    n_bootstrap = _get_n_samples_bootstrap(n_samples, forest.max_samples)
    rng = check_random_state(random_state)

    # Trees are fitted on label indices into `forest.classes_`, so compare
    # against encoded y rather than mapping predictions back each time.
    y_encoded = np.searchsorted(forest.classes_, y)

    drops = np.zeros((len(forest.estimators_), n_features))
    for t, tree in enumerate(forest.estimators_):
        oob = _generate_unsampled_indices(tree.random_state, n_samples, n_bootstrap)
        if oob.size == 0:
            continue
        # Fancy indexing copies, so this block is safe to permute in place.
        X_oob = X[oob]
        y_oob = y_encoded[oob]
        baseline = np.mean(tree.predict(X_oob) == y_oob)
        for j in range(n_features):
            original = X_oob[:, j].copy()
            X_oob[:, j] = original[rng.permutation(oob.size)]
            drops[t, j] = baseline - np.mean(tree.predict(X_oob) == y_oob)
            X_oob[:, j] = original

    mean_drop = drops.mean(axis=0)
    standard_error = drops.std(axis=0, ddof=1) / np.sqrt(drops.shape[0])
    return np.divide(
        mean_drop, standard_error, out=np.zeros_like(mean_drop), where=standard_error > 0
    )


def gini_importance(
    X,
    y,
    n_estimators: int = 500,
    max_features="sqrt",
    random_state=None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Mean decrease in impurity -- the fast alternative importance.

    Not what R uses (see :func:`oob_permutation_importance`), and biased
    toward continuous and high-cardinality features, but roughly two
    orders of magnitude cheaper. Offered for large feature sets or a
    quick first look, via :class:`BorutaSelector`'s ``importance="gini"``.
    """
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    ).fit(np.asarray(X, dtype=float), np.asarray(y))
    return forest.feature_importances_


_IMPORTANCE_BACKENDS = {
    "permutation": oob_permutation_importance,
    "gini": gini_importance,
}


@dataclass
class _ForestImportance:
    """Picklable binding of a backend to its forest settings.

    Deliberately a module-level class rather than a closure over
    `BorutaSelector`: `joblib` ships this to worker processes, and a
    closure would drag the whole selector along with it.
    """

    backend: object
    n_estimators: int
    n_jobs: int | None

    def __call__(self, X, y, random_state=None):
        return self.backend(
            X,
            y,
            n_estimators=self.n_estimators,
            random_state=random_state,
            n_jobs=self.n_jobs,
        )


# --------------------------------------------------------------------------
# The Boruta loop
# --------------------------------------------------------------------------


@dataclass
class BorutaRun:
    """One complete Boruta run: its decisions and its full history.

    `importance_history` uses ``-inf`` for iterations in which a feature
    had already been rejected and so was not evaluated -- matching R's
    own `ImpHistory` sentinel exactly (confirmed against a real R run's
    export; R uses ``-Inf``, not ``NA``, and `Boruta::attStats` filters
    on ``is.finite``). Reductions over this array must therefore mask on
    ``np.isfinite``, not on ``np.isnan``.

    Attributes
    ----------
    feature_names : list of str
    decisions : np.ndarray of object
        Raw decisions, before tentative resolution: one of
        ``"Confirmed"``, ``"Tentative"``, ``"Rejected"`` per feature.
    rough_fixed_decisions : np.ndarray of object
        Decisions after :func:`tentative_rough_fix` -- no ``"Tentative"``
        remains. This is what Paper 1 read its 8 features off.
    importance_history : np.ndarray of shape (n_iterations, n_features)
    shadow_history : np.ndarray of shape (n_iterations, 3)
        Per-iteration ``(min, mean, max)`` over the shadow features.
    hits : np.ndarray of shape (n_features,)
        How many iterations each feature beat the best shadow feature.
    n_iterations : int
    """

    feature_names: list[str]
    decisions: np.ndarray
    rough_fixed_decisions: np.ndarray
    importance_history: np.ndarray
    shadow_history: np.ndarray
    hits: np.ndarray
    n_iterations: int = field(default=0)

    @property
    def shadow_max(self) -> np.ndarray:
        """Per-iteration maximum shadow importance -- Boruta's threshold."""
        return self.shadow_history[:, 2]

    def confirmed(self, rough_fixed: bool = True) -> list[str]:
        """Features decided ``"Confirmed"``, after rough fix by default."""
        decisions = self.rough_fixed_decisions if rough_fixed else self.decisions
        return [f for f, d in zip(self.feature_names, decisions) if d == CONFIRMED]


def _make_shadows(X_active: np.ndarray, rng) -> np.ndarray:
    """Independently permuted copies of the still-relevant features.

    Duplicates the block until it has at least `_MIN_SHADOWS` columns
    *before* permuting, so duplicated columns still get independent
    shuffles -- R does the same, and it matters once few features remain.
    """
    shadows = X_active
    while shadows.shape[1] < _MIN_SHADOWS:
        shadows = np.hstack([shadows, shadows])
    return np.column_stack([rng.permutation(column) for column in shadows.T])


def _do_tests(
    decisions: np.ndarray, hits: np.ndarray, n_iterations: int, alpha: float
) -> np.ndarray:
    """Two-sided binomial test of hit counts against chance, per R's `mcAdj`.

    Under the null that a feature is no better than a shadow, its hit
    count is Binomial(`n_iterations`, 0.5). A feature is confirmed when
    it has significantly *more* hits than chance and rejected when it has
    significantly fewer, with Bonferroni correction over the **total**
    number of attributes -- the constant R's ``p.adjust(..., "bonferroni")``
    applies, since it is handed the full-length p-value vector every
    round.

    This replaces `BorutaPy`'s two-step Benjamini-Hochberg-then-Bonferroni
    correction, which its own docstring notes is less conservative than
    R's, and which was previously documented here as an unfixable
    divergence.
    """
    n_attributes = len(decisions)
    p_confirm = binom.sf(hits - 1, n_iterations, 0.5)  # P(X >= hits)
    p_reject = binom.cdf(hits, n_iterations, 0.5)  # P(X <= hits)

    tentative = decisions == TENTATIVE
    to_confirm = tentative & (np.minimum(p_confirm * n_attributes, 1.0) < alpha)
    to_reject = tentative & (np.minimum(p_reject * n_attributes, 1.0) < alpha)

    updated = decisions.copy()
    updated[to_confirm] = CONFIRMED
    updated[to_reject] = REJECTED
    return updated


def tentative_rough_fix(
    decisions: np.ndarray, importance_history: np.ndarray, shadow_max: np.ndarray
) -> np.ndarray:
    """Resolve tentative features the way R's ``TentativeRoughFix`` does.

    A tentative feature is promoted to ``"Confirmed"`` if its median
    importance across all iterations exceeds the median of the
    per-iteration maximum shadow importance, and demoted to
    ``"Rejected"`` otherwise. Confirmed and rejected features are left
    alone.

    This is a real port of the rule, not the coarse stand-in the previous
    `BorutaPy`-based implementation had to use (which kept *every*
    tentative feature outright, since `BorutaPy` never exposes the shadow
    history this comparison needs). On Paper 1's real data the difference
    is visible: R's seed-12345 run leaves 6 tentative features and this
    rule promotes 5 of them, giving the paper's 8.

    Parameters
    ----------
    decisions : np.ndarray of object
    importance_history : np.ndarray of shape (n_iterations, n_features)
    shadow_max : np.ndarray of shape (n_iterations,)

    Returns
    -------
    np.ndarray of object
        A copy of `decisions` with no ``"Tentative"`` entries remaining.
    """
    resolved = decisions.copy()
    tentative = np.flatnonzero(decisions == TENTATIVE)
    if tentative.size == 0:
        return resolved

    # A tentative feature was never rejected, so it has a finite
    # importance in every iteration -- no masking needed here.
    median_importance = np.median(importance_history[:, tentative], axis=0)
    threshold = np.median(shadow_max)
    resolved[tentative[median_importance > threshold]] = CONFIRMED
    resolved[tentative[median_importance <= threshold]] = REJECTED
    return resolved


def _boruta_run(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    importance_fn,
    alpha: float,
    max_iter: int,
    random_state,
) -> BorutaRun:
    """A single Boruta run: iterate until nothing is tentative, or `max_iter`.

    Implemented from the algorithm as published in Kursa & Rudnicki
    (2010), *Feature Selection with the Boruta Package*, JSS 36(11) --
    the project's licensing rule is to implement from published
    definitions rather than transcribe GPL source. Parameter values were
    confirmed against a live R run's observable behaviour (iteration
    count, the ``-Inf`` history sentinel, the shadow-padding minimum).
    """
    rng = check_random_state(random_state)
    n_features = X.shape[1]

    decisions = np.full(n_features, TENTATIVE, dtype=object)
    hits = np.zeros(n_features, dtype=int)
    importance_history: list[np.ndarray] = []
    shadow_history: list[tuple[float, float, float]] = []

    n_iterations = 0
    # R's loop increments *then* compares, so `maxRuns=100` runs 99
    # iterations. Matched here so iteration counts line up with R's.
    while (decisions == TENTATIVE).any() and n_iterations + 1 < max_iter:
        n_iterations += 1

        # Shadows are built from every non-rejected feature (confirmed
        # ones included), not only the undecided ones -- R does the same,
        # and it keeps the threshold from collapsing as features resolve.
        active = decisions != REJECTED
        X_active = X[:, active]
        X_shadow = _make_shadows(X_active, rng)

        seed = rng.randint(np.iinfo(np.int32).max)
        importances = importance_fn(
            np.hstack([X_active, X_shadow]), y, random_state=seed
        )
        n_active = X_active.shape[1]
        real_importance = importances[:n_active]
        shadow_importance = importances[n_active:]

        # Rejected features are not evaluated at all; -inf records that
        # without them ever counting as a hit.
        iteration_importance = np.full(n_features, -np.inf)
        iteration_importance[active] = real_importance

        hits += iteration_importance > shadow_importance.max()
        decisions = _do_tests(decisions, hits, n_iterations, alpha)

        importance_history.append(iteration_importance)
        shadow_history.append(
            (shadow_importance.min(), shadow_importance.mean(), shadow_importance.max())
        )

    history = (
        np.array(importance_history)
        if importance_history
        else np.empty((0, n_features))
    )
    shadows = np.array(shadow_history) if shadow_history else np.empty((0, 3))

    return BorutaRun(
        feature_names=list(feature_names),
        decisions=decisions,
        rough_fixed_decisions=tentative_rough_fix(decisions, history, shadows[:, 2]),
        importance_history=history,
        shadow_history=shadows,
        hits=hits,
        n_iterations=n_iterations,
    )


# --------------------------------------------------------------------------
# Correlated-feature clustering
# --------------------------------------------------------------------------


def correlated_feature_clusters(
    X: pd.DataFrame,
    feature_columns: list[str] | None = None,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Group near-duplicate features into correlated clusters.

    **This is the diagnostic that explains unstable Boruta results.**
    Boruta, like any random-forest importance method, splits credit
    between correlated predictors more or less arbitrarily: if five
    features carry the same signal, each run may crown a different one.
    Selection then looks unstable at the feature level while being
    perfectly stable at the cluster level. On Paper 1's data the AU12
    autocorrelation family correlates at up to ``|r| = 0.88`` and
    accounts for essentially all of the churn between seeds.

    Read alongside :class:`BorutaSelector`'s ``stability_``, which
    carries a matching ``cluster`` column, or plot it with
    :func:`plot_feature_clusters`.

    Uses average-linkage hierarchical clustering on ``1 - |r|``, cut at
    ``1 - threshold``: two groups merge when the *mean* absolute
    correlation between their members reaches `threshold`. Average
    linkage rather than single linkage deliberately -- single linkage
    merges on the strongest pair alone, which on Paper 1's data chains
    13 features into one cluster at the same cutoff where average
    linkage's largest is 6.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data (metadata columns are ignored if `feature_columns`
        is given).
    feature_columns : list of str, optional
        Which columns to cluster. Defaults to every numeric column --
        pass explicitly whenever a metadata column might be numeric.
    threshold : float, default 0.8
        Mean absolute correlation at which a group counts as
        near-duplicate.

        **Not 0.9, and the difference matters.** `caret::findCorrelation`
        was called at 0.9 in `final_analysis_NMF_check.Rmd`, but that
        cutoff finds nothing useful here: on Paper 1's real data it
        yields only 2-member clusters and leaves every one of the AU12
        autocorrelation features a singleton -- including the four that
        demonstrably trade places between Boruta runs
        (``diff2_acf1``, ``diff2_acf10``, ``diff1x_pacf5``,
        ``diff2x_pacf5``, whose pairwise ``|r|`` tops out at 0.88 and
        medians 0.66). At 0.8, three of those four group; all four need
        0.7. Raise it toward 0.9 for a stricter near-duplicate
        definition, lower it to find broader families -- but check what
        actually groups on your data rather than trusting the default,
        which is calibrated to one dataset.

    Returns
    -------
    pd.DataFrame
        One row per feature: ``feature``, ``cluster`` (integer id),
        ``cluster_size``, ``max_abs_corr`` (its strongest absolute
        correlation with another member of its cluster; ``NaN`` for
        singletons). Sorted by descending cluster size, so the
        near-duplicate groups worth worrying about come first.
    """
    columns = _resolve_feature_columns(X, feature_columns)
    correlation = X[columns].corr().abs().to_numpy()
    # A constant column gives NaN correlations; treat it as unrelated to
    # everything rather than letting NaN propagate through the linkage.
    np.fill_diagonal(correlation, 1.0)
    correlation = np.nan_to_num(correlation, nan=0.0)

    if len(columns) < 2:
        labels = np.ones(len(columns), dtype=int)
    else:
        distance = np.clip(1.0 - correlation, 0.0, None)
        # Enforce exact symmetry/zero diagonal; squareform is strict and
        # floating-point asymmetry in a corr matrix will trip it.
        distance = (distance + distance.T) / 2.0
        np.fill_diagonal(distance, 0.0)
        linkage_matrix = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(linkage_matrix, t=1.0 - threshold, criterion="distance")

    sizes = pd.Series(labels).value_counts()
    max_abs_corr = []
    for i in range(len(columns)):
        members = np.flatnonzero(labels == labels[i])
        others = members[members != i]
        max_abs_corr.append(correlation[i, others].max() if others.size else np.nan)

    clusters = pd.DataFrame({
        "feature": columns,
        "cluster": labels,
        "cluster_size": [sizes[label] for label in labels],
        "max_abs_corr": max_abs_corr,
    })
    return clusters.sort_values(
        ["cluster_size", "cluster", "feature"], ascending=[False, True, True]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# Selector
# --------------------------------------------------------------------------


class BorutaSelector(BaseEstimator, TransformerMixin):
    """All-relevant feature selection via Boruta, repeated across seeds.

    A native implementation of the Boruta algorithm (Kursa & Rudnicki,
    2010), replacing the `BorutaPy` wrapper this class used to be. Going
    native fixed three things that could not be fixed from outside the
    library: the importance measure is now R's actual one
    (:func:`oob_permutation_importance`), the multiple-testing correction
    is R's Bonferroni rather than `BorutaPy`'s laxer two-step variant,
    and `TentativeRoughFix` and ``normHits`` are real implementations
    rather than stand-ins, since the loop now owns the shadow history
    both of them need.

    **This is the first supervised step in the pipeline** -- every
    earlier transformer's ``fit(X, y=None)`` ignores ``y``; here ``y`` is
    required, since Boruta decides relevance against a real target.

    Expects one row per group (e.g. per video), with every series'
    features already pivoted into separate columns -- see
    :func:`~facedyn.features.reshape.pivot_features_wide`, the step
    immediately before this one.

    **Runs across `n_repeats` seeds by default, and this is the point of
    the class.** A single Boruta run on correlated features returns a
    confident-looking list that is substantially noise: Paper 1's
    published 8 features come from one run at ``set.seed(12345)``, and
    re-running the identical R code at other seeds recovers only 3-5 of
    them. Nothing in a single run's output reveals that. Here,
    ``selection_frequency_`` and ``stability_`` do, and
    ``selected_columns_`` keeps only what survives `selection_threshold`
    of the runs. Set ``n_repeats=1`` for the literal single-run
    procedure; it warns, because that is the configuration that hides
    the problem.

    Pair it with :func:`correlated_feature_clusters` (whose output
    ``stability_`` carries as a ``cluster`` column) to see *why*
    particular features churn -- near-duplicate features share credit
    arbitrarily, so a cluster can be selected in every single run while
    no individual member is.

    Parameters
    ----------
    feature_columns : list of str, optional
        Which columns to select from. If not given, every numeric column
        in `X` is treated as a feature column.
    n_repeats : int, default 20
        How many independently seeded Boruta runs to perform. ``1``
        reproduces R's single-run procedure and warns.
    selection_threshold : float, default 0.8
        Fraction of runs in which a feature must be confirmed (after
        tentative rough fix) to be kept by `transform`. Ignored when
        ``n_repeats=1``, where any confirmed feature is kept.
    importance : {"permutation", "gini"} or callable, default "permutation"
        Importance backend. ``"permutation"`` is
        :func:`oob_permutation_importance`, matching R. ``"gini"`` is
        much faster and much less faithful. A callable must accept
        ``(X, y, random_state=...)`` and return one importance per
        column.
    n_estimators : int, default 500
        Trees per forest. Matches R's real default (`ranger`'s
        ``num.trees=500`` via `Boruta`).
    alpha : float, default 0.01
        Significance level for the per-iteration binomial test. Matches
        R Boruta's real default `pValue` (confirmed from the R package
        source: ``0.01``, not `BorutaPy`'s own default of ``0.05``).
    max_iter : int, default 100
        Maximum Boruta iterations per run. Matches R Boruta's default
        `maxRuns`. As in R, the loop performs at most ``max_iter - 1``
        iterations.
    correlation_threshold : float, default 0.8
        Passed to :func:`correlated_feature_clusters` to annotate
        ``stability_``. Set to ``None`` to skip clustering. See that
        function on why the default is 0.8 rather than
        `caret::findCorrelation`'s 0.9.
    random_state : int, optional
        Seeds the whole procedure, including the per-repeat seeds.
    n_jobs : int, optional
        Parallelism across repeats. Forests run single-threaded when
        repeats are parallelised, to avoid nested parallelism; with
        ``n_repeats=1`` the forest gets `n_jobs` instead.

    Attributes
    ----------
    feature_columns_ : list of str
        Columns Boruta was run over.
    runs_ : list of BorutaRun
        Every run's decisions and full importance history.
    stability_ : pd.DataFrame
        One row per feature: ``feature``, ``selection_freq``,
        ``n_confirmed``, ``n_tentative``, ``n_rejected``, ``median_imp``,
        ``selected``, plus ``cluster``/``cluster_size`` when clustering
        is enabled. Sorted by descending ``selection_freq``.
        ``n_confirmed``/``n_rejected`` count *rough-fixed* decisions,
        matching what R's plot showed and what `selection_freq` is
        computed from; ``n_tentative`` counts runs whose **raw** decision
        was tentative, i.e. runs where the rough fix had to break the tie
        rather than Boruta resolving it outright. The three therefore
        need not sum to the number of runs -- a high ``n_tentative`` next
        to a middling ``selection_freq`` is the signature of a feature
        Boruta genuinely could not decide.
    selection_frequency_ : dict of str to float
        Fraction of runs confirming each feature.
    selected_columns_ : list of str
        The columns `transform` keeps.
    support_, support_weak_ : np.ndarray of bool
        Confirmed / tentative masks from the first run, in
        `feature_columns_` order, for continuity with sklearn's selector
        conventions and with R's raw (pre-rough-fix) decisions.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_repeats: int = 20,
        selection_threshold: float = 0.8,
        importance="permutation",
        n_estimators: int = 500,
        alpha: float = 0.01,
        max_iter: int = 100,
        correlation_threshold: float | None = 0.8,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.feature_columns = feature_columns
        self.n_repeats = n_repeats
        self.selection_threshold = selection_threshold
        self.importance = importance
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.max_iter = max_iter
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _resolve_importance_fn(self, forest_n_jobs: int | None):
        if callable(self.importance):
            return self.importance
        try:
            backend = _IMPORTANCE_BACKENDS[self.importance]
        except KeyError:
            raise ValueError(
                f"importance must be one of {sorted(_IMPORTANCE_BACKENDS)} or a "
                f"callable, got {self.importance!r}"
            ) from None
        return _ForestImportance(backend, self.n_estimators, forest_n_jobs)

    def fit(self, X: pd.DataFrame, y) -> "BorutaSelector":
        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {self.n_repeats}")
        if self.n_repeats == 1:
            warnings.warn(
                "n_repeats=1 runs Boruta once, which cannot show how stable its "
                "selection is. On correlated features a single run routinely "
                "confirms a different set each seed -- Paper 1's published "
                "8-feature result recovers only 3-5 of itself when reseeded. "
                "Use n_repeats > 1 (the default is 20) unless you specifically "
                "want to reproduce a single-run result.",
                UserWarning,
                stacklevel=2,
            )

        self.feature_columns_ = _resolve_feature_columns(X, self.feature_columns)
        X_array = X[self.feature_columns_].to_numpy(dtype=float)
        y_array = np.asarray(y)

        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_repeats)

        forest_n_jobs = self.n_jobs if self.n_repeats == 1 else 1
        importance_fn = self._resolve_importance_fn(forest_n_jobs)

        self.runs_ = list(
            Parallel(n_jobs=self.n_jobs)(
                delayed(_boruta_run)(
                    X_array,
                    y_array,
                    self.feature_columns_,
                    importance_fn,
                    self.alpha,
                    self.max_iter,
                    int(seed),
                )
                for seed in seeds
            )
        )

        self._build_stability(X)
        first = self.runs_[0]
        self.support_ = first.decisions == CONFIRMED
        self.support_weak_ = first.decisions == TENTATIVE
        return self

    def _build_stability(self, X: pd.DataFrame) -> None:
        features = self.feature_columns_
        n_runs = len(self.runs_)

        rough_fixed = np.array([run.rough_fixed_decisions for run in self.runs_])
        raw = np.array([run.decisions for run in self.runs_])

        n_confirmed = (rough_fixed == CONFIRMED).sum(axis=0)
        selection_freq = n_confirmed / n_runs

        # Each run's per-feature median importance, ignoring the -inf
        # iterations where the feature had already been rejected. A
        # feature rejected in its run's very first iteration has no
        # finite values at all, hence the empty-slice guard.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            per_run_median = np.array([
                [
                    np.median(column[np.isfinite(column)])
                    if np.isfinite(column).any()
                    else np.nan
                    for column in run.importance_history.T
                ]
                for run in self.runs_
            ])
            median_imp = np.nanmedian(per_run_median, axis=0)

        threshold = self.selection_threshold if n_runs > 1 else 0.0
        selected = selection_freq >= threshold if n_runs > 1 else n_confirmed > 0

        stability = pd.DataFrame({
            "feature": features,
            "selection_freq": selection_freq,
            "n_confirmed": n_confirmed,
            "n_tentative": (raw == TENTATIVE).sum(axis=0),
            "n_rejected": (rough_fixed == REJECTED).sum(axis=0),
            "median_imp": median_imp,
            "selected": selected,
        })

        if self.correlation_threshold is not None:
            clusters = correlated_feature_clusters(
                X, self.feature_columns_, threshold=self.correlation_threshold
            )
            stability = stability.merge(
                clusters[["feature", "cluster", "cluster_size"]], on="feature", how="left"
            )

        self.stability_ = stability.sort_values(
            ["selection_freq", "median_imp"], ascending=False
        ).reset_index(drop=True)
        self.selection_frequency_ = dict(zip(features, selection_freq))
        self.selected_columns_ = [f for f, keep in zip(features, selected) if keep]

    def cluster_stability(self) -> pd.DataFrame:
        """Roll `stability_` up to correlated-cluster level.

        The view that separates real instability from apparent
        instability. A cluster of near-duplicate features can be
        confirmed in every single run while no individual member clears
        `selection_threshold`, because the runs disagree about *which*
        member to credit. Seeing ``cluster_freq = 1.0`` next to a best
        member at ``0.45`` is the signal that the feature-level result is
        splitting credit, not that the signal is weak.

        Returns
        -------
        pd.DataFrame
            One row per cluster: ``cluster``, ``n_features``,
            ``cluster_freq`` (fraction of runs confirming *any* member),
            ``best_feature`` and ``best_feature_freq``, ``members``.
            Sorted by descending ``cluster_freq``.
        """
        check_is_fitted(self, "stability_")
        if "cluster" not in self.stability_:
            raise ValueError(
                "cluster_stability() needs clustering enabled -- set "
                "correlation_threshold to a value rather than None."
            )

        rough_fixed = np.array([run.rough_fixed_decisions for run in self.runs_])
        index = {f: i for i, f in enumerate(self.feature_columns_)}

        rows = []
        for cluster, group in self.stability_.groupby("cluster", sort=False):
            members = list(group["feature"])
            columns = [index[f] for f in members]
            # A run "confirms the cluster" if it confirmed any member.
            confirmed_any = (rough_fixed[:, columns] == CONFIRMED).any(axis=1)
            best = group.loc[group["selection_freq"].idxmax()]
            rows.append({
                "cluster": cluster,
                "n_features": len(members),
                "cluster_freq": confirmed_any.mean(),
                "best_feature": best["feature"],
                "best_feature_freq": best["selection_freq"],
                "members": members,
            })

        return (
            pd.DataFrame(rows)
            .sort_values(["cluster_freq", "n_features"], ascending=False)
            .reset_index(drop=True)
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_columns_")
        metadata = X.drop(columns=self.feature_columns_).reset_index(drop=True)
        selected = X[self.selected_columns_].reset_index(drop=True)
        return pd.concat([metadata, selected], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "selected_columns_")
        return np.asarray(self.selected_columns_, dtype=object)


def boruta_feature_stats(selector: BorutaSelector, run: int = 0) -> pd.DataFrame:
    """Per-feature importance summary for one run of a fitted selector.

    Replicates R's ``Boruta::attStats()``
    (`final_analysis_NMF_check.Rmd`'s ``feature_stats``/``ranked_features``,
    printed there but never saved to disk), now including ``normHits``,
    which the previous `BorutaPy`-based implementation had to omit for
    want of a shadow history.

    All reductions run over *finite* history entries only. Iterations in
    which a feature had already been rejected are stored as ``-inf``
    (R's own sentinel), and including them would drag every rejected
    feature's ``minImp`` to ``-inf`` and its mean with it -- R's
    `attStats` filters the same way, on ``is.finite``.

    ``normHits`` follows R exactly: the fraction of *all* iterations in
    which the feature beat that iteration's maximum shadow importance.
    The denominator is the run's total iteration count, not the number
    the feature survived, so a feature rejected early is scored against
    the full run rather than flattered by its short life.

    Parameters
    ----------
    selector : BorutaSelector
        A fitted selector.
    run : int, default 0
        Which of ``selector.runs_`` to summarise. With the default
        ``n_repeats=20`` there are twenty; run 0 is as arbitrary as any
        single Boruta run, which is the point -- use
        ``selector.stability_`` for the across-run picture.

    Returns
    -------
    pd.DataFrame
        Columns ``feature``, ``meanImp``, ``medianImp``, ``minImp``,
        ``maxImp``, ``normHits``, ``decision`` (raw, pre-rough-fix) and
        ``roughFixedDecision``, one row per feature, sorted descending by
        ``medianImp`` (matching R's ``ranked_features``).
    """
    check_is_fitted(selector, "runs_")
    boruta_run = selector.runs_[run]
    history = boruta_run.importance_history
    shadow_max = boruta_run.shadow_max
    n_iterations = len(shadow_max)

    rows = []
    for j, feature in enumerate(boruta_run.feature_names):
        column = history[:, j]
        finite = np.isfinite(column)
        values = column[finite]
        if values.size == 0:
            rows.append({
                "feature": feature,
                "meanImp": np.nan,
                "medianImp": np.nan,
                "minImp": np.nan,
                "maxImp": np.nan,
                "normHits": 0.0,
            })
            continue
        rows.append({
            "feature": feature,
            "meanImp": values.mean(),
            "medianImp": np.median(values),
            "minImp": values.min(),
            "maxImp": values.max(),
            "normHits": float(np.sum(shadow_max[: values.size] < values) / n_iterations),
        })

    stats = pd.DataFrame(rows)
    stats["decision"] = boruta_run.decisions
    stats["roughFixedDecision"] = boruta_run.rough_fixed_decisions
    return stats.sort_values("medianImp", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _require_matplotlib(function_name: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            f"{function_name} requires matplotlib. Install with: pip install facedyn[viz]"
        ) from e
    return plt


def plot_boruta_importance(
    selector: BorutaSelector,
    run: int = 0,
    max_features: int | None = 30,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Box plot of per-feature importance across a run's iterations.

    The equivalent of R's ``plot(TentativeRoughFix(boruta_output))`` --
    the plot Paper 1's 8 features were read off (its Figure 2A). Each box
    summarises one feature's importance over the iterations it survived,
    coloured by its rough-fixed decision, with the shadow
    minimum/mean/maximum distributions drawn alongside as the reference
    Boruta actually tests against.

    **Shows one run.** Read it with :func:`plot_boruta_stability`, which
    shows whether that run's answer holds up across seeds; taking a plot
    like this one at face value is exactly how the original analysis
    ended up with an 8-feature list that reseeds to 3-5 of itself.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    selector : BorutaSelector
        A fitted selector.
    run : int, default 0
        Which of ``selector.runs_`` to plot.
    max_features : int, optional
        Show only this many top features by median importance, plus the
        shadow references. ``None`` shows all. Defaults to 30, since the
        full set is usually unreadable.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    save_path : str or pathlib.Path, optional
        Filename to save the figure to. Format is inferred from the
        extension. Not saved if ``None``.
    output_dir : str or pathlib.Path, default "."
        Directory `save_path` is written into, created if needed.
    dpi : int, default 300
        Resolution for raster formats. Ignored for vector formats.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_boruta_importance")
    check_is_fitted(selector, "runs_")
    boruta_run = selector.runs_[run]

    stats = boruta_feature_stats(selector, run=run).dropna(subset=["medianImp"])
    if max_features is not None:
        stats = stats.head(max_features)
    order = list(stats["feature"])[::-1]  # highest importance at the top

    index = {f: i for i, f in enumerate(boruta_run.feature_names)}
    data, colours, labels = [], [], []
    for name, column, colour in [
        ("shadowMin", 0, _SHADOW_COLOUR),
        ("shadowMean", 1, _SHADOW_COLOUR),
        ("shadowMax", 2, _SHADOW_COLOUR),
    ]:
        data.append(boruta_run.shadow_history[:, column])
        colours.append(colour)
        labels.append(name)
    for feature in order:
        values = boruta_run.importance_history[:, index[feature]]
        data.append(values[np.isfinite(values)])
        colours.append(
            _DECISION_COLOURS[boruta_run.rough_fixed_decisions[index[feature]]]
        )
        labels.append(feature)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.28 * len(data) + 1.5))

    boxes = ax.boxplot(
        data, vert=False, patch_artist=True, widths=0.6, showfliers=False,
        medianprops={"color": "black"},
    )
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)

    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Importance (Z-score)")
    ax.set_title(f"Boruta importance - run {run} of {len(selector.runs_)}")
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=colour, label=label)
        for label, colour in [
            (CONFIRMED, _DECISION_COLOURS[CONFIRMED]),
            (TENTATIVE, _DECISION_COLOURS[TENTATIVE]),
            (REJECTED, _DECISION_COLOURS[REJECTED]),
            ("shadow", _SHADOW_COLOUR),
        ]
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_boruta_stability(
    selector: BorutaSelector,
    max_features: int | None = 30,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Selection frequency across seeds -- how often each feature survives.

    **The plot to read before trusting any single Boruta run.** A feature
    confirmed in 20 of 20 runs and one confirmed in 9 of 20 are
    indistinguishable in a single run's output and look identical in
    :func:`plot_boruta_importance`; here they don't. Bars are coloured by
    whether they clear `selection_threshold`, which is drawn as a
    reference line.

    Features that hover near the middle are usually not weak but
    *shared*: several near-duplicate features splitting credit between
    runs. Confirm that with :meth:`BorutaSelector.cluster_stability` or
    :func:`plot_feature_clusters` before concluding the signal is absent.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    selector : BorutaSelector
        A fitted selector.
    max_features : int, optional
        Show only this many features, by descending selection frequency.
        ``None`` shows all. Features never selected in any run are always
        omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    save_path : str or pathlib.Path, optional
        Filename to save the figure to. Format is inferred from the
        extension. Not saved if ``None``.
    output_dir : str or pathlib.Path, default "."
        Directory `save_path` is written into, created if needed.
    dpi : int, default 300
        Resolution for raster formats. Ignored for vector formats.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_boruta_stability")
    check_is_fitted(selector, "stability_")

    shown = selector.stability_[selector.stability_["selection_freq"] > 0]
    if max_features is not None:
        shown = shown.head(max_features)
    shown = shown.iloc[::-1]  # most frequent at the top

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.3 * max(len(shown), 1) + 1.5))

    colours = [
        _DECISION_COLOURS[CONFIRMED] if selected else _DECISION_COLOURS[TENTATIVE]
        for selected in shown["selected"]
    ]
    ax.barh(shown["feature"], shown["selection_freq"], color=colours)
    ax.axvline(
        selector.selection_threshold, color="black", linestyle="--", linewidth=1,
        label=f"threshold = {selector.selection_threshold:g}",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Fraction of {len(selector.runs_)} seeded runs confirming the feature")
    ax.set_title("Boruta selection stability across seeds")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="lower right", fontsize=8)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_feature_clusters(
    X: pd.DataFrame,
    clusters: pd.DataFrame,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Correlation heatmap ordered so correlated clusters form blocks.

    Takes :func:`correlated_feature_clusters`'s output. Near-duplicate
    families show up as bright blocks on the diagonal -- these are the
    groups whose members trade places between Boruta runs, and seeing
    their size is usually enough to explain an unstable selection.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    X : pd.DataFrame
        The same data `clusters` was computed from.
    clusters : pd.DataFrame
        Output of :func:`correlated_feature_clusters`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    save_path : str or pathlib.Path, optional
        Filename to save the figure to. Format is inferred from the
        extension. Not saved if ``None``.
    output_dir : str or pathlib.Path, default "."
        Directory `save_path` is written into, created if needed.
    dpi : int, default 300
        Resolution for raster formats. Ignored for vector formats.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib("plot_feature_clusters")

    features = list(clusters["feature"])
    correlation = X[features].corr().abs()
    matrix = np.nan_to_num(correlation.to_numpy(), nan=0.0)
    np.fill_diagonal(matrix, 1.0)

    # Order within the heatmap by the dendrogram, not just by cluster id,
    # so blocks read as blocks even where a cut split a broader family.
    if len(features) > 2:
        distance = np.clip(1.0 - matrix, 0.0, None)
        distance = (distance + distance.T) / 2.0
        np.fill_diagonal(distance, 0.0)
        order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    else:
        order = np.arange(len(features))

    ordered = matrix[np.ix_(order, order)]
    labels = [features[i] for i in order]

    if ax is None:
        _, ax = plt.subplots(figsize=(0.22 * len(labels) + 3, 0.22 * len(labels) + 2.5))

    image = ax.imshow(ordered, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Absolute feature correlation, ordered by cluster")
    ax.figure.colorbar(image, ax=ax, label="|r|", fraction=0.046)
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax
