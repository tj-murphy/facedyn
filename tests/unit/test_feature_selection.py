"""Unit tests for the native Boruta implementation.

Going native from `BorutaPy` made the algorithm exactly testable for the
first time: the loop takes an `importance` callable, so these tests
inject a deterministic fake importance function and assert against
hand-computed binomial decisions, rather than asserting soft properties
of whatever a real random forest happened to produce.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom
from sklearn.ensemble import RandomForestClassifier

from facedyn.feature_selection import (
    CONFIRMED,
    REJECTED,
    TENTATIVE,
    BorutaSelector,
    _boruta_run,
    _do_tests,
    _make_shadows,
    _oob_indices,
    boruta_feature_stats,
    correlated_feature_clusters,
    gini_importance,
    oob_permutation_importance,
    plot_boruta_importance,
    plot_boruta_stability,
    plot_feature_clusters,
    tentative_rough_fix,
)

#: 58 s: every test here fits Boruta at least once. Skipped unless `pytest --runslow`; always run in CI.
pytestmark = pytest.mark.slow


def constant_importance(values):
    """An importance function returning fixed per-column values.

    `values` gives the importance of the real (active) columns; shadow
    columns always score 0. Lets a test dictate exactly which features
    beat the shadow threshold each iteration, and therefore exactly what
    the binomial test should decide.
    """

    def importance_fn(X, y, random_state=None):
        n_active = len(values)
        return np.array(list(values) + [0.0] * (X.shape[1] - n_active))

    return importance_fn


# --------------------------------------------------------------------------
# Shadow generation
# --------------------------------------------------------------------------


def test_make_shadows_pads_to_at_least_five_columns():
    # R duplicates the shadow block until it has 5 columns, so that
    # max(shadow) isn't taken over a degenerate 1-2 column sample once
    # most features have been rejected.
    rng = np.random.RandomState(0)
    X = np.arange(20, dtype=float).reshape(10, 2)

    shadows = _make_shadows(X, rng)

    assert shadows.shape == (10, 8)  # 2 -> 4 -> 8, doubling past 5


def test_make_shadows_permutes_without_changing_column_values():
    rng = np.random.RandomState(0)
    X = np.arange(60, dtype=float).reshape(10, 6)

    shadows = _make_shadows(X, rng)

    assert shadows.shape == X.shape
    for j in range(6):
        # A shadow is a permutation of its source column: same multiset,
        # different order (with 10 distinct values, a fixed permutation
        # is vanishingly unlikely and this seed doesn't produce one).
        assert sorted(shadows[:, j]) == sorted(X[:, j])
    assert not np.array_equal(shadows, X)


def test_make_shadows_duplicated_columns_are_independently_permuted():
    # Padding duplicates columns *before* permuting, so the copies must
    # not come out identical -- otherwise max(shadow) would be computed
    # over fewer effective samples than it appears to be.
    rng = np.random.RandomState(1)
    X = np.arange(40, dtype=float).reshape(20, 2)

    shadows = _make_shadows(X, rng)

    assert not np.array_equal(shadows[:, 0], shadows[:, 4])


# --------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------


def test_do_tests_confirms_and_rejects_at_the_binomial_extremes():
    decisions = np.array([TENTATIVE] * 3, dtype=object)
    # 20 iterations: 20 hits is as extreme as possible, 0 hits likewise.
    hits = np.array([20, 10, 0])

    updated = _do_tests(decisions, hits, n_iterations=20, alpha=0.01)

    assert list(updated) == [CONFIRMED, TENTATIVE, REJECTED]


def test_do_tests_applies_bonferroni_over_total_attribute_count():
    """R's `mcAdj=TRUE` corrects by the total attribute count, not by the
    number still undecided -- it hands `p.adjust` the full-length p-value
    vector every round. That makes the correction constant, and stricter
    than `BorutaPy`'s two-step BH-then-Bonferroni."""
    n_iterations, hits = 12, 11
    p_confirm = binom.sf(hits - 1, n_iterations, 0.5)
    # Chosen so the raw p-value clears alpha but 3x the p-value does not:
    # confirms with 1 attribute, stays tentative with 3.
    alpha = 0.006
    assert p_confirm < alpha < p_confirm * 3

    one = _do_tests(np.array([TENTATIVE], dtype=object), np.array([hits]), n_iterations, alpha)
    three = _do_tests(
        np.array([TENTATIVE] * 3, dtype=object),
        np.array([hits, 0, 0]),
        n_iterations,
        alpha,
    )

    assert one[0] == CONFIRMED
    assert three[0] == TENTATIVE


def test_do_tests_leaves_already_decided_features_alone():
    decisions = np.array([CONFIRMED, REJECTED], dtype=object)

    updated = _do_tests(decisions, np.array([0, 20]), n_iterations=20, alpha=0.01)

    assert list(updated) == [CONFIRMED, REJECTED]


# --------------------------------------------------------------------------
# Tentative rough fix
# --------------------------------------------------------------------------


def test_tentative_rough_fix_splits_on_median_shadow_max():
    # R's rule: a tentative feature is confirmed iff its median
    # importance exceeds the median of the per-iteration shadowMax.
    decisions = np.array([TENTATIVE, TENTATIVE, CONFIRMED, REJECTED], dtype=object)
    history = np.array([
        [3.0, 1.0, 9.0, -np.inf],
        [4.0, 1.0, 9.0, -np.inf],
        [5.0, 1.0, 9.0, -np.inf],
    ])
    shadow_max = np.array([2.0, 2.0, 2.0])

    resolved = tentative_rough_fix(decisions, history, shadow_max)

    # median 4.0 > 2.0 -> confirmed; median 1.0 <= 2.0 -> rejected.
    assert list(resolved) == [CONFIRMED, REJECTED, CONFIRMED, REJECTED]


def test_tentative_rough_fix_is_a_no_op_without_tentatives():
    decisions = np.array([CONFIRMED, REJECTED], dtype=object)

    resolved = tentative_rough_fix(
        decisions, np.array([[1.0, -np.inf]]), np.array([0.5])
    )

    assert list(resolved) == [CONFIRMED, REJECTED]
    assert resolved is not decisions


# --------------------------------------------------------------------------
# The run loop
# --------------------------------------------------------------------------


def alternating_importance(reference_column, hit_every=2):
    """Importance function giving one feature a 50% hit rate.

    Scores every column ``-1.0`` except the column matching
    `reference_column`, which scores ``1.0`` on every `hit_every`-th
    iteration. Shadows are permutations of their source, so they never
    match the reference and stay at ``-1.0`` -- meaning the shadow
    threshold is ``-1.0``, every other real feature ties with it and so
    never scores a hit, and the reference feature hits exactly half the
    time. A hit rate of 0.5 is precisely the null the binomial test
    cannot reject in either direction, so that feature stays tentative
    forever while the rest are rejected early.
    """
    state = {"calls": 0}

    def importance_fn(X, y, random_state=None):
        state["calls"] += 1
        importances = np.full(X.shape[1], -1.0)
        if state["calls"] % hit_every == 0:
            for j in range(X.shape[1]):
                if np.array_equal(X[:, j], reference_column):
                    importances[j] = 1.0
        return importances

    return importance_fn


def _make_X(n_features):
    return np.random.RandomState(0).normal(size=(30, n_features))


def _run(importance_fn, n_features, X=None, max_iter=100, alpha=0.01):
    X = _make_X(n_features) if X is None else X
    y = np.array(["a", "b"] * 15)
    names = [f"f{i}" for i in range(n_features)]
    return _boruta_run(X, y, names, importance_fn, alpha, max_iter, random_state=0)


def _run_with(values, n_features, max_iter=100, alpha=0.01):
    return _run(constant_importance(values), n_features, max_iter=max_iter, alpha=alpha)


def test_boruta_run_confirms_features_that_always_beat_the_shadows():
    # f0 scores 1.0 against shadows fixed at 0.0 every iteration, so it
    # hits every time; f1 and f2 score 0.0 and never hit.
    run = _run_with([1.0, 0.0, 0.0], n_features=3)

    assert run.decisions[0] == CONFIRMED
    assert list(run.decisions[1:]) == [REJECTED, REJECTED]


def test_boruta_run_stops_early_once_nothing_is_tentative():
    run = _run_with([1.0, 0.0, 0.0], n_features=3)

    # Far fewer than max_iter: ~11 iterations suffice for a Bonferroni-
    # corrected binomial test over 3 attributes to resolve everything.
    assert 0 < run.n_iterations < 30


def test_boruta_run_honours_max_iter_and_matches_rs_off_by_one():
    """R's loop increments then compares, so `maxRuns=100` performs 99
    iterations. Matched here so iteration counts line up with R's."""
    X = _make_X(2)
    # f0 hits exactly half the time -- the one hit rate the binomial test
    # can never resolve -- so it stays tentative and the loop runs to its
    # cap rather than stopping early.
    run = _run(alternating_importance(X[:, 0]), n_features=2, X=X, max_iter=100)

    assert run.n_iterations == 99
    assert run.decisions[0] == TENTATIVE


def test_boruta_run_records_minus_inf_for_rejected_features():
    """Rejected features stop being evaluated, and the history records
    that with R's own `-Inf` sentinel rather than NaN -- confirmed
    against a real R run's `ImpHistory` export."""
    X = _make_X(2)
    # f0 stays tentative to the cap, so f1's early rejection leaves a
    # long -inf tail behind it. (With every feature resolving at once the
    # loop stops immediately and there is no tail to inspect.)
    run = _run(alternating_importance(X[:, 0]), n_features=2, X=X)

    rejected_column = run.importance_history[:, 1]
    assert run.decisions[1] == REJECTED
    assert np.isneginf(rejected_column[-1])
    # It was evaluated before it was rejected, so not every entry is -inf.
    assert np.isfinite(rejected_column[0])
    assert np.isfinite(run.importance_history[:, 0]).all()


def test_boruta_run_history_shapes_line_up_with_iteration_count():
    run = _run_with([1.0, 0.0, 0.0], n_features=3)

    assert run.importance_history.shape == (run.n_iterations, 3)
    assert run.shadow_history.shape == (run.n_iterations, 3)
    assert len(run.shadow_max) == run.n_iterations
    assert run.hits[0] == run.n_iterations  # f0 hit every iteration


def test_boruta_run_confirmed_reports_rough_fixed_decisions_by_default():
    run = _run_with([1.0, 0.0, 0.0], n_features=3)

    assert run.confirmed() == ["f0"]
    assert run.confirmed(rough_fixed=False) == ["f0"]


# --------------------------------------------------------------------------
# Importance backends
# --------------------------------------------------------------------------


def test_oob_indices_match_sklearns_own_bootstrap():
    """`_oob_indices` reproduces sklearn's bootstrap draw rather than
    calling its private helper -- that helper's signature changed under us
    and broke every permutation-importance call on newer scikit-learn.

    Skipped rather than removed if the private function disappears: the
    two behavioural tests below pin what actually matters without touching
    sklearn internals, but while this import still works it is the
    sharpest available check on divergence.
    """
    import inspect

    sklearn_helper = pytest.importorskip("sklearn.ensemble._forest")
    generate_unsampled = getattr(sklearn_helper, "_generate_unsampled_indices", None)
    if generate_unsampled is None:
        pytest.skip("sklearn no longer exposes _generate_unsampled_indices")

    # The signature change that caused the original breakage: newer
    # scikit-learn requires `sample_weight`. Adapt to whichever is present
    # so this check keeps working across both.
    parameters = inspect.signature(generate_unsampled).parameters
    extra = {"sample_weight": None} if "sample_weight" in parameters else {}

    for seed in (0, 1, 42, 12345):
        expected = generate_unsampled(seed, 100, 100, **extra)
        assert list(_oob_indices(seed, 100, 100)) == list(expected)


def test_oob_indices_leave_out_the_expected_fraction():
    """Sampling n of n with replacement leaves ~1/e ~ 36.8% unpicked. A
    version-independent check that these really are bootstrap complements."""
    fractions = [len(_oob_indices(seed, 500, 500)) / 500 for seed in range(20)]

    assert 0.33 < np.mean(fractions) < 0.40


def test_oob_indices_really_are_held_out_from_the_tree():
    """The behavioural check that matters: an unrestricted tree memorises
    its training data, so if these indices are genuinely out-of-bag its
    accuracy on them must be clearly worse than on its in-bag samples. A
    wrong (e.g. arbitrary) index set would collapse that gap."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + rng.normal(scale=2.0, size=n) > 0).astype(int)

    forest = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    in_bag_scores, oob_scores = [], []
    for tree in forest.estimators_:
        oob = _oob_indices(tree.random_state, n, n)
        in_bag = np.setdiff1d(np.arange(n), oob)
        in_bag_scores.append(np.mean(tree.predict(X[in_bag]) == y[in_bag]))
        oob_scores.append(np.mean(tree.predict(X[oob]) == y[oob]))

    assert np.mean(in_bag_scores) > 0.95  # memorised its own training rows
    assert np.mean(oob_scores) < 0.85  # but generalises far less well


def test_correlated_feature_clusters_accepts_read_only_input():
    """pandas 3 hands back read-only arrays from `.to_numpy()`, which broke
    the in-place diagonal fill this used to do. Asserted directly so the
    fix cannot regress on older pandas, where the array is writable and
    the bug is invisible."""
    frame = _correlated_frame()
    values = frame.to_numpy()
    values.flags.writeable = False
    read_only = pd.DataFrame(values, columns=frame.columns, copy=False)

    clusters = correlated_feature_clusters(read_only, threshold=0.9)

    assert len(clusters) == len(frame.columns)


def test_oob_permutation_importance_ranks_the_informative_feature_first():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(size=n)
    X = np.column_stack([signal, rng.normal(size=n), rng.normal(size=n)])
    y = (signal > 0).astype(int)

    importance = oob_permutation_importance(X, y, n_estimators=100, random_state=0)

    assert importance.argmax() == 0
    assert importance[0] > importance[1:].max()


def test_gini_importance_ranks_the_informative_feature_first():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(size=n)
    X = np.column_stack([signal, rng.normal(size=n), rng.normal(size=n)])
    y = (signal > 0).astype(int)

    importance = gini_importance(X, y, n_estimators=100, random_state=0)

    assert importance.argmax() == 0


# --------------------------------------------------------------------------
# Correlated-feature clustering
# --------------------------------------------------------------------------


def _correlated_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(size=n)
    return pd.DataFrame({
        # Three near-duplicates of one signal -- the shape that makes
        # Boruta's feature-level selection unstable.
        "dup_a": base,
        "dup_b": base + rng.normal(scale=0.05, size=n),
        "dup_c": base + rng.normal(scale=0.05, size=n),
        "independent": rng.normal(size=n),
    })


def test_correlated_feature_clusters_groups_near_duplicates():
    clusters = correlated_feature_clusters(_correlated_frame(), threshold=0.9)

    by_feature = clusters.set_index("feature")["cluster"]
    assert by_feature["dup_a"] == by_feature["dup_b"] == by_feature["dup_c"]
    assert by_feature["independent"] != by_feature["dup_a"]
    assert by_feature.index.size == 4


def test_correlated_feature_clusters_reports_sizes_and_max_correlation():
    clusters = correlated_feature_clusters(_correlated_frame(), threshold=0.9).set_index(
        "feature"
    )

    assert clusters.loc["dup_a", "cluster_size"] == 3
    assert clusters.loc["independent", "cluster_size"] == 1
    assert clusters.loc["dup_a", "max_abs_corr"] > 0.9
    # A singleton has no other member to correlate with.
    assert np.isnan(clusters.loc["independent", "max_abs_corr"])


def test_correlated_feature_clusters_separates_everything_at_a_high_threshold():
    clusters = correlated_feature_clusters(_correlated_frame(), threshold=0.999)

    assert clusters["cluster"].nunique() == 4


def test_correlated_feature_clusters_handles_a_constant_column():
    frame = _correlated_frame()
    frame["constant"] = 1.0

    clusters = correlated_feature_clusters(frame, threshold=0.9).set_index("feature")

    # NaN correlations must not propagate into the linkage; a constant
    # column is simply unrelated to everything.
    assert clusters.loc["constant", "cluster_size"] == 1


# --------------------------------------------------------------------------
# Selector
# --------------------------------------------------------------------------


def _selector_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    signal = rng.normal(size=n)
    frame = pd.DataFrame({
        "video_filename": [f"v{i}" for i in range(n)],
        "informative": signal,
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
    })
    frame["isfakeorreal"] = np.where(signal > 0, "fake", "real")
    return frame


FEATURES = ["informative", "noise_a", "noise_b"]


def _fit_selector(**kwargs) -> BorutaSelector:
    frame = _selector_frame()
    defaults = dict(
        feature_columns=FEATURES,
        n_repeats=4,
        importance="gini",
        n_estimators=50,
        random_state=0,
    )
    selector = BorutaSelector(**{**defaults, **kwargs})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(frame, frame["isfakeorreal"])
    return selector


def test_selector_runs_once_per_repeat_and_selects_the_informative_feature():
    selector = _fit_selector(n_repeats=4)

    assert len(selector.runs_) == 4
    assert selector.selected_columns_ == ["informative"]
    assert selector.selection_frequency_["informative"] == 1.0


def test_selector_stability_reports_one_row_per_feature_sorted_by_frequency():
    selector = _fit_selector()

    stability = selector.stability_
    assert set(stability["feature"]) == set(FEATURES)
    assert stability["selection_freq"].is_monotonic_decreasing
    assert stability.iloc[0]["feature"] == "informative"
    assert set(stability.columns) >= {
        "feature", "selection_freq", "n_confirmed", "n_tentative",
        "n_rejected", "median_imp", "selected", "cluster", "cluster_size",
    }


def test_selector_threshold_controls_what_transform_keeps():
    """`selection_threshold` is the whole point of repeating runs -- a
    feature confirmed in some runs but not most should not survive."""
    selector = _fit_selector()
    frequencies = selector.selection_frequency_

    strict = _fit_selector(selection_threshold=1.01)
    permissive = _fit_selector(selection_threshold=0.0)

    assert strict.selected_columns_ == []
    assert set(permissive.selected_columns_) == set(FEATURES)
    assert frequencies["informative"] >= frequencies["noise_a"]


def test_selector_warns_on_a_single_run():
    """n_repeats=1 is the configuration that hides selection instability
    -- the exact trap that produced Paper 1's unreproducible 8 features."""
    frame = _selector_frame()
    selector = BorutaSelector(
        feature_columns=FEATURES, n_repeats=1, importance="gini",
        n_estimators=50, random_state=0,
    )

    with pytest.warns(UserWarning, match="cannot show how stable"):
        selector.fit(frame, frame["isfakeorreal"])

    # With one run there is no frequency to threshold, so anything
    # confirmed is kept.
    assert selector.selected_columns_ == ["informative"]


def test_selector_single_repeat_does_not_start_a_parallel_backend():
    """A single repeat has nothing to parallelise across, and starting a
    `loky` pool anyway deadlocked a Jupyter kernel (the worker's forest
    also requests `n_jobs`). Single-run selection is what a nested
    cross-validation loop does on every fold, so this path must stay
    inline."""
    frame = _selector_frame()
    calls = []

    def tracking_parallel(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("Parallel must not be used for n_repeats=1")

    import facedyn.feature_selection as module

    original = module.Parallel
    module.Parallel = tracking_parallel
    try:
        selector = BorutaSelector(
            feature_columns=FEATURES, n_repeats=1, importance="gini",
            n_estimators=50, random_state=0, n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            selector.fit(frame, frame["isfakeorreal"])
    finally:
        module.Parallel = original

    assert calls == []
    assert len(selector.runs_) == 1


def test_selector_rejects_a_non_positive_repeat_count():
    frame = _selector_frame()
    selector = BorutaSelector(feature_columns=FEATURES, n_repeats=0)

    with pytest.raises(ValueError, match="n_repeats must be >= 1"):
        selector.fit(frame, frame["isfakeorreal"])


def test_selector_rejects_an_unknown_importance_backend():
    frame = _selector_frame()
    selector = BorutaSelector(feature_columns=FEATURES, n_repeats=2, importance="shap")

    with pytest.raises(ValueError, match="importance must be one of"):
        selector.fit(frame, frame["isfakeorreal"])


def test_selector_accepts_a_callable_importance():
    frame = _selector_frame()
    selector = BorutaSelector(
        feature_columns=FEATURES, n_repeats=2, importance=constant_importance([1.0, 0.0, 0.0]),
        random_state=0,
    )

    selector.fit(frame, frame["isfakeorreal"])

    assert selector.selected_columns_ == ["informative"]


def test_selector_is_reproducible_for_a_fixed_random_state():
    first = _fit_selector(random_state=7)
    second = _fit_selector(random_state=7)

    assert first.selected_columns_ == second.selected_columns_
    pd.testing.assert_frame_equal(first.stability_, second.stability_)


def test_selector_transform_keeps_metadata_and_selected_columns():
    selector = _fit_selector()
    frame = _selector_frame()

    out = selector.transform(frame)

    assert set(out.columns) == {"video_filename", "isfakeorreal", "informative"}
    assert len(out) == len(frame)


def test_selector_transform_handles_nothing_clearing_the_threshold():
    """"No feature is stable enough" is a real, correct outcome, not an
    error case -- it is exactly what facedyn's own 31-feature set produces
    on the Paper 1 training data. `transform` must return the metadata
    columns and no features rather than raising."""
    selector = _fit_selector(selection_threshold=1.01)
    frame = _selector_frame()

    out = selector.transform(frame)

    assert selector.selected_columns_ == []
    assert list(out.columns) == ["video_filename", "isfakeorreal"]
    assert len(out) == len(frame)
    assert list(selector.get_feature_names_out()) == []


def test_selector_get_feature_names_out_matches_selected_columns():
    selector = _fit_selector()

    assert list(selector.get_feature_names_out()) == selector.selected_columns_


def test_selector_exposes_first_run_support_masks():
    selector = _fit_selector()
    first = selector.runs_[0]

    assert list(selector.support_) == [d == CONFIRMED for d in first.decisions]
    assert list(selector.support_weak_) == [d == TENTATIVE for d in first.decisions]


def test_selector_cluster_stability_rolls_frequencies_up_to_clusters():
    """The view that separates real instability from credit-splitting: a
    cluster can be confirmed in every run while no single member is."""
    rng = np.random.default_rng(0)
    n = 150
    signal = rng.normal(size=n)
    frame = pd.DataFrame({
        "dup_a": signal,
        "dup_b": signal + rng.normal(scale=0.01, size=n),
        "noise": rng.normal(size=n),
    })
    frame["isfakeorreal"] = np.where(signal > 0, "fake", "real")

    selector = BorutaSelector(
        feature_columns=["dup_a", "dup_b", "noise"], n_repeats=4,
        importance="gini", n_estimators=50, random_state=0,
    )
    selector.fit(frame, frame["isfakeorreal"])
    clusters = selector.cluster_stability()

    duplicate_cluster = clusters[clusters["n_features"] == 2].iloc[0]
    assert set(duplicate_cluster["members"]) == {"dup_a", "dup_b"}
    # Confirming *either* duplicate confirms the cluster, so cluster
    # frequency is at least the best individual member's.
    assert duplicate_cluster["cluster_freq"] >= duplicate_cluster["best_feature_freq"]
    assert duplicate_cluster["cluster_freq"] == 1.0


def test_selector_cluster_stability_requires_clustering_enabled():
    selector = _fit_selector(correlation_threshold=None)

    assert "cluster" not in selector.stability_.columns
    with pytest.raises(ValueError, match="needs clustering enabled"):
        selector.cluster_stability()


# --------------------------------------------------------------------------
# attStats replication
# --------------------------------------------------------------------------


def test_boruta_feature_stats_ignores_the_minus_inf_sentinel():
    """R's `attStats` reduces over `is.finite` entries only. Including the
    `-Inf` iterations would drag every rejected feature's minImp to -inf
    and its mean with it."""
    selector = _fit_selector(importance=constant_importance([1.0, 0.0, 0.0]))
    run = selector.runs_[0]
    # Craft a history with a known finite prefix and a rejected tail.
    run.importance_history = np.array([
        [0.5, 0.2, 0.1],
        [0.6, 0.3, 0.2],
        [0.4, 0.1, -np.inf],
        [0.5, -np.inf, -np.inf],
    ])
    run.shadow_history = np.array([[0.0, 0.0, 0.15]] * 4)
    run.decisions = np.array([CONFIRMED, TENTATIVE, REJECTED], dtype=object)
    run.rough_fixed_decisions = np.array([CONFIRMED, CONFIRMED, REJECTED], dtype=object)
    run.feature_names = ["confirmed_feat", "tentative_feat", "rejected_feat"]

    stats = boruta_feature_stats(selector).set_index("feature")

    assert stats.loc["confirmed_feat", "minImp"] == 0.4
    assert stats.loc["confirmed_feat", "maxImp"] == 0.6
    # Only its two finite iterations count, not the -inf ones.
    assert stats.loc["rejected_feat", "minImp"] == 0.1
    assert stats.loc["rejected_feat", "maxImp"] == 0.2
    assert stats.loc["rejected_feat", "meanImp"] == pytest.approx(0.15)


def test_boruta_feature_stats_normhits_divides_by_total_iterations():
    """R's `normHits` denominator is the run's full iteration count, not
    the number of iterations the feature survived -- so a feature
    rejected early is scored against the whole run, not flattered by its
    short life."""
    selector = _fit_selector(importance=constant_importance([1.0, 0.0, 0.0]))
    run = selector.runs_[0]
    run.importance_history = np.array([
        [0.5, 0.2, 0.1],
        [0.6, 0.3, 0.2],
        [0.4, 0.1, -np.inf],
        [0.5, -np.inf, -np.inf],
    ])
    run.shadow_history = np.array([[0.0, 0.0, 0.15]] * 4)
    run.decisions = np.array([CONFIRMED, TENTATIVE, REJECTED], dtype=object)
    run.rough_fixed_decisions = np.array([CONFIRMED, CONFIRMED, REJECTED], dtype=object)
    run.feature_names = ["confirmed_feat", "tentative_feat", "rejected_feat"]

    stats = boruta_feature_stats(selector).set_index("feature")

    # confirmed_feat beats shadowMax=0.15 in all 4 of 4 iterations.
    assert stats.loc["confirmed_feat", "normHits"] == 1.0
    # tentative_feat: 0.2 and 0.3 beat 0.15, 0.1 does not -> 2 of 4.
    assert stats.loc["tentative_feat", "normHits"] == pytest.approx(0.5)
    # rejected_feat: only 0.2 beats 0.15, over the full 4 iterations.
    assert stats.loc["rejected_feat", "normHits"] == pytest.approx(0.25)


def test_boruta_feature_stats_reports_both_raw_and_rough_fixed_decisions():
    selector = _fit_selector()

    stats = boruta_feature_stats(selector)

    assert {"decision", "roughFixedDecision"} <= set(stats.columns)
    assert TENTATIVE not in set(stats["roughFixedDecision"])
    assert stats["medianImp"].is_monotonic_decreasing


def test_boruta_feature_stats_selects_the_requested_run():
    selector = _fit_selector(n_repeats=3)

    for run in range(3):
        stats = boruta_feature_stats(selector, run=run)
        assert list(stats["decision"]) == [
            selector.runs_[run].decisions[selector.runs_[run].feature_names.index(f)]
            for f in stats["feature"]
        ]


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _agg():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_plot_boruta_importance_draws_features_and_shadow_references():
    _agg()
    selector = _fit_selector()

    ax = plot_boruta_importance(selector)

    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert {"shadowMin", "shadowMean", "shadowMax"} <= set(labels)
    assert "informative" in labels


def test_plot_boruta_importance_respects_max_features():
    _agg()
    selector = _fit_selector()

    ax = plot_boruta_importance(selector, max_features=1)

    # One feature plus the three shadow references.
    assert len(ax.get_yticklabels()) == 4


def test_plot_boruta_stability_marks_the_threshold():
    _agg()
    selector = _fit_selector()

    ax = plot_boruta_stability(selector)

    assert ax.get_xlim() == (0, 1)
    assert "informative" in [t.get_text() for t in ax.get_yticklabels()]
    assert any(
        line.get_xdata()[0] == selector.selection_threshold for line in ax.get_lines()
    )


def test_plot_feature_clusters_renders_a_square_correlation_matrix():
    _agg()
    frame = _correlated_frame()
    clusters = correlated_feature_clusters(frame, threshold=0.9)

    ax = plot_feature_clusters(frame, clusters)

    assert ax.images[0].get_array().shape == (4, 4)


def test_plots_raise_a_helpful_error_without_matplotlib(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    selector = _fit_selector()

    with pytest.raises(ImportError, match=r"pip install facedyn\[viz\]"):
        plot_boruta_stability(selector)
