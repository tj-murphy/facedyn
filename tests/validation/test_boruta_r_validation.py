"""Validation of the native Boruta implementation against R's real result.

**What "agreement with R" can mean here, and why the earlier version of
this file asserted the wrong thing.** It pinned one exact 4-feature
overlap with R's published 8, on the reasoning that a real measured
number beats a threshold picked to make a test pass. The number was real;
the target was not stable enough to pin. Re-running the *identical R
code* at other seeds recovers only 3-5 of those same 8 features, and at
`ntree=500` `ranger`'s importance correlates with itself across seeds at
Spearman 0.30-0.45. The old assertion was pinning noise, and it framed a
result inside R's own self-agreement band as a shortfall.

So these tests assert against R's *measured* self-agreement instead:

- overlap with R's 8 at or above 3, the floor R itself hits when reseeded
- `diff1x_pacf5_AU17_chin_raiser` ranked top, which it is in every run
  observed, R and Python alike, by a wide margin

The stability test pins the finding that motivated this module's design:
Paper 1's published 8 features do not reproduce across seeds, and the
package now says so out loud instead of returning a confident list.

See PIPELINE.md step 8 for the full investigation, including the R-oracle
numbers quoted above and the classifier check showing the choice of
feature set does not move downstream performance.
"""

from pathlib import Path

import pandas as pd
import pytest

from facedyn.feature_selection import (
    BorutaSelector,
    boruta_feature_stats,
    correlated_feature_clusters,
)

DATA_PATH = Path(__file__).parents[2] / "R Validation Data" / "r_cmfts_output_imputed_zerosd.csv"

pytestmark = [
    pytest.mark.skipif(
        not DATA_PATH.exists(),
        reason="r_cmfts_output_imputed_zerosd.csv not available locally",
    ),
    # 339 s: twenty-seed Boruta fits on the real 108-feature table. Skipped unless `pytest --runslow`; always run in CI.
    pytest.mark.slow,
]

METADATA_COLUMNS = ["video_filename", "isfakeorreal", "emotion", "valence"]

# R's real selection (final_analysis_NMF_check.Rmd's `boruta_selected_features`,
# repeated identically 4 times in the file). Reproduced exactly by a live R
# run at set.seed(12345): 3 Confirmed + 6 Tentative, then TentativeRoughFix.
R_SELECTED_FEATURES = [
    "diff1x_pacf5_AU17_chin_raiser",
    "diff1x_pacf5_AU01_inner_brow_raiser",
    "diff2_acf1_AU01_inner_brow_raiser",
    "diff2_acf10_AU12_lip_corner_puller",
    "max_kl_shift_AU12_lip_corner_puller",
    "diff1_acf1_AU12_lip_corner_puller",
    "lumpiness_AU12_lip_corner_puller",
    "diff2_acf1_AU12_lip_corner_puller",
]

# The lowest overlap R's own code achieves against its own published 8
# when only the seed changes (measured: 3/8 at seeds 1 and 999).
R_SELF_AGREEMENT_FLOOR = 3

# The four AU12 autocorrelation features that demonstrably trade places
# between R runs -- the churn this module's clustering exists to explain.
CHURNING_AU12_FEATURES = [
    "diff2_acf1_AU12_lip_corner_puller",
    "diff2_acf10_AU12_lip_corner_puller",
    "diff1x_pacf5_AU12_lip_corner_puller",
    "diff2x_pacf5_AU12_lip_corner_puller",
]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def feature_columns(data) -> list[str]:
    return [c for c in data.columns if c not in METADATA_COLUMNS]


@pytest.fixture(scope="module")
def single_run_selector(data, feature_columns) -> BorutaSelector:
    """One R-faithful run: R's own seed, R's importance measure, R's defaults."""
    selector = BorutaSelector(
        feature_columns=feature_columns, n_repeats=1, random_state=12345, n_jobs=-1
    )
    with pytest.warns(UserWarning, match="cannot show how stable"):
        selector.fit(data, data["isfakeorreal"])
    return selector


def test_single_run_overlaps_r_within_its_own_self_agreement_band(single_run_selector):
    overlap = set(single_run_selector.selected_columns_) & set(R_SELECTED_FEATURES)

    assert len(overlap) >= R_SELF_AGREEMENT_FLOOR, (
        f"selected {single_run_selector.selected_columns_}, overlapping R's 8 in "
        f"{sorted(overlap)} -- below the {R_SELF_AGREEMENT_FLOOR}/8 floor R itself "
        f"hits when only the seed changes"
    )


def test_single_run_ranks_the_same_top_feature_as_r(single_run_selector):
    """`diff1x_pacf5_AU17_chin_raiser` leads by a wide margin in every run
    observed -- R at five seeds, `ranger` alone, and this implementation.
    It is the one part of the selection that is genuinely stable."""
    stats = boruta_feature_stats(single_run_selector)

    assert stats.iloc[0]["feature"] == "diff1x_pacf5_AU17_chin_raiser"
    assert stats.iloc[0]["decision"] == "Confirmed"


def test_single_run_matches_rs_iteration_structure(single_run_selector):
    """R performs 99 iterations here (maxRuns=100, and its loop increments
    before comparing). Matching that is a check on the loop itself, not on
    the stochastic outcome."""
    run = single_run_selector.runs_[0]

    assert run.n_iterations == 99
    assert run.importance_history.shape == (99, len(single_run_selector.feature_columns_))
    # Rough fix leaves nothing undecided, as R's does.
    assert "Tentative" not in set(run.rough_fixed_decisions)


def test_rejected_features_are_dropped_from_later_iterations(single_run_selector):
    """R stops evaluating rejected features and records -Inf for them.
    Reproducing that is what makes a run affordable: only ~9-11 of 108
    features are still active after iteration 20."""
    import numpy as np

    history = single_run_selector.runs_[0].importance_history
    still_active = np.isfinite(history).sum(axis=1)

    assert still_active[0] == len(single_run_selector.feature_columns_)
    assert still_active[-1] < 20
    assert still_active[-1] < still_active[0]


@pytest.fixture(scope="module")
def stability_selector(data, feature_columns) -> BorutaSelector:
    """Several seeds, at a reduced tree count to keep the test affordable.

    `n_estimators=100` rather than the default 500: the finding under test
    is about disagreement *between* seeds, which a lower tree count makes
    more pronounced, not less. The full-fidelity run (20 repeats, 500
    trees, ~12 minutes) is recorded in PIPELINE.md rather than run here.
    """
    selector = BorutaSelector(
        feature_columns=feature_columns,
        n_repeats=8,
        n_estimators=100,
        random_state=12345,
        n_jobs=-1,
    )
    return selector.fit(data, data["isfakeorreal"])


def test_stability_shows_rs_published_eight_do_not_reproduce(stability_selector):
    """The finding this module exists for.

    Paper 1 read 8 features off a single Boruta run and hardcoded them
    downstream. Across seeds, almost none of them survive: at the full
    20-repeat / 500-tree setting exactly one of the eight clears a 0.8
    selection frequency. A single run cannot show that, which is why
    `BorutaSelector` repeats by default.
    """
    frequencies = stability_selector.selection_frequency_
    published = {f: frequencies[f] for f in R_SELECTED_FEATURES}

    # 0.8 here is a fixed reference level for "confirmed in the large
    # majority of runs", not the shipped default -- that moved to 0.5 on
    # 2026-08-03. The claim under test is about R's eight features, so it
    # should not move when a package default does.
    survivors = [f for f, freq in published.items() if freq >= 0.8]
    assert len(survivors) <= 3, (
        f"expected almost none of R's 8 to survive reseeding, got {survivors} "
        f"from {published}"
    )
    # At least one of the published eight is close to a coin flip.
    assert min(published.values()) <= 0.5


def test_stability_keeps_the_one_genuinely_stable_feature(stability_selector):
    """`diff1x_pacf5_AU17_chin_raiser` stands alone at the top.

    **What this asserts, and why it is not a selection-frequency
    threshold.** An earlier version required ``selection_freq >= 0.8``,
    which is the shipped default's value and was never chosen with any
    backing -- and the frequency is not stable enough to carry that
    weight. Measured on this exact fixture, the same feature comes first
    on both scikit-learn versions but at very different levels, because
    the two draw random-forest bootstraps differently:

    | scikit-learn | top feature | runner-up |
    |---|---|---|
    | 1.7.2 | 0.750 (6/8 runs) | 0.250 |
    | 1.8.0 | 1.000 (8/8 runs) | 0.500 |

    What survives both is the *ranking* and the *margin*: first place, a
    clear majority of runs, and a lead of 2-3x over whatever is second.
    Those are the claims PIPELINE.md actually makes about this feature, so
    those are what is pinned here. The 2x multiplier below is the lower of
    the two observed margins -- an observed floor, not a principled
    constant, and worth re-measuring rather than trusting if it ever
    fails.
    """
    top = stability_selector.stability_.iloc[0]
    runner_up = stability_selector.stability_.iloc[1]

    assert top["feature"] == "diff1x_pacf5_AU17_chin_raiser"
    assert top["selection_freq"] > 0.5, "not even confirmed more often than not"
    assert top["selection_freq"] >= 2 * runner_up["selection_freq"], (
        f"expected a clear lead, got {top['selection_freq']} against "
        f"{runner_up['feature']} at {runner_up['selection_freq']}"
    )
    # Whatever the shipped default keeps here, this feature is in it and
    # leads it. Asserted as membership rather than as the whole of
    # `selected_columns_`: the default moved from 0.8 to 0.5 on 2026-08-03
    # (see PIPELINE.md "Step 8b"), and on scikit-learn 1.8 the runner-up
    # sits exactly at 0.5, so what else comes along is a property of the
    # threshold rather than of this fixture.
    assert "diff1x_pacf5_AU17_chin_raiser" in stability_selector.selected_columns_
    assert all(
        stability_selector.selection_frequency_[f] <= top["selection_freq"]
        for f in stability_selector.selected_columns_
    )


def test_cluster_stability_never_understates_its_members(stability_selector):
    """Confirming any member confirms the cluster, so a cluster's
    frequency is at least its best member's. Where it is *higher*, the
    runs are splitting credit between near-duplicates rather than
    disagreeing about whether signal exists."""
    clusters = stability_selector.cluster_stability()

    assert (clusters["cluster_freq"] >= clusters["best_feature_freq"]).all()


def test_clustering_groups_the_features_that_churn_between_runs(data, feature_columns):
    """The mechanism behind the instability, asserted directly on the data.

    `caret::findCorrelation`'s 0.9 cutoff finds none of this -- every one
    of these is a singleton there, which is why `correlated_feature_clusters`
    defaults to 0.8 instead.
    """
    at_default = correlated_feature_clusters(data, feature_columns, threshold=0.8)
    at_default = at_default.set_index("feature").loc[CHURNING_AU12_FEATURES, "cluster"]
    at_findcorrelation = correlated_feature_clusters(data, feature_columns, threshold=0.9)
    at_findcorrelation = at_findcorrelation.set_index("feature")

    # Three of the four group at the default; all four need 0.7.
    assert at_default.nunique() == 2
    assert at_default.value_counts().max() == 3
    # At R's 0.9 cutoff the diagnostic finds nothing at all here.
    assert (at_findcorrelation.loc[CHURNING_AU12_FEATURES, "cluster_size"] == 1).all()


def test_clustering_groups_all_four_churning_features_at_a_lower_threshold(
    data, feature_columns
):
    clusters = correlated_feature_clusters(data, feature_columns, threshold=0.7)
    clusters = clusters.set_index("feature").loc[CHURNING_AU12_FEATURES, "cluster"]

    assert clusters.nunique() == 1
