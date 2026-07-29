"""Regression test: `feature_diagnostics`/`FeatureCleaner` vs. real R output.

Two independent real fixtures, covering different pieces of this step:

1. `cmfts_output_subset.csv` (10 real rows, long format: one row per
   video x AU, un-suffixed feature names -- reused from
   `test_cmfts_r_bridge_validation.py`'s CMFTS bridge validation). Carries
   exactly the pathology the all-NaN auto-drop rule exists to handle:
   `permutation_entropy` is NA in all 10 real rows (root-caused, elsewhere
   in this project, to a real bug in `tsExpKit::permutationEntropy`), and
   several other columns (`sample_entropy`, `Kurtosis`/`Skewness`, the
   ACF/PACF family, Burg `entropy`, `nonlinearity`, `unitroot_kpss`/`_pp`,
   `e_acf1`/`e_acf10`) are NA in exactly 3 of the 10 rows, from the
   fixture's three deliberately-included constant-series rows.

2. `cmfts_output_imputed_subset.csv`/`cmfts_output_imputed_zerosd_subset.csv`
   (20-row subsets of `R Validation Data/r_cmfts_output_imputed.csv` and
   `..._imputed_zerosd.csv` -- the *real*, full 370-video training set, wide
   format: `dta_cmfts_output_imputed` and its zero-SD-pruned successor
   `dta_cmfts_output_imputed_zerosd` from `final_analysis_NMF_check.Rmd`).
   Confirmed safe to subset to 20 rows: `length`/`nperiods`/`seasonal_period`
   are a single value (241/0/1) across *all* 370 real rows, not just this
   subset, so no row-selection artifact is possible here. This is the
   step's own authoritative real fixture (its R variable names match this
   step exactly), and lets `is_constant`'s zero-SD decision be checked
   against R's real `sd(x, na.rm=T) == 0` decision directly, rather than
   inferred from a differently-shaped fixture borrowed from another step.

Both validate the deterministic pieces (NaN counting, the all-NaN auto-drop
rule, the zero-SD drop) against real R output directly. Neither validates
imputation *quality* against R's `missForest` -- that's a distinct,
stochastic comparison, out of scope here (see `PIPELINE.md` step 7).
"""

from pathlib import Path

import pandas as pd

from facedyn.features.cleaning import FeatureCleaner, feature_diagnostics

FIXTURES = Path(__file__).parent / "fixtures"
METADATA_COLUMNS = ["video_filename", "isfakeorreal", "emotion", "valence", "AU"]


def _load_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "cmfts_output_subset.csv")
    feature_columns = [c for c in df.columns if c not in METADATA_COLUMNS]
    return df, feature_columns


def _load_imputed_zerosd_fixtures() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    imputed = pd.read_csv(FIXTURES / "cmfts_output_imputed_subset.csv")
    zerosd = pd.read_csv(FIXTURES / "cmfts_output_imputed_zerosd_subset.csv")
    metadata = ["video_filename", "isfakeorreal", "emotion", "valence"]
    feature_columns = [c for c in imputed.columns if c not in metadata]
    return imputed, zerosd, feature_columns


# The 9 columns real R's `sds_train`/`features_to_keep` logic actually
# dropped from `dta_cmfts_output_imputed`, on the full 370-video real
# training set -- confirmed by diffing that file's columns against
# `dta_cmfts_output_imputed_zerosd`'s.
REAL_ZERO_SD_COLUMNS = [
    "length_AU01_inner_brow_raiser",
    "length_AU12_lip_corner_puller",
    "length_AU17_chin_raiser",
    "nperiods_AU01_inner_brow_raiser",
    "nperiods_AU12_lip_corner_puller",
    "nperiods_AU17_chin_raiser",
    "seasonal_period_AU01_inner_brow_raiser",
    "seasonal_period_AU12_lip_corner_puller",
    "seasonal_period_AU17_chin_raiser",
]


def test_feature_diagnostics_matches_known_real_nan_counts():
    df, feature_columns = _load_fixture()

    report = feature_diagnostics(df, feature_columns=feature_columns).set_index("column")

    # permutation_entropy: NA in all 10 real rows (tsExpKit bug, see PIPELINE.md).
    assert report.loc["permutation_entropy", "n_nan"] == 10
    assert report.loc["permutation_entropy", "all_nan"]

    # The fixture's three constant-series rows send these to NA, and only these.
    constant_series_casualties = [
        "sample_entropy", "Kurtosis", "Skewness", "x_acf1", "x_acf10",
        "diff1_acf1", "diff1_acf10", "diff2_acf1", "diff2_acf10", "x_pacf5",
        "diff1x_pacf5", "diff2x_pacf5", "entropy", "nonlinearity",
        "unitroot_kpss", "unitroot_pp", "e_acf1", "e_acf10",
    ]
    for col in constant_series_casualties:
        assert report.loc[col, "n_nan"] == 3, col
        assert not report.loc[col, "all_nan"], col

    # Everything else in this fixture is fully observed.
    fully_observed = set(feature_columns) - set(constant_series_casualties) - {"permutation_entropy"}
    for col in fully_observed:
        assert report.loc[col, "n_nan"] == 0, col


def test_feature_cleaner_auto_drops_only_the_all_nan_column_by_default():
    df, feature_columns = _load_fixture()

    cleaner = FeatureCleaner(feature_columns=feature_columns, random_state=0)
    out = cleaner.fit_transform(df)

    # Real R drops permutation_entropy for exactly this reason (all-NA,
    # unimputable) -- the default `max_nan_fraction=1.0` rule reproduces
    # that specific drop without hardcoding the column name.
    assert cleaner.dropped_columns_ == ["permutation_entropy"]
    assert "permutation_entropy" not in out.columns

    # `length`/`nperiods`/`seasonal_period` are also genuinely constant
    # across all 10 real rows in this fixture (every series here has the
    # same length and frequency) -- a real, correct zero-SD drop, not an
    # artifact of the fixture being small.
    assert cleaner.zero_sd_columns_ == ["length", "nperiods", "seasonal_period"]

    # Every surviving feature column comes back fully imputed, no NaN left.
    assert not out[cleaner.feature_names_out_].isna().any().any()


def test_feature_diagnostics_matches_real_zero_sd_drop_on_full_scale_data():
    """`is_constant` reproduces R's real `sd(x, na.rm=T) == 0` zero-SD
    decision exactly -- on a subset of the real, full-scale (370-video)
    training set, not just a small synthetic or 10-row fixture."""
    imputed, zerosd, feature_columns = _load_imputed_zerosd_fixtures()

    report = feature_diagnostics(imputed, feature_columns=feature_columns)
    our_constant = sorted(report.loc[report["is_constant"], "column"])
    r_dropped = sorted(set(feature_columns) - set(zerosd.columns))

    assert our_constant == r_dropped == sorted(REAL_ZERO_SD_COLUMNS)


def test_feature_cleaner_drops_exactly_the_real_zero_sd_columns():
    imputed, zerosd, feature_columns = _load_imputed_zerosd_fixtures()

    # Already-imputed real data (no NaN left) -- impute=False isolates the
    # zero-SD step from IterativeImputer's own stochasticity, and changes
    # nothing here since there's nothing left to impute anyway.
    cleaner = FeatureCleaner(feature_columns=feature_columns, impute=False, random_state=0)
    out = cleaner.fit_transform(imputed)

    assert sorted(cleaner.zero_sd_columns_) == sorted(REAL_ZERO_SD_COLUMNS)
    assert set(out.columns) - {"video_filename", "isfakeorreal", "emotion", "valence"} == (
        set(zerosd.columns) - {"video_filename", "isfakeorreal", "emotion", "valence"}
    )
