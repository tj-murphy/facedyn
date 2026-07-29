"""Regression test: `feature_diagnostics`/`FeatureCleaner` vs. real R output.

Reuses the already-committed `cmfts_output_subset.csv` fixture (10 real
rows from `final_analysis_NMF_check.Rmd`'s CMFTS output, see
`test_cmfts_r_bridge_validation.py` for provenance) rather than a new
fixture, since it already carries exactly the pathology this step exists to
handle: `permutation_entropy` is NA in all 10 real rows (root-caused,
elsewhere in this project, to a real bug in `tsExpKit::permutationEntropy`),
and several other columns (`sample_entropy`, `Kurtosis`/`Skewness`, the
ACF/PACF family, Burg `entropy`, `nonlinearity`, `unitroot_kpss`/`_pp`,
`e_acf1`/`e_acf10`) are NA in exactly 3 of the 10 rows, from the fixture's
three deliberately-included constant-series rows.

This validates the deterministic pieces (NaN counting, the all-NaN auto-drop
rule) against real R output directly. It does not validate imputation
*quality* against R's `missForest` -- that's a distinct, stochastic
comparison, out of scope here (see `PIPELINE.md` step 7).
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
