import numpy as np
import pandas as pd
import pytest

from facedyn.features.cleaning import FeatureCleaner, feature_diagnostics


def test_feature_diagnostics_reports_nan_inf_sd_and_constancy():
    df = pd.DataFrame({
        "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
        "has_nan": [1.0, np.nan, 3.0, np.nan, 5.0],
        "has_inf": [1.0, np.inf, -np.inf, 4.0, 5.0],
        "all_nan": [np.nan] * 5,
        "constant": [2.0] * 5,
    })
    original = df.copy()

    report = feature_diagnostics(df)

    pd.testing.assert_frame_equal(df, original)  # read-only

    by_col = report.set_index("column")
    assert by_col.loc["normal", "n_nan"] == 0
    assert not by_col.loc["normal", "is_constant"]
    assert by_col.loc["has_nan", "n_nan"] == 2
    assert by_col.loc["has_nan", "pct_nan"] == 2 / 5
    assert not by_col.loc["has_nan", "all_nan"]
    assert by_col.loc["has_inf", "n_inf"] == 2
    assert by_col.loc["has_inf", "n_nan"] == 0  # not converted here -- read-only report
    assert by_col.loc["all_nan", "n_nan"] == 5
    assert by_col.loc["all_nan", "all_nan"]
    assert by_col.loc["constant", "is_constant"]
    assert by_col.loc["constant", "sd"] == 0.0


def test_feature_diagnostics_infers_numeric_columns_when_not_given():
    df = pd.DataFrame({
        "video_filename": ["a", "b", "c"],
        "feature_x": [1.0, 2.0, 3.0],
    })

    report = feature_diagnostics(df)

    assert list(report["column"]) == ["feature_x"]


def _sample_features(n: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "video_filename": [f"vid_{i}" for i in range(n)],
        "series": ["AU12"] * n,
        "feature_a": rng.normal(size=n),
        "feature_b": rng.normal(size=n),
    })


def test_feature_cleaner_drops_all_nan_column_and_warns():
    df = _sample_features()
    df["broken"] = np.nan

    cleaner = FeatureCleaner(
        feature_columns=["feature_a", "feature_b", "broken"], random_state=0
    )
    with pytest.warns(UserWarning, match="missing values"):
        cleaner.fit(df)

    assert cleaner.dropped_columns_ == ["broken"]
    assert "broken" not in cleaner.feature_names_out_


def test_feature_cleaner_converts_inf_to_nan_before_imputing():
    df = _sample_features()
    df.loc[0, "feature_a"] = np.inf
    df.loc[1, "feature_a"] = -np.inf

    cleaner = FeatureCleaner(feature_columns=["feature_a", "feature_b"], random_state=0)
    out = cleaner.fit_transform(df)

    assert np.isfinite(out["feature_a"]).all()
    assert not out["feature_a"].isna().any()


def test_feature_cleaner_drops_near_zero_sd_columns_after_imputation():
    n = 15
    df = pd.DataFrame({
        "video_filename": [f"vid_{i}" for i in range(n)],
        "varying": np.linspace(0, 1, n),
        # constant, but with float noise -- not exactly `sd == 0`, matching
        # what real constant AU-derived series look like upstream
        "near_constant": np.full(n, 0.5) + np.random.default_rng(1).normal(scale=1e-30, size=n),
    })

    cleaner = FeatureCleaner(
        feature_columns=["varying", "near_constant"], impute=False, random_state=0
    )
    with pytest.warns(UserWarning, match="near-zero-variance"):
        out = cleaner.fit_transform(df)

    assert cleaner.zero_sd_columns_ == ["near_constant"]
    assert "near_constant" not in out.columns
    assert "varying" in out.columns


def test_feature_cleaner_respects_explicit_drop_columns():
    df = _sample_features()
    df["unwanted"] = np.linspace(0, 1, len(df))

    cleaner = FeatureCleaner(
        feature_columns=["feature_a", "feature_b", "unwanted"],
        drop_columns=["unwanted"],
        impute=False,
        drop_zero_sd=False,
        random_state=0,
    )
    with pytest.warns(UserWarning, match="Dropping"):
        out = cleaner.fit_transform(df)

    assert cleaner.dropped_columns_ == ["unwanted"]
    assert "unwanted" not in out.columns


def test_feature_cleaner_impute_false_leaves_nan_in_place():
    df = _sample_features()
    df.loc[0, "feature_a"] = np.nan

    cleaner = FeatureCleaner(
        feature_columns=["feature_a", "feature_b"],
        impute=False,
        drop_zero_sd=False,
        random_state=0,
    )
    out = cleaner.fit_transform(df)

    assert cleaner.imputer_ is None
    assert out["feature_a"].isna().sum() == 1


def test_feature_cleaner_zero_sd_decision_fit_on_train_reused_on_transform():
    train = pd.DataFrame({
        "video_filename": [f"vid_{i}" for i in range(10)],
        "au": np.linspace(0, 1, 10),  # varies in train
    })
    test = pd.DataFrame({
        "video_filename": [f"vid_{i}" for i in range(10, 15)],
        "au": np.full(5, 0.3),  # constant in test only
    })

    cleaner = FeatureCleaner(feature_columns=["au"], impute=False, random_state=0)
    cleaner.fit(train)
    assert cleaner.zero_sd_columns_ == []  # train varies -> decision is "keep"

    out_test = cleaner.transform(test)

    # Reuses train's decision rather than recomputing on test, so "au"
    # survives even though it's constant in this particular test batch.
    assert "au" in out_test.columns
    pd.testing.assert_series_equal(
        out_test["au"].reset_index(drop=True), test["au"].reset_index(drop=True)
    )


def test_feature_cleaner_metadata_columns_pass_through_unchanged():
    df = _sample_features()

    cleaner = FeatureCleaner(
        feature_columns=["feature_a", "feature_b"],
        impute=False,
        drop_zero_sd=False,
        random_state=0,
    )
    out = cleaner.fit_transform(df)

    pd.testing.assert_series_equal(
        out["video_filename"].reset_index(drop=True), df["video_filename"].reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        out["series"].reset_index(drop=True), df["series"].reset_index(drop=True)
    )


def test_feature_cleaner_get_feature_names_out_matches_output_columns():
    df = _sample_features()
    df["broken"] = np.nan

    cleaner = FeatureCleaner(
        feature_columns=["feature_a", "feature_b", "broken"], impute=False, random_state=0
    )
    with pytest.warns(UserWarning):
        out = cleaner.fit_transform(df)

    metadata_cols = {"video_filename", "series"}
    feature_cols_out = [c for c in out.columns if c not in metadata_cols]
    assert list(cleaner.get_feature_names_out()) == feature_cols_out
