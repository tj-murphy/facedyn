"""Unit tests for facedyn's own feature set.

Each feature is checked against a series whose value it has in closed form
(a ramp's trend, a square wave's episodes, a monotone series' permutation
entropy) rather than against a stored snapshot, so a wrong implementation
fails rather than a changed one.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from facedyn.features.timeseries import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    TimeSeriesFeatureExtractor,
    extract_features,
    extract_timeseries_features,
)


@pytest.fixture
def noise() -> np.ndarray:
    return np.random.default_rng(0).normal(size=200)


def test_returns_all_31_features_in_declared_order(noise):
    result = extract_features(noise)

    assert list(result.index) == FEATURE_NAMES
    assert len(result) == 31
    assert result["length"] == 200


def test_feature_groups_partition_feature_names():
    grouped = [name for group in FEATURE_GROUPS.values() for name in group]
    assert grouped == FEATURE_NAMES


# --------------------------------------------------------------------------
# distribution
# --------------------------------------------------------------------------


def test_distribution_features_match_their_definitions(noise):
    result = extract_features(noise)

    assert result["mean"] == pytest.approx(noise.mean())
    assert result["sd"] == pytest.approx(noise.std(ddof=1))
    assert result["min"] == pytest.approx(noise.min())
    assert result["max"] == pytest.approx(noise.max())
    assert result["range"] == pytest.approx(noise.max() - noise.min())
    assert result["median"] == pytest.approx(np.median(noise))
    q75, q25 = np.percentile(noise, [75, 25])
    assert result["iqr"] == pytest.approx(q75 - q25)
    assert result["skewness"] == pytest.approx(stats.skew(noise))
    assert result["kurtosis"] == pytest.approx(stats.kurtosis(noise))


def test_kurtosis_is_excess_so_normal_data_sits_near_zero():
    x = np.random.default_rng(3).normal(size=20000)
    assert extract_features(x)["kurtosis"] == pytest.approx(0, abs=0.1)


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------


def test_acf1_matches_hand_computed_biased_estimator(noise):
    xc = noise - noise.mean()
    expected = np.dot(xc[1:], xc[:-1]) / np.dot(xc, xc)

    assert extract_features(noise)["acf1"] == pytest.approx(expected)


def test_pacf1_equals_acf1(noise):
    """Levinson-Durbin's first step is the identity phi_11 = rho_1 -- there
    are no intervening lags to partial out."""
    result = extract_features(noise)
    assert result["pacf1"] == pytest.approx(result["acf1"])


def test_smooth_series_has_higher_acf1_than_white_noise(noise):
    smooth = np.cumsum(noise)
    assert extract_features(smooth)["acf1"] > extract_features(noise)["acf1"]


def test_diff_features_match_their_definitions(noise):
    result = extract_features(noise)
    d1 = np.diff(noise)

    assert result["diff1_sd"] == pytest.approx(d1.std(ddof=1))
    assert result["mean_abs_diff"] == pytest.approx(np.abs(d1).mean())


def test_sumsq_features_are_non_negative(noise):
    result = extract_features(noise)
    assert result["acf10_sumsq"] >= 0
    assert result["pacf5_sumsq"] >= 0


# --------------------------------------------------------------------------
# trend
# --------------------------------------------------------------------------


def test_perfect_ramp_has_exact_slope_and_unit_r2():
    x = 3.0 * np.arange(100, dtype=float) + 7.0

    result = extract_features(x)

    assert result["trend_slope"] == pytest.approx(3.0)
    assert result["trend_r2"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "x",
    [
        np.sin(np.linspace(0, 8 * np.pi, 400, endpoint=False)),
        np.random.default_rng(0).normal(size=200),
        np.arange(200) * 0.3 + np.random.default_rng(1).normal(size=200),
    ],
    ids=["sine", "noise", "ramp_plus_noise"],
)
def test_trend_matches_an_independent_ols_reference(x):
    """Cross-checked against ``scipy.stats.linregress`` rather than a
    hand-derived expectation, so the hand-rolled ``polyfit``/R-squared path
    is verified against a separate implementation of the same fit."""
    reference = stats.linregress(np.arange(len(x), dtype=float), x)

    result = extract_features(x)

    assert result["trend_slope"] == pytest.approx(reference.slope)
    assert result["trend_r2"] == pytest.approx(reference.rvalue**2)


def test_oscillation_has_far_weaker_linear_trend_than_a_ramp():
    sine = np.sin(np.linspace(0, 8 * np.pi, 400, endpoint=False))
    ramp = np.arange(400) * 0.3 + np.random.default_rng(1).normal(size=400)

    assert extract_features(sine)["trend_r2"] < 0.1
    assert extract_features(ramp)["trend_r2"] > 0.9


def test_ramp_crosses_its_mean_once_and_stays_above_for_half_its_length():
    x = np.arange(100, dtype=float)

    result = extract_features(x)

    assert result["mean_crossings"] == pytest.approx(1 / 100)
    assert result["longest_run_above_mean"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# complexity
# --------------------------------------------------------------------------


def test_monotone_series_has_zero_permutation_entropy():
    """Every ordinal window of a strictly increasing series has the same
    rank pattern, so the pattern distribution is a point mass."""
    assert extract_features(np.arange(100, dtype=float))["permutation_entropy"] == pytest.approx(0)


def test_white_noise_permutation_entropy_approaches_one(noise):
    """All order-3 patterns are near-equiprobable in i.i.d. noise, and the
    feature is normalised by log(3!)."""
    assert extract_features(noise)["permutation_entropy"] == pytest.approx(1.0, abs=0.05)


def test_permutation_entropy_is_bounded_in_unit_interval(noise):
    for x in (noise, np.arange(50, dtype=float), np.sin(np.linspace(0, 20, 300))):
        pe = extract_features(x)["permutation_entropy"]
        assert 0 <= pe <= 1


def test_sine_has_lower_spectral_entropy_than_noise(noise):
    """A single-frequency signal concentrates all its power in one bin;
    white noise spreads it evenly."""
    sine = np.sin(np.linspace(0, 8 * np.pi, 200, endpoint=False))

    assert extract_features(sine)["spectral_entropy"] < 0.1
    assert extract_features(noise)["spectral_entropy"] > 0.8


def test_sine_has_lower_sample_entropy_than_noise(noise):
    """Sample entropy measures how unpredictable the next value is given a
    matching recent history -- near-zero for a deterministic oscillation."""
    sine = np.sin(np.linspace(0, 8 * np.pi, 200, endpoint=False))

    assert extract_features(sine)["sample_entropy"] < extract_features(noise)["sample_entropy"]


# --------------------------------------------------------------------------
# activation
# --------------------------------------------------------------------------


@pytest.fixture
def square_wave() -> np.ndarray:
    """60 frames, 6 cycles of 5 high then 5 low frames."""
    return np.tile(np.concatenate([np.ones(5), np.zeros(5)]), 6)


def test_square_wave_activation_features_are_exact(square_wave):
    result = extract_features(square_wave)

    assert result["prop_active"] == pytest.approx(0.5)
    assert result["n_episodes"] == 6
    assert result["mean_episode_len"] == pytest.approx(5 / 60)
    assert result["max_episode_len"] == pytest.approx(5 / 60)
    # every rise is 0 -> 1 and every fall 1 -> 0
    assert result["mean_onset_slope"] == pytest.approx(1.0)
    assert result["mean_offset_slope"] == pytest.approx(-1.0)


def test_absolute_activation_threshold_overrides_the_mean(square_wave):
    """A cut above every value leaves nothing active, which the default
    mean-relative threshold can never do."""
    result = extract_features(square_wave, activation_threshold=5.0)

    assert result["prop_active"] == 0
    assert result["n_episodes"] == 0
    assert np.isnan(result["mean_episode_len"])


def test_median_activation_threshold_is_accepted(square_wave):
    result = extract_features(square_wave, activation_threshold="median")
    assert 0 <= result["prop_active"] <= 1


def test_unknown_activation_threshold_string_raises(square_wave):
    with pytest.raises(ValueError, match="activation_threshold"):
        extract_features(square_wave, activation_threshold="q90")


def test_peak_rate_counts_one_local_maximum_per_sine_cycle():
    n, cycles = 400, 8
    x = np.sin(np.linspace(0, 2 * np.pi * cycles, n, endpoint=False))

    assert extract_features(x)["peak_rate"] == pytest.approx(cycles / n)


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------


def test_constant_series_is_nan_where_undefined_and_finite_where_not():
    """Constant representative-AU series are real (10/1110 videos in the
    Paper 1 training set), so this is a routine case, not a pathology."""
    result = extract_features(np.full(120, 0.5))

    undefined = [
        "skewness", "kurtosis", "acf1", "acf2", "acf10_sumsq", "pacf1",
        "pacf5_sumsq", "diff1_sd", "mean_abs_diff", "trend_slope", "trend_r2",
        "mean_crossings", "longest_run_above_mean", "sample_entropy",
        "permutation_entropy", "spectral_entropy", "mean_episode_len",
        "max_episode_len", "mean_onset_slope", "mean_offset_slope",
    ]
    for name in undefined:
        assert np.isnan(result[name]), f"{name} should be NaN for a constant series"

    still_defined = {
        "mean": 0.5, "sd": 0.0, "min": 0.5, "max": 0.5, "range": 0.0,
        "median": 0.5, "iqr": 0.0, "prop_active": 0.0, "n_episodes": 0.0,
        "peak_rate": 0.0, "length": 120.0,
    }
    for name, expected in still_defined.items():
        assert result[name] == pytest.approx(expected), f"{name} should stay defined"


def test_near_constant_series_is_treated_as_constant():
    """Series that are constant apart from ~1e-30 of floating-point noise
    from upstream smoothing and normalisation must not slip past the
    constancy check -- an exact `std(x) == 0` test misses them and pushes
    a near-zero denominator through several divisions."""
    x = np.full(120, 0.37) + np.random.default_rng(0).normal(scale=1e-30, size=120)

    assert np.std(x) > 0  # not bit-identical
    assert np.isnan(extract_features(x)["acf1"])


def test_nan_input_raises_rather_than_silently_dropping_frames():
    x = np.arange(50, dtype=float)
    x[10] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        extract_features(x)


def test_very_short_series_returns_nan_not_a_crash():
    result = extract_features([1.0, 2.0, 3.0])

    assert result["length"] == 3
    assert np.isnan(result["sample_entropy"])
    assert np.isnan(result["spectral_entropy"])
    assert result["mean"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# frame-level API
# --------------------------------------------------------------------------


def make_wide(n_rows: int = 3, n_frames: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "video_filename": [f"v{i}" for i in range(n_rows)],
        "isfakeorreal": ["real"] * n_rows,
        "series": ["smth_AU01_r"] * n_rows,
        **{f"fr_{j + 1}": rng.random(n_rows) for j in range(n_frames)},
    })


def test_extract_timeseries_features_carries_metadata_and_adds_features():
    wide = make_wide()

    result = extract_timeseries_features(wide, n_jobs=1)

    assert len(result) == len(wide)
    assert list(result.columns) == ["video_filename", "isfakeorreal", "series", *FEATURE_NAMES]
    assert (result["length"] == 80).all()


def test_extract_timeseries_features_parallel_matches_serial():
    wide = make_wide(n_rows=4, seed=2)

    pd.testing.assert_frame_equal(
        extract_timeseries_features(wide, n_jobs=1),
        extract_timeseries_features(wide, n_jobs=2),
    )


def test_transformer_matches_the_plain_function():
    wide = make_wide()

    pd.testing.assert_frame_equal(
        TimeSeriesFeatureExtractor(n_jobs=1).fit_transform(wide),
        extract_timeseries_features(wide, n_jobs=1),
    )


def test_transformer_forwards_activation_threshold():
    wide = make_wide()

    result = TimeSeriesFeatureExtractor(activation_threshold=5.0, n_jobs=1).fit_transform(wide)

    assert (result["prop_active"] == 0).all()


def test_transformer_reports_feature_names_out():
    extractor = TimeSeriesFeatureExtractor().fit(make_wide())
    assert list(extractor.get_feature_names_out()) == FEATURE_NAMES


def test_transformer_requires_fit_before_transform():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        TimeSeriesFeatureExtractor().transform(make_wide())
