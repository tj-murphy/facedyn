import numpy as np
import pandas as pd
import pytest

from facedyn.features.cmfts import cmfts_features, extract_cmfts_features, reshape_for_cmfts

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

EXPECTED_COLUMNS = [
    "lempel_ziv", "aproximation_entropy", "sample_entropy", "permutation_entropy",
    "shannon_entropy_CS", "shannon_entropy_SG", "spectral_entropy", "nforbiden",
    "Kurtosis", "Skewness", "length", "x_acf1", "x_acf10", "diff1_acf1", "diff1_acf10",
    "diff2_acf1", "diff2_acf10", "x_pacf5", "diff1x_pacf5", "diff2x_pacf5", "entropy",
    "nonlinearity", "hurst", "stability", "lumpiness", "unitroot_kpss", "unitroot_pp",
    "nperiods", "seasonal_period", "trend", "spike", "linearity", "curvature",
    "e_acf1", "e_acf10", "max_level_shift", "time_level_shift", "max_var_shift",
    "time_var_shift", "max_kl_shift", "time_kl_shift",
]


def make_wide_input(n_videos: int = 3, n_frames: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_videos):
        for au in ["smth_AU01_r", "smth_AU06_r"]:
            rows.append({
                "video_filename": f"vid_{v}",
                "isfakeorreal": "real",
                "AU": au,
                **{f"fr_{i + 1}": rng.random() for i in range(n_frames)},
            })
    return pd.DataFrame(rows)


def test_reshape_for_cmfts_pivots_long_to_wide():
    n = 20
    df = pd.DataFrame({
        "video_filename": ["v1"] * n + ["v2"] * n,
        "isfakeorreal": ["real"] * (2 * n),
        "frame": list(range(1, n + 1)) * 2,
        "smth_AU01_r": np.arange(2 * n, dtype=float),
        "smth_AU06_r": np.arange(2 * n, dtype=float) * 2,
    })

    wide = reshape_for_cmfts(df, value_cols=["smth_AU01_r", "smth_AU06_r"])

    assert wide.shape[0] == 4  # 2 videos x 2 series
    assert set(wide["series"]) == {"smth_AU01_r", "smth_AU06_r"}
    frame_cols = [f"fr_{i}" for i in range(1, n + 1)]
    assert list(wide.columns[-n:]) == frame_cols
    row = wide[(wide["video_filename"] == "v1") & (wide["series"] == "smth_AU01_r")].iloc[0]
    np.testing.assert_array_equal(row[frame_cols].to_numpy(), np.arange(n, dtype=float))


def test_reshape_for_cmfts_keeps_extra_metadata_columns_by_default():
    df = pd.DataFrame({
        "video_filename": ["v1", "v1"],
        "frame": [1, 2],
        "emotion": ["happy", "happy"],
        "smth_AU01_r": [0.1, 0.2],
    })
    wide = reshape_for_cmfts(df, value_cols=["smth_AU01_r"])
    assert "emotion" in wide.columns


def test_extract_cmfts_features_returns_all_41_features_in_order():
    rng = np.random.default_rng(0)
    series = rng.random(120)

    result = extract_cmfts_features(series)

    assert list(result.index) == EXPECTED_COLUMNS
    assert len(result) == 41
    assert result["length"] == 120


def test_extract_cmfts_features_permutation_entropy_always_nan():
    """Reproduces a real (confirmed, not our bug) upstream tsExpKit bug --
    see cmfts.py's module docstring."""
    rng = np.random.default_rng(1)
    result = extract_cmfts_features(rng.random(100))
    assert np.isnan(result["permutation_entropy"])


def test_extract_cmfts_features_constant_series_matches_real_r_nan_pattern():
    """Matches the exact NaN-vs-finite pattern confirmed against a real
    constant representative-AU series in R's actual training-set output
    (see test_cmfts_validation.py and cmfts.py's `_is_constant` handling)."""
    # 0.5 (exactly representable in binary) round-trips through mean/std
    # with exactly zero floating-point noise, matching the real constant
    # AU rows this mirrors (also exactly std==0, not just "very small") --
    # an arbitrary non-power-of-2 constant like 0.37 would instead pick up
    # tiny sum/divide rounding noise and *not* reproduce this NaN pattern,
    # for reasons that are about float64 representability, not this code.
    series = np.full(120, 0.5)

    result = extract_cmfts_features(series)

    should_be_nan = [
        "sample_entropy", "permutation_entropy", "Kurtosis", "Skewness",
        "x_acf1", "x_acf10", "diff1_acf1", "diff1_acf10", "diff2_acf1", "diff2_acf10",
        "x_pacf5", "diff1x_pacf5", "diff2x_pacf5", "entropy", "nonlinearity",
        "unitroot_kpss", "unitroot_pp", "e_acf1", "e_acf10",
    ]
    for col in should_be_nan:
        assert np.isnan(result[col]), f"{col} should be NaN for a constant series"

    should_be_finite = [
        "lempel_ziv", "aproximation_entropy", "shannon_entropy_SG", "spectral_entropy",
        "nforbiden", "length", "hurst", "stability", "lumpiness", "nperiods",
        "seasonal_period", "trend", "spike", "linearity", "curvature",
        "max_level_shift", "max_var_shift", "max_kl_shift",
    ]
    for col in should_be_finite:
        assert np.isfinite(result[col]), f"{col} should be finite for a constant series"


def test_cmfts_features_end_to_end_shape_and_metadata():
    wide = make_wide_input(n_videos=2, n_frames=50)

    result = cmfts_features(wide, n_jobs=1)

    assert len(result) == len(wide)
    assert set(wide.columns) - {c for c in wide.columns if c.startswith("fr_")} <= set(result.columns)
    for col in EXPECTED_COLUMNS:
        assert col in result.columns


def test_cmfts_features_n_jobs_matches_serial():
    wide = make_wide_input(n_videos=2, n_frames=50, seed=2)

    serial = cmfts_features(wide, n_jobs=1)
    parallel = cmfts_features(wide, n_jobs=2)

    pd.testing.assert_frame_equal(serial, parallel)
