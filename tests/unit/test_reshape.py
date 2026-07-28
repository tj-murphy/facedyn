import numpy as np
import pandas as pd
import pytest

from facedyn.features.reshape import apply_rowwise, reshape_to_wide, split_wide


def test_reshape_to_wide_one_row_per_video_and_series():
    n = 20
    df = pd.DataFrame({
        "video_filename": ["v1"] * n + ["v2"] * n,
        "isfakeorreal": ["real"] * (2 * n),
        "frame": list(range(1, n + 1)) * 2,
        "smth_AU01_r": np.arange(2 * n, dtype=float),
        "smth_AU06_r": np.arange(2 * n, dtype=float) * 2,
    })

    wide = reshape_to_wide(df, value_cols=["smth_AU01_r", "smth_AU06_r"])

    assert wide.shape[0] == 4  # 2 videos x 2 series
    assert set(wide["series"]) == {"smth_AU01_r", "smth_AU06_r"}
    frame_cols = [f"fr_{i}" for i in range(1, n + 1)]
    assert list(wide.columns[-n:]) == frame_cols
    row = wide[(wide["video_filename"] == "v1") & (wide["series"] == "smth_AU01_r")].iloc[0]
    np.testing.assert_array_equal(row[frame_cols].to_numpy(), np.arange(n, dtype=float))


def test_reshape_to_wide_drops_frame_varying_columns_not_fragments_pivot():
    """Regression test for a real bug: RepresentativeAUSelector.transform's
    output (this function's intended input) carries through *every*
    non-factorized column, including ones that vary per-frame (raw AU
    intensities, timestamp, confidence, ...), not just genuine per-video
    metadata. Auto-including a frame-varying column in `id_vars` used to
    silently fragment the pivot into one row per (video, frame, series)
    instead of (video, series) -- confirmed to actually happen against
    real data (200x too many rows, ~99.6% NaN) before this was fixed."""
    n = 20
    df = pd.DataFrame({
        "video_filename": ["v1"] * n,
        "isfakeorreal": ["real"] * n,  # constant per video -> safe, should be kept
        "timestamp": np.arange(n, dtype=float),  # varies per frame -> must be dropped
        "frame": list(range(1, n + 1)),
        "smth_AU01_r": np.arange(n, dtype=float),
    })

    with pytest.warns(UserWarning, match="timestamp"):
        wide = reshape_to_wide(df, value_cols=["smth_AU01_r"])

    assert wide.shape[0] == 1  # one video x one series, not one row per frame
    assert "isfakeorreal" in wide.columns
    assert "timestamp" not in wide.columns
    frame_cols = [f"fr_{i}" for i in range(1, n + 1)]
    assert wide[frame_cols].notna().all(axis=None)  # no fragmented-NaN rows


def test_reshape_to_wide_keeps_extra_metadata_columns_by_default():
    df = pd.DataFrame({
        "video_filename": ["v1", "v1"],
        "frame": [1, 2],
        "emotion": ["happy", "happy"],
        "smth_AU01_r": [0.1, 0.2],
    })
    wide = reshape_to_wide(df, value_cols=["smth_AU01_r"])
    assert "emotion" in wide.columns


def make_wide(n_rows: int = 3, n_frames: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "video_filename": [f"v{i}" for i in range(n_rows)],
        "series": ["smth_AU01_r"] * n_rows,
        **{f"fr_{j + 1}": rng.random(n_rows) for j in range(n_frames)},
    })


def test_split_wide_separates_metadata_from_values():
    wide = make_wide(n_rows=3, n_frames=10)

    metadata, values = split_wide(wide)

    assert list(metadata.columns) == ["video_filename", "series"]
    assert values.shape == (3, 10)
    np.testing.assert_allclose(values[:, 0], wide["fr_1"].to_numpy())


def test_split_wide_raises_when_no_frame_columns_match():
    df = pd.DataFrame({"video_filename": ["v1"], "t_1": [0.5]})

    with pytest.raises(ValueError, match="No frame-value columns matched"):
        split_wide(df)


def _row_sum(row):
    """Module-level (picklable) so the n_jobs>1 test can actually fork."""
    return pd.Series({"total": float(np.sum(row)), "n": float(len(row))})


def test_apply_rowwise_carries_metadata_and_appends_features():
    wide = make_wide(n_rows=3, n_frames=10)

    out = apply_rowwise(wide, _row_sum, n_jobs=1)

    assert len(out) == len(wide)
    assert list(out.columns) == ["video_filename", "series", "total", "n"]
    assert (out["n"] == 10).all()
    np.testing.assert_allclose(
        out["total"].to_numpy(),
        wide[[c for c in wide.columns if c.startswith("fr_")]].to_numpy().sum(axis=1),
    )


def test_apply_rowwise_parallel_matches_serial():
    wide = make_wide(n_rows=4, n_frames=10)

    pd.testing.assert_frame_equal(
        apply_rowwise(wide, _row_sum, n_jobs=1),
        apply_rowwise(wide, _row_sum, n_jobs=2),
    )
