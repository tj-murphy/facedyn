import pandas as pd
import pytest

from facedyn.smoothing import RollingSmoother


def naive_left_rolling_mean_extend(values: list[float], k: int) -> list[float]:
    """Brute-force reference: left-aligned rolling mean, edge-extended.

    Independent of the vectorized reverse-roll-reverse implementation under
    test, so it serves as a meaningful correctness check rather than a
    restatement of the same trick.
    """
    n = len(values)
    out: list[float | None] = [None] * n
    for i in range(n):
        window = values[i : i + k]
        if len(window) == k:
            out[i] = sum(window) / k
    # forward-fill then back-fill to emulate R's fill="extend"
    last = None
    for i in range(n):
        if out[i] is None:
            out[i] = last
        else:
            last = out[i]
    nxt = None
    for i in range(n - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return out


@pytest.mark.parametrize("values,k", [
    ([1, 2, 3, 4, 5, 6], 4),
    ([1, 2, 3, 4, 5, 6], 2),
    ([5, 5, 5], 4),  # shorter than window
    ([1.0, 3.0, 2.0, 8.0, 4.0, 4.0, 9.0], 3),
])
def test_matches_naive_reference(values, k):
    df = pd.DataFrame({
        "video_filename": ["v1"] * len(values),
        "AU01_r": values,
    })
    smoother = RollingSmoother(window=k)
    out = smoother.fit_transform(df)

    expected = naive_left_rolling_mean_extend(values, k)
    actual = out["smth_AU01_r"].tolist()
    for a, e in zip(actual, expected):
        if e is None:
            assert pd.isna(a)
        else:
            assert a == pytest.approx(e)


def test_does_not_leak_across_video_groups():
    df = pd.DataFrame({
        "video_filename": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "AU01_r": [1, 1, 1, 1, 100, 100, 100, 100],
    })
    out = RollingSmoother(window=4).fit_transform(df)

    assert (out.loc[out["video_filename"] == "a", "smth_AU01_r"] == 1).all()
    assert (out.loc[out["video_filename"] == "b", "smth_AU01_r"] == 100).all()


def test_column_selection_by_pattern():
    df = pd.DataFrame({
        "video_filename": ["a", "a"],
        "AU01_r": [1, 2],
        "AU01_c": [0, 1],  # OpenFace presence column, should NOT be smoothed
        "frame": [1, 2],
    })
    out = RollingSmoother(window=2).fit_transform(df)

    assert "smth_AU01_r" in out.columns
    assert "smth_AU01_c" not in out.columns
    assert "smth_frame" not in out.columns


def test_explicit_columns_override_pattern():
    df = pd.DataFrame({
        "video_filename": ["a", "a"],
        "custom_signal": [1, 2],
    })
    out = RollingSmoother(window=2, columns=["custom_signal"]).fit_transform(df)

    assert "smth_custom_signal" in out.columns
