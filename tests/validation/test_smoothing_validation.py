"""Regression test: Python smoothing output vs. the R reference implementation.

Fixtures are a single video's rows (241 frames) extracted from the full
preprocessed dataset and from `zoo::rollmean(..., k=4, fill="extend",
align="left")` output produced by the original R pipeline, kept small so
this runs fast in CI instead of diffing the full ~112k-row dataset.
"""

from pathlib import Path

import pandas as pd

from facedyn.smoothing import RollingSmoother

FIXTURES = Path(__file__).parent / "fixtures"


def test_rolling_smoother_matches_r_output():
    raw = pd.read_csv(FIXTURES / "raw_20__exit_phone_room.csv")
    r_smoothed = pd.read_csv(FIXTURES / "r_smoothed_20__exit_phone_room.csv")

    python_smoothed = RollingSmoother(window=4).fit_transform(raw)

    smth_cols = [c for c in r_smoothed.columns if c.startswith("smth_")]
    assert smth_cols, "fixture should contain R-smoothed columns"

    for col in smth_cols:
        pd.testing.assert_series_equal(
            python_smoothed[col].reset_index(drop=True),
            r_smoothed[col].reset_index(drop=True),
            check_exact=False,
            atol=1e-8,
            check_names=False,
        )
