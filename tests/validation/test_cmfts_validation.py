"""Regression test: cmfts_features vs. real R `cmfts::cmfts()` output.

Fixtures (`cmfts_input_subset.csv`/`cmfts_output_subset.csv`,
`tests/validation/fixtures/`) are a 10-row subset of the real
`R Validation Data/r_cmfts_input.csv` -> `r_cmfts_output.csv` pair (370
real training videos x 3 representative AUs, from
`final_analysis_NMF_check.Rmd`) -- deliberately including several of the
10 rows whose representative-AU series is exactly constant for that whole
video (indices 40, 165, 187 in the original data), since that's the edge
case most CMFTS measures go NaN on, plus one row (847) with the single
largest observed `nonlinearity` discrepancy in the full 1110-row set.

A full 1110-row validation was run interactively (not committed -- see
`PIPELINE.md` step 6): 24 of the 41 features match real R to
near-floating-point precision on essentially every row. The remaining 17
fall into three documented categories, reflected in the per-feature
tolerances below rather than asserted as exact:

1. **`hurst`** and the **`stl_features` group** (`trend`, `spike`,
   `linearity`, `curvature`, `e_acf1`, `e_acf10`) are *approximations*, not
   faithful ports -- R's real algorithms (`fracdiff`'s exact/truncated
   Gaussian MLE; `stats::supsmu`, Friedman's variable-span smoother) are
   implemented in Fortran with no closed form to transcribe. See
   `_hurst`/`_stl_features` in `src/facedyn/features/cmfts.py` for the
   approximation strategy and validated typical error magnitude.
2. `max_level_shift`/`max_var_shift`/`max_kl_shift` (+ their `time_*`
   companions) occasionally (~1-3% of real rows) pick a different
   near-tied local peak than the given fixture -- confirmed this is
   floating-point tie sensitivity in R itself (a *fresh* live-R run on the
   same input can also disagree with the given fixture on those same
   rows), not a bug in this port.
3. `shannon_entropy_CS` and `nonlinearity` can reach very large magnitudes
   (Chao-Shen's coverage correction is pathological on continuous data
   fed to it as if it were bin counts -- see module docstring) where
   `atol`-style comparisons are meaningless; checked with a relative
   tolerance instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from facedyn.features.cmfts import cmfts_features

FIXTURES = Path(__file__).parent / "fixtures"

# Match to near floating-point precision on every fixture row.
EXACT_COLUMNS = [
    "lempel_ziv", "aproximation_entropy", "sample_entropy", "shannon_entropy_SG",
    "spectral_entropy", "nforbiden", "Kurtosis", "Skewness", "length",
    "x_acf1", "x_acf10", "diff1_acf1", "diff1_acf10", "diff2_acf1", "diff2_acf10",
    "x_pacf5", "diff1x_pacf5", "diff2x_pacf5", "entropy", "stability", "lumpiness",
    "unitroot_kpss", "nperiods", "seasonal_period", "time_kl_shift",
]
# Occasional argmax tie-sensitivity (see module docstring, category 2).
LOOSE_COLUMNS = {
    "unitroot_pp": 1e-3,
    "max_level_shift": 0.3,
    "time_level_shift": 200,
    "max_var_shift": 0.4,
    "time_var_shift": 200,
    "max_kl_shift": 5.0,
}
# Documented approximations (category 1) -- generous, order-of-magnitude tolerances.
APPROXIMATE_COLUMNS = {
    "hurst": 0.4,  # the constant-series rows in this fixture are the worst case observed
    "trend": 0.25,
    "spike": 0.001,
    "linearity": 1.0,
    "curvature": 1.5,
    "e_acf1": 0.2,
    "e_acf10": 1.0,
}
# Large-magnitude columns (category 3) -- relative, not absolute, tolerance.
RELATIVE_COLUMNS = {"shannon_entropy_CS": 1e-6, "nonlinearity": 0.25}


def _assert_columns_close(actual: pd.Series, expected: pd.Series, atol: float, label: str):
    a, e = actual.to_numpy(dtype=float), expected.to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(e)
    both_inf = np.isinf(a) & np.isinf(e) & (np.sign(a) == np.sign(e))
    skip = both_nan | both_inf
    assert not (np.isnan(a) != np.isnan(e)).any(), f"{label}: NaN pattern mismatch"
    assert not (np.isinf(a) != np.isinf(e)).any(), f"{label}: Inf pattern mismatch"
    np.testing.assert_allclose(a[~skip], e[~skip], atol=atol, rtol=0, err_msg=label)


def test_cmfts_features_matches_real_r_output():
    inp = pd.read_csv(FIXTURES / "cmfts_input_subset.csv")
    expected = pd.read_csv(FIXTURES / "cmfts_output_subset.csv").reset_index(drop=True)

    result = cmfts_features(inp, n_jobs=1).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result[["video_filename", "AU"]], expected[["video_filename", "AU"]]
    )

    assert result["permutation_entropy"].isna().all()

    for col in EXACT_COLUMNS:
        _assert_columns_close(result[col], expected[col], atol=1e-6, label=col)
    for col, atol in LOOSE_COLUMNS.items():
        _assert_columns_close(result[col], expected[col], atol=atol, label=col)
    for col, atol in APPROXIMATE_COLUMNS.items():
        _assert_columns_close(result[col], expected[col], atol=atol, label=col)
    for col, rtol in RELATIVE_COLUMNS.items():
        a, e = result[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float)
        both_inf = np.isinf(a) & np.isinf(e)
        np.testing.assert_allclose(a[~both_inf], e[~both_inf], rtol=rtol, err_msg=col)
