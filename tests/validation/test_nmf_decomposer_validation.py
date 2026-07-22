"""Regression test: NMFDecomposer vs. real RcppML output, both the
activation matrix and the basis matrix.

All fixtures live in `R Validation Data/` (gitignored, not committed --
large private exports) and were pulled from the same real run of
`final_analysis.Rmd`'s "Run NMF" section on the actual 370-video training
set (89,170 rows), after `set.seed(12345)`, `RcppML::nmf(input_matrix, k =
3, tol = 1e-6, maxit = 750)`:

- `r_optimal_k_input.csv` -- the real post-normalisation training input
  (human-readable AU column names).
- `r_NMF.csv` -- R's real `dta_nmf_output` (per-frame activations, already
  row-aligned with the input above by `video_filename`/`frame`, confirmed
  below).
- `r_model_nmf_w.csv` -- R's real `model_nmf$w` (the raw AU-by-component
  basis matrix, 17x3, row order matching `r_optimal_k_input.csv`'s AU
  columns).
- `r_model_nmf_h.csv` / `r_model_nmf_d.csv` -- R's real `model_nmf$h` / `$d`
  (raw, un-rescaled activations and the scale vector). Not used directly by
  either test below, but were used interactively to confirm
  `w @ diag(d) @ h` reconstructs `r_optimal_k_input.csv`'s AU matrix with
  the exact same MSE (~0.619) as `nmf_rank_mse_sweep` computes independently
  -- i.e. these three files are mutually consistent and correctly oriented,
  not just individually plausible.

Unconstrained NMF only identifies components up to (a) a permutation and
(b) an arbitrary positive rescaling of each component -- for any positive
diagonal S, `(W, H)` and `(W @ S, inv(S) @ H)` reconstruct the same data
equally well. R's parameterization (`A ~= W . diag(d) . H`) pins that scale
down via `d`; sklearn's solver doesn't separate scale out at all, so an
exact numeric match isn't expected even on a correct fit -- see
`NMFDecomposer`'s docstring for the full writeup, including an earlier
untested assumption the first test below disproved (that sklearn's output
would already equal R's scaled H with no rescaling needed -- empirically
each component lands on its own, different, arbitrary scale). So both
tests follow the project's standard approach for stochastic/
non-identifiable steps (see PIPELINE.md's validation protocol): match
components between the two fits via Hungarian assignment on
cross-correlation, then check each matched pair is related by a clean
proportional scale, not just a loose linear one -- i.e. the two fits found
the same underlying structure, modulo the expected scale ambiguity. As a
sanity cross-check, both tests independently recover the *same* matched
permutation (py component 0 <-> R's nmf1, py 1 <-> R's nmf3, py 2 <-> R's
nmf2) and their per-component scale factors are each other's reciprocal, as
the `(W @ S, inv(S) @ H)` identity above predicts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from facedyn.nmf import NMFDecomposer

FIXTURES_DIR = Path(__file__).parents[2] / "R Validation Data"
INPUT_PATH = FIXTURES_DIR / "r_optimal_k_input.csv"
ACTIVATIONS_PATH = FIXTURES_DIR / "r_NMF.csv"
BASIS_PATH = FIXTURES_DIR / "r_model_nmf_w.csv"

pytestmark = pytest.mark.skipif(
    not (INPUT_PATH.exists() and ACTIVATIONS_PATH.exists() and BASIS_PATH.exists()),
    reason="R Validation Data/*.csv not available locally",
)


def _assert_matched_components(py_matrix: np.ndarray, r_matrix: np.ndarray, residual_tol: float):
    """Match columns of `py_matrix`/`r_matrix` via Hungarian assignment on
    cross-correlation, then assert each matched pair is a clean
    through-origin proportional relationship. Returns the matched index
    pairs so callers can cross-check permutations against each other."""
    n = py_matrix.shape[1]
    corr = np.corrcoef(py_matrix.T, r_matrix.T)[:n, n:]
    py_idx, r_idx = linear_sum_assignment(-np.abs(corr))
    assert sorted(py_idx) == list(range(n))
    assert sorted(r_idx) == list(range(n))

    matched_corrs = corr[py_idx, r_idx]
    assert (matched_corrs > 0.999).all(), matched_corrs

    for py_i, r_i in zip(py_idx, r_idx):
        py_col = py_matrix[:, py_i]
        r_col = r_matrix[:, r_i]
        slope = np.dot(py_col, r_col) / np.dot(py_col, py_col)
        residual = r_col - slope * py_col
        relative_residual_norm = np.linalg.norm(residual) / np.linalg.norm(r_col)
        assert relative_residual_norm < residual_tol, (py_i, r_i, relative_residual_norm)

    return py_idx, r_idx


def test_activations_match_rcppml_components_up_to_scale():
    X = pd.read_csv(INPUT_PATH)
    r_out = pd.read_csv(ACTIVATIONS_PATH)

    assert len(X) == len(r_out)
    assert (X["video_filename"] == r_out["video_filename"]).all()
    assert (X["frame"] == r_out["frame"]).all()

    au_cols = [c for c in X.columns if c.startswith("AU")]
    decomposer = NMFDecomposer(n_components=3, columns=au_cols, random_state=0)
    py_activations = decomposer.fit_transform(X)[["nmf1", "nmf2", "nmf3"]].to_numpy()
    r_activations = r_out[["nmf1", "nmf2", "nmf3"]].to_numpy()

    _assert_matched_components(py_activations, r_activations, residual_tol=0.02)


def test_basis_matrix_matches_rcppml_w_up_to_scale():
    X = pd.read_csv(INPUT_PATH)
    au_cols = [c for c in X.columns if c.startswith("AU")]
    r_w = pd.read_csv(BASIS_PATH).to_numpy()  # (n_features, n_components), no row names
    assert r_w.shape == (len(au_cols), 3)

    decomposer = NMFDecomposer(n_components=3, columns=au_cols, random_state=0).fit(X)
    py_w = decomposer.components_.T  # (n_features, n_components)

    _assert_matched_components(py_w, r_w, residual_tol=0.01)


def test_normalized_basis_matrix_closely_matches_rcppml():
    """The version actually published (`apply(model_nmf$w, 2, fn_maxnormalise)`
    in final_analysis.Rmd, and facedyn.nmf.plot_nmf_basis_heatmap's default):
    per-column min-max scaling should cancel the scale ambiguity entirely,
    so this is a much tighter check than the raw-scale ones above."""
    X = pd.read_csv(INPUT_PATH)
    au_cols = [c for c in X.columns if c.startswith("AU")]
    r_w = pd.read_csv(BASIS_PATH).to_numpy()

    decomposer = NMFDecomposer(n_components=3, columns=au_cols, random_state=0).fit(X)
    py_w = decomposer.components_.T

    def max_normalize(m):
        col_min, col_max = m.min(axis=0), m.max(axis=0)
        return (m - col_min) / (col_max - col_min)

    py_idx, r_idx = _assert_matched_components(py_w, r_w, residual_tol=0.01)

    py_norm = max_normalize(py_w)[:, py_idx]
    r_norm = max_normalize(r_w)[:, r_idx]
    assert np.abs(py_norm - r_norm).max() < 0.01
