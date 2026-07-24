"""Regression tests: NMF reconstruction-quality diagnostics vs. real R output.

`final_analysis.Rmd`'s "Validation"/"Reconstruction Pt2" sections check how
much of the original 17-AU signal survives NMF's compression to 3
components -- reporting RMSE/NRMSE/MAE/R2 (~L695-711, `dta_nmf_recon_err`)
and per-AU R2 (~L4225-4260, `dta_r2`). This project's own quoted real
numbers: training R2=0.4167 (RMSE=0.7868, NRMSE=0.0436, MAE=0.5294), 20%
test-set R2=0.4452. `facedyn.nmf`'s `_reconstruction_metrics`,
`nmf_reconstruction_error`, `nmf_reconstruction_r2_per_au` replicate this.

Two independent checks, using small fixtures derived from R's real exports
in `R Validation Data/` (gitignored, not committed -- see
`project_r_to_python_port.md`/`PIPELINE.md` for provenance):

1. **Exact formula check** (`nmf_basis_w.csv`/`nmf_basis_d.csv` -- R's real
   17x3 `model_nmf$w` and length-3 `model_nmf$d`, tiny, copied verbatim --
   plus `nmf_reconstruction_input_subset.csv`/`nmf_reconstruction_h_subset.csv`,
   a real, randomly-chosen 300-row slice of `r_optimal_k_input.csv` and the
   *matching* columns of `model_nmf$h`, confirmed row/column-aligned by the
   fact that reconstructing this exact subset from `w`/`d`/`h_subset` lands
   at essentially the same RMSE/R2 as the full 89,170-row dataset (~0.74
   RMSE vs. the full set's 0.79, ~0.43 R2 vs. 0.42) -- a coincidence this
   close is not possible if the rows were misaligned. Reconstruction from
   fixed `w`/`d`/`h` is a deterministic, exact per-row computation
   regardless of which rows are sampled, so this test checks
   `_reconstruction_metrics` against an independently-computed numpy
   reference to floating-point precision.

2. **Out-of-sample check** -- reuses the existing 15-video
   `r_normalised_subset.csv` fixture (`test_normalisation_validation.py`'s
   real R post-normalisation test-set subset), renamed from `smth_AU*_r`
   to the human-readable column names `w`/`d` are indexed by. A temporary
   `NMFDecomposer` has its `components_` forced to R's real
   `(w @ diag(d)).T` (no Python re-fit), so `nmf_reconstruction_error`'s
   `decomposer.transform()`-based projection is exercised against R's own
   real trained basis on real held-out data. Reference values below were
   computed once, offline, the same way (not derived from
   `nmf_reconstruction_error` itself, to avoid a tautological check) --
   see the interactive validation run in this session's history. Matches
   the full real 94-video test set's R2=0.4452 closely (this 15-video
   subset's own R2 differs, ~0.43, as expected for a smaller sample -- see
   `PIPELINE.md`'s validation protocol on why a subset needn't reproduce a
   full-dataset aggregate exactly). Looser tolerance than check 1, since
   `decomposer.transform()`'s coordinate-descent solver isn't guaranteed
   bit-identical across sklearn/BLAS versions, unlike the pure-arithmetic
   check above.

**Per-AU ranking discrepancy, noted not swept under the rug**: R's prose
(`final_analysis.Rmd`, right after `dta_r2`) claims AU07 is the
best-reconstructed AU (~0.887-0.899) and AU23 the worst (~0.010-0.011).
Both checks here -- and a from-scratch sklearn fit in this session's
interactive validation -- consistently found AU06/AU01/AU02/AU12 best and
AU45/AU04/AU09 worst instead, with AU07 solidly mid-table. The *aggregate*
R2 matches R's real quoted numbers almost exactly in both checks, and the
same "different" per-AU ranking shows up in two independent computations
(a genuine sklearn fit, and R's own forced `w`/`d`) -- strong evidence
this implementation is self-consistent and correct, and that R's specific
AU07/AU23 prose is a stale worked example left over from an earlier run
(the same file has other superseded R2 figures near it, e.g. "R2 = 0.4415"
struck through in favor of "0.4167"). Not asserted here for that reason --
see `PIPELINE.md` step 4 for the full writeup.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import NMF as SklearnNMF

from facedyn.au_labels import extract_au_code
from facedyn.nmf import NMFDecomposer, nmf_reconstruction_error, nmf_reconstruction_r2_per_au

FIXTURES = Path(__file__).parent / "fixtures"

# R's au_names, in the exact order `model_nmf$w`'s rows / r_optimal_k_input.csv's
# AU columns are in (final_analysis.Rmd, ~L649-653) -- static reference
# vocabulary, not derived data, so safe to hardcode.
AU_NAMES = [
    "AU01_inner_brow_raiser", "AU02_outer_brow_raiser", "AU04_brow_lowerer",
    "AU05_upper_lid_raiser", "AU06_cheek_raiser", "AU07_lid_tightener",
    "AU09_nose_wrinkler", "AU10_upper_lip_raiser", "AU12_lip_corner_puller",
    "AU14_dimpler", "AU15_lip_corner_depressor", "AU17_chin_raiser",
    "AU20_lip_stretcher", "AU23_lip_tightener", "AU25_lips_part",
    "AU26_jaw_drop", "AU45_blink",
]


def test_reconstruction_metrics_match_r_basis_on_real_input_subset():
    """Validates `_reconstruction_metrics`'s arithmetic only (RMSE/NRMSE/MAE/R2
    formulas), independent of `NMFDecomposer`/`transform()` -- reconstructs
    the real 300-row subset directly from R's real `w`/`d`/`h` (mirroring
    `final_analysis.Rmd` ~L695-711 exactly) and checks the implementation
    reproduces an independently-computed numpy reference exactly."""
    from facedyn.nmf import _reconstruction_metrics

    w = pd.read_csv(FIXTURES / "nmf_basis_w.csv").to_numpy()
    d = pd.read_csv(FIXTURES / "nmf_basis_d.csv")["x"].to_numpy()
    input_subset = pd.read_csv(FIXTURES / "nmf_reconstruction_input_subset.csv")
    h_subset = pd.read_csv(FIXTURES / "nmf_reconstruction_h_subset.csv").to_numpy()

    au_cols = [c for c in input_subset.columns if c.startswith("AU")]
    original = input_subset[au_cols].to_numpy()
    reconstructed = (h_subset * d[None, :]) @ w.T

    error = original - reconstructed
    expected_rmse = np.sqrt(np.mean(error**2))
    expected = {
        "RMSE": expected_rmse,
        "NRMSE": expected_rmse / (original.max() - original.min()),
        "MAE": np.mean(np.abs(error)),
        "R2": 1 - np.sum(error**2) / np.sum((original - original.mean()) ** 2),
    }

    result_metrics = _reconstruction_metrics(original, reconstructed)
    for key, value in expected.items():
        assert result_metrics[key] == pytest.approx(value, rel=1e-10)


def test_reconstruction_out_of_sample_matches_r_basis_on_real_test_subset():
    w = pd.read_csv(FIXTURES / "nmf_basis_w.csv").to_numpy()
    d = pd.read_csv(FIXTURES / "nmf_basis_d.csv")["x"].to_numpy()

    code_to_name = {extract_au_code(name): name for name in AU_NAMES}
    subset = pd.read_csv(FIXTURES / "r_normalised_subset.csv")
    smth_cols = [c for c in subset.columns if c.startswith("smth_AU")]
    subset = subset.rename(columns={c: code_to_name[extract_au_code(c)] for c in smth_cols})

    decomposer = NMFDecomposer(n_components=3, columns=AU_NAMES)
    decomposer.columns_ = AU_NAMES
    decomposer.prefix = "nmf"
    components_forced = (w * d[None, :]).T  # (3, 17)
    model = SklearnNMF(n_components=3, init="nndsvda", max_iter=750, tol=1e-6)
    model.components_ = components_forced
    model.n_components_ = 3
    decomposer.model_ = model
    decomposer.components_ = components_forced

    result = nmf_reconstruction_error(decomposer, subset)
    r2 = result.loc[result["metric"] == "R2", "value"].item()
    # Reference computed offline the same way (see module docstring) --
    # this 15-video subset's own R2, not the full 94-video set's 0.4452.
    assert r2 == pytest.approx(0.433112, abs=1e-3)

    per_au = nmf_reconstruction_r2_per_au(decomposer, subset)
    best = per_au.iloc[0]
    worst = per_au.iloc[-1]
    assert best["au"] == "AU06_cheek_raiser"
    assert best["r2"] == pytest.approx(0.8137, abs=0.02)
    assert worst["au"] == "AU45_blink"
    assert worst["r2"] == pytest.approx(-0.0538, abs=0.02)
