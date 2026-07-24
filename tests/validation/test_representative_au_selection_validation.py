"""Regression test: select_representative_aus vs. R's real, actual choice.

`final_analysis_NMF_check.Rmd` never runs an argmax computation for this
step -- it hardcodes the three representative AU column names directly into
a `select()`, based on the researcher eyeballing the basis-matrix heatmap.
Its own intro text states the result: "Component 1 = AU12, Component 2 =
AU17, Component 3 = AU01".

Reuses `tests/validation/fixtures/nmf_basis_w.csv` (R's real `model_nmf$w`,
copied in for the reconstruction-diagnostics validation work -- see
`test_nmf_reconstruction_validation.py`) rather than a new fixture. Since
`select_representative_aus` is a deterministic argmax, this is an exact
check, not a tolerance-based one: it either reproduces R's real historical
choice exactly, or it doesn't.
"""

from pathlib import Path

import pandas as pd

from facedyn.nmf import NMFDecomposer
from facedyn.representative_aus import select_representative_aus

FIXTURES = Path(__file__).parent / "fixtures"

AU_NAMES = [
    "AU01_inner_brow_raiser", "AU02_outer_brow_raiser", "AU04_brow_lowerer",
    "AU05_upper_lid_raiser", "AU06_cheek_raiser", "AU07_lid_tightener",
    "AU09_nose_wrinkler", "AU10_upper_lip_raiser", "AU12_lip_corner_puller",
    "AU14_dimpler", "AU15_lip_corner_depressor", "AU17_chin_raiser",
    "AU20_lip_stretcher", "AU23_lip_tightener", "AU25_lips_part",
    "AU26_jaw_drop", "AU45_blink",
]


def test_matches_final_analysis_nmf_check_rmds_real_selection():
    w = pd.read_csv(FIXTURES / "nmf_basis_w.csv").to_numpy()  # (17, 3)

    decomposer = NMFDecomposer(n_components=3, columns=AU_NAMES)
    decomposer.columns_ = AU_NAMES
    decomposer.components_ = w.T  # (3, 17), sklearn's components_ convention

    result = select_representative_aus(decomposer)

    expected = {
        "nmf1": "AU12_lip_corner_puller",
        "nmf2": "AU17_chin_raiser",
        "nmf3": "AU01_inner_brow_raiser",
    }
    actual = dict(zip(result["component"], result["au"]))
    assert actual == expected
