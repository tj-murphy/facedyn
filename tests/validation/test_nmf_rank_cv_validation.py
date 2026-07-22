"""Regression test: nmf_rank_cv_sweep's optimal rank vs. real RcppML output.

`r_optimal_k_input.csv` is the exact training data RcppML's built-in
cross-validation rank search was run on for real (see
`r_rank_selection/optimal_k_rcppml.Rmd` and
`r_rank_selection/cv_rank_results.csv`), which selected k=3.

An earlier row-holdout CV mechanism was tried first and failed this exact
check (it picked k=10, the highest rank tried, with train/test MSE nearly
identical at every rank -- see nmf.py's docstring for why). The current
entry-masking mechanism was validated once, interactively, against the
*full* real dataset (89,170 rows, all 9 ranks, 3 replicates) and correctly
selected k=3, with an even sharper overfitting signal than RcppML's own
run (test MSE reaching ~20,546 at k=10 vs. RcppML's ~4.9) -- see
PIPELINE.md step 4 for the full writeup.

That full-scale run took ~25 minutes (the custom masked multiplicative-
update solver is markedly slower than sklearn's optimized solver at this
data scale -- a known, documented tradeoff, not addressed here). This test
uses a reduced configuration (5,000-row subsample, ranks 2-6, 2 replicates,
looser convergence) calibrated to still correctly select k=3 in ~1 second,
for practical use in the regular test suite -- not a substitute for the
full-scale check, which is documented but not re-run automatically.
"""

from pathlib import Path

import pandas as pd
import pytest

from facedyn.nmf import nmf_rank_cv_sweep

DATA_PATH = Path(__file__).parents[2] / "R Validation Data" / "r_optimal_k_input.csv"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="r_optimal_k_input.csv not available locally"
)


def test_cv_sweep_selects_rank_3_like_rcppml():
    df = pd.read_csv(DATA_PATH)
    au_cols = [c for c in df.columns if c.startswith("AU")]
    sample = df.sample(n=5000, random_state=0).reset_index(drop=True)

    result = nmf_rank_cv_sweep(
        sample, ranks=range(2, 7), test_fraction=0.1, n_replicates=2,
        columns=au_cols, random_state=12345, max_iter=150, tol=1e-4,
    )
    agg = result.groupby("rank")["test_mse"].mean()

    assert agg.idxmin() == 3
