"""Regression test: ZScoreShiftNormalizer.transform() vs. real R output.

R's actual train/test split was inferred by elimination: `r_normalised.csv`
(exported from R immediately after the normalisation step in
final_analysis_NMF_check.Rmd) contains exactly 94 videos; `r_smoothed.csv`
(pre-normalisation, full 464-video dataset) contains all of those plus 370
more — which must be R's actual training set, since a real/fake pair split
can't overlap. Confirmed end-to-end interactively against the FULL data
(max abs diff ~1.5e-13 across all 17 AU columns, 22,654 rows) before this
fixture was built; this test captures a compact, committable slice of that
same real R run for ongoing CI protection.

The fixture videos are specifically the ones containing the TRUE global
minimum z-score for each of the 17 AU columns within R's actual 94-video
test set — chosen deliberately, not arbitrarily, so a small subset still
reproduces the exact same shift constant R used. (transform()'s shift is
recomputed from whatever data it's given — see normalisation.py's
docstring — so an arbitrary small subset would produce a different, smaller
local minimum than R's real 94-video run did, and legitimately fail to
match despite correct code.)

The mean/SD fixture stores R's actual fitted training parameters directly,
rather than shipping the full 370-video training set (~89k rows) just to
regenerate 17 numbers. fit()'s correctness is already covered by
tests/unit/test_normalisation.py's synthetic tests; this test is
deliberately scoped to transform() only.
"""

from pathlib import Path

import pandas as pd

from facedyn.normalisation import ZScoreShiftNormalizer

FIXTURES = Path(__file__).parent / "fixtures"


def test_transform_matches_r_output():
    params = pd.read_csv(FIXTURES / "normalisation_train_params.csv")
    raw = pd.read_csv(FIXTURES / "raw_smoothed_normalisation_subset.csv")
    expected = pd.read_csv(FIXTURES / "r_normalised_subset.csv")

    normalizer = ZScoreShiftNormalizer()
    # Inject R's actual fitted training parameters directly, bypassing
    # fit() -- see module docstring for why.
    normalizer.columns_ = params["column"].tolist()
    normalizer.means_ = pd.Series(params["mean"].values, index=params["column"])
    normalizer.sds_ = pd.Series(params["sd"].values, index=params["column"])

    result = normalizer.transform(raw)

    key = ["video_filename", "frame"]
    merged = result.merge(expected, on=key, suffixes=("_py", "_r"))
    assert len(merged) == len(expected)

    smth_cols = [c for c in expected.columns if c.startswith("smth_")]
    for col in smth_cols:
        pd.testing.assert_series_equal(
            merged[f"{col}_py"],
            merged[f"{col}_r"],
            check_exact=False,
            atol=1e-8,
            check_names=False,
        )
