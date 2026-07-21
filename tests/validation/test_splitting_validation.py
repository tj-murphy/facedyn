"""Sanity check: split video counts vs. the R reference run.

Unlike smoothing, this step is stochastic (NumPy's RNG differs from R's
`sample()`), so exact video-identity matching against R isn't meaningful —
see PIPELINE.md's stochastic-step validation policy. Instead this checks
that, on the real dataset, the resulting split sizes structurally match
what final_analysis_NMF_check.Rmd reports ("Training set: 370 videos, Test
set: 94 videos") and that the pairing invariant holds.

Skipped when the (gitignored, locally-provided) full dataset isn't present.
"""

from pathlib import Path

import pandas as pd
import pytest

from facedyn.splitting import paired_train_test_split

DATA_PATH = Path(__file__).parents[2] / "Data" / "full_dataset_preprocessed.csv"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="full dataset not available locally"
)


def test_split_counts_match_r_reference():
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = paired_train_test_split(df, train_size=0.8, random_state=12345)

    assert train_df["video_filename"].nunique() == 370
    assert test_df["video_filename"].nunique() == 94

    pairing = dict(zip(df["video_filename"], df["corresponding_video"]))
    train_videos = set(train_df["video_filename"])
    test_videos = set(test_df["video_filename"])
    for video in train_videos:
        assert pairing[video] in train_videos
    for video in test_videos:
        assert pairing[video] in test_videos
