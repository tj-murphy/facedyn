"""Train/test splitting for video-level (frame-level) AU data.

Two splitters are provided:

- :func:`group_train_test_split` — the general-purpose default. Makes no
  assumption about study design beyond "a video's frames shouldn't be split
  across train and test." Works for any condition-comparison (emotion
  categories, unpaired real/fake, patient/control, ...).
- :func:`paired_train_test_split` — a specialization for designs with
  explicit matched pairs (e.g. this package's original real/fake deepfake
  pairing), where a pair must land on the same side of the split together.
  Only use this if your data actually has that structure; most users should
  start with :func:`group_train_test_split`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_train_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int | None = None,
    video_col: str = "video_filename",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split that keeps each video's rows together.

    No assumption is made about matched pairs or class balance — this is
    the right default for most datasets. A thin wrapper around
    :class:`sklearn.model_selection.GroupShuffleSplit`, grouping rows by
    ``video_col`` so a single video's frames always stay on one side.

    Parameters
    ----------
    df : pd.DataFrame
        Video-level (frame-level) data, one row per frame. See
        ``DATA_FORMAT.md`` for the expected shape.
    train_size : float, default 0.8
        Fraction of videos assigned to the training set.
    random_state : int, optional
        Seed for the split. Same seed -> same split.
    video_col : str, default "video_filename"
        Column identifying each video; all of a video's rows are kept
        together on one side of the split.

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Row-level subsets of ``df``, disjoint and exhaustive.
    """
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[video_col]))
    return df.iloc[train_idx], df.iloc[test_idx]


def paired_train_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int | None = None,
    video_col: str = "video_filename",
    label_col: str = "isfakeorreal",
    pair_col: str = "corresponding_video",
    real_label: str = "real",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split video-level data into train/test sets without splitting pairs.

    Only use this if your data has explicit matched pairs (e.g. each real
    video has exactly one corresponding fake video that must land on the
    same side of the split). If your data doesn't have that structure, use
    :func:`group_train_test_split` instead.

    Replicates the R pipeline's split: a fraction of *real* videos is
    sampled, and each selected real video's paired fake (identified via
    ``pair_col``, which is symmetric — a real row's ``pair_col`` value is
    its fake's ``video_col`` value, and vice versa) is carried into the same
    split. This guarantees a real video and its one deepfake counterpart
    always land on the same side of the split.

    This step is stochastic: sampling uses NumPy's RNG, which is a
    different algorithm from R's ``sample()``. Even with a numerically
    "matched" seed, the exact videos selected will differ from the R
    output — only the split proportions and the pairing invariant are
    meant to replicate R (see PIPELINE.md's stochastic-step validation
    policy).

    Parameters
    ----------
    df : pd.DataFrame
        Video-level (frame-level) data, one row per frame.
    train_size : float, default 0.8
        Fraction of real videos assigned to the training set.
        ``floor(train_size * n_real)`` real videos are sampled for train.
    random_state : int, optional
        Seed for the sampling RNG. Same seed -> same split.
    video_col : str, default "video_filename"
        Column identifying each video.
    label_col : str, default "isfakeorreal"
        Column distinguishing real vs. fake videos.
    pair_col : str, default "corresponding_video"
        Column giving each video's paired counterpart's ``video_col`` value.
    real_label : str, default "real"
        Value in ``label_col`` identifying real videos.

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Row-level subsets of ``df``, disjoint and exhaustive.

    Raises
    ------
    ValueError
        If a sampled real video's ``pair_col`` value does not match any
        video present in ``df`` (a broken real/fake pairing).
    """
    real_rows = df.loc[df[label_col] == real_label, [video_col, pair_col]].drop_duplicates(video_col)
    real_to_fake = dict(zip(real_rows[video_col], real_rows[pair_col]))
    real_videos = np.array(sorted(real_to_fake))

    n_train = math.floor(train_size * len(real_videos))
    rng = np.random.default_rng(random_state)
    train_real_videos = rng.choice(real_videos, size=n_train, replace=False)

    known_videos = set(df[video_col])
    train_fake_videos = set()
    for real_video in train_real_videos:
        fake_video = real_to_fake[real_video]
        if fake_video not in known_videos:
            raise ValueError(
                f"Broken pairing: real video {real_video!r} points to "
                f"{fake_video!r} via '{pair_col}', which is not present in df."
            )
        train_fake_videos.add(fake_video)

    train_videos = set(train_real_videos) | train_fake_videos
    train_mask = df[video_col].isin(train_videos)
    return df.loc[train_mask], df.loc[~train_mask]
