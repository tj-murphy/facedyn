"""Train/test splitting for frame-level AU data.

- :func:`group_train_test_split` is the general-purpose default. It assumes
  only that a video's frames should not be split across train and test.
- :func:`paired_train_test_split` is for designs with explicit matched
  pairs, such as real/fake deepfake pairing, where both members must land
  on the same side. Use it only if your data has that structure.
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

    Assumes nothing about matched pairs or class balance, so it is the
    right default for most datasets. Wraps
    :class:`sklearn.model_selection.GroupShuffleSplit`, grouping by
    `video_col`.

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

    Use only if your data has explicit matched pairs. Otherwise use
    :func:`group_train_test_split`.

    Samples a fraction of real videos and carries each one's paired fake
    into the same split, so a pair never straddles the split. `pair_col` is
    symmetric: a real row's value is its fake's `video_col` value, and vice
    versa.

    Sampling uses NumPy's RNG, which differs from R's ``sample()``, so the
    exact videos selected will not match R even with the same seed. Only
    the proportions and the pairing invariant replicate.

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
