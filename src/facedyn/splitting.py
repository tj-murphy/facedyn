"""Train/test splitting and cross-validation grouping for AU data.

- :func:`group_train_test_split` is the general-purpose default. It assumes
  only that a video's frames should not be split across train and test.
- :func:`paired_train_test_split` is for designs with explicit matched
  pairs, such as real/fake deepfake pairing, where both members must land
  on the same side. Use it only if your data has that structure.
- :func:`pair_groups` (and :func:`pair_groups_from_filenames`) produce the
  ``groups`` array cross-validation needs so a pair is never split across
  folds. On matched-pairs data this is not optional -- see `PIPELINE.md`
  step 8 and :class:`RepeatedStratifiedGroupKFold`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.utils.validation import check_random_state


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


def _video_to_pair(df: pd.DataFrame, video_col: str, pair_col: str) -> dict[str, object]:
    """Map each video to its counterpart, checking one counterpart per video."""
    unique = df[[video_col, pair_col]].drop_duplicates()
    conflicting = unique[video_col].duplicated(keep=False)
    if conflicting.any():
        offenders = sorted(unique.loc[conflicting, video_col].unique())[:5]
        raise ValueError(
            f"Each video needs a single '{pair_col}' value, but these have more "
            f"than one: {offenders}. Frames of the same video disagree about "
            f"which video they are paired with."
        )
    return dict(zip(unique[video_col], unique[pair_col]))


def pair_groups(
    df: pd.DataFrame,
    video_col: str = "video_filename",
    pair_col: str = "corresponding_video",
) -> np.ndarray:
    """Group ids that keep matched pairs together during cross-validation.

    Pass the result as ``groups`` to any grouped splitter
    (:class:`RepeatedStratifiedGroupKFold`,
    :class:`sklearn.model_selection.StratifiedGroupKFold`, ...).

    **Why this matters on paired data.** In a real/fake design each fake is
    generated from a specific real video and shares its scene and
    performance, so the two carry near-identical dynamics under *opposite*
    labels. Split a pair across folds and the model learns "dynamics like
    this -> real" from one member, then meets its twin labelled fake:
    predictions are systematically inverted rather than merely uninformed.
    Measured on Paper 1's data with all 108 features, plain
    `StratifiedKFold` scores a random forest at **0.394** ROC-AUC -- below
    chance -- against **0.640** with the pairs grouped. Leakage usually
    inflates a score; here it deflates it, which is what makes it easy to
    miss: it looks like a weak model, not like leakage.

    Pointers are followed as undirected links, so both members of a pair
    get the same id whether ``pair_col`` is filled in on both rows (as in
    facedyn's own data) or only on one side. The id is the alphabetically
    first video in the group.

    Parameters
    ----------
    df : pd.DataFrame
        Row-level data (frames or one row per video). One group id is
        returned per row, in the same order.
    video_col : str, default "video_filename"
        Column identifying each video.
    pair_col : str, default "corresponding_video"
        Column giving each video's paired counterpart's ``video_col``
        value. Missing values mark a video as unpaired.

    Returns
    -------
    np.ndarray of str
        Group id per row of ``df``.

    Raises
    ------
    ValueError
        If a video's rows disagree about its counterpart.

    Warns
    -----
    UserWarning
        If a video's counterpart is not present in ``df`` (that video
        becomes its own group -- safe, since an absent partner cannot leak,
        but usually a sign that rows were dropped upstream), or if the
        pointers chain more than two videos together (grouped anyway, which
        is the conservative reading, but the pairing is not one-to-one).

    See Also
    --------
    pair_groups_from_filenames : derive the pairing when there is no
        ``pair_col``, e.g. from an export that kept only the filename.

    Examples
    --------
    >>> groups = pair_groups(features)
    >>> cross_val_score(model, X, y, groups=groups,
    ...                 cv=RepeatedStratifiedGroupKFold())  # doctest: +SKIP
    """
    for column in (video_col, pair_col):
        if column not in df:
            raise KeyError(f"df has no column {column!r}")

    video_to_pair = _video_to_pair(df, video_col, pair_col)
    known = set(video_to_pair)

    missing = {
        video
        for video, partner in video_to_pair.items()
        if pd.notna(partner) and partner not in known
    }
    if missing:
        warnings.warn(
            f"{len(missing)} video(s) name a '{pair_col}' that is not in df, "
            f"e.g. {sorted(missing)[:3]}. They are treated as unpaired (their "
            "own group). This is safe for cross-validation but usually means "
            "rows were dropped upstream, so check the pairing is intact.",
            UserWarning,
            stacklevel=2,
        )

    # Union-find over the pointers, treated as undirected. Following them
    # in one direction only would leave a fake whose own `pair_col` is
    # blank in a group of its own, i.e. in a different fold from the real
    # video that points at it -- the exact split this function prevents.
    parent = {video: video for video in known}

    def find(video):
        while parent[video] != video:
            parent[video] = parent[parent[video]]
            video = parent[video]
        return video

    for video, partner in video_to_pair.items():
        if pd.isna(partner) or partner not in known:
            continue
        root_a, root_b = find(video), find(partner)
        if root_a != root_b:
            parent[max(root_a, root_b, key=str)] = min(root_a, root_b, key=str)

    keys = {video: str(find(video)) for video in known}

    oversized = pd.Series(list(keys.values())).value_counts()
    oversized = oversized[oversized > 2]
    if not oversized.empty:
        warnings.warn(
            f"{len(oversized)} group(s) contain more than two videos, e.g. "
            f"{oversized.index[0]!r} with {int(oversized.iloc[0])}. The "
            f"'{pair_col}' pointers chain several videos together, so the "
            "pairing is not one-to-one. They are grouped together anyway "
            "(the conservative choice), but check the pairing.",
            UserWarning,
            stacklevel=2,
        )

    return df[video_col].map(keys).to_numpy()


def _deepfake_detection_pair_key(video_filename: str, label: str, real_label: str) -> str:
    """Google DFD naming rule: fake ``{a}_{b}__{scene}__{hash}`` -> ``{a}__{scene}``."""
    if label == real_label:
        return video_filename
    if "__" not in video_filename:
        return video_filename
    actors, rest = video_filename.split("__", 1)
    return f"{actors.split('_')[0]}__{rest.rsplit('__', 1)[0]}"


def pair_groups_from_filenames(
    df: pd.DataFrame,
    video_col: str = "video_filename",
    label_col: str = "isfakeorreal",
    real_label: str = "real",
    key_fn: Callable[[str, str], str] | None = None,
) -> np.ndarray:
    """Recover pair groups from video filenames when there is no pair column.

    :func:`pair_groups` is the right function whenever the data carries an
    explicit pointer. Use this one for tables that kept only the filename
    -- for instance R's exported CMFTS feature table, which has
    ``video_filename``/``isfakeorreal``/``emotion``/``valence`` and nothing
    else to pair on.

    **The default rule is specific to the Google DeepFakeDetection naming
    scheme** used by Paper 1: a real video is ``{actor}__{scene}`` and the
    fake built from it is ``{actor}_{other}__{scene}__{hash}``, so dropping
    the second actor id and the hash recovers the source video. On Paper
    1's real files this resolves 100% of fakes, giving exactly 185 groups
    of 2 in the training set and 47 in the test set. **Check
    :func:`pair_group_report` on your own data before trusting it**, and
    pass ``key_fn`` for any other naming convention.

    Parameters
    ----------
    df : pd.DataFrame
        Row-level data. One group id is returned per row, in order.
    video_col : str, default "video_filename"
        Column identifying each video.
    label_col : str, default "isfakeorreal"
        Column distinguishing the two roles in a pair.
    real_label : str, default "real"
        Value in ``label_col`` marking the source (unmodified) video, whose
        filename is the group id.
    key_fn : callable, optional
        ``key_fn(video_filename, label) -> group_id``, replacing the
        default rule entirely.

    Returns
    -------
    np.ndarray of str
        Group id per row of ``df``.

    Warns
    -----
    UserWarning
        If any derived key fails to match a video carrying ``real_label``,
        i.e. the naming rule does not fit this data.
    """
    for column in (video_col, label_col):
        if column not in df:
            raise KeyError(f"df has no column {column!r}")

    using_default_rule = key_fn is None
    if key_fn is None:
        def key_fn(video_filename, label):  # noqa: E306 - local default
            return _deepfake_detection_pair_key(video_filename, label, real_label)

    groups = np.array([
        key_fn(video, label) for video, label in zip(df[video_col], df[label_col])
    ])

    # Only the default rule promises that a derived key *is* a real video's
    # filename; a custom key_fn is free to invent ids of its own.
    real_videos = set(df.loc[df[label_col] == real_label, video_col])
    derived = groups[(df[label_col] != real_label).to_numpy()]
    if using_default_rule and len(derived):
        matched = np.isin(derived, list(real_videos)).mean()
        if matched < 1.0:
            warnings.warn(
                f"Only {matched:.0%} of non-{real_label!r} videos resolve to a "
                f"{real_label!r} video present in df. The filename rule may not "
                "fit this dataset -- inspect pair_group_report() and pass a "
                "key_fn if it does not.",
                UserWarning,
                stacklevel=2,
            )
    return groups


def pair_group_report(
    df: pd.DataFrame,
    groups: np.ndarray,
    video_col: str = "video_filename",
) -> pd.DataFrame:
    """Summarise how many videos ended up in groups of each size.

    The check to run before cross-validating: on a complete matched-pairs
    design every group holds exactly 2 videos, and any row of size 1 is a
    video whose partner was not recognised.

    Parameters
    ----------
    df : pd.DataFrame
        The data ``groups`` was derived from.
    groups : array-like
        Group ids, one per row of ``df``.
    video_col : str, default "video_filename"
        Column identifying each video. Sizes count distinct videos, not
        rows, so the report reads the same on frame-level and video-level
        tables.

    Returns
    -------
    pd.DataFrame
        Columns ``group_size``, ``n_groups``, ``n_videos``, ascending by
        size.

    Examples
    --------
    >>> pair_group_report(cmfts, pair_groups_from_filenames(cmfts))
       group_size  n_groups  n_videos
    0           2       185       370
    """
    groups = np.asarray(groups)
    if len(groups) != len(df):
        raise ValueError(
            f"groups has {len(groups)} entries but df has {len(df)} rows."
        )

    per_video = pd.DataFrame({
        "video": df[video_col].to_numpy(),
        "group": groups,
    }).drop_duplicates()
    sizes = per_video.groupby("group").size()

    report = (
        sizes.value_counts()
        .rename_axis("group_size")
        .reset_index(name="n_groups")
        .sort_values("group_size")
        .reset_index(drop=True)
    )
    report["n_videos"] = report["group_size"] * report["n_groups"]
    return report


class RepeatedStratifiedGroupKFold(BaseCrossValidator):
    """Repeated stratified k-fold that never splits a group across folds.

    The cross-validator R's ``trainControl(method="repeatedcv", number=5,
    repeats=3)`` implies on paired data. scikit-learn ships
    :class:`~sklearn.model_selection.StratifiedGroupKFold` and
    :class:`~sklearn.model_selection.RepeatedStratifiedKFold` but nothing
    that is both, so this composes them: each repeat is an independently
    shuffled `StratifiedGroupKFold`.

    ``groups`` is **required**. Passing group ids is the whole point of the
    class, and a silent fallback to ungrouped folds is exactly the failure
    this exists to prevent (see :func:`pair_groups`).

    Parameters
    ----------
    n_splits : int, default 5
        Folds per repeat. Matches R's ``number``.
    n_repeats : int, default 3
        Independently shuffled repeats. Matches R's ``repeats``.
    random_state : int, optional
        Seeds the per-repeat shuffles. Same seed -> same folds.

    Examples
    --------
    >>> cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=3, random_state=0)
    >>> cross_val_score(model, X, y, cv=cv, groups=pair_groups(df),
    ...                 scoring="roc_auc")  # doctest: +SKIP
    """

    def __init__(self, n_splits: int = 5, n_repeats: int = 3, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats

    def split(self, X, y=None, groups=None):
        """Yield ``(train_idx, test_idx)`` for every fold of every repeat."""
        if groups is None:
            raise ValueError(
                "RepeatedStratifiedGroupKFold requires groups. Build them with "
                "facedyn.splitting.pair_groups (or pair_groups_from_filenames) "
                "so matched pairs stay in the same fold; without grouping, a "
                "pair split across folds drives ROC-AUC below chance on this "
                "kind of data."
            )
        if y is None:
            raise ValueError("RepeatedStratifiedGroupKFold requires y to stratify on.")

        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_repeats)
        for seed in seeds:
            splitter = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=int(seed)
            )
            yield from splitter.split(X, y, groups)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # BaseCrossValidator's default `split` builds folds from this hook;
        # `split` is overridden above, so it exists only to satisfy the ABC.
        raise NotImplementedError
