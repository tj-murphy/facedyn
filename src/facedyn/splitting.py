"""Train/test splitting and cross-validation grouping for AU data.

Choosing the unit of independence
---------------------------------
Rows that are not independent must stay on the same side of every split.
Frames of one video are never independent, so the video is the *minimum*
unit. In repeated-measures data -- several videos per person, possibly
across several sessions -- the **participant** is the unit: a person's
idiosyncratic face and expressive style are shared by all their videos, so
a model tested on someone it also trained on is partly recognising the
person. Sessions (e.g. drug and placebo days) nest within participant, so
group by participant, not by session.

- :func:`group_train_test_split` is the general-purpose default. It keeps a
  video's frames together; pass ``group_col`` (e.g. ``"participant"``) to
  keep every video from one participant together instead.
- :func:`groups_from_column` produces the ``groups`` array cross-validation
  needs from an existing id column (participant, scene, ...), and
  :func:`group_report` summarises the groups and their class balance.
- :func:`paired_train_test_split` is for designs with explicit matched
  pairs, where both members must land on the same side -- for example a
  real video and the deepfake made from it. Use it only if your data has
  that structure. Its defaults (``isfakeorreal``, ``real``) follow the
  deepfake study's column names; override them for your own data.
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


def _video_to_group(df: pd.DataFrame, group_col: str, video_col: str) -> pd.Series:
    """Map each video to its group id, checking one non-missing id per video.

    Returns a Series indexed by video id (in order of first appearance).
    """
    for column in (video_col, group_col):
        if column not in df:
            raise KeyError(f"df has no column {column!r}")

    if group_col == video_col:
        videos = pd.unique(df[video_col])
        return pd.Series(videos, index=pd.Index(videos, name=video_col), name=group_col)

    missing = df[group_col].isna()
    if missing.any():
        offenders = sorted(df.loc[missing, video_col].astype(str).unique())[:5]
        raise ValueError(
            f"'{group_col}' is missing for {df.loc[missing, video_col].nunique()} "
            f"video(s), e.g. {offenders}. Every video needs a group id; a "
            "missing one cannot be kept apart from the rest of its group."
        )

    unique = df[[video_col, group_col]].drop_duplicates()
    conflicting = unique[video_col].duplicated(keep=False)
    if conflicting.any():
        offenders = sorted(unique.loc[conflicting, video_col].astype(str).unique())[:5]
        raise ValueError(
            f"Each video needs a single '{group_col}' value, but these have more "
            f"than one: {offenders}. Groups must nest videos (e.g. every video "
            "belongs to exactly one participant)."
        )
    return unique.set_index(video_col)[group_col]


def group_train_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int | None = None,
    video_col: str = "video_filename",
    group_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split that keeps each video's (or group's) rows together.

    Assumes nothing about matched pairs or class balance, so it is the
    right default for most datasets. Wraps
    :class:`sklearn.model_selection.GroupShuffleSplit`, grouping by
    `video_col`, or by `group_col` when given.

    **Repeated-measures data: pass** ``group_col``. When each participant
    contributes several videos (and possibly several sessions), the
    participant is the unit of independence. Splitting by video alone puts
    the same person on both sides of the split, so the test score partly
    measures how well the model recognises people it has already seen.
    Sessions nest within participant, so group by the participant column,
    not a session column.

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level data (one row per frame) or video-level data (one row
        per video). See ``DATA_FORMAT.md`` for the expected shape.
    train_size : float, default 0.8
        Fraction of groups assigned to the training set: videos when
        ``group_col`` is None, otherwise ``group_col`` values (e.g.
        participants). With unequal group sizes the fraction of *videos* in
        train will differ somewhat from ``train_size``.
    random_state : int, optional
        Seed for the split. Same seed -> same split.
    video_col : str, default "video_filename"
        Column identifying each video; all of a video's rows are kept
        together on one side of the split.
    group_col : str, optional
        Column identifying the unit of independence, e.g. ``"participant"``.
        All rows sharing a value land on the same side. Each video must map
        to exactly one value. If None (default), each video is its own group.

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Row-level subsets of ``df``, disjoint and exhaustive.

    Raises
    ------
    ValueError
        If ``group_col`` is given and a video has a missing value or more
        than one value in it.

    Examples
    --------
    >>> train, test = group_train_test_split(
    ...     frames, group_col="participant", random_state=0)  # doctest: +SKIP
    """
    if group_col is None:
        groups = df[video_col]
    else:
        _video_to_group(df, group_col, video_col)  # validate nesting
        groups = df[group_col]
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx], df.iloc[test_idx]


def groups_from_column(
    df: pd.DataFrame,
    group_col: str,
    video_col: str = "video_filename",
    videos=None,
) -> np.ndarray:
    """Group ids for cross-validation, read from an existing id column.

    The repeated-measures counterpart of :func:`pair_groups`: pass the
    result as ``groups`` to :class:`RepeatedStratifiedGroupKFold`,
    :func:`facedyn.evaluation.cross_validate_grouped` or any scikit-learn
    splitter that accepts ``groups``, so that every video from one
    participant (or scene, site, ...) lands in the same fold.

    For repeated-measures data the participant is the unit of independence.
    Sessions nest within participant, so use the participant column even
    when each participant was recorded on several days.

    Parameters
    ----------
    df : pd.DataFrame
        Data carrying both ``video_col`` and ``group_col``: frame-level
        data, or a video-level feature table (e.g. from
        :func:`facedyn.features.pivot_features_wide`, which carries a
        participant column through as long as it is constant per video).
    group_col : str
        Column identifying the unit of independence, e.g. ``"participant"``.
    video_col : str, default "video_filename"
        Column identifying each video.
    videos : array-like, optional
        Video ids to return group ids for, in this order. Use it when the
        feature matrix you will cross-validate does not itself carry
        ``group_col`` -- e.g. ``videos=features["video_filename"]`` with
        ``df`` the frame-level data. If None (default), one id is returned
        per row of ``df``, in row order.

    Returns
    -------
    np.ndarray
        One group id per row of ``df``, or per entry of ``videos``.

    Raises
    ------
    KeyError
        If a column is absent, or an entry of ``videos`` is not in ``df``.
    ValueError
        If a video has a missing group id, or more than one.

    See Also
    --------
    group_report : check the groups before cross-validating.
    pair_groups : for matched-pairs designs.

    Examples
    --------
    >>> wide = pivot_features_wide(features)          # keeps "participant"
    >>> groups = groups_from_column(wide, "participant")
    >>> cross_validate_grouped(model, X, y, groups=groups)  # doctest: +SKIP
    """
    video_to_group = _video_to_group(df, group_col, video_col)
    if videos is None:
        return df[group_col].to_numpy()

    videos = pd.Index(np.asarray(videos))
    absent = videos.difference(video_to_group.index)
    if len(absent):
        raise KeyError(
            f"{len(absent)} video(s) in `videos` are not in df, e.g. "
            f"{sorted(map(str, absent))[:3]}."
        )
    return video_to_group.reindex(videos).to_numpy()


def group_report(
    df: pd.DataFrame,
    group_col: str,
    video_col: str = "video_filename",
    label_col: str | None = None,
) -> pd.DataFrame:
    """One row per group: how many videos it holds and, optionally, of which class.

    The check to run before cross-validating repeated-measures data. Very
    unequal group sizes make fold sizes uneven, and groups that contribute
    only one class make stratification harder: with few groups,
    :class:`RepeatedStratifiedGroupKFold` may not be able to give every
    fold every class. Counts are of distinct videos, so the report reads
    the same on frame-level and video-level tables.

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level or video-level data.
    group_col : str
        Column identifying the unit of independence, e.g. ``"participant"``.
    video_col : str, default "video_filename"
        Column identifying each video.
    label_col : str, optional
        Class label column. If given, adds one ``n_<class>`` count column
        per class and an ``n_classes`` column.

    Returns
    -------
    pd.DataFrame
        Columns ``group_col``, ``n_videos``, then (with ``label_col``) one
        ``n_<class>`` column per class in sorted order and ``n_classes``.
        Sorted by group id.

    Raises
    ------
    ValueError
        If a video has a missing or more than one group id, or (with
        ``label_col``) more than one label.

    See Also
    --------
    pair_group_report : the equivalent summary for matched pairs.

    Examples
    --------
    >>> group_report(frames, "participant", label_col="emotion")  # doctest: +SKIP
      participant  n_videos  n_angry  n_happy  n_sad  n_classes
    0        P01        22        8        7      7          3
    """
    video_to_group = _video_to_group(df, group_col, video_col)
    per_video = video_to_group.rename(group_col).reset_index()

    report = (
        per_video.groupby(group_col, sort=True)
        .size()
        .rename("n_videos")
        .reset_index()
    )
    if label_col is None:
        return report

    if label_col not in df:
        raise KeyError(f"df has no column {label_col!r}")
    labels = df[[video_col, label_col]].drop_duplicates()
    conflicting = labels[video_col].duplicated(keep=False)
    if conflicting.any():
        offenders = sorted(labels.loc[conflicting, video_col].astype(str).unique())[:5]
        raise ValueError(
            f"Each video needs a single '{label_col}' value, but these have more "
            f"than one: {offenders}."
        )
    per_video = per_video.merge(labels, on=video_col, how="left")
    counts = pd.crosstab(per_video[group_col], per_video[label_col])
    counts = counts.reindex(sorted(counts.columns, key=str), axis=1)
    counts.columns = [f"n_{label}" for label in counts.columns]
    counts["n_classes"] = (counts > 0).sum(axis=1)
    return report.merge(counts.reset_index(), on=group_col, how="left")


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

    Each pair has a *source* member (``label_col == real_label``; the real
    video in a deepfake design) and a partner. Samples a fraction of source
    videos and carries each one's partner into the same split, so a pair
    never straddles the split. `pair_col` is symmetric: a source row's
    value is its partner's `video_col` value, and vice versa.

    Sampling uses NumPy's RNG, which differs from R's ``sample()``, so the
    exact videos selected will not match R even with the same seed. Only
    the proportions and the pairing invariant replicate.

    Parameters
    ----------
    df : pd.DataFrame
        Video-level (frame-level) data, one row per frame.
    train_size : float, default 0.8
        Fraction of source (``real_label``) videos assigned to the training
        set. ``floor(train_size * n_source)`` are sampled for train.
    random_state : int, optional
        Seed for the sampling RNG. Same seed -> same split.
    video_col : str, default "video_filename"
        Column identifying each video.
    label_col : str, default "isfakeorreal"
        Column distinguishing the two members of a pair (real vs. fake in
        a deepfake design).
    pair_col : str, default "corresponding_video"
        Column giving each video's paired counterpart's ``video_col`` value.
    real_label : str, default "real"
        Value in ``label_col`` identifying the source member of each pair.

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Row-level subsets of ``df``, disjoint and exhaustive.

    Raises
    ------
    ValueError
        If a sampled source video's ``pair_col`` value does not match any
        video present in ``df`` (a broken pairing).
    """
    source_rows = df.loc[df[label_col] == real_label, [video_col, pair_col]].drop_duplicates(video_col)
    source_to_partner = dict(zip(source_rows[video_col], source_rows[pair_col]))
    source_videos = np.array(sorted(source_to_partner))

    n_train = math.floor(train_size * len(source_videos))
    rng = np.random.default_rng(random_state)
    train_source_videos = rng.choice(source_videos, size=n_train, replace=False)

    known_videos = set(df[video_col])
    train_partner_videos = set()
    for source_video in train_source_videos:
        partner_video = source_to_partner[source_video]
        if partner_video not in known_videos:
            raise ValueError(
                f"Broken pairing: source video {source_video!r} points to "
                f"{partner_video!r} via '{pair_col}', which is not present in df."
            )
        train_partner_videos.add(partner_video)

    train_videos = set(train_source_videos) | train_partner_videos
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

    **Why this matters on paired data.** When the two members of a pair
    share their source material but carry *opposite* labels -- in a
    deepfake design each fake is generated from a specific real video and
    shares its scene and performance -- they have near-identical dynamics.
    Split a pair across folds and the model learns "dynamics like this ->
    class A" from one member, then meets its twin labelled class B:
    predictions are systematically inverted rather than merely uninformed.
    Measured on the deepfake study's data with all 108 features, plain
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
    # in one direction only would leave a partner whose own `pair_col` is
    # blank in a group of its own, i.e. in a different fold from the source
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
    scheme** used by the deepfake study: a real video is
    ``{actor}__{scene}`` and the fake built from it is
    ``{actor}_{other}__{scene}__{hash}``, so dropping the second actor id
    and the hash recovers the source video. On that study's files this
    resolves 100% of fakes, giving exactly 185 groups
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
    that is both, so this composes them: each repeat runs
    `StratifiedGroupKFold` over an independently randomised group order.

    ``groups`` is **required**. Passing group ids is the whole point of the
    class, and a silent fallback to ungrouped folds is exactly the failure
    this exists to prevent (see :func:`pair_groups`).

    Notes
    -----
    The repeats are randomised **here** rather than by delegating to
    `StratifiedGroupKFold`'s own ``shuffle``, because that argument does
    nothing on balanced matched-pairs data. Up to scikit-learn 1.7 it
    shuffles the per-group *class-count matrix* without carrying the
    group identities along; on a balanced matched-pairs design every
    group contributes the same counts (one of each class), so shuffling
    rows of an array of identical rows changes nothing and every seed
    returns identical folds. scikit-learn 1.8 shuffles a permutation
    array instead and does vary. Rather than behave differently by
    version, each repeat here relabels the groups through its own random
    permutation -- which drives the greedy assignment order on every
    version -- and asks for ``shuffle=False``. Verified to give distinct,
    balanced, still-pair-respecting partitions on both 1.7.2 and 1.8.0.

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
                "facedyn.splitting.groups_from_column (repeated measures, e.g. "
                "one group per participant) or pair_groups (matched pairs), so "
                "dependent observations stay in the same fold; without grouping, "
                "scores can be inflated or, on matched pairs, driven below chance."
            )
        if y is None:
            raise ValueError("RepeatedStratifiedGroupKFold requires y to stratify on.")

        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        codes = np.searchsorted(unique_groups, groups)

        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_repeats)
        for seed in seeds:
            # Relabelling is a bijection on the group ids, so which rows
            # share a group is untouched -- only the order the greedy
            # assignment meets them in changes. See the class Notes for
            # why this is not left to `shuffle=True`.
            relabelled = check_random_state(int(seed)).permutation(len(unique_groups))[codes]
            splitter = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=False)
            yield from splitter.split(X, y, relabelled)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # BaseCrossValidator's default `split` builds folds from this hook;
        # `split` is overridden above, so it exists only to satisfy the ABC.
        raise NotImplementedError
