import numpy as np
import pandas as pd
import pytest

from facedyn.splitting import (
    RepeatedStratifiedGroupKFold,
    group_train_test_split,
    pair_group_report,
    pair_groups,
    pair_groups_from_filenames,
    paired_train_test_split,
)


def make_paired_df(n_pairs: int, frames_per_video: int = 3) -> pd.DataFrame:
    """Synthetic frame-level data: n_pairs real/fake video pairs."""
    rows = []
    for i in range(n_pairs):
        real_name = f"real_{i}"
        fake_name = f"fake_{i}"
        for frame in range(frames_per_video):
            rows.append({
                "video_filename": real_name,
                "corresponding_video": fake_name,
                "isfakeorreal": "real",
                "frame": frame,
            })
            rows.append({
                "video_filename": fake_name,
                "corresponding_video": real_name,
                "isfakeorreal": "fake",
                "frame": frame,
            })
    return pd.DataFrame(rows)


def make_unpaired_df(n_videos: int, frames_per_video: int = 3) -> pd.DataFrame:
    """Synthetic frame-level data with no pairing structure at all —
    e.g. independent videos labeled by an arbitrary condition."""
    rows = []
    conditions = ["happy", "sad", "neutral"]
    for i in range(n_videos):
        for frame in range(frames_per_video):
            rows.append({
                "video_filename": f"video_{i}",
                "condition": conditions[i % len(conditions)],
                "frame": frame,
            })
    return pd.DataFrame(rows)


def test_group_split_never_splits_a_videos_rows():
    df = make_unpaired_df(n_videos=20)
    train_df, test_df = group_train_test_split(df, train_size=0.8, random_state=0)

    train_videos = set(train_df["video_filename"])
    test_videos = set(test_df["video_filename"])
    assert train_videos.isdisjoint(test_videos)

    for video, group in df.groupby("video_filename"):
        assert group.index.isin(train_df.index).all() or group.index.isin(test_df.index).all()


def test_group_split_disjoint_and_exhaustive():
    df = make_unpaired_df(n_videos=20)
    train_df, test_df = group_train_test_split(df, random_state=0)

    assert len(train_df) + len(test_df) == len(df)
    combined_index = set(train_df.index) | set(test_df.index)
    assert combined_index == set(df.index)


def test_group_split_proportion_is_approximately_train_size():
    df = make_unpaired_df(n_videos=20)
    train_df, test_df = group_train_test_split(df, train_size=0.8, random_state=0)

    n_train_videos = train_df["video_filename"].nunique()
    n_test_videos = test_df["video_filename"].nunique()
    assert n_train_videos == 16
    assert n_test_videos == 4


def test_group_split_same_seed_is_deterministic():
    df = make_unpaired_df(n_videos=20)
    train_a, _ = group_train_test_split(df, random_state=7)
    train_b, _ = group_train_test_split(df, random_state=7)

    assert set(train_a["video_filename"]) == set(train_b["video_filename"])


def test_group_split_requires_no_pairing_column():
    # Sanity check: works on data with no pairing/label columns whatsoever.
    df = make_unpaired_df(n_videos=10)
    train_df, test_df = group_train_test_split(df, train_size=0.7, random_state=0)
    assert len(train_df) > 0
    assert len(test_df) > 0


def test_pairs_never_split_across_train_and_test():
    df = make_paired_df(n_pairs=10)
    train_df, test_df = paired_train_test_split(df, train_size=0.8, random_state=0)

    train_videos = set(train_df["video_filename"])
    test_videos = set(test_df["video_filename"])
    pairing = dict(zip(df["video_filename"], df["corresponding_video"]))

    for video in train_videos:
        assert pairing[video] in train_videos
    for video in test_videos:
        assert pairing[video] in test_videos


def test_train_and_test_disjoint_and_exhaustive():
    df = make_paired_df(n_pairs=10)
    train_df, test_df = paired_train_test_split(df, random_state=0)

    assert set(train_df["video_filename"]).isdisjoint(set(test_df["video_filename"]))
    assert len(train_df) + len(test_df) == len(df)
    combined_index = set(train_df.index) | set(test_df.index)
    assert combined_index == set(df.index)


def test_train_proportion_matches_floor_of_real_videos():
    n_pairs = 20
    df = make_paired_df(n_pairs=n_pairs)
    train_df, _ = paired_train_test_split(df, train_size=0.8, random_state=0)

    n_train_real = train_df.loc[train_df["isfakeorreal"] == "real", "video_filename"].nunique()
    assert n_train_real == 16  # floor(0.8 * 20)


def test_same_seed_is_deterministic():
    df = make_paired_df(n_pairs=10)
    train_a, test_a = paired_train_test_split(df, random_state=42)
    train_b, test_b = paired_train_test_split(df, random_state=42)

    assert set(train_a["video_filename"]) == set(train_b["video_filename"])
    assert set(test_a["video_filename"]) == set(test_b["video_filename"])


def test_broken_pairing_raises():
    df = make_paired_df(n_pairs=3)
    # Break the pairing for one real video by pointing it at a video that doesn't exist.
    df.loc[df["video_filename"] == "real_0", "corresponding_video"] = "does_not_exist"

    # train_size=1.0 samples every real video, guaranteeing real_0 (and its
    # broken pairing) is included regardless of RNG seed.
    with pytest.raises(ValueError, match="Broken pairing"):
        paired_train_test_split(df, train_size=1.0, random_state=0)


# --- pair grouping for cross-validation ----------------------------------


def test_pair_groups_gives_both_members_one_id():
    df = make_paired_df(n_pairs=5)
    groups = pair_groups(df)

    per_video = df.assign(group=groups).drop_duplicates("video_filename")
    for i in range(5):
        pair = per_video.loc[per_video["video_filename"].isin([f"real_{i}", f"fake_{i}"])]
        assert pair["group"].nunique() == 1
    assert per_video["group"].nunique() == 5


def test_pair_groups_follows_a_one_directional_pointer():
    # Only the real rows name their counterpart; the fakes have no pointer.
    # Following the link in one direction only would leave each fake in a
    # group of its own -- i.e. in a different fold from its twin.
    df = make_paired_df(n_pairs=4)
    df.loc[df["isfakeorreal"] == "fake", "corresponding_video"] = np.nan

    groups = pair_groups(df)
    report = pair_group_report(df, groups)

    assert report["group_size"].tolist() == [2]
    assert report["n_groups"].tolist() == [4]


def test_pair_groups_warns_when_a_counterpart_is_absent():
    df = make_paired_df(n_pairs=3)
    df = df[df["video_filename"] != "fake_0"]  # drop one member of a pair

    with pytest.warns(UserWarning, match="not in df"):
        groups = pair_groups(df)

    per_video = df.assign(group=groups).drop_duplicates("video_filename")
    assert (per_video.loc[per_video["video_filename"] == "real_0", "group"] == "real_0").all()


def test_pair_groups_warns_when_pointers_chain_more_than_two_videos():
    df = make_paired_df(n_pairs=2)
    # fake_1 now points at real_0, chaining all four videos together.
    df.loc[df["video_filename"] == "fake_1", "corresponding_video"] = "real_0"

    with pytest.warns(UserWarning, match="more than two videos"):
        groups = pair_groups(df)

    # Chained videos are grouped together, which is the conservative choice.
    assert len(set(groups)) == 1


def test_pair_groups_rejects_a_video_with_two_counterparts():
    df = make_paired_df(n_pairs=2)
    df.loc[df.index[0], "corresponding_video"] = "fake_1"  # one frame disagrees

    with pytest.raises(ValueError, match="single 'corresponding_video' value"):
        pair_groups(df)


def make_deepfake_named_df(n_pairs: int = 4) -> pd.DataFrame:
    """Google DFD filenames: real `{actor}__{scene}`, fake `{a}_{b}__{scene}__{hash}`."""
    rows = []
    for i in range(n_pairs):
        rows.append({
            "video_filename": f"{i:02d}__scene_{i}",
            "isfakeorreal": "real",
        })
        rows.append({
            "video_filename": f"{i:02d}_{i + 50:02d}__scene_{i}__ABCD{i}",
            "isfakeorreal": "fake",
        })
    return pd.DataFrame(rows)


def test_pair_groups_from_filenames_recovers_the_source_video():
    df = make_deepfake_named_df(n_pairs=4)
    groups = pair_groups_from_filenames(df)

    assert pair_group_report(df, groups)["group_size"].tolist() == [2]
    # A fake's group id is its source real video's filename.
    assert groups[1] == "00__scene_0"


def test_pair_groups_from_filenames_warns_when_the_rule_does_not_fit():
    df = pd.DataFrame({
        "video_filename": ["subject_a", "subject_b"],
        "isfakeorreal": ["real", "fake"],
    })
    with pytest.warns(UserWarning, match="filename rule may not fit"):
        pair_groups_from_filenames(df)


def test_pair_groups_from_filenames_accepts_a_custom_key():
    df = pd.DataFrame({
        "video_filename": ["a-real", "a-fake", "b-real", "b-fake"],
        "isfakeorreal": ["real", "fake", "real", "fake"],
    })
    groups = pair_groups_from_filenames(
        df, key_fn=lambda video, label: video.split("-")[0]
    )
    assert list(groups) == ["a", "a", "b", "b"]


def test_pair_group_report_counts_videos_not_rows():
    df = make_paired_df(n_pairs=6, frames_per_video=10)
    report = pair_group_report(df, pair_groups(df))

    assert report.to_dict("records") == [{"group_size": 2, "n_groups": 6, "n_videos": 12}]


# --- RepeatedStratifiedGroupKFold ----------------------------------------


def make_video_level_df(n_pairs: int = 20) -> pd.DataFrame:
    df = make_paired_df(n_pairs=n_pairs, frames_per_video=1)
    rng = np.random.default_rng(0)
    df["feature"] = rng.normal(size=len(df))
    return df.reset_index(drop=True)


def test_repeated_grouped_cv_never_splits_a_pair():
    df = make_video_level_df()
    groups = pair_groups(df)
    y = (df["isfakeorreal"] == "real").to_numpy()

    cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=3, random_state=0)
    folds = list(cv.split(df[["feature"]], y, groups))

    assert len(folds) == cv.get_n_splits() == 15
    for train_idx, test_idx in folds:
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_repeated_grouped_cv_is_deterministic_for_a_given_seed():
    df = make_video_level_df()
    groups = pair_groups(df)
    y = (df["isfakeorreal"] == "real").to_numpy()

    folds_a = list(RepeatedStratifiedGroupKFold(random_state=0).split(df[["feature"]], y, groups))
    folds_b = list(RepeatedStratifiedGroupKFold(random_state=0).split(df[["feature"]], y, groups))

    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)


def test_repeated_grouped_cv_repeats_are_genuinely_different_partitions():
    """Each repeat must re-partition the data, or `n_repeats` buys nothing.

    This is not automatic. `StratifiedGroupKFold(shuffle=True)` up to
    scikit-learn 1.7 shuffles the per-group class-count matrix without the
    group identities attached, so on a balanced matched-pairs design --
    every group contributing one real and one fake -- every seed returns
    identical folds. `RepeatedStratifiedGroupKFold` therefore randomises
    the group order itself; this test fails if that is ever dropped in
    favour of delegating to `shuffle`.
    """
    df = make_video_level_df()
    groups = pair_groups(df)
    y = (df["isfakeorreal"] == "real").to_numpy()

    folds = list(
        RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=3, random_state=0)
        .split(df[["feature"]], y, groups)
    )
    partitions = [
        frozenset(frozenset(test.tolist()) for _, test in folds[i:i + 5])
        for i in (0, 5, 10)
    ]

    assert len(set(partitions)) == 3
    # Every repeat still covers every sample exactly once across its folds.
    for partition in partitions:
        assert sorted(i for fold in partition for i in fold) == list(range(len(df)))


def test_repeated_grouped_cv_folds_stay_balanced_and_stratified():
    df = make_video_level_df(n_pairs=25)
    groups = pair_groups(df)
    y = (df["isfakeorreal"] == "real").to_numpy()

    for _, test_idx in RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=2,
                                                    random_state=0).split(
        df[["feature"]], y, groups
    ):
        assert len(test_idx) == pytest.approx(len(df) / 5, abs=4)
        assert y[test_idx].mean() == pytest.approx(0.5, abs=0.2)


def test_repeated_grouped_cv_requires_groups():
    df = make_video_level_df(n_pairs=10)
    y = (df["isfakeorreal"] == "real").to_numpy()

    with pytest.raises(ValueError, match="requires groups"):
        list(RepeatedStratifiedGroupKFold().split(df[["feature"]], y))
