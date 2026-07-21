import pandas as pd
import pytest

from facedyn.splitting import group_train_test_split, paired_train_test_split


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
