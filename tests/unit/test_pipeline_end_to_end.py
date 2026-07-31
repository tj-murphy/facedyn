"""The whole pipeline, fitted on train and applied to a held-out test set.

Every stage is unit-tested on its own elsewhere. What this file guards is
the join between them, and specifically the property the individual tests
cannot see: that a test set can be pushed through the *fitted* pipeline
without any stage refitting itself on it. That is what makes the final
held-out number meaningful, and it is easy to break silently -- a stage
that quietly recomputes its parameters in `transform` still returns a
plausible-looking frame.

The data is synthetic and deliberately small (matched real/fake pairs, a
handful of AUs, short series), so this runs in seconds rather than the
tens of minutes the real pipeline takes.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from facedyn.classifiers import fit_classifiers, make_classifiers
from facedyn.evaluation import evaluate_models
from facedyn.feature_selection import BorutaSelector
from facedyn.features.cleaning import FeatureCleaner
from facedyn.features.reshape import pivot_features_wide, reshape_to_wide
from facedyn.features.timeseries import TimeSeriesFeatureExtractor
from facedyn.nmf import NMFDecomposer
from facedyn.normalisation import ZScoreShiftNormalizer
from facedyn.representative_aus import RepresentativeAUSelector
from facedyn.smoothing import RollingSmoother
from facedyn.splitting import pair_groups, paired_train_test_split

AU_COLUMNS = ["AU01_r", "AU06_r", "AU12_r", "AU17_r"]
N_FRAMES = 40


def make_frame_level_data(n_pairs: int = 32, seed: int = 0) -> pd.DataFrame:
    """Frame-level AU data for `n_pairs` matched real/fake video pairs.

    Fakes move with a larger amplitude than their partner, giving a real
    signal for the classifier to find, while both members of a pair share
    a per-AU baseline -- mirroring the way a deepfake inherits its source
    video's scene, which is what makes pair-aware splitting necessary.

    The amplitude difference is applied to *every* AU rather than one,
    because step 5 keeps only the AUs that dominate the NMF components: a
    signal planted in a single AU is discarded before feature extraction
    if that AU is not selected, which says nothing about the pipeline.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pair in range(n_pairs):
        baseline = rng.normal(scale=0.3, size=len(AU_COLUMNS))
        for label in ("real", "fake"):
            video = f"{label}_{pair}"
            partner = f"{'fake' if label == 'real' else 'real'}_{pair}"
            phase = rng.uniform(0, np.pi)
            amplitude = 1.4 if label == "fake" else 0.5
            for frame in range(N_FRAMES):
                signal = np.sin(frame / 6 + phase) * amplitude
                values = baseline + signal + rng.normal(scale=0.15, size=len(AU_COLUMNS))
                rows.append({
                    "video_filename": video,
                    "corresponding_video": partner,
                    "isfakeorreal": label,
                    "frame": frame,
                    **dict(zip(AU_COLUMNS, np.abs(values))),
                })
    return pd.DataFrame(rows)


def features_for(
    frames: pd.DataFrame,
    normalizer: ZScoreShiftNormalizer,
    decomposer: NMFDecomposer,
    au_selector: RepresentativeAUSelector,
    extractor: TimeSeriesFeatureExtractor,
    cleaner: FeatureCleaner,
) -> pd.DataFrame:
    """Apply already-fitted stages to a frame-level set. No fitting here."""
    normalised = normalizer.transform(frames)
    representative = au_selector.transform(normalised)
    wide = reshape_to_wide(representative, value_cols=au_selector.selected_columns_)
    extracted = extractor.transform(wide)
    cleaned = cleaner.transform(extracted)
    return pivot_features_wide(cleaned, feature_columns=list(cleaner.feature_names_out_))


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Fit every stage on the training half only, then transform both."""
    frames = make_frame_level_data()
    smoother = RollingSmoother()
    smoothed = smoother.fit_transform(frames)

    train_frames, test_frames = paired_train_test_split(
        smoothed, train_size=0.75, random_state=0
    )

    normalizer = ZScoreShiftNormalizer().fit(train_frames)
    train_norm = normalizer.transform(train_frames)

    decomposer = NMFDecomposer(n_components=2, random_state=0).fit(train_norm)
    au_selector = RepresentativeAUSelector(decomposer).fit(train_norm)
    extractor = TimeSeriesFeatureExtractor()

    train_wide = reshape_to_wide(
        au_selector.transform(train_norm), value_cols=au_selector.selected_columns_
    )
    train_extracted = extractor.fit_transform(train_wide)
    cleaner = FeatureCleaner().fit(train_extracted)

    stages = (normalizer, decomposer, au_selector, extractor, cleaner)
    train_features = features_for(train_frames, *stages)
    test_features = features_for(test_frames, *stages)

    return {
        "frames": frames,
        "train_frames": train_frames,
        "test_frames": test_frames,
        "stages": stages,
        "train_features": train_features,
        "test_features": test_features,
    }


def test_the_split_keeps_pairs_together(fitted_pipeline):
    train_videos = set(fitted_pipeline["train_frames"]["video_filename"])
    test_videos = set(fitted_pipeline["test_frames"]["video_filename"])

    assert train_videos.isdisjoint(test_videos)
    for video in train_videos:
        partner = video.replace("real_", "TMP_").replace("fake_", "real_").replace("TMP_", "fake_")
        assert partner in train_videos


def test_test_set_reaches_one_row_per_video_with_the_same_columns(fitted_pipeline):
    train_features = fitted_pipeline["train_features"]
    test_features = fitted_pipeline["test_features"]

    assert list(train_features.columns) == list(test_features.columns)
    assert len(test_features) == fitted_pipeline["test_frames"]["video_filename"].nunique()
    assert test_features["video_filename"].is_unique


def test_no_stage_refits_itself_on_the_test_set(fitted_pipeline):
    """Fitted state must be identical before and after transforming test data."""
    normalizer, decomposer, au_selector, _, cleaner = fitted_pipeline["stages"]

    before = {
        "means": normalizer.means_.copy(),
        "sds": normalizer.sds_.copy(),
        "components": decomposer.components_.copy(),
        "representative_aus": list(au_selector.selected_columns_),
        "kept_features": list(cleaner.feature_names_out_),
    }

    features_for(fitted_pipeline["test_frames"], *fitted_pipeline["stages"])

    pd.testing.assert_series_equal(normalizer.means_, before["means"])
    pd.testing.assert_series_equal(normalizer.sds_, before["sds"])
    np.testing.assert_allclose(decomposer.components_, before["components"])
    assert list(au_selector.selected_columns_) == before["representative_aus"]
    assert list(cleaner.feature_names_out_) == before["kept_features"]


def test_nmf_projects_the_test_set_onto_the_training_basis(fitted_pipeline):
    """`transform` must project, not re-factorise: R fits NMF on train only."""
    normalizer, decomposer, *_ = fitted_pipeline["stages"]
    test_norm = normalizer.transform(fitted_pipeline["test_frames"])

    components = decomposer.components_.copy()
    activations = decomposer.transform(test_norm)

    np.testing.assert_allclose(decomposer.components_, components)
    assert (activations[[c for c in activations.columns if c.startswith("nmf")]] >= 0).all().all()


def test_a_classifier_trains_on_train_features_and_scores_the_test_set(fitted_pipeline):
    train_features = fitted_pipeline["train_features"]
    test_features = fitted_pipeline["test_features"]

    metadata = ["video_filename", "corresponding_video", "isfakeorreal"]
    feature_columns = [c for c in train_features.columns if c not in metadata]

    models = make_classifiers(["random_forest"], random_state=0, n_jobs=1)
    models["random_forest"].set_params(classifier__n_estimators=50)
    fit_classifiers(models, train_features[feature_columns], train_features["isfakeorreal"])

    results = evaluate_models(
        models, test_features[feature_columns], test_features["isfakeorreal"],
        positive_label="fake",
    )

    assert results.loc["random_forest", "n"] == len(test_features)
    assert 0.0 <= results.loc["random_forest", "roc_auc"] <= 1.0
    # The planted AU12 signal is strong, so this should be well above chance.
    assert results.loc["random_forest", "roc_auc"] > 0.6


def test_feature_selection_runs_on_train_and_subsets_the_test_set(fitted_pipeline):
    train_features = fitted_pipeline["train_features"]
    test_features = fitted_pipeline["test_features"]

    metadata = ["video_filename", "corresponding_video", "isfakeorreal"]
    feature_columns = [c for c in train_features.columns if c not in metadata]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the n_repeats=1 warning
        selector = BorutaSelector(
            feature_columns=feature_columns, n_repeats=1, n_estimators=60,
            importance="gini", random_state=0,
        ).fit(train_features, train_features["isfakeorreal"])

    selected_test = selector.transform(test_features)

    for column in selector.selected_columns_:
        assert column in selected_test.columns
    assert len(selected_test) == len(test_features)


def test_pair_groups_survive_the_whole_pipeline(fitted_pipeline):
    """Grouping must still work on the one-row-per-video feature table."""
    train_features = fitted_pipeline["train_features"]
    groups = pair_groups(train_features)

    assert len(groups) == len(train_features)
    assert len(set(groups)) == len(train_features) / 2
