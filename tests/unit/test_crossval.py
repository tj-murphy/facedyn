"""cross_validate_pipeline: every fitted step refitted inside each fold.

The tests that matter most here are the two leakage guards:

- `test_every_fitted_step_sees_only_training_videos` spies on each step's
  `fit` and checks it never received a test-fold video.
- `test_selection_inside_folds_does_not_find_signal_in_noise` shows the
  consequence: on pure noise, supervised selection fitted on all videos
  gives a well-above-chance score, and the fold-internal pipeline does not.

The data is synthetic and small (short series, a handful of AUs), and the
cleaner uses a SimpleImputer and the model a logistic regression, so the
whole file runs in well under a minute.
"""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from facedyn.classifiers import make_classifier
from facedyn.crossval import CVPipelineResult, cross_validate_pipeline
from facedyn.evaluation import cross_validate_grouped
from facedyn.feature_selection import BorutaSelector
from facedyn.features.cleaning import FeatureCleaner
from facedyn.features.reshape import pivot_features_wide, reshape_to_wide
from facedyn.features.timeseries import TimeSeriesFeatureExtractor
from facedyn.nmf import NMFDecomposer
from facedyn.normalisation import ZScoreShiftNormalizer
from facedyn.representative_aus import RepresentativeAUSelector
from facedyn.smoothing import RollingSmoother
from facedyn.splitting import RepeatedStratifiedGroupKFold

AU_COLUMNS = ["AU01_r", "AU06_r", "AU12_r", "AU17_r"]
N_FRAMES = 40

pytestmark = pytest.mark.filterwarnings(
    "ignore::UserWarning", "ignore::sklearn.exceptions.ConvergenceWarning"
)


def make_frames(
    n_participants=12,
    videos_per_participant=4,
    classes=("a", "b"),
    effect=1.0,
    seed=0,
    au_columns=AU_COLUMNS,
) -> pd.DataFrame:
    """Repeated-measures frame data: several videos per participant.

    Each participant has a per-AU baseline shared by all their videos (the
    reason to group by participant). Class k moves with amplitude
    ``0.5 + effect * k`` on every AU, so ``effect=0`` gives pure noise.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for p in range(n_participants):
        baseline = rng.normal(scale=0.3, size=len(au_columns))
        labels = rng.permutation([classes[v % len(classes)] for v in range(videos_per_participant)])
        for v, label in enumerate(labels):
            amplitude = 0.5 + effect * classes.index(label)
            phase = rng.uniform(0, np.pi)
            t = np.arange(N_FRAMES)[:, None]
            values = (
                baseline
                + np.sin(t / 6 + phase) * amplitude
                + rng.normal(scale=0.15, size=(N_FRAMES, len(au_columns)))
            )
            video = pd.DataFrame(np.abs(values), columns=au_columns)
            video.insert(0, "frame", np.arange(N_FRAMES))
            video.insert(0, "label", label)
            video.insert(0, "participant", f"p{p}")
            video.insert(0, "video_filename", f"p{p}_v{v}")
            parts.append(video)
    return pd.concat(parts, ignore_index=True)


def fast_kwargs(**overrides):
    """Cheap steps for tests: no iterative imputation, a linear model."""
    kwargs = {
        "cleaner": FeatureCleaner(impute=SimpleImputer()),
        "selector": False,
        "model": make_classifier("logistic_regression"),
        "cv": RepeatedStratifiedGroupKFold(n_splits=4, n_repeats=2, random_state=0),
        "random_state": 0,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module")
def frames():
    return make_frames()


@pytest.fixture(scope="module")
def binary_result(frames):
    return cross_validate_pipeline(
        frames, "label", "participant", keep_bases=True, keep_models=True, **fast_kwargs()
    )


# --------------------------------------------------------------------------
# Out-of-fold predictions and grouping
# --------------------------------------------------------------------------


def test_every_video_is_predicted_exactly_once_per_repeat(frames, binary_result):
    oof = binary_result.oof
    n_videos = frames["video_filename"].nunique()

    assert binary_result.n_repeats == 2
    for _, part in oof.groupby("repeat"):
        assert len(part) == n_videos
        assert part["video"].is_unique
        assert set(part["video"]) == set(frames["video_filename"])


def test_groups_are_never_split_across_folds(binary_result):
    folds_per_group = binary_result.oof.groupby(["repeat", "group"])["fold"].nunique()
    assert (folds_per_group == 1).all()


def test_oof_carries_true_labels_and_valid_probabilities(frames, binary_result):
    oof = binary_result.oof
    truth = frames.groupby("video_filename")["label"].first()

    assert list(oof.columns) == [
        "video",
        "group",
        "repeat",
        "fold",
        "y_true",
        "y_pred",
        "proba_a",
        "proba_b",
    ]
    assert (oof["y_true"].to_numpy() == truth.loc[oof["video"]].to_numpy()).all()
    np.testing.assert_allclose(oof[["proba_a", "proba_b"]].sum(axis=1), 1.0)
    expected_pred = np.where(oof["proba_b"] > 0.5, "b", "a")
    assert (oof["y_pred"].to_numpy() == expected_pred).all()


def test_planted_signal_is_found(binary_result):
    assert binary_result.summary().loc["roc_auc", "mean"] > 0.9
    metrics = binary_result.oof_metrics()
    assert list(metrics.index) == [0, 1]
    assert (metrics["roc_auc"] > 0.9).all()
    assert (metrics["roc_auc_ci_lower"] <= metrics["roc_auc"]).all()
    assert (metrics["n"] == 48).all()


def test_scores_have_one_row_per_fold(binary_result):
    scores = binary_result.scores
    assert scores.index.names == ["repeat", "fold"]
    assert len(scores) == 8 == binary_result.n_folds
    assert list(scores.columns) == ["roc_auc", "accuracy", "balanced_accuracy", "n_test"]
    assert scores.loc[0, "n_test"].sum() == 48


def test_oof_metrics_do_not_depend_on_positive_label(binary_result):
    a = binary_result.oof_metrics(positive_label="a")
    b = binary_result.oof_metrics(positive_label="b")
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="not one of"):
        binary_result.oof_metrics(positive_label="c")


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


class _Spy:
    """Records the videos each step's `fit` was given, in call order."""

    calls: ClassVar[dict] = {}

    @classmethod
    def record(cls, step, videos):
        cls.calls.setdefault(step, []).append(set(videos))


class SpyNormalizer(ZScoreShiftNormalizer):
    def fit(self, X, y=None):
        _Spy.record("normalizer", X["video_filename"])
        return super().fit(X, y)


class SpyDecomposer(NMFDecomposer):
    def fit(self, X, y=None):
        _Spy.record("decomposer", X["video_filename"])
        return super().fit(X, y)


class SpyExtractor(TimeSeriesFeatureExtractor):
    def fit(self, X, y=None):
        _Spy.record("extractor", X["video_filename"])
        return super().fit(X, y)


class SpyCleaner(FeatureCleaner):
    def fit(self, X, y=None):
        _Spy.record("cleaner", X["video_filename"])
        return super().fit(X, y)


class SpySelector(SelectKBest):
    def fit(self, X, y=None):
        _Spy.record("selector", X.index)
        return super().fit(X, y)


class SpyModel(LogisticRegression):
    def fit(self, X, y=None, sample_weight=None):
        _Spy.record("model", X.index)
        return super().fit(X, y, sample_weight)


def test_every_fitted_step_sees_only_training_videos(frames):
    _Spy.calls = {}
    result = cross_validate_pipeline(
        frames,
        "label",
        "participant",
        normalizer=SpyNormalizer(),
        decomposer=SpyDecomposer(n_components=2, random_state=0),
        feature_extractor=SpyExtractor(),
        cleaner=SpyCleaner(impute=SimpleImputer()),
        selector=SpySelector(f_classif, k=5),
        model=SpyModel(max_iter=1000),
        cv=RepeatedStratifiedGroupKFold(n_splits=4, n_repeats=2, random_state=0),
        n_jobs=1,
    )

    all_videos = set(frames["video_filename"])
    expected_train = [
        all_videos - set(part["video"])
        for _, part in result.oof.groupby(["repeat", "fold"], sort=False)
    ]
    assert set(_Spy.calls) == {
        "normalizer",
        "decomposer",
        "extractor",
        "cleaner",
        "selector",
        "model",
    }
    for step, calls in _Spy.calls.items():
        assert calls == expected_train, step


def test_selection_inside_folds_does_not_find_signal_in_noise():
    """REFORMS L1.3: selection fitted on all videos manufactures a score.

    On pure noise, selecting 10 of 240 features with the labels of every
    video and then cross-validating gives a clearly above-chance ROC-AUC.
    The same selector inside `cross_validate_pipeline` does not.
    """
    au_columns = [f"AU{i:02d}_r" for i in (1, 2, 4, 6, 9, 12, 15, 17)]
    frames = make_frames(n_participants=10, effect=0.0, seed=0, au_columns=au_columns)
    cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=3, random_state=0)
    smoothed_cols = [f"smth_{c}" for c in au_columns]

    honest = cross_validate_pipeline(
        frames,
        "label",
        "participant",
        decomposer=False,
        **fast_kwargs(selector=SelectKBest(f_classif, k=10), cv=cv),
    )

    # The leaky reference: every step, including selection, fitted on all videos.
    normalised = ZScoreShiftNormalizer().fit_transform(RollingSmoother().fit_transform(frames))
    wide = reshape_to_wide(
        normalised[["video_filename", "frame", *smoothed_cols]],
        value_cols=smoothed_cols,
        id_vars=[],
    )
    extracted = TimeSeriesFeatureExtractor().fit_transform(wide)
    feature_cols = [c for c in extracted.columns if c not in ("video_filename", "series")]
    cleaner = FeatureCleaner(feature_columns=feature_cols, impute=SimpleImputer()).fit(extracted)
    X = pivot_features_wide(
        cleaner.transform(extracted), feature_columns=list(cleaner.feature_names_out_), id_vars=[]
    ).set_index("video_filename")
    per_video = frames.groupby("video_filename")[["label", "participant"]].first().loc[X.index]
    kept = SelectKBest(f_classif, k=10).fit(X, per_video["label"]).get_feature_names_out()
    leaky = cross_validate_grouped(
        make_classifier("logistic_regression"),
        X[kept],
        per_video["label"],
        groups=per_video["participant"].to_numpy(),
        cv=cv,
    )

    assert X.shape[1] > 200
    honest_auc = honest.oof_metrics()["roc_auc"].mean()
    leaky_auc = leaky["roc_auc"].mean()
    assert leaky_auc > 0.7
    assert honest_auc < 0.6
    assert leaky_auc - honest_auc > 0.2


# --------------------------------------------------------------------------
# Equivalence with a hand-written loop
# --------------------------------------------------------------------------


def test_matches_a_manual_loop_without_selection(frames):
    cv = RepeatedStratifiedGroupKFold(n_splits=4, n_repeats=1, random_state=3)
    result = cross_validate_pipeline(
        frames,
        "label",
        "participant",
        decomposer=NMFDecomposer(n_components=2, random_state=0),
        **fast_kwargs(cv=cv),
    )

    smoothed = RollingSmoother().fit_transform(frames)
    per_video = frames.groupby("video_filename", sort=False)[["label", "participant"]].first()
    videos = per_video.index
    expected = []
    for train_idx, test_idx in cv.split(videos, per_video["label"], per_video["participant"]):
        train = smoothed[smoothed["video_filename"].isin(videos[train_idx])]
        test = smoothed[smoothed["video_filename"].isin(videos[test_idx])]
        normalizer = ZScoreShiftNormalizer().fit(train)
        train, test = normalizer.transform(train), normalizer.transform(test)
        decomposer = NMFDecomposer(n_components=2, random_state=0).fit(train)
        aus = list(dict.fromkeys(RepresentativeAUSelector(decomposer).fit(train).selected_columns_))

        tables = []
        for side in (train, test):
            wide = reshape_to_wide(
                side[["video_filename", "frame", *aus]], value_cols=aus, id_vars=[]
            )
            tables.append(TimeSeriesFeatureExtractor().fit_transform(wide))
        feature_cols = [c for c in tables[0].columns if c not in ("video_filename", "series")]
        cleaner = FeatureCleaner(feature_columns=feature_cols, impute=SimpleImputer())
        cleaner.fit(tables[0])
        X_train, X_test = (
            pivot_features_wide(
                cleaner.transform(t), feature_columns=list(cleaner.feature_names_out_), id_vars=[]
            )
            .set_index("video_filename")
            .reindex(videos[idx])
            for t, idx in zip(tables, (train_idx, test_idx), strict=True)
        )
        model = make_classifier("logistic_regression").fit(
            X_train, per_video["label"].iloc[train_idx]
        )
        expected.append(pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index))

    expected = pd.concat(expected)
    got = result.oof.set_index("video")["proba_b"]
    pd.testing.assert_series_equal(
        got, expected.loc[got.index], check_names=False, check_index_type=False, rtol=1e-10
    )


# --------------------------------------------------------------------------
# Multiclass
# --------------------------------------------------------------------------


def test_three_classes():
    frames = make_frames(
        n_participants=10, videos_per_participant=6, classes=("happy", "sad", "angry"), seed=1
    )
    result = cross_validate_pipeline(frames, "label", "participant", **fast_kwargs())

    assert result.classes == ["angry", "happy", "sad"]
    proba_cols = ["proba_angry", "proba_happy", "proba_sad"]
    assert proba_cols == [c for c in result.oof.columns if c.startswith("proba_")]
    np.testing.assert_allclose(result.oof[proba_cols].sum(axis=1), 1.0)
    assert set(result.oof["y_pred"]) <= set(result.classes)

    metrics = result.oof_metrics()
    assert "roc_auc_ci_lower" not in metrics.columns
    assert (metrics["roc_auc"] > 0.8).all()
    assert result.scores["roc_auc"].notna().all()
    per_class = metrics[[f"roc_auc_{c}" for c in result.classes]]
    np.testing.assert_allclose(per_class.mean(axis=1), metrics["roc_auc"])


def test_three_class_oof_feeds_classification_metrics():
    """OOF predictions of one repeat give the full multiclass statistics."""
    from facedyn.evaluation import classification_metrics

    frames = make_frames(
        n_participants=10, videos_per_participant=6, classes=("happy", "sad", "angry"), seed=1
    )
    result = cross_validate_pipeline(frames, "label", "participant", **fast_kwargs())
    oof = result.oof[result.oof["repeat"] == 0]
    proba = oof[[f"proba_{c}" for c in result.classes]]

    m = classification_metrics(oof["y_true"], oof["y_pred"], proba, labels=result.classes)
    pooled = result.oof_metrics().loc[0]
    assert m["roc_auc"] == pytest.approx(pooled["roc_auc"])
    assert m["accuracy"] == pytest.approx(pooled["accuracy"])
    assert m["balanced_accuracy"] == pytest.approx(pooled["balanced_accuracy"])


@pytest.mark.slow
def test_three_classes_with_boruta_and_random_forest():
    """The default selector and model families handle three classes."""
    frames = make_frames(
        n_participants=8, videos_per_participant=6, classes=("happy", "sad", "angry"), seed=2
    )
    result = cross_validate_pipeline(
        frames,
        "label",
        "participant",
        **fast_kwargs(
            selector=BorutaSelector(
                n_repeats=2, n_estimators=50, importance="gini", max_iter=20, random_state=0
            ),
            model=make_classifier(
                "random_forest", preset="default", random_state=0, n_estimators=50
            ),
            cv=RepeatedStratifiedGroupKFold(n_splits=4, n_repeats=1, random_state=0),
        ),
    )

    assert "selection_freq" in result.selected_features.columns
    assert result.selected_features["selection_freq"].between(0, 1).all()
    assert result.oof_metrics()["roc_auc"].iloc[0] > 0.7


# --------------------------------------------------------------------------
# Selection records
# --------------------------------------------------------------------------


def test_selected_aus_record_each_folds_choice(binary_result):
    aus = binary_result.selected_aus
    assert list(aus.columns) == ["repeat", "fold", "component", "au"]
    assert len(aus) == 3 * binary_result.n_folds
    assert set(aus["au"]) <= {f"smth_{c}" for c in AU_COLUMNS}

    frequency = binary_result.au_frequency()
    assert frequency.between(0, 1).all()
    assert frequency.index.name == "au"


def test_feature_frequency_without_selector_is_all_ones(binary_result):
    assert binary_result.selected_features["selected"].all()
    frequency = binary_result.feature_frequency()
    # Features from an AU only some folds chose are absent from the other folds.
    assert frequency.max() == 1.0
    assert frequency.between(0, 1).all()


def test_feature_frequency_with_a_selector(frames):
    result = cross_validate_pipeline(
        frames, "label", "participant", **fast_kwargs(selector=SelectKBest(f_classif, k=4))
    )
    kept_per_fold = result.selected_features.groupby(["repeat", "fold"])["selected"].sum()
    assert (kept_per_fold == 4).all()
    assert result.feature_frequency().sum() == pytest.approx(4.0)


def test_keep_bases_and_keep_models(binary_result):
    assert set(binary_result.nmf_bases) == set(binary_result.scores.index)
    basis = binary_result.nmf_bases[(0, 0)]
    assert list(basis.index) == ["nmf1", "nmf2", "nmf3"]
    assert list(basis.columns) == [f"smth_{c}" for c in AU_COLUMNS]

    entry = binary_result.fold_models[(0, 0)]
    test_videos = binary_result.oof.query("repeat == 0 and fold == 0")["video"]
    assert list(entry["X_test"].index) == list(test_videos)
    assert list(entry["X_test"].columns) == entry["features"]
    np.testing.assert_allclose(
        entry["model"].predict_proba(entry["X_test"])[:, 1],
        binary_result.oof.query("repeat == 0 and fold == 0")["proba_b"],
    )


def test_bases_and_models_are_not_kept_by_default(frames):
    result = cross_validate_pipeline(frames, "label", "participant", **fast_kwargs())
    assert result.nmf_bases == {}
    assert result.fold_models == {}


def test_empty_selection_warns_and_predicts_class_frequencies(frames):
    class SelectNothing(SelectKBest):
        def get_feature_names_out(self, input_features=None):
            return np.array([], dtype=object)

    with pytest.warns(UserWarning, match="kept no features in 8 of 8 folds"):
        result = cross_validate_pipeline(
            frames,
            "label",
            "participant",
            **fast_kwargs(selector=SelectNothing(f_classif, k=1)),
        )
    np.testing.assert_allclose(result.oof["proba_a"], 0.5)


# --------------------------------------------------------------------------
# Alternative routes through the pipeline
# --------------------------------------------------------------------------


def test_nmf_activations_instead_of_representative_aus(frames):
    result = cross_validate_pipeline(
        frames, "label", "participant", representative_aus=False, **fast_kwargs()
    )
    features = set(result.selected_features["feature"])
    assert any(f.endswith("_nmf1") for f in features)
    assert not any("AU" in f for f in features)
    assert result.selected_aus.empty
    assert result.au_frequency().empty


def test_without_decomposer_uses_every_normalised_au(frames):
    result = cross_validate_pipeline(
        frames, "label", "participant", decomposer=False, **fast_kwargs()
    )
    features = set(result.selected_features["feature"])
    for au in AU_COLUMNS:
        assert f"mean_smth_{au}" in features


def test_already_smoothed_input_with_explicit_series(frames):
    smoothed = RollingSmoother().fit_transform(frames)
    result = cross_validate_pipeline(
        smoothed,
        "label",
        "participant",
        smoother=False,
        normalizer=False,
        decomposer=False,
        series_columns=["smth_AU12_r"],
        **fast_kwargs(),
    )
    assert result.selected_features["feature"].str.endswith("_smth_AU12_r").all()
    assert result.summary().loc["roc_auc", "mean"] > 0.9


def test_all_frame_steps_off_needs_series_columns(frames):
    with pytest.raises(ValueError, match="series_columns"):
        cross_validate_pipeline(
            frames,
            "label",
            "participant",
            smoother=False,
            normalizer=False,
            decomposer=False,
            **fast_kwargs(),
        )


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def test_labels_and_groups_as_series_match_column_names(frames, binary_result):
    per_video = frames.groupby("video_filename")[["label", "participant"]].first()
    result = cross_validate_pipeline(
        frames.drop(columns=["label", "participant"]),
        per_video["label"],
        per_video["participant"],
        keep_bases=True,
        keep_models=True,
        **fast_kwargs(),
    )
    pd.testing.assert_frame_equal(result.oof, binary_result.oof)


def test_parallel_folds_match_sequential(frames, binary_result):
    result = cross_validate_pipeline(frames, "label", "participant", n_jobs=2, **fast_kwargs())
    pd.testing.assert_frame_equal(result.oof, binary_result.oof)
    pd.testing.assert_frame_equal(result.scores, binary_result.scores)


def test_same_seed_same_result(frames, binary_result):
    again = cross_validate_pipeline(frames, "label", "participant", **fast_kwargs())
    pd.testing.assert_frame_equal(again.oof, binary_result.oof)


def test_missing_label_for_a_video_raises(frames):
    per_video = frames.groupby("video_filename")["label"].first().iloc[1:]
    with pytest.raises(ValueError, match="labels has no value for 1 video"):
        cross_validate_pipeline(frames, per_video, "participant", **fast_kwargs())


def test_label_that_varies_within_a_video_raises(frames):
    bad = frames.copy()
    bad.loc[bad.index[0], "label"] = "b" if bad.loc[bad.index[0], "label"] == "a" else "a"
    with pytest.raises(ValueError, match="more than one"):
        cross_validate_pipeline(bad, "label", "participant", **fast_kwargs())


def test_bad_inputs_raise(frames):
    with pytest.raises(KeyError, match="no column 'clip'"):
        cross_validate_pipeline(frames, "label", "participant", video_col="clip")
    with pytest.raises(TypeError, match="column name or a pd.Series"):
        cross_validate_pipeline(frames, ["a"], "participant")
    with pytest.raises(ValueError, match="at least two classes"):
        cross_validate_pipeline(frames.assign(label="a"), "label", "participant")
    with pytest.raises(TypeError, match="predict_proba"):
        cross_validate_pipeline(frames, "label", "participant", model=SVC(probability=False))


def test_result_type(binary_result):
    assert isinstance(binary_result, CVPipelineResult)
    summary = binary_result.summary()
    assert list(summary.columns) == ["mean", "std"]
    assert "roc_auc" in summary.index
