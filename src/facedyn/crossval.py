"""Cross-validation that refits every fitted pipeline step inside each fold.

:func:`facedyn.evaluation.cross_validate_grouped` takes a finished
video-level feature table. Anything fitted to produce that table --
normalisation, NMF, representative-AU choice, imputation and, above all,
supervised feature selection -- has then already seen the videos that
later act as test folds. Selection fitted outside the folds is the
textbook case of feature-selection leakage (Ambroise & McLachlan, 2002;
REFORMS item L1.3): on pure noise it produces a well-above-chance score.

:func:`cross_validate_pipeline` starts from frame-level data instead and
runs the whole pipeline per fold::

    split videos by group
      -> fit normaliser on training frames, transform train and test
      -> fit NMF on training frames, pick representative AUs
      -> extract per-video time-series features
      -> fit cleaner, feature selector and classifier on training videos
      -> predict the test videos

A plain scikit-learn ``Pipeline`` cannot express this, because the
frame -> video step changes the number of rows. Smoothing is the one step
run once up front: it works within each video and learns nothing, so it
cannot carry information between videos.

The result holds out-of-fold predictions for every video in every repeat,
per-fold scores, and which AUs and features each fold picked, so that the
stability of the selection can be reported alongside the score.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from facedyn.classifiers import make_classifier
from facedyn.evaluation import roc_auc_delong_ci
from facedyn.feature_selection import BorutaSelector
from facedyn.features.cleaning import FeatureCleaner
from facedyn.features.reshape import pivot_features_wide, reshape_to_wide
from facedyn.features.timeseries import TimeSeriesFeatureExtractor
from facedyn.nmf import NMFDecomposer
from facedyn.normalisation import ZScoreShiftNormalizer
from facedyn.representative_aus import RepresentativeAUSelector
from facedyn.smoothing import RollingSmoother
from facedyn.splitting import RepeatedStratifiedGroupKFold, _video_to_group

__all__ = [
    "CVPipelineResult",
    "cross_validate_pipeline",
]

_SERIES_COL = "series"


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class CVPipelineResult:
    """Output of :func:`cross_validate_pipeline`.

    Attributes
    ----------
    oof : pd.DataFrame
        Out-of-fold predictions, one row per video per repeat. Columns
        ``video``, ``group``, ``repeat``, ``fold``, ``y_true``, ``y_pred``
        and one ``proba_<class>`` column per class (in ``classes`` order).
    scores : pd.DataFrame
        One row per fold (indexed by ``repeat`` and ``fold``), with
        ``roc_auc``, ``accuracy``, ``balanced_accuracy`` and ``n_test``.
        ``roc_auc`` is one-vs-rest macro-averaged when there are more than
        two classes, and ``NaN`` for a test fold that holds only one class.
    selected_aus : pd.DataFrame
        One row per fold per NMF component: ``repeat``, ``fold``,
        ``component``, ``au``. Empty when representative AUs were not used.
    selected_features : pd.DataFrame
        One row per fold per candidate feature: ``repeat``, ``fold``,
        ``feature``, ``selected``, plus ``selection_freq`` (the fraction of
        Boruta runs confirming it in that fold) when the selector reports
        one. Every candidate is ``selected`` when there is no selector.
    classes : list
        Class labels, sorted. Fixes the order of the ``proba_`` columns.
    nmf_bases : dict
        ``(repeat, fold) -> DataFrame`` of the fold's NMF basis matrix
        (components x AU columns). Only filled with ``keep_bases=True``.
    fold_models : dict
        ``(repeat, fold) -> dict`` with the fitted ``model``, its
        ``features`` (column order) and ``X_test`` (the fold's test feature
        table, indexed by video). Only filled with ``keep_models=True``.
    """

    oof: pd.DataFrame
    scores: pd.DataFrame
    selected_aus: pd.DataFrame
    selected_features: pd.DataFrame
    classes: list
    nmf_bases: dict = field(default_factory=dict)
    fold_models: dict = field(default_factory=dict)

    @property
    def n_repeats(self) -> int:
        """Number of CV repeats in the result."""
        return int(self.oof["repeat"].nunique())

    @property
    def n_folds(self) -> int:
        """Total number of folds, summed over repeats."""
        return len(self.scores)

    def summary(self) -> pd.DataFrame:
        """Mean and SD of each per-fold score, across all folds."""
        return self.scores.agg(["mean", "std"]).T

    def au_frequency(self) -> pd.Series:
        """Fraction of folds in which each AU was chosen as representative.

        An AU chosen for two components in one fold counts once.
        """
        if self.selected_aus.empty:
            return pd.Series(dtype=float, name="frequency")
        per_fold = self.selected_aus.drop_duplicates(["repeat", "fold", "au"])
        return (
            (per_fold["au"].value_counts() / self.n_folds)
            .rename("frequency")
            .rename_axis("au")
        )

    def feature_frequency(self) -> pd.Series:
        """Fraction of folds in which each feature was kept by the selector.

        The denominator is every fold, so a feature that only existed in
        some folds (e.g. built from an AU only some folds chose) is
        penalised for the folds it was absent from.
        """
        kept = self.selected_features.loc[self.selected_features["selected"], "feature"]
        return (
            (kept.value_counts() / self.n_folds)
            .reindex(self.selected_features["feature"].unique(), fill_value=0.0)
            .sort_values(ascending=False, kind="stable")
            .rename("frequency")
            .rename_axis("feature")
        )

    def oof_metrics(self, positive_label=None, alpha: float = 0.05) -> pd.DataFrame:
        """Scores of the pooled out-of-fold predictions, one row per repeat.

        Pooling scores every video once per repeat, by a model that never
        saw it, which is the number to report alongside the per-fold
        spread.

        Parameters
        ----------
        positive_label : optional
            Binary only: the class whose probability is scored. ROC-AUC
            and its interval do not depend on the choice, so the default
            (the last of ``classes``) is safe.
        alpha : float, default 0.05
            1 - confidence level of the DeLong interval.

        Returns
        -------
        pd.DataFrame
            Indexed by ``repeat``. Columns ``n``, ``roc_auc``,
            ``accuracy``, ``balanced_accuracy``, and for binary labels
            ``roc_auc_ci_lower``/``_upper`` (DeLong). DeLong is binary
            only, so with more than two classes ``roc_auc`` is the
            one-vs-rest macro average and has no interval, and
            ``roc_auc_<class>`` gives each class's one-vs-rest AUC. For
            the full multiclass statistics of one repeat (per-class
            DeLong intervals, macro F1, kappa), pass its rows to
            :func:`facedyn.evaluation.classification_metrics`.
        """
        binary = len(self.classes) == 2
        if binary:
            positive_label = self.classes[-1] if positive_label is None else positive_label
            if positive_label not in self.classes:
                raise ValueError(
                    f"positive_label={positive_label!r} is not one of {self.classes}."
                )
        proba_cols = [f"proba_{c}" for c in self.classes]

        rows = {}
        for repeat, part in self.oof.groupby("repeat"):
            y_true = part["y_true"].to_numpy()
            row = {
                "n": len(part),
                "accuracy": accuracy_score(y_true, part["y_pred"]),
                "balanced_accuracy": balanced_accuracy_score(y_true, part["y_pred"]),
            }
            if binary:
                auc, lower, upper = roc_auc_delong_ci(
                    y_true, part[f"proba_{positive_label}"], positive_label=positive_label,
                    alpha=alpha,
                )
                row.update(roc_auc=auc, roc_auc_ci_lower=lower, roc_auc_ci_upper=upper)
            else:
                proba = part[proba_cols].to_numpy()
                row["roc_auc"] = _roc_auc(y_true, proba, self.classes)
                for k, cls in enumerate(self.classes):
                    is_class = y_true == cls
                    row[f"roc_auc_{cls}"] = (
                        float(roc_auc_score(is_class, proba[:, k]))
                        if 0 < is_class.sum() < len(is_class) else float("nan")
                    )
            rows[repeat] = row
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("repeat")


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def cross_validate_pipeline(
    frames: pd.DataFrame,
    labels,
    groups,
    *,
    smoother=None,
    normalizer=None,
    decomposer=None,
    representative_aus: bool = True,
    series_columns: list[str] | None = None,
    feature_extractor=None,
    cleaner=None,
    selector=None,
    model=None,
    cv=None,
    random_state: int | None = None,
    n_jobs: int | None = None,
    keep_bases: bool = False,
    keep_models: bool = False,
    video_col: str = "video_filename",
    frame_col: str = "frame",
) -> CVPipelineResult:
    """Cross-validate by group, refitting every fitted step inside each fold.

    Use this, not :func:`facedyn.evaluation.cross_validate_grouped`, when
    starting from frame-level AU data: every step that learns from data is
    fitted on the training videos of each fold only. See the module
    docstring for the per-fold sequence.

    Each step argument takes ``None`` for facedyn's default, an unfitted
    estimator of your own (cloned per fold, so it is never modified), or
    ``False`` to skip the step. ``random_state`` seeds the defaults and the
    default splitter only; estimators you pass keep their own seeds.

    Parameters
    ----------
    frames : pd.DataFrame
        Frame-level data, one row per frame (see "Input data" in the README), with
        ``video_col``, ``frame_col`` and the raw AU columns.
    labels : str or pd.Series
        The class of each video: a column of ``frames`` (constant within
        each video), or a Series indexed by video id.
    groups : str or pd.Series
        The unit of independence of each video, e.g. ``"participant"`` for
        repeated measures or a pair id for matched pairs (see
        :mod:`facedyn.splitting`). A column of ``frames`` or a Series
        indexed by video id. Folds never split a group.
    smoother : estimator or False, optional
        Default :class:`~facedyn.smoothing.RollingSmoother`. Applied once
        to all frames before splitting, which is safe because it works
        within each video and learns nothing. ``False`` if ``frames`` is
        already smoothed.
    normalizer : estimator or False, optional
        Default :class:`~facedyn.normalisation.ZScoreShiftNormalizer`.
    decomposer : NMFDecomposer or False, optional
        Default ``NMFDecomposer(n_components=3)``. ``False`` extracts
        features from ``series_columns`` directly.
    representative_aus : bool, default True
        With a decomposer: ``True`` extracts features from each
        component's representative AU (chosen per fold with
        :class:`~facedyn.representative_aus.RepresentativeAUSelector`);
        ``False`` extracts them from the NMF activations instead.
    series_columns : list of str, optional
        The frame columns to extract features from when
        ``decomposer=False``. Defaults to the normaliser's columns, or the
        smoother's output columns if there is no normaliser.
    feature_extractor : estimator, optional
        Default :class:`~facedyn.features.TimeSeriesFeatureExtractor`. Any
        extractor taking the shape of
        :func:`~facedyn.features.reshape_to_wide` works.
    cleaner : estimator or False, optional
        Default :class:`~facedyn.features.FeatureCleaner`. Fitted on the
        training videos' extracted features.
    selector : estimator or False, optional
        Default :class:`~facedyn.feature_selection.BorutaSelector`. Any
        scikit-learn selector with ``get_feature_names_out`` works (e.g.
        ``SelectKBest``). ``False`` keeps every feature, which is much
        faster for seed sweeps. A fold in which nothing is selected warns
        and predicts the training class frequencies.
    model : estimator, optional
        Default ``make_classifier("random_forest", preset="default")``.
        Must implement ``predict_proba``.
    cv : cross-validator, optional
        Default ``RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=3)``.
        Splits videos; it receives ``groups``. Repeats are detected from
        when a video is tested a second time.
    random_state : int, optional
        Seeds the default splitter and default steps.
    n_jobs : int, optional
        Folds run in parallel. Leave the steps' own ``n_jobs`` at 1 when
        using this, to avoid nested parallelism.
    keep_bases : bool, default False
        Store each fold's NMF basis matrix in ``nmf_bases``.
    keep_models : bool, default False
        Store each fold's fitted model and test feature table in
        ``fold_models`` (e.g. for averaging effect curves over folds).
    video_col, frame_col : str
        Column names in ``frames``.

    Returns
    -------
    CVPipelineResult

    Notes
    -----
    Runtime is dominated by the selector: the default Boruta runs 20
    repeats of up to 99 forests in every fold. For quick runs pass
    ``selector=False`` or a cheaper ``BorutaSelector`` (fewer repeats,
    ``importance="gini"``), and use ``n_jobs``.

    ``ZScoreShiftNormalizer`` recomputes its non-negativity shift on each
    ``transform`` call, so the test videos of a fold are shifted by their
    own minimum, exactly as the original analysis treated its test set.

    Examples
    --------
    >>> result = cross_validate_pipeline(
    ...     frames, labels="emotion", groups="participant", random_state=0
    ... )  # doctest: +SKIP
    >>> result.oof_metrics()        # pooled out-of-fold AUC per repeat
    >>> result.feature_frequency()  # how often each feature was selected
    """
    for column in (video_col, frame_col):
        if column not in frames:
            raise KeyError(f"frames has no column {column!r}")

    videos = pd.Index(pd.unique(frames[video_col]), name=video_col)
    y = _per_video(frames, labels, videos, video_col, "labels")
    group_ids = _per_video(frames, groups, videos, video_col, "groups")
    classes = sorted(pd.unique(y).tolist())
    if len(classes) < 2:
        raise ValueError(f"labels must contain at least two classes, got {classes}.")

    steps = _resolve_steps(
        smoother, normalizer, decomposer, feature_extractor, cleaner, selector, model,
        random_state, video_col,
    )
    if cv is None:
        cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=3, random_state=random_state)

    if steps["smoother"] is not None:
        frames = clone(steps["smoother"]).fit_transform(frames)

    splits = list(cv.split(np.zeros(len(videos)), y, group_ids))
    fold_ids = _repeat_and_fold(splits, len(videos))

    config = {
        "representative_aus": representative_aus,
        "series_columns": series_columns,
        "classes": classes,
        "keep_bases": keep_bases,
        "keep_models": keep_models,
        "video_col": video_col,
        "frame_col": frame_col,
    }
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(
            frames, videos, y, train_idx, test_idx, steps, config
        )
        for train_idx, test_idx in splits
    )

    empty = [ids for ids, out in zip(fold_ids, outputs, strict=True) if not out["features"]["selected"].any()]
    if empty:
        warnings.warn(
            f"The selector kept no features in {len(empty)} of {len(splits)} folds "
            f"(repeat, fold: {empty[:5]}), so those folds predict the training class "
            "frequencies and score at chance. Consider a laxer selector or "
            "selector=False.",
            UserWarning,
            stacklevel=2,
        )

    oof_parts, score_rows, au_parts, feature_parts = [], [], [], []
    nmf_bases, fold_models = {}, {}
    for (repeat, fold), (_, test_idx), out in zip(fold_ids, splits, outputs, strict=True):
        test_videos = videos[test_idx]
        oof = pd.DataFrame({
            "video": test_videos,
            "group": group_ids[test_idx],
            "repeat": repeat,
            "fold": fold,
            "y_true": y[test_idx],
            "y_pred": out["y_pred"],
        })
        for k, cls in enumerate(classes):
            oof[f"proba_{cls}"] = out["proba"][:, k]
        oof_parts.append(oof)
        score_rows.append({"repeat": repeat, "fold": fold, **_fold_scores(oof, classes)})
        au_parts.append(out["aus"].assign(repeat=repeat, fold=fold))
        feature_parts.append(out["features"].assign(repeat=repeat, fold=fold))
        if keep_bases and out["basis"] is not None:
            nmf_bases[(repeat, fold)] = out["basis"]
        if keep_models:
            fold_models[(repeat, fold)] = out["model"]

    lead = ["repeat", "fold"]
    au_table = pd.concat(au_parts, ignore_index=True)
    feature_table = pd.concat(feature_parts, ignore_index=True)
    return CVPipelineResult(
        oof=pd.concat(oof_parts, ignore_index=True),
        scores=pd.DataFrame(score_rows).set_index(lead),
        selected_aus=au_table[lead + [c for c in au_table.columns if c not in lead]],
        selected_features=feature_table[
            lead + [c for c in feature_table.columns if c not in lead]
        ],
        classes=classes,
        nmf_bases=nmf_bases,
        fold_models=fold_models,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _per_video(frames, spec, videos, video_col, name) -> np.ndarray:
    """One value per video, from a column name or a video-indexed Series."""
    if isinstance(spec, str):
        mapping = _video_to_group(frames, spec, video_col)
    elif isinstance(spec, pd.Series):
        if not spec.index.is_unique:
            raise ValueError(f"{name} must have one entry per video (duplicate index values).")
        mapping = spec
    else:
        raise TypeError(
            f"{name} must be a column name or a pd.Series indexed by video id, "
            f"got {type(spec).__name__}."
        )
    values = mapping.reindex(videos)
    missing = values.isna()
    if missing.any():
        offenders = sorted(map(str, videos[missing.to_numpy()]))[:5]
        raise ValueError(
            f"{name} has no value for {int(missing.sum())} video(s), e.g. {offenders}."
        )
    return values.to_numpy()


def _resolve_steps(smoother, normalizer, decomposer, feature_extractor, cleaner,
                   selector, model, random_state, video_col) -> dict:
    """Fill in defaults; ``False`` becomes ``None`` (step skipped)."""
    def pick(value, default):
        if value is False:
            return None
        return default() if value is None else value

    steps = {
        "smoother": pick(smoother, lambda: RollingSmoother(group_col=video_col)),
        "normalizer": pick(normalizer, ZScoreShiftNormalizer),
        "decomposer": pick(
            decomposer, lambda: NMFDecomposer(n_components=3, random_state=random_state)
        ),
        "extractor": pick(feature_extractor, TimeSeriesFeatureExtractor),
        "cleaner": pick(cleaner, lambda: FeatureCleaner(random_state=random_state)),
        "selector": pick(selector, lambda: BorutaSelector(random_state=random_state)),
        "model": pick(
            model,
            lambda: make_classifier(
                "random_forest", preset="default", random_state=random_state
            ),
        ),
    }
    if steps["extractor"] is None:
        raise ValueError("feature_extractor cannot be False: features are required.")
    if steps["model"] is None:
        raise ValueError("model cannot be False.")
    if not hasattr(steps["model"], "predict_proba"):
        raise TypeError(
            f"{type(steps['model']).__name__} has no predict_proba; out-of-fold "
            "probabilities are needed for ROC-AUC."
        )
    return steps


def _repeat_and_fold(splits, n_videos) -> list[tuple[int, int]]:
    """Label each split with its repeat and its fold within that repeat.

    A new repeat starts when a test fold contains a video already tested
    in the current repeat, which holds for any repeated k-fold splitter.
    """
    ids, seen, repeat, fold = [], np.zeros(n_videos, dtype=bool), 0, 0
    for _, test_idx in splits:
        if seen[test_idx].any():
            repeat, fold = repeat + 1, 0
            seen[:] = False
        seen[test_idx] = True
        ids.append((repeat, fold))
        fold += 1
    return ids


def _roc_auc(y_true, proba, classes) -> float:
    present = set(pd.unique(y_true).tolist())
    if len(present) < 2:
        return float("nan")
    if len(classes) == 2:
        return float(roc_auc_score(y_true == classes[1], proba[:, 1]))
    if present != set(classes):
        # OvR macro AUC is undefined for a class absent from the fold.
        return float("nan")
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                               labels=classes))


def _fold_scores(oof: pd.DataFrame, classes) -> dict:
    y_true = oof["y_true"].to_numpy()
    proba = oof[[f"proba_{c}" for c in classes]].to_numpy()
    return {
        "roc_auc": _roc_auc(y_true, proba, classes),
        "accuracy": accuracy_score(y_true, oof["y_pred"]),
        "balanced_accuracy": balanced_accuracy_score(y_true, oof["y_pred"]),
        "n_test": len(oof),
    }


def _series_frames(train, test, steps, config):
    """Normalise and decompose frame data; return both sides + series names."""
    video_col, frame_col = config["video_col"], config["frame_col"]
    basis, aus = None, pd.DataFrame(columns=["component", "au"])

    normalizer = steps["normalizer"]
    if normalizer is not None:
        normalizer = clone(normalizer).fit(train)
        train, test = normalizer.transform(train), normalizer.transform(test)

    decomposer = steps["decomposer"]
    if decomposer is not None:
        decomposer = clone(decomposer).fit(train)
        basis = pd.DataFrame(
            decomposer.components_,
            index=[f"{decomposer.prefix}{i + 1}" for i in range(decomposer.n_components)],
            columns=decomposer.columns_,
        )
        if config["representative_aus"]:
            au_selector = RepresentativeAUSelector(decomposer).fit(train)
            series = list(dict.fromkeys(au_selector.selected_columns_))
            aus = au_selector.selection_[["component", "au"]].copy()
        else:
            train, test = decomposer.transform(train), decomposer.transform(test)
            series = list(basis.index)
    else:
        series = _default_series(config["series_columns"], normalizer, steps["smoother"],
                                 train)

    keep = [video_col, frame_col, *series]
    return train[keep], test[keep], series, basis, aus


def _default_series(series_columns, normalizer, smoother, frames) -> list[str]:
    if series_columns is not None:
        return list(series_columns)
    if normalizer is not None:
        return list(normalizer.columns_)
    if smoother is not None:
        smoothed_from = clone(smoother).fit(frames).columns_
        return [f"{smoother.prefix}{c}" for c in smoothed_from]
    raise ValueError(
        "With decomposer=False, normalizer=False and smoother=False, pass "
        "series_columns to say which frame columns to extract features from."
    )


def _wide_series(frames, series, video_col, frame_col) -> pd.DataFrame:
    return reshape_to_wide(frames, value_cols=series, id_vars=[], group_col=video_col,
                           frame_col=frame_col)


def _feature_columns(table, video_col) -> list[str]:
    return [c for c in table.columns if c not in (video_col, _SERIES_COL)]


def _to_video_table(long, feature_cols, video_order, video_col) -> pd.DataFrame:
    """One row per video, indexed by video id, in ``video_order``."""
    wide = pivot_features_wide(long, group_col=video_col, series_col=_SERIES_COL,
                               feature_columns=feature_cols, id_vars=[])
    return wide.set_index(video_col).reindex(video_order)


def _run_fold(frames, videos, y, train_idx, test_idx, steps, config) -> dict:
    """Fit every step on the training videos of one fold; predict the rest."""
    video_col, frame_col = config["video_col"], config["frame_col"]
    train_videos, test_videos = videos[train_idx], videos[test_idx]
    y_train = y[train_idx]

    in_train = frames[video_col].isin(train_videos)
    in_test = frames[video_col].isin(test_videos)
    train_frames, test_frames, series, basis, aus = _series_frames(
        frames.loc[in_train].reset_index(drop=True),
        frames.loc[in_test].reset_index(drop=True),
        steps, config,
    )

    train_wide = _wide_series(train_frames, series, video_col, frame_col)
    test_wide = _wide_series(test_frames, series, video_col, frame_col)
    extractor = clone(steps["extractor"]).fit(train_wide)
    train_long, test_long = extractor.transform(train_wide), extractor.transform(test_wide)

    feature_cols = _feature_columns(train_long, video_col)
    if steps["cleaner"] is not None:
        cleaner = clone(steps["cleaner"])
        if "feature_columns" in cleaner.get_params() and cleaner.feature_columns is None:
            cleaner.set_params(feature_columns=feature_cols)
        cleaner.fit(train_long)
        train_long, test_long = cleaner.transform(train_long), cleaner.transform(test_long)
        feature_cols = list(cleaner.get_feature_names_out())

    X_train = _to_video_table(train_long, feature_cols, train_videos, video_col)
    X_test = _to_video_table(test_long, feature_cols, test_videos, video_col)
    candidates = list(X_train.columns)

    features = pd.DataFrame({"feature": candidates, "selected": True})
    if steps["selector"] is not None:
        selector = clone(steps["selector"]).fit(X_train, y_train)
        kept = set(selector.get_feature_names_out())
        features["selected"] = features["feature"].isin(kept)
        frequency = getattr(selector, "selection_frequency_", None)
        if frequency is not None:
            features["selection_freq"] = features["feature"].map(frequency)
    kept_cols = features.loc[features["selected"], "feature"].tolist()

    # No features left: predict the training class frequencies.
    # The caller warns, since warnings raised inside joblib workers are lost.
    model = clone(steps["model"]) if kept_cols else DummyClassifier(strategy="prior")
    X_fit = X_train[kept_cols] if kept_cols else np.zeros((len(X_train), 1))
    X_pred = X_test[kept_cols] if kept_cols else np.zeros((len(X_test), 1))
    model.fit(X_fit, y_train)

    proba = np.zeros((len(test_videos), len(config["classes"])))
    model_proba = model.predict_proba(X_pred)
    for k, cls in enumerate(model.classes_):
        # A class missing from this training fold keeps probability 0.
        proba[:, config["classes"].index(cls)] = model_proba[:, k]

    return {
        "y_pred": model.predict(X_pred),
        "proba": proba,
        "aus": aus,
        "features": features,
        "basis": basis if config["keep_bases"] else None,
        "model": (
            {"model": model, "features": kept_cols, "X_test": X_test[kept_cols]}
            if config["keep_models"] else None
        ),
    }
