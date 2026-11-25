"""Interpretable, leakage-aware classification of facial action unit (AU) time series.

The pipeline runs from frame-level AU intensities to a video-level
classifier: smoothing, normalisation, NMF or representative-AU reduction,
interpretable time-series features, Boruta selection with stability across
seeds, binary or multiclass classification, and evaluation. Splitting and
cross-validation keep the unit of independence intact -- the participant for
repeated-measures data, the pair for matched-pairs designs -- and
:func:`cross_validate_pipeline` refits every fitted step inside each fold.

facedyn began as the Python port of the R analysis in a deepfake-detection
study (Murphy, Cook & Cuve, in prep). References in the docs to "the
original R analysis" or "the deepfake study" mean that work. It is kept as
a validated reference, not as the package's scope.
"""

__version__ = "1.0.0"

from facedyn.au_labels import humanise_au_label, humanise_au_labels, humanise_feature_label
from facedyn.classifiers import (
    CLASSIFIER_NAMES,
    fit_classifiers,
    make_classifier,
    make_classifiers,
)
from facedyn.crossval import CVPipelineResult, cross_validate_pipeline
from facedyn.effects import (
    ale_grid,
    feature_effect_ale,
    feature_effect_ale_cv,
    plot_feature_effect,
    summarise_feature_effects,
)
from facedyn.evaluation import (
    classification_metrics,
    confusion_matrix_frame,
    cross_validate_grouped,
    delong_roc_test,
    evaluate_models,
    pairwise_contrasts,
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_probability_distributions,
    plot_roc_curves,
    plot_roc_ovr,
    roc_auc_delong_ci,
)
from facedyn.face_maps import plot_nmf_face_maps
from facedyn.feature_selection import (
    BorutaRun,
    BorutaSelector,
    boruta_feature_stats,
    correlated_feature_clusters,
    gini_importance,
    oob_permutation_importance,
    plot_boruta_importance,
    plot_boruta_stability,
    plot_feature_clusters,
    tentative_rough_fix,
)
from facedyn.features import (
    CallableFeatureExtractor,
    FeatureCleaner,
    RFeatureExtractor,
    TimeSeriesFeatureExtractor,
    cmfts_r_features,
    extract_timeseries_features,
    feature_diagnostics,
    pivot_features_wide,
    reshape_to_wide,
)
from facedyn.nmf import (
    NMFDecomposer,
    nmf_cophenetic_correlation,
    nmf_rank_cv_sweep,
    nmf_rank_mse_sweep,
)
from facedyn.normalisation import ZScoreShiftNormalizer
from facedyn.representative_aus import RepresentativeAUSelector, select_representative_aus
from facedyn.smoothing import RollingSmoother
from facedyn.splitting import (
    RepeatedStratifiedGroupKFold,
    group_report,
    group_train_test_split,
    groups_from_column,
    pair_group_report,
    pair_groups,
    pair_groups_from_filenames,
    paired_train_test_split,
)
from facedyn.tuning import (
    ThresholdTuningResult,
    plot_threshold_sweep,
    threshold_grid,
    threshold_sweep,
    tune_selection_threshold,
)

__all__ = [
    "CLASSIFIER_NAMES",
    "BorutaRun",
    "BorutaSelector",
    "CVPipelineResult",
    "CallableFeatureExtractor",
    "FeatureCleaner",
    "NMFDecomposer",
    "RFeatureExtractor",
    "RepeatedStratifiedGroupKFold",
    "RepresentativeAUSelector",
    "RollingSmoother",
    "ThresholdTuningResult",
    "TimeSeriesFeatureExtractor",
    "ZScoreShiftNormalizer",
    "ale_grid",
    "boruta_feature_stats",
    "classification_metrics",
    "cmfts_r_features",
    "confusion_matrix_frame",
    "correlated_feature_clusters",
    "cross_validate_grouped",
    "cross_validate_pipeline",
    "delong_roc_test",
    "evaluate_models",
    "extract_timeseries_features",
    "feature_diagnostics",
    "feature_effect_ale",
    "feature_effect_ale_cv",
    "fit_classifiers",
    "gini_importance",
    "group_report",
    "group_train_test_split",
    "groups_from_column",
    "humanise_au_label",
    "humanise_au_labels",
    "humanise_feature_label",
    "make_classifier",
    "make_classifiers",
    "nmf_cophenetic_correlation",
    "nmf_rank_cv_sweep",
    "nmf_rank_mse_sweep",
    "oob_permutation_importance",
    "pair_group_report",
    "pair_groups",
    "pair_groups_from_filenames",
    "paired_train_test_split",
    "pairwise_contrasts",
    "pivot_features_wide",
    "plot_accuracy_comparison",
    "plot_boruta_importance",
    "plot_boruta_stability",
    "plot_confusion_matrix",
    "plot_decision_boundary",
    "plot_feature_clusters",
    "plot_feature_effect",
    "plot_nmf_face_maps",
    "plot_probability_distributions",
    "plot_roc_curves",
    "plot_roc_ovr",
    "plot_threshold_sweep",
    "reshape_to_wide",
    "roc_auc_delong_ci",
    "select_representative_aus",
    "summarise_feature_effects",
    "tentative_rough_fix",
    "threshold_grid",
    "threshold_sweep",
    "tune_selection_threshold",
]
