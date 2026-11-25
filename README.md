# facedyn

**Interpretable, leakage-aware classification of facial action unit (AU) time series.**
facedyn takes frame-level AU intensities (e.g. from OpenFace) and builds
video-level classifiers whose features can be read in terms of facial
movement. Every step that learns from data is fitted inside the training
folds, and splitting respects the unit of independence, so the scores you
report are not inflated by leakage.

It works with binary or multiclass labels and with any grouping structure:
- **Repeated measures**: several videos per participant, possibly across
  sessions. The participant is the unit.
- **Matched pairs**: e.g. a real video and the deepfake made from it. The
  pair is the unit.

```bash
pip install facedyn          # core
pip install facedyn[viz]     # + matplotlib, for the plot_* functions
```

## Applications

- **Emotion expression** (happy / sad / angry, repeated measures with
  participant grouping): the worked example in the methods paper (in prep).
- **Deepfake detection** (real / fake, matched pairs): Murphy, Cook & Cuve
  (in prep). facedyn began as the Python port of this study's R analysis.
  Regression tests still check the binary path against that analysis's
  output, number for number.

## The pipeline

| Stage | What it does | Main API |
|---|---|---|
| **Smoothing** | Rolling mean per video, edge-extended | `RollingSmoother` |
| **Splitting** | Train/test and CV groups that keep a participant (or a matched pair) on one side | `group_train_test_split`, `groups_from_column`, `group_report`, `paired_train_test_split`, `pair_groups` |
| **Normalisation** | Z-score plus the non-negativity shift NMF needs | `ZScoreShiftNormalizer` |
| **Dimensionality reduction** | NMF, with rank selection and reconstruction diagnostics | `NMFDecomposer`, `nmf_rank_cv_sweep`, `nmf_cophenetic_correlation` |
| **Representative AUs** | Keep the top-loading AU per component | `RepresentativeAUSelector` |
| **Feature extraction** | 31 interpretable features per series, or bridge to another package | `TimeSeriesFeatureExtractor`, `RFeatureExtractor`, `CallableFeatureExtractor` |
| **Feature cleanup** | NaN/Inf/zero-variance report, then optional imputation | `feature_diagnostics`, `FeatureCleaner` |
| **Feature selection** | Native Boruta, repeated across seeds for stability | `BorutaSelector`, `correlated_feature_clusters` |
| **Threshold tuning** | Picks Boruta's stability threshold on data held out of training | `tune_selection_threshold` |
| **Classification** | Random forest, RBF SVM, logistic regression, C5.0 substitute; binary or multiclass | `make_classifiers`, `fit_classifiers` |
| **Cross-validation** | Refits every step above inside each fold; out-of-fold predictions and per-fold selections | `cross_validate_pipeline`, `cross_validate_grouped` |
| **Evaluation** | `caret`/`pROC`-equivalent statistics, macro one-vs-rest AUC, DeLong tests, figures | `classification_metrics`, `roc_auc_delong_ci`, `delong_roc_test`, `pairwise_contrasts`, `plot_roc_ovr`, `plot_confusion_matrix` |
| **Interpretation** | Accumulated local effects (ALE) per feature, averaged over CV folds | `plot_feature_effect`, `feature_effect_ale_cv` |

Every transformer follows the scikit-learn `fit`/`transform` contract, so
you can swap any stage for your own.

## Quickstart

`frames` has one row per video frame: a `video_filename` column, a `frame`
column, OpenFace AU intensity columns (`AU01_r`, `AU02_r`, ...), plus the
label and the participant id (any column names; pass them as arguments).

```python
from facedyn import (
    group_report, cross_validate_pipeline,
    plot_roc_ovr, plot_confusion_matrix, plot_feature_effect,
)

# The participant is the unit of independence: check the groups first.
group_report(frames, "participant", label_col="emotion")

# Smoothing, normalisation, NMF, AU choice, features, cleaning, Boruta and
# the classifier are all refitted on the training participants of each fold.
result = cross_validate_pipeline(
    frames, labels="emotion", groups="participant",
    random_state=0, n_jobs=-1, keep_models=True,
)

result.summary()             # mean +- SD of the per-fold scores
result.oof_metrics()         # pooled out-of-fold AUC per repeat (macro OvR for 3+ classes)
result.au_frequency()        # how often each AU was chosen across folds
result.feature_frequency()   # how often each feature survived selection

oof = result.oof[result.oof["repeat"] == 0]
plot_roc_ovr(oof["y_true"], oof[[f"proba_{c}" for c in result.classes]],
             labels=result.classes)
plot_confusion_matrix(oof["y_true"], oof["y_pred"], labels=result.classes,
                      normalize="reference", annotate="both")
plot_feature_effect(result, features=result.feature_frequency().index[:3])
```

For matched-pairs data, build the groups with `pair_groups` (or
`pair_groups_from_filenames`) and pass them as `groups`. If you already have
a feature table that nothing has been fitted to, `cross_validate_grouped`
cross-validates it directly.

## Scope

Classification of AU time series, binary or multiclass, from OpenFace
output. Continuous outcomes (regression) and other AU extractors are planned
for later versions.

## Development

```bash
pip install -e ".[dev,viz]"
pytest              # fast tests, ~15 s
pytest --runslow    # everything, including Boruta and R validation, ~5 min
```

`pytest` skips tests marked `slow` (Boruta fits, `missForest`-equivalent
imputation, the rpy2 bridge) so the edit loop stays in seconds. CI runs
`--runslow`, so nothing is exempt from gating a push. The suite leaves two
cores free on machines with more than four, since several tests ask for
`n_jobs=-1`; add `nice -n 19` for a full run you want to forget about.

The R-bridge tests need rpy2 and a working R install, and skip otherwise:

```bash
pip install -e ".[dev,r]"     # on R 4.4 or older, pin: pip install 'rpy2<3.6'
```

## Credits

NMF component face maps adapt the visualisation approach from
[py-feat](https://py-feat.org/) (Cheong et al., 2023).

Licensed under the MIT License.
