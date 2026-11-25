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

## Input data

facedyn reads long-format, frame-level data: one row per video frame.

| Column | Default name | Notes |
|---|---|---|
| Video id | `video_filename` | Groups the frames of one video. Every function that needs it takes `video_col`. |
| Frame number | `frame` | Orders frames within a video (`frame_col`). |
| AU intensities | any column ending `_r` (e.g. `AU01_r`) | OpenFace's naming. Override with `column_pattern=` or `columns=[...]`. |
| Label | any name | Constant within a video. Binary or multiclass. |
| Group | any name | The unit of independence, e.g. `participant`. Constant within a video. |

| video_filename | frame | participant | emotion | AU01_r | AU06_r | AU12_r |
|---|---|---|---|---|---|---|
| p01_happy_1 | 1 | p01 | happy | 0.4 | 0.1 | 0.0 |
| p01_happy_1 | 2 | p01 | happy | 0.5 | 0.2 | 0.1 |
| p02_sad_1 | 1 | p02 | sad | 0.0 | 0.3 | 0.6 |

## Quickstart

`frames` is a table in the shape above.

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
python -m pip install -U pip           # dependency groups need pip >= 25.1
pip install --group dev -e ".[viz]"   # tests, ruff, mypy, pre-commit
pre-commit install                    # lint and format on every commit
pytest                                # fast tests, ~15 s
pytest --runslow                      # everything, including Boruta and R validation
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Credits

NMF component face maps adapt the visualisation approach from
[py-feat](https://py-feat.org/) (Cheong et al., 2023).

Licensed under the MIT License.
