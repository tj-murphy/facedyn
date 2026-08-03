# facedyn

**Interpretable analysis of facial Action Unit (AU) time series.** Provides researchers with a complete suite of tools for understanding their time series data and designing transparent classification pipelines.

facedyn packages the pipeline from Murphy, Cook & Cuve (in prep), which used
facial dynamics to distinguish real videos from deepfakes.

```bash
pip install facedyn          # core
pip install facedyn[viz]     # + matplotlib, for the plot_* functions
```

## The pipeline

| Stage | What it does | Main API |
|---|---|---|
| **Smoothing** | Rolling mean per video, edge-extended | `RollingSmoother` |
| **Splitting** | Train/test that never splits a matched pair | `paired_train_test_split`, `group_train_test_split` |
| **Normalisation** | Z-score plus the non-negativity shift NMF needs | `ZScoreShiftNormalizer` |
| **Dimensionality reduction** | NMF, with rank selection and reconstruction diagnostics | `NMFDecomposer`, `nmf_rank_cv_sweep`, `nmf_cophenetic_correlation` |
| **Representative AUs** | Keep the top-loading AU per component | `RepresentativeAUSelector` |
| **Feature extraction** | 31 interpretable features per series, or bridge to another package | `TimeSeriesFeatureExtractor`, `RFeatureExtractor`, `CallableFeatureExtractor` |
| **Feature cleanup** | NaN/Inf/zero-variance report, then optional imputation | `feature_diagnostics`, `FeatureCleaner` |
| **Feature selection** | Native Boruta, repeated across seeds for stability | `BorutaSelector`, `correlated_feature_clusters` |
| **Threshold tuning** | Picks Boruta's stability threshold on held-out-of-training data | `tune_selection_threshold` |
| **Classification** | Random forest, RBF SVM, logistic regression, C5.0 substitute | `make_classifiers`, `fit_classifiers` |
| **Evaluation** | `caret`/`pROC`-equivalent statistics, grouped CV, figures | `classification_metrics`, `roc_auc_delong_ci`, `delong_roc_test`, `cross_validate_grouped` |

Every transformer follows the scikit-learn `fit`/`transform` contract, so
stages compose with `Pipeline` and can be swapped for your own.

## Quickstart

```python
from facedyn import (
    RollingSmoother, ZScoreShiftNormalizer, NMFDecomposer,
    RepresentativeAUSelector, paired_train_test_split, pair_groups,
    BorutaSelector, make_classifiers, fit_classifiers, evaluate_models,
)

train_df, test_df = paired_train_test_split(frames, random_state=0)

smoother = RollingSmoother().fit(train_df)
normalizer = ZScoreShiftNormalizer().fit(smoother.transform(train_df))
decomposer = NMFDecomposer(n_components=3, random_state=0).fit(
    normalizer.transform(smoother.transform(train_df))
)
aus = RepresentativeAUSelector(decomposer)   # reads the fitted decomposer

# ... feature extraction and cleanup, then one row per video ...

selector = BorutaSelector(n_repeats=20, random_state=0).fit(wide, y_train)
print(selector.stability_)          # how often each feature survived

models = fit_classifiers(make_classifiers(random_state=0), wide[selector.selected_columns_], y_train)
evaluate_models(models, test_wide[selector.selected_columns_], y_test, positive_label="fake")
```

## Scope

Version 1.0.0 covers the pipeline above end to end.

Additions and improvements ship in later versions.

## Development

```bash
pip install -e ".[dev,viz]"
pytest              # fast tests, ~15 s
pytest --runslow    # everything, including Boruta and R validation, ~5 min
```

`pytest` skips tests marked `slow` — Boruta fits, `missForest`-equivalent
imputation, the rpy2 bridge — so the edit loop stays in seconds. CI runs
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
