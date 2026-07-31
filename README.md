# facedyn

facedyn is a Python toolkit for interpretable analysis of facial Action Unit (AU) time series.
The aim is to provide researchers with a complete suite of tools for understanding their time series
data and designing transparent classification pipelines.


It packages the entire pipeline featured in Murphy, Cook & Cuve (in prep), including:

- Temporal smoothing (inc. visualisations)
- Normalisation
- Pair-matched test/train splitting
- Dimensionality reduction (Non-Negative Matrix Factorisation)
    - Including functions to find optimal k
    - Fitting NMF
    - Visualising NMF matrices
    - Visualising NMF component face maps (implementation from py-feat; Cheong et al., 2023)
    - Reconstruction error calculations (including per-AU)
- Representative AU selection
- Interpretable time series feature extraction: 31 built-in features across five groups
  (distribution, dynamics, trend/stationarity, complexity, and facial-activation dynamics),
  plus bridges to external feature packages
- Feature diagnostics and optional cleanup: NaN/Inf/near-zero-variance reporting,
  plus an opt-in transformer for dropping, imputing (missForest-equivalent) and pruning,
  working the same way regardless of which feature-extraction route produced the data
- Feature selection: a native Boruta implementation, with out-of-bag permutation importance matching R's `ranger` backend. Repeats the
  run across seeds by default and reports how often each feature survives, because a single
  Boruta run on correlated features returns a confident-looking list that largely does not
  reproduce. Ships correlated-cluster diagnostics to show why features churn,
  plus importance, stability and cluster plots
- Classification: random forest, RBF-kernel SVM, unpenalised logistic regression and a
  documented C5.0 substitute (AdaBoost, since C5.0 has no Python implementation), each
  built as a scaler + estimator pipeline with the original analysis's hyperparameters
- Evaluation: `caret::confusionMatrix`-equivalent statistics (accuracy with exact
  Clopper-Pearson intervals, no-information-rate test, kappa, McNemar, sensitivity and
  specificity) and `pROC`-equivalent ROC statistics (AUC with DeLong intervals, paired and
  unpaired DeLong tests), validated against the original R output; ROC, confusion-matrix,
  predicted-probability, decision-boundary and accuracy-comparison plots
- Pair-aware cross-validation: group-id helpers and a repeated stratified grouped k-fold
  splitter. On matched-pairs data (each deepfake generated from a specific real video), folds
  that split a pair train on one twin and test on its opposite-labelled counterpart, which
  drove ROC-AUC below chance in the original analysis's data, so grouping is the default
  and the ungrouped path warns

TO DO:
- Boruta selection-threshold tuning
- Emotion-subset and valence analyses

Project is under active development.

## Development install

```bash
pip install -e ".[dev]"
pytest
```

To also run the R-bridge validation tests (skipped otherwise):

```bash
pip install -e ".[dev,r]"     # on R 4.4 or older, pin: pip install 'rpy2<3.6'
```
