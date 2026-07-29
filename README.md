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

TO DO:
- Feature selection
- Classification

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
