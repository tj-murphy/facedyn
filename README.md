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

TO DO:
- Interpretable time series feature extraction (CMFTS; Báldan & Benítez, 2023)
- Feature selection
- Classification

Project is under active development.

## Development install

```bash
pip install -e ".[dev]"
pytest
```
