# Core Pipeline Reference

Reference spec for the Python port of the Paper 1 analysis pipeline
("Interpretable facial dynamics as behavioral and perceptual traces of
deepfakes", Murphy, Cook & Cuve). Use this document, not `final_analysis.Rmd`,
as the source of truth for what to implement — `final_analysis.Rmd` is
largely superseded and contains an abandoned PCA-based branch.

**Ground truth sources, in order of authority:**
1. The published paper's Methods section + Figure 5 flowchart (p.14) —
   `TM-Paper1-Combined-Final.pdf`
2. `Final Analysis/final_analysis_NMF_check.Rmd` in the Paper 1 R repo — the
   more current working file (treat commented-out sections in it as
   abandoned exploratory branches, not spec)
3. ~~`final_analysis.Rmd`~~ — outdated, do not use as a stage reference

**Explicitly out of the final pipeline** (confirmed abandoned/superseded):
PCA branch, separate real-vs-fake NMF models, correlation-cutoff feature
selection, RFE, SMOTE/bootstrap class balancing, dominant-AU averaging.

## Steps

| # | Step | What it does | R source | Python status | Notes / open decisions |
|---|------|---------------|----------|----------------|--------------------------|
| 0 | AU extraction | OpenFace 2.0 → 17 AU intensity series/video, 24fps, truncated to 241 frames (10s) | External tool (OpenFace), not R | **Out of package scope (proposed)** | Package likely assumes AU CSVs already extracted; confirm this scope boundary before writing docs/README |
| 1 | Smoothing | 4-frame left-aligned rolling mean, edge-extended, per video | `zoo::rollmean(k=4, fill="extend", align="left")` | ✅ **Done** — `src/facedyn/smoothing.py` (`RollingSmoother`, sklearn transformer) | Validated two ways: (1) original full-dataset comparison, `r_smoothed.csv` vs `python_smoothed.csv`, identical row counts (111,825), match to FP precision; (2) small-fixture regression test in `tests/validation/` (single video, 241 frames) runs in CI |
| 2 | Train/test split | 80/20 split, preserving real/fake video pairing | `final_analysis_NMF_check.Rmd` ~L229 | Not started | Deterministic once seeded; straightforward |
| 3 | Normalisation | Z-score per video using **train-set-learned** params, then shift by training set's global min z-score (NMF non-negativity) | ~L392 | Not started | Must implement as fit (on train) / transform (on any set) — mirrors the paper's held-out-video reuse pattern |
| 4 | NMF | Rank-3 model fit on training data only; rank chosen via reconstruction error across ranks 2–10 | ~L330 (in old file); NMF fit in `_NMF_check.Rmd` | Not started | `sklearn.decomposition.NMF` candidate; validate rank-selection curve reproduces, and reconstruction error is comparable (stochastic init — expect statistical, not bit-exact, match) |
| 5 | Representative AU selection | Per NMF component, take the AU with the highest loading → 3 representative AU series/video | ~L489 | Not started | Simple argmax over basis matrix; low risk |
| 6 | CMFTS feature extraction | 111 time-series metrics (entropy, autocorrelation, Lempel-Ziv complexity, etc.) per representative AU | R `CMFTS` package | Not started | **Highest-risk step** — no Python port exists. Plan: download CMFTS R source, check whether underlying metrics already exist in Python (`antropy`, `nolds`, `statsmodels`), port only the CMFTS-specific glue/parameterization |
| 7 | CMFTS cleanup | Drop permutation-entropy columns, convert Inf→NA, impute via `missForest`, drop zero-SD features | ~L658–769 | Not started | missForest: check existing Python reimplementations before porting from R source; iterative RF imputation is simple enough to reimplement if needed |
| 8 | Feature selection | **Boruta** only (final method) → 8 features retained for real/fake classifier | R `Boruta` package | Not started | `BorutaPy` (scikit-learn-contrib) candidate — validate against R Boruta; stochastic (random shadow features), so expect similar-but-not-identical selected feature sets — compare via overlap/statistical equivalence |
| 9 | Classifier training + CV | Random Forest, C5.0 Boosted Trees, SVM (RBF), Logistic Regression; 5-fold ×3-repeat CV on train (n=370), evaluated on held-out test (n=94) | `caret::train(...)`, ~L1328 onward | Not started | RF/SVM/LogReg → direct `sklearn` equivalents. **C5.0 has no Python equivalent** — decision: substitute a native classifier (e.g. gradient-boosted trees) rather than bridge via rpy2, per the "well-optimized, easy to use" goal for the shipped package; document the substitution explicitly in the paper/README |
| 10 | Emotion/No-Emotion subset analysis | Post-hoc split of test predictions by emotion annotation | ~L2066 onward | Not started | Simple slicing + re-scoring; low risk |
| 11 | Valence classification | Separate RF trained only on real videos, class-balanced by downsampling to n=72/class, predicting valence (positive/neutral/negative); evaluated on real vs. fake test subsets | `_NMF_check.Rmd` valence branch (this file's primary focus) | Not started | Reuses steps 1–8 pipeline machinery with a different target/subset |
| 12 | Human PLD comparison | Point-light-display perceptual experiment + LOPO/LOSO correspondence analysis | Requires separately-collected human data | **Likely out of scope** | Depends on human data collection, not purely computational — confirm with user whether this belongs in the package or stays paper-only |

## Validation protocol

For every step, run the R implementation and the Python port on the same
input and compare outputs, following the pattern established for smoothing
(`r_smoothed.csv` vs `python_smoothed.csv`):

- **Deterministic steps** (smoothing, split, normalisation, representative-AU
  selection, CMFTS feature values, Inf/zero-SD handling): expect exact or
  near-exact (floating-point-tolerance) numeric match.
- **Stochastic steps** (NMF initialization, Random Forest training,
  missForest imputation, Boruta's shadow-feature sampling): R and NumPy use
  different RNGs, so bit-identical output is not achievable even with
  matched seeds. Validate via statistical equivalence instead — comparable
  reconstruction error / overlapping performance confidence intervals /
  overlapping selected-feature sets — not exact diffing.
- Check in small reference fixtures (not the full private dataset) so these
  comparisons can run as automated regression tests in CI — this also
  strengthens the JOSS submission by demonstrating rigorous validation
  against the original implementation.

## rpy2 policy

`rpy2` (or standalone R scripts) may be used as a **development-time
validation oracle only** — to generate reference outputs to diff the Python
port against. It must not be a runtime dependency of the shipped package:
that would require end users to install R, which conflicts with the goal of
a well-optimized, easy-to-install pure-Python package suitable for JOSS.
