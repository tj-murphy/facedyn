# Project Structure Plan

Target: a standard, installable Python package suitable for a JOSS (or
similar open-science software journal) submission — not a collection of
analysis scripts. This document is the proposed layout and the reasoning
behind it; see `PIPELINE.md` for what each module needs to implement.

## Why a "regular" installable package, not a script collection

JOSS reviewers specifically check for: an OSI-approved open license,
`pip`-installable packaging, automated tests with CI, documentation
(installation + usage example + API reference), a clear statement of need,
and a community contribution process. A folder of standalone scripts (like
the current `smoothing.py`/`diagnose.py`) doesn't satisfy any of these — the
project needs to become a proper package before submission is realistic.

## API design: scikit-learn-style estimators

Each pipeline step (smoothing, normalisation, NMF, representative-AU
selection, CMFTS extraction, feature selection) should be implemented as a
scikit-learn-compatible transformer (`fit`/`transform`, subclassing
`BaseEstimator`/`TransformerMixin`), composed into a single `Pipeline` object
for the end-to-end flow. Reasons:

- **Matches the R code's own fit/apply pattern** — `final_analysis_NMF_check.Rmd`
  learns normalisation/NMF/feature-selection parameters on the training set
  and re-applies them verbatim to a separate 40-held-out-video set. A
  fit/transform split is the natural Python equivalent, and lets users
  apply the trained pipeline to *their own* new videos — a meaningful
  usability win over a monolithic script.
- **Idiomatic for the target audience** — behavioral/psych researchers doing
  ML in Python already know scikit-learn; an unfamiliar bespoke framework
  adds friction for exactly the people JOSS submissions need to be usable by.
- **Composability and testability** — each transformer can be unit-tested in
  isolation, and the full pipeline can be built/inspected/swapped via
  standard `sklearn.pipeline.Pipeline` tooling.

## Proposed layout

```
<package-name>/                     # e.g. facedyn, deepfake-au-pipeline — TBD
├── src/
│   └── <package_name>/
│       ├── __init__.py
│       ├── smoothing.py            # RollingSmoother transformer
│       ├── splitting.py            # paired train/test split utility
│       ├── normalisation.py        # ZScoreNonNegShift transformer
│       ├── nmf.py                  # NMFDecomposer + rank-selection helper
│       ├── representative_aus.py   # RepresentativeAUSelector
│       ├── features/
│       │   ├── __init__.py
│       │   └── cmfts.py            # ported CMFTS metrics
│       ├── preprocessing.py        # Inf handling, imputation, zero-SD pruning
│       ├── feature_selection.py    # Boruta wrapper (BorutaPy-based)
│       ├── classifiers.py          # RF / SVM / LogReg + documented C5.0 substitute
│       ├── pipeline.py             # assembles the full sklearn Pipeline
│       └── evaluation.py           # CV harness, metrics, emotion-subset analysis
├── tests/
│   ├── unit/                       # one test module per src module
│   └── validation/                 # R-vs-Python regression tests + small fixture data
├── docs/                           # mkdocs (or Sphinx) source; PIPELINE.md content migrates here
├── examples/                       # example notebooks/scripts on public or synthetic data
├── paper/
│   ├── paper.md                    # JOSS paper (statement of need, functionality, etc.)
│   ├── paper.bib
│   └── figures/
├── .github/
│   └── workflows/ci.yml            # lint + pytest on push/PR
├── pyproject.toml                  # packaging (hatch or setuptools), pinned dependencies
├── README.md                       # statement of need, install, quickstart example
├── LICENSE                         # OSI-approved (e.g. MIT)
├── CONTRIBUTING.md
├── CITATION.cff
└── CHANGELOG.md
```

`PIPELINE.md` and this file stay at the repo root for now as living planning
docs; content migrates into `docs/` once a documentation site is set up.

## Other open-science / JOSS-readiness items to resolve early

- **Version control**: this directory is not yet a git repository. JOSS
  expects a version-controlled history; user is handling `git init` /
  first commit manually.
- **Data availability**: the real dataset (Google DFD, via FaceForensics++)
  likely cannot be redistributed. Plan to ship a small synthetic or public
  sample dataset for docs/tests/examples, with clear instructions for users
  to obtain the real dataset separately.
- **Dependency reproducibility**: pin dependency versions in `pyproject.toml`
  (and/or a lockfile) so results are reproducible by reviewers and future
  users, not just "install whatever's latest."
- **Test depth for CI**: unit tests per transformer, plus lightweight
  regression tests against checked-in R reference outputs (small fixtures,
  not the full 111k-row CSVs) so CI runs fast.
- **Package naming**: not yet decided — needs a name before `pyproject.toml`
  / PyPI registration.

## Scaffolding status (as of 2026-07-21)

Done:
- `pyproject.toml` (hatchling build backend, `src/` layout, `dev` extra for
  pytest), package installs editable (`pip install -e ".[dev]"`)
- `src/facedyn/` — package name decided: **facedyn**
- `LICENSE` — **MIT**, decided; referenced from `pyproject.toml` via SPDX
  `license = "MIT"` / `license-files = ["LICENSE"]`
- `src/facedyn/smoothing.py` — `RollingSmoother`, step 1 (smoothing)
  reimplemented as a scikit-learn `BaseEstimator`/`TransformerMixin`,
  replacing the old root-level `smoothing.py` script (removed)
- `tests/unit/test_smoothing.py` — unit tests incl. a brute-force reference
  implementation (independent of the vectorized trick under test) and a
  cross-video-leakage check
- `tests/validation/` — small single-video (241-frame) fixture pair
  extracted from the full dataset + R output, regression-tests the Python
  transformer against R's actual `zoo::rollmean` output (fast enough for CI,
  unlike the original full 111k-row CSVs)
- All 8 tests pass (`pytest`)

Remaining (not yet actioned — user is handling git manually):
1. `git init` / first commit (user's responsibility)
2. `.github/workflows/` CI (lint + pytest on push/PR)
4. Continue implementing pipeline steps in the order listed in `PIPELINE.md`,
   following the same pattern: sklearn-style transformer in `src/facedyn/` +
   unit tests + a small-fixture R-vs-Python regression test in
   `tests/validation/`
5. Decide what to do with the original full-size `python_smoothed.csv` /
   `r_smoothed.csv` at the repo root (37–42MB) — no longer needed for tests
   now that small fixtures exist in `tests/validation/fixtures/`; candidates
   are deleting them or moving them outside the package if kept as raw
   validation artifacts
