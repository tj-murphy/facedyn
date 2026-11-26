# Changelog

All notable changes to facedyn will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Participant-level grouping for repeated-measures data, where the participant (not the video) is the unit of independence and sessions nest within participant:
  - `group_train_test_split(..., group_col=None)`: pass e.g. `group_col="participant"` to keep all of a participant's videos on one side of the split. `train_size` is then a fraction of participants. The default (`None`) behaves exactly as before.
  - `groups_from_column(df, group_col, video_col, videos=None)`: builds the `groups` array for `RepeatedStratifiedGroupKFold` / `cross_validate_grouped` from an id column, optionally aligned to a given video order (e.g. a `pivot_features_wide` table).
  - `group_report(df, group_col, video_col, label_col=None)`: videos per group and, with `label_col`, class counts per group.
  - All three raise if a video has a missing group id or more than one.

- `cross_validate_pipeline(frames, labels, groups, ...)` (new module `facedyn.crossval`): grouped cross-validation from frame-level data that refits every fitted step inside each fold (normaliser, NMF, representative-AU choice, feature extraction, cleaner, feature selector and classifier), so nothing fitted ever sees a test-fold video. Smoothing runs once up front, since it is per-video and learns nothing. Each step takes `None` (default), your own unfitted estimator, or `False` to skip it; `representative_aus=False` uses NMF activations instead. Works with two or more classes, and folds run in parallel with `n_jobs`.
- `CVPipelineResult`: `oof` (out-of-fold predictions for every video in every repeat, with one `proba_<class>` column per class), `scores` (per fold), `selected_aus` and `selected_features` (per fold, with Boruta's per-fold `selection_freq`), optional `nmf_bases` (`keep_bases=True`) and fitted fold models (`keep_models=True`); plus `.summary()`, `.au_frequency()`, `.feature_frequency()` and `.oof_metrics()` (pooled out-of-fold ROC-AUC per repeat, with a DeLong CI for binary labels).
- Multiclass evaluation:
  - `classification_metrics` handles more than two classes. It reports accuracy and its exact CI, the no-information rate, kappa, MCC, balanced accuracy, macro F1 and macro one-vs-rest ROC-AUC. It also gives each class `c` its own `precision_c`, `recall_c`, `f1_c` and `roc_auc_c`, the last with a DeLong CI. Pass `y_score` as a 2-D probability array, with the new `labels=` argument giving its column order.
  - A binary `y_score` may also be 2-D now. The binary path is otherwise unchanged; the R-validation tests still pass.
  - `evaluate_models` scores models with more than two classes using the multiclass statistics.
  - `plot_roc_curves` draws one one-vs-rest curve per class for multiclass models.
  - New `plot_roc_ovr(y_true, y_score, labels)` does the same from probabilities alone, e.g. pooled out-of-fold predictions.
  - `plot_confusion_matrix` takes `normalize="reference"` (each true class sums to 1, so recall is on the diagonal) or `"prediction"` (the same as `True`). It also takes `annotate="count" | "proportion" | "both"`. Normalised shading now runs over a fixed 0–1 range, and a row or column with no entries shades as 0 rather than NaN.
  - New `pairwise_contrasts(data, label_col, pairs=None)` returns the binary subsets (e.g. happy–sad, happy–angry, sad–angry) for pairwise DeLong tests, since DeLong itself stays binary.
  - `CVPipelineResult.oof_metrics()` adds a one-vs-rest `roc_auc_<class>` for each class when there are more than two.

- Feature effects with accumulated local effects (new module `facedyn.effects`). ALE is used instead of partial dependence because facedyn's time-series features are strongly correlated (Apley & Zhu, 2020). It is implemented from the paper's definition, with no `ALEPlot`/`alibi`/`PyALE` code or dependency.
  - `feature_effect_ale(model, X, features, grid_size=20, classes=None, grid=None)`: centred first-order ALE on an empirical-quantile grid. Classifiers give one probability curve per class (the positive class only for binary models); other models use `predict`. Returns a long table with `feature`, `class`, `value`, `effect`, `n`.
  - `feature_effect_ale_cv(result, features, X=None, ...)`: the same for every fold model of a `cross_validate_pipeline(..., keep_models=True)` run, on one grid shared by all folds. Each fold is estimated over its own test videos by default. Folds whose selector dropped the feature are left out, so compare with `feature_frequency()`.
  - `summarise_feature_effects(curves)`: mean, SD, min and max across folds at each grid value, plus `n_folds`.
  - `plot_feature_effect(model_or_result, X=None, features=None, ...)`: one panel per feature with a rug of the data. For a CV result it draws the mean curve with an SD or min-max band (and optionally each fold's curve), and the title says in how many folds the feature was used.
  - `ale_grid(values, grid_size)`: the interval edges used.
- `humanise_feature_label("kurtosis_smth_AU12_r")` gives `"AU12 - Lip Corner Puller — kurtosis"`.

- `facedyn.__version__`. The version is now defined once, in `src/facedyn/__init__.py`, and read from there by the build.
- Every public module defines `__all__`.
- README section "Input data" describing the expected frame-level shape. Docstrings point to it.
- `CONTRIBUTING.md`: setup, checks, coding conventions and the release steps.
- Development tooling, following the Scientific Python Development Guide:
  - ruff (lint and format) and mypy, configured in `pyproject.toml` and run through pre-commit (`.pre-commit-config.yaml`).
  - Development dependencies are now dependency groups (`test`, `lint`, `dev`): `pip install --group dev -e .`. The `dev` extra is removed.
  - Stricter pytest configuration: `--strict-markers`, `--strict-config`, `xfail_strict`, and warnings raised as errors.
  - CI: a lint and type-check job, a build-and-`twine check` job, Python 3.14 in the test matrix, cancellation of superseded runs, an automatic import check over every submodule, and one "All checks passed" job to require. Dependabot keeps the action versions current.

### Changed

- `"svm_rbf"` is now `CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=5, ensemble=False)` instead of `SVC(probability=True)`, which scikit-learn 1.9 deprecates (removal in 1.11). Both do Platt scaling on 5-fold cross-validated decision values; the wrapper is used on every scikit-learn version so results do not depend on the installed one. SVM probabilities will differ slightly from 1.0.0. `predict` now always agrees with `predict_proba`. The SVC's own parameters move one level down: `estimator__C` in `make_classifier` overrides and `PAPER1_PARAMS`, `classifier__estimator__C` in `PAPER1_PARAM_GRIDS`. The SVM is deterministic and ignores `random_state`.
- `classification_metrics` and `evaluate_models` now raise if `positive_label` is given with more than two classes. Previously they quietly computed one-vs-rest binary statistics. Use the per-class columns or `pairwise_contrasts` instead.
- `threshold_sweep` and `tune_selection_threshold` raise a clear error for `scoring="roc_auc"` with more than two classes, because DeLong is binary. The tuner raises before any Boruta fit. The error points to `scoring="roc_auc_ovr"`.
- `plot_probability_distributions` and `plot_decision_boundary` label the negative group "rest" when it contains more than one class.
- `cross_validate_grouped`'s docstring now says it is for feature tables nothing has been fitted to, and points to `cross_validate_pipeline` otherwise.
- The missing-`groups` messages from `RepeatedStratifiedGroupKFold` and `cross_validate_grouped` now point to `groups_from_column` as well as `pair_groups`.
- General-purpose framing. facedyn is now described as a tool for classifying AU time series in general. Deepfake detection is one documented application; it is no longer the package's scope.
  - `pyproject.toml`: new description ("Interpretable, leakage-aware classification of facial action unit time series."), plus keywords, trove classifiers and `project.urls` (Homepage, Documentation, Issues, Changelog).
  - README: rewritten around repeated-measures and matched-pairs designs. The emotion and deepfake studies are listed as two applications. The quickstart uses participant grouping, `cross_validate_pipeline`, the multiclass plots and ALE.
  - The package has a module docstring. It says once that "the original R analysis" and "the deepfake study" refer to Murphy, Cook & Cuve (in prep).
  - Docstrings and messages no longer mention "Paper 1". They speak of classes or labels rather than real/fake, except for the matched-pairs helpers, whose real/fake defaults are kept and documented as the deepfake case.
  - `cross_validate_grouped`'s missing-`groups` warning is generic: it says splitting repeated measures or matched pairs can inflate or deflate a score. The `positive_label` and `tune_selection_threshold` errors no longer cite one study's choices.
  - `paired_train_test_split`'s broken-pairing error says "source video" instead of "real video".
  - No change to the API, defaults or behaviour. `make_classifier`'s default `preset="paper1"` stays as it is, so existing code still reproduces the published numbers.

### Fixed

- `plot_nmf_rank_cv(robust=True)`: the bold summary line is now clipped to the robust axis range like the individual replicates, and the y-axis starts at 0. Previously, when most replicates at a rank were degenerate (so the median was too), the unclipped line pushed the lower axis limit far below zero and flattened the real curve.
- Feature extraction now works on videos of different lengths. `reshape_to_wide` pads shorter videos with trailing NaNs up to the longest video, and every extractor then rejected them ("Series contains N NaN value(s)"), so `cross_validate_pipeline` failed on any dataset with unequal video lengths. `apply_rowwise`, which all extractors share, now drops trailing padding before computing features. A gap *inside* a series is still passed through and still rejected.
- Every transformer listed `BaseEstimator` before `TransformerMixin`. On scikit-learn 1.6 or later that made them report that they were not transformers (`__sklearn_tags__().transformer_tags` was `None`). They now list the mixin first, as scikit-learn requires.
- Lists that must be the same length are zipped with `strict=True`, so a mismatch raises instead of silently dropping items.
- Docstrings no longer refer to development notes that are not in the repository.

## [1.0.0]

Initial release. Covers the full pipeline from Murphy, Cook, & Cuve (in prep): smoothing, train/test splitting, z-score normalisation, NMF decomposition and rank selection, feature extraction and selection (including a Boruta implementation), classification, evaluation, and threshold tuning.

### Added

- `plot_nmf_face_maps(..., full_range=False)`: opt in with `full_range=True` to shade regions using the colormap's full `[0, 1]` range instead of py-feat's capped-at-67% quantised palette, so color intensity matches `plot_nmf_basis_heatmap` for the same underlying value. Default behaviour (py-feat-faithful rendering) is unchanged.
- `plot_nmf_reconstruction(..., au_label=None)` and `plot_nmf_reconstruction_extremes(..., humanise=False)`: caption plots with a humanised AU name (e.g. `"AU07 - Lid Tightener"`) instead of the raw column name, without affecting which column is plotted.

### Fixed

- `plot_nmf_face_maps(..., normalize=False)` raised `ValueError: alpha (...) is outside 0-1 range` whenever a raw (unnormalized) component value exceeded 1 — the alpha calculation assumed values were already bounded to `[0, 1]`, which only holds after normalization. Values are now clipped to `[0, 1]` before use, matching the behaviour the docstring already claimed ("saturate every region to the same colour").
- `plot_nmf_reconstruction_extremes`'s titles (AU name + "Original vs. Reconstructed" + video ID) overlapped into an unreadable jumble for longer video filenames. Titles now split across two lines and use `wrap=True` to further wrap anything still too long, and the default figure is wider (`(11, 4.5)` instead of `(9, 4)`).
