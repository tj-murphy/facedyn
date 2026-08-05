# Changelog

All notable changes to facedyn will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [1.0.0]

Initial release. Covers the full pipeline from Murphy, Cook, & Cuve (in prep): smoothing, train/test splitting, z-score normalisation, NMF decomposition and rank selection, feature extraction and selection (including a Boruta implementation), classification, evaluation, and threshold tuning.

### Added

- `plot_nmf_face_maps(..., full_range=False)`: opt in with `full_range=True` to shade regions using the colormap's full `[0, 1]` range instead of py-feat's capped-at-67% quantised palette, so color intensity matches `plot_nmf_basis_heatmap` for the same underlying value. Default behaviour (py-feat-faithful rendering) is unchanged.
- `plot_nmf_reconstruction(..., au_label=None)` and `plot_nmf_reconstruction_extremes(..., humanise=False)`: caption plots with a humanised AU name (e.g. `"AU07 - Lid Tightener"`) instead of the raw column name, without affecting which column is plotted.

### Fixed

- `plot_nmf_face_maps(..., normalize=False)` raised `ValueError: alpha (...) is outside 0-1 range` whenever a raw (unnormalized) component value exceeded 1 — the alpha calculation assumed values were already bounded to `[0, 1]`, which only holds after normalization. Values are now clipped to `[0, 1]` before use, matching the behaviour the docstring already claimed ("saturate every region to the same colour").
- `plot_nmf_reconstruction_extremes`'s titles (AU name + "Original vs. Reconstructed" + video ID) overlapped into an unreadable jumble for longer video filenames. Titles now split across two lines and use `wrap=True` to further wrap anything still too long, and the default figure is wider (`(11, 4.5)` instead of `(9, 4)`).


