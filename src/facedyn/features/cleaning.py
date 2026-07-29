"""Generic diagnostics and optional cleanup for feature-extractor output.

Works on the output of any of the three routes in this subpackage
(:class:`~facedyn.features.timeseries.TimeSeriesFeatureExtractor`,
:class:`~facedyn.features.external.RFeatureExtractor`,
:class:`~facedyn.features.external.CallableFeatureExtractor`), or any other
extractor returning the same shape, since it only looks at column values --
never at how they were produced.

Two independent pieces, so you can inspect without committing to any
automatic decision:

- :func:`feature_diagnostics`. A read-only report of NaN/Inf/near-zero-
  variance issues per column, to decide what to keep, drop or impute by
  hand.
- :class:`FeatureCleaner`. An opt-in transformer that acts on that same kind
  of information automatically, with Paper-1-like defaults, fully
  overridable.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.utils.validation import check_is_fitted

from facedyn._plot_utils import save_figure
from facedyn.features._utils import is_constant


def _resolve_feature_columns(X: pd.DataFrame, feature_columns: list[str] | None) -> list[str]:
    if feature_columns is not None:
        return list(feature_columns)
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def feature_diagnostics(
    X: pd.DataFrame, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Per-column report of NaN, Inf, and near-zero-variance issues.

    Read-only -- does not modify `X`. Safe to run on the output of any
    feature extractor, since it only looks at column values, not how they
    were produced. Inspect the result to decide what to keep, drop or
    impute -- manually, or by feeding the same decisions into
    :class:`FeatureCleaner` (its ``drop_columns``/`max_nan_fraction`).

    Parameters
    ----------
    X : pd.DataFrame
        Feature-extractor output (metadata columns plus feature columns).
    feature_columns : list of str, optional
        Which columns to report on. If not given, every numeric column in
        `X` is treated as a feature column -- pass this explicitly (e.g.
        ``extractor.get_feature_names_out()``) whenever a metadata column
        might also be numeric (an integer-coded label, say), since nothing
        in a feature extractor's output otherwise marks which columns are
        metadata and which are features.

    Returns
    -------
    pd.DataFrame
        One row per feature column: ``column``, ``n_nan``, ``pct_nan``,
        ``n_inf``, ``pct_inf``, ``sd``, ``is_constant``, ``all_nan``.
        ``n_nan``/``pct_nan`` count only genuine ``NaN`` -- Inf is reported
        separately, since nothing has converted it yet at this point.
        ``sd``/``is_constant`` are computed over the finite (non-NaN,
        non-Inf) values only.
    """
    columns = _resolve_feature_columns(X, feature_columns)
    n = len(X)
    rows = []
    for col in columns:
        values = X[col].to_numpy(dtype=float)
        nan_mask = np.isnan(values)
        inf_mask = np.isinf(values)
        finite = values[~nan_mask & ~inf_mask]
        n_nan = int(nan_mask.sum())
        n_inf = int(inf_mask.sum())
        rows.append(
            {
                "column": col,
                "n_nan": n_nan,
                "pct_nan": n_nan / n if n else np.nan,
                "n_inf": n_inf,
                "pct_inf": n_inf / n if n else np.nan,
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                "is_constant": is_constant(finite) if len(finite) else True,
                "all_nan": n_nan == n,
            }
        )
    return pd.DataFrame(rows)


def plot_feature_diagnostics(
    report: pd.DataFrame,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Stacked bar chart of missing/infinite values per feature column.

    Takes :func:`feature_diagnostics`'s output. Stacks each column's
    ``pct_nan`` and ``pct_inf`` so the worst-affected columns are obvious at
    a glance, sorted with the worst at the top.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    report : pd.DataFrame
        Output of :func:`feature_diagnostics`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    save_path : str or pathlib.Path, optional
        Filename to save the figure to. Format is inferred from the
        extension. Not saved if ``None``.
    output_dir : str or pathlib.Path, default "."
        Directory `save_path` is written into, created if needed.
    dpi : int, default 300
        Resolution for raster formats. Ignored for vector formats.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_feature_diagnostics requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    ordered = report.sort_values(["pct_nan", "pct_inf"], ascending=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.35 * len(ordered) + 1.5))

    ax.barh(ordered["column"], ordered["pct_nan"], color="#D55E00", label="NaN")
    ax.barh(
        ordered["column"], ordered["pct_inf"], left=ordered["pct_nan"],
        color="#0072B2", label="Inf",
    )
    ax.set_xlabel("Proportion of rows")
    ax.set_title("Feature diagnostics - missing / infinite values per column")
    ax.legend()
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Optional, opt-in cleanup of feature-extractor output.

    Generic over which extractor produced the features, since it only
    operates on already-extracted feature columns. Always safe to skip: run
    :func:`feature_diagnostics` on its own and decide what to do by hand
    instead, if you prefer.

    Chains four independently-toggleable steps, in this order (matching the
    real R analysis's order in ``final_analysis_NMF_check.Rmd`` ~L658-769):

    1. **Inf -> NaN**, unconditionally, in every feature column.
    2. **Drop unusable columns** -- the union of `drop_columns` (explicit,
       for anything identified from :func:`feature_diagnostics`) and any
       column whose fraction of missing values (after step 1) is at or
       above `max_nan_fraction`. The default, ``1.0``, only drops columns
       that are entirely missing -- a technical necessity, since such a
       column has no signal to impute from, not merely a style choice.
    3. **Impute** (optional, on by default) -- an
       ``sklearn.impute.IterativeImputer`` with a ``RandomForestRegressor``
       estimator, sklearn's own documented equivalent of R's ``missForest``.
       Pass ``impute=False`` to leave NaN in place instead, or your own
       fitted-or-unfitted sklearn transformer to use something else (e.g.
       ``SimpleImputer``). Unlike R, which reruns ``missForest`` from
       scratch on every dataset it touches (train, test, held-out, each
       independently -- not a reusable fitted model), this fits once on the
       training data and calls ``.transform()`` on new data, which
       ``IterativeImputer`` supports natively. Deliberate, not a silent
       behaviour change.
    4. **Drop near-zero-variance columns**, checked on the *imputed*
       training data (imputation can itself create a constant column) using
       the same floating-point-tolerant constant check used elsewhere in
       this package (see :func:`facedyn.features._utils.is_constant`), not a
       strict ``sd == 0``. If ``impute=False``, a column with any remaining
       NaN is never dropped here (missing values make it ineligible for
       this check, not mistakenly constant). The decision is made once in
       `fit` and reapplied in `transform`.

    R's own cleanup differs from this in two ways, both deliberate:

    - It drops CMFTS's ``permutation_entropy``/``shannon_entropy_CS``/``_SG``
      columns by **hardcoded position** (``[, -14:-16]``), specific to
      whatever columns CMFTS happened to produce in that one run. This drops
      by *how much data is missing*, which generalizes to any extractor's
      output. ``shannon_entropy_CS``/``_SG`` are majority-``Inf`` rather
      than fully missing, so replicating R's exact drop needs either a
      lower `max_nan_fraction` or listing them in `drop_columns` once
      :func:`feature_diagnostics` has flagged them -- not the default.
    - Its zero-SD decision is fit-on-train for the 20%-test-set but
      recomputed independently for the 40-held-out set -- an inconsistency
      in the source. This always reuses the fit-time decision, resolving it
      one way rather than reproducing the inconsistency.

    Parameters
    ----------
    feature_columns : list of str, optional
        Which columns to clean. If not given, every numeric column in the
        data passed to `fit` is treated as a feature column.
    drop_columns : list of str, optional
        Columns to drop unconditionally, in addition to any dropped by
        `max_nan_fraction`.
    max_nan_fraction : float, default 1.0
        Drop any feature column whose fraction of missing values (after
        Inf -> NaN) is at or above this. ``1.0`` drops only fully-missing
        columns.
    impute : bool or sklearn transformer, default True
        ``True`` imputes with ``IterativeImputer(RandomForestRegressor(...))``.
        ``False`` skips imputation, leaving NaN in place. Anything else is
        used directly as the imputer (must implement ``fit``/``transform``).
    drop_zero_sd : bool, default True
        Drop near-zero-variance columns, checked after imputation.
    random_state : int, optional
        Passed to the default imputer's ``RandomForestRegressor`` and to
        ``IterativeImputer`` itself, for reproducibility.

    Attributes
    ----------
    diagnostics_ : pd.DataFrame
        :func:`feature_diagnostics` computed on the data passed to `fit`,
        before any cleaning -- available even when using the automated path.
    dropped_columns_ : list of str
        Columns dropped in step 2 (missing-data threshold and/or explicit).
    zero_sd_columns_ : list of str
        Columns dropped in step 4.
    imputer_ : sklearn transformer or None
        The fitted imputer, or ``None`` if ``impute=False``.
    feature_names_out_ : list of str
        Feature columns retained after all cleaning steps.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        max_nan_fraction: float = 1.0,
        impute: bool | BaseEstimator = True,
        drop_zero_sd: bool = True,
        random_state: int | None = None,
    ):
        self.feature_columns = feature_columns
        self.drop_columns = drop_columns
        self.max_nan_fraction = max_nan_fraction
        self.impute = impute
        self.drop_zero_sd = drop_zero_sd
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureCleaner":
        self.feature_columns_ = _resolve_feature_columns(X, self.feature_columns)
        self.diagnostics_ = feature_diagnostics(X, self.feature_columns_)

        converted = self._inf_to_nan(X[self.feature_columns_])

        explicit = set(self.drop_columns or [])
        nan_fraction = converted.isna().mean()
        threshold_dropped = set(nan_fraction[nan_fraction >= self.max_nan_fraction].index)
        self.dropped_columns_ = sorted(explicit | threshold_dropped)
        if self.dropped_columns_:
            warnings.warn(
                f"Dropping {len(self.dropped_columns_)} feature column(s) with "
                f">= {self.max_nan_fraction:.0%} missing values (after "
                f"Inf -> NaN) and/or listed in `drop_columns`: "
                f"{self.dropped_columns_}.",
                stacklevel=2,
            )

        remaining = [c for c in self.feature_columns_ if c not in self.dropped_columns_]
        working = converted[remaining]

        if self.impute is False:
            self.imputer_ = None
            imputed = working
        else:
            self.imputer_ = self._build_imputer()
            imputed = pd.DataFrame(
                self.imputer_.fit_transform(working), columns=remaining, index=working.index
            )

        if self.drop_zero_sd:
            self.zero_sd_columns_ = sorted(
                c for c in remaining if is_constant(imputed[c].to_numpy())
            )
            if self.zero_sd_columns_:
                warnings.warn(
                    f"Dropping {len(self.zero_sd_columns_)} near-zero-variance "
                    f"feature column(s) (checked after imputation): "
                    f"{self.zero_sd_columns_}.",
                    stacklevel=2,
                )
        else:
            self.zero_sd_columns_ = []

        self.feature_names_out_ = [c for c in remaining if c not in self.zero_sd_columns_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        metadata = X.drop(columns=self.feature_columns_).reset_index(drop=True)

        remaining = [c for c in self.feature_columns_ if c not in self.dropped_columns_]
        converted = self._inf_to_nan(X[remaining]).reset_index(drop=True)

        if self.imputer_ is not None:
            imputed = pd.DataFrame(
                self.imputer_.transform(converted), columns=remaining, index=converted.index
            )
        else:
            imputed = converted

        cleaned = imputed[self.feature_names_out_]
        return pd.concat([metadata, cleaned], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _inf_to_nan(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.replace([np.inf, -np.inf], np.nan)

    def _build_imputer(self):
        if self.impute is True:
            return IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                max_iter=10,
                random_state=self.random_state,
            )
        return self.impute
