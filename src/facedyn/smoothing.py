"""Per-video temporal smoothing of Action Unit intensity signals."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RollingSmoother(BaseEstimator, TransformerMixin):
    """Left-aligned rolling mean, edge-extended, applied within each video.

    Replicates R's ``zoo::rollmean(x, k, fill = "extend", align = "left")``
    applied per-group via ``dplyr::group_by``. pandas has no native
    left-aligned rolling window, so this reverses the series, applies a
    standard (right-aligned) rolling mean, and reverses back; edge frames
    that fall short of a full window are filled by carrying the nearest
    valid value outward ("extend").

    Parameters
    ----------
    window : int, default 4
        Number of frames in the rolling window.
    group_col : str, default "video_filename"
        Column identifying which rows belong to the same video; smoothing
        is applied independently within each group.
    columns : list of str, optional
        Explicit columns to smooth. If not given, columns are selected via
        ``column_pattern``.
    column_pattern : str, default r"_r$"
        Regex used to select columns to smooth when ``columns`` is not
        given. The default matches OpenFace AU intensity columns (e.g.
        ``AU01_r``).
    prefix : str, default "smth_"
        Prefix used for the new smoothed columns.
    """

    def __init__(
        self,
        window: int = 4,
        group_col: str = "video_filename",
        columns: list[str] | None = None,
        column_pattern: str = r"_r$",
        prefix: str = "smth_",
    ):
        self.window = window
        self.group_col = group_col
        self.columns = columns
        self.column_pattern = column_pattern
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y=None) -> "RollingSmoother":
        self.columns_ = self._resolve_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        columns = getattr(self, "columns_", None) or self._resolve_columns(X)
        grouped = X.groupby(self.group_col)
        result = X.copy()
        for col in columns:
            result[f"{self.prefix}{col}"] = grouped[col].transform(
                self._rolling_mean_left_extend
            )
        return result

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return list(self.columns)
        pattern = re.compile(self.column_pattern)
        return [col for col in X.columns if pattern.search(col)]

    def _rolling_mean_left_extend(self, series: pd.Series) -> pd.Series:
        reversed_series = series[::-1]
        rolled = reversed_series.rolling(
            window=self.window, min_periods=self.window
        ).mean()[::-1]
        return rolled.ffill().bfill()
