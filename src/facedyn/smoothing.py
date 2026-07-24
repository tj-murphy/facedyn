"""Per-video temporal smoothing of Action Unit intensity signals."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from facedyn._plot_utils import save_figure


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


def plot_smoothing_comparison(
    df: pd.DataFrame,
    column: str,
    video_filename: str,
    smoothed_column: str | None = None,
    group_col: str = "video_filename",
    frame_col: str = "frame",
    mode: str = "stacked",
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot a raw AU column against its smoothed counterpart for one video.

    Requires matplotlib (``pip install facedyn[viz]``).

    Intended for picking a rolling ``window`` size: run
    :class:`RollingSmoother` with a candidate window, then use this to see
    how aggressively it flattens the signal for a given video/column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``column``, ``smoothed_column``, ``group_col`` and
        ``frame_col``. Typically the output of
        :meth:`RollingSmoother.transform`.
    column : str
        Raw column to plot, e.g. ``"AU07_r"``.
    video_filename : str
        Value of ``group_col`` identifying which video's rows to plot.
    smoothed_column : str, optional
        Smoothed column to plot. Defaults to ``f"{self.prefix}{column}"``
        using :class:`RollingSmoother`'s default prefix, i.e.
        ``f"smth_{column}"``.
    group_col : str, default "video_filename"
        Column identifying which rows belong to which video.
    frame_col : str, default "frame"
        Column used for the x-axis.
    mode : {"stacked", "overlay"}, default "stacked"
        ``"stacked"`` draws raw and smoothed on two separate, vertically
        stacked axes sharing an x-axis. ``"overlay"`` draws both as lines on
        a single axes. ``ax`` is ignored when ``mode="stacked"`` since two
        axes are required.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on when ``mode="overlay"``. A new figure/axes is
        created if not given.
    save_path : str or pathlib.Path, optional
        If given, save the figure to this filename (e.g.
        ``"smoothing.pdf"`` or ``"smoothing.png"``) -- format is inferred
        from the extension. Not saved if left as ``None`` (the default).
    output_dir : str or pathlib.Path, default "."
        Directory ``save_path`` is written into (created if it doesn't
        already exist). Ignored if ``save_path`` is None.
    dpi : int, default 300
        Resolution used when saving to a raster format (e.g. PNG); ignored
        for vector formats (e.g. PDF) and if ``save_path`` is None.

    Returns
    -------
    matplotlib.axes.Axes or list of matplotlib.axes.Axes
        A single Axes for ``mode="overlay"``, or ``[raw_ax, smoothed_ax]``
        for ``mode="stacked"``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_smoothing_comparison requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    if mode not in ("stacked", "overlay"):
        raise ValueError(f"mode must be 'stacked' or 'overlay', got {mode!r}")

    if smoothed_column is None:
        smoothed_column = f"smth_{column}"

    video_data = df[df[group_col] == video_filename].sort_values(frame_col)
    raw_color, smoothed_color = "#009E73", "#D55E00"

    if mode == "overlay":
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(
            video_data[frame_col], video_data[column],
            color=raw_color, alpha=0.6, linewidth=1, label="Raw",
        )
        ax.plot(
            video_data[frame_col], video_data[smoothed_column],
            color=smoothed_color, linewidth=1.5, label="Smoothed",
        )
        ax.set_xlabel(frame_col)
        ax.set_ylabel(column)
        ax.set_title(f"{column} - Raw vs. Smoothed ({video_filename})")
        ax.legend()
        save_figure(ax.figure, save_path, output_dir, dpi)
        return ax

    _, (raw_ax, smoothed_ax) = plt.subplots(2, 1, sharex=True, sharey=True)
    raw_ax.plot(video_data[frame_col], video_data[column], color=raw_color, linewidth=1)
    raw_ax.set_ylabel(column)
    raw_ax.set_title(f"Raw ({video_filename})")

    smoothed_ax.plot(
        video_data[frame_col], video_data[smoothed_column],
        color=smoothed_color, linewidth=1,
    )
    smoothed_ax.set_xlabel(frame_col)
    smoothed_ax.set_ylabel(column)
    smoothed_ax.set_title("Smoothed")

    save_figure(raw_ax.figure, save_path, output_dir, dpi)
    return [raw_ax, smoothed_ax]
