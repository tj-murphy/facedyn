"""Reshaping long per-frame data into the wide shape the extractors consume.

The wide shape is one row per (video, series), with values in ``fr_1..fr_N``
columns and per-video metadata alongside. Every extractor in this subpackage
takes it, so one reshaped frame works with all of them.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _resolve_id_vars(df: pd.DataFrame, candidates: list[str], group_col: str) -> list[str]:
    """Auto-detect metadata columns that are constant within each group.

    Shared by `reshape_to_wide` and `pivot_features_wide`: both carry
    per-group metadata through a pivot, and both drop (with a warning) any
    candidate column that actually varies within a group, since including
    one would fragment the pivot into more than one row per group.
    """
    constant_per_group = df.groupby(group_col, sort=False)[candidates].nunique(dropna=False).max() <= 1
    id_vars = constant_per_group.index[constant_per_group].tolist()
    dropped = [c for c in candidates if c not in id_vars]
    if dropped:
        warnings.warn(
            f"Dropping columns that vary within a {group_col!r} group (not "
            f"safe to carry through as per-group metadata): {dropped}. Pass "
            f"`id_vars` explicitly to control this.",
            stacklevel=3,
        )
    return id_vars


def reshape_to_wide(
    df: pd.DataFrame,
    value_cols: list[str],
    id_vars: list[str] | None = None,
    group_col: str = "video_filename",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """Pivot long per-frame data to one row per video and series.

    Takes :class:`~facedyn.representative_aus.RepresentativeAUSelector`'s
    output directly.

    Parameters
    ----------
    df : pd.DataFrame
        One row per (video, frame), with one column per series.
    value_cols : list of str
        Columns to treat as separate series. Each becomes its own row per
        group, labelled in a new ``"series"`` column.
    id_vars : list of str, optional
        Metadata columns to carry through. Defaults to those taking exactly
        one value within every group. Columns that vary within a video are
        dropped with a warning, since including one fragments the pivot into
        one row per frame.
    group_col : str, default "video_filename"
        Column identifying which rows belong to which video.
    frame_col : str, default "frame"
        Column giving each row's frame index within its video.

    Returns
    -------
    pd.DataFrame
        One row per (group, series). Columns are `id_vars`, `group_col`,
        ``"series"``, then ``fr_1, fr_2, ...`` in ascending frame order.
    """
    if id_vars is None:
        candidates = [c for c in df.columns if c not in value_cols and c not in (group_col, frame_col)]
        id_vars = _resolve_id_vars(df, candidates, group_col)
    keys = [group_col, *id_vars]

    long = df.melt(
        id_vars=[*keys, frame_col], value_vars=value_cols, var_name="series", value_name="activation"
    )
    wide = long.pivot(index=[*keys, "series"], columns=frame_col, values="activation")
    wide = wide.reindex(sorted(wide.columns), axis=1)
    wide.columns = [f"fr_{c}" for c in wide.columns]
    return wide.reset_index()


def split_wide(
    wide_df: pd.DataFrame, frame_pattern: str = r"^fr_"
) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a wide frame into its metadata columns and its value matrix.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Typically :func:`reshape_to_wide`'s output.
    frame_pattern : str, default r"^fr_"
        Regex identifying the frame-value columns.

    Returns
    -------
    metadata : pd.DataFrame
        The non-frame columns, index reset.
    values : np.ndarray
        ``(n_rows, n_frames)`` float array of the frame columns.
    """
    frame_cols = [c for c in wide_df.columns if re.search(frame_pattern, c)]
    if not frame_cols:
        raise ValueError(
            f"No frame-value columns matched {frame_pattern!r} in columns "
            f"{list(wide_df.columns)}. Pass `frame_pattern` to match your "
            f"own naming, or reshape with `reshape_to_wide` first."
        )
    metadata = wide_df.drop(columns=frame_cols).reset_index(drop=True)
    values = wide_df[frame_cols].to_numpy(dtype=float)
    return metadata, values


def apply_rowwise(
    wide_df: pd.DataFrame,
    fn,
    frame_pattern: str = r"^fr_",
    n_jobs: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """Apply a per-series feature function to every row of a wide frame.

    The shared execution path behind every extractor here, so all of them
    return the same shape from the same input.

    Parameters
    ----------
    wide_df : pd.DataFrame
        One row per series. Non-frame columns are carried through unchanged.
    fn : callable
        Maps a 1-D float array to a ``pd.Series`` or dict of named features.
    frame_pattern : str, default r"^fr_"
        Regex identifying the frame-value columns.
    n_jobs : int, optional
        Parallel worker processes. ``None`` or ``1`` is sequential, ``-1``
        uses all cores. Each worker's BLAS calls are pinned to one thread to
        avoid oversubscribing the machine.
    verbose : int, default 0
        Forwarded to ``joblib.Parallel``. Set to ``10`` to print progress.

    Returns
    -------
    pd.DataFrame
        The non-frame columns, plus one column per feature.
    """
    from threadpoolctl import threadpool_limits

    metadata, values = split_wide(wide_df, frame_pattern)

    with threadpool_limits(limits=1):
        rows = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(fn)(row) for row in values)

    features = pd.DataFrame(list(rows)).reset_index(drop=True)
    return pd.concat([metadata, features], axis=1)


def pivot_features_wide(
    df: pd.DataFrame,
    group_col: str = "video_filename",
    series_col: str = "series",
    feature_columns: list[str] | None = None,
    id_vars: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot one-row-per-(group, series) feature output to one row per group.

    Takes the shape every extractor in this subpackage returns (one row per
    video x series -- e.g. :class:`~facedyn.features.timeseries.TimeSeriesFeatureExtractor`
    or :class:`~facedyn.features.cleaning.FeatureCleaner`'s output) and
    produces the shape a cross-series learner needs instead: one row per
    video, with every series' features as separate, series-suffixed columns
    (e.g. ``mean_smth_AU01_r``, ``mean_smth_AU12_r``, ...). Mirrors the
    pivot the real R analysis does (``dta_cmfts_output_concat``,
    `final_analysis_NMF_check.Rmd`) immediately before Boruta feature
    selection -- see :mod:`facedyn.feature_selection`.

    Parameters
    ----------
    df : pd.DataFrame
        One row per (group, series).
    group_col : str, default "video_filename"
        Column identifying which rows belong to which group (video).
    series_col : str, default "series"
        Column identifying which series each row's features came from.
    feature_columns : list of str, optional
        Which columns to pivot. If not given, every numeric column (other
        than `group_col`/`series_col`) is treated as a feature column --
        pass this explicitly (e.g. ``extractor.get_feature_names_out()``)
        whenever a metadata column might also be numeric, since a numeric
        metadata column would otherwise be swept up and pivoted as if it
        were a real feature rather than checked against `id_vars`.
    id_vars : list of str, optional
        Metadata columns to carry through, one value per group. Defaults to
        those taking exactly one value within every group. Columns that
        vary within a group are dropped with a warning, since including one
        would fragment the pivot into more than one row per group.

    Returns
    -------
    pd.DataFrame
        One row per `group_col` value. Columns are `id_vars`, `group_col`,
        then one ``f"{feature}_{series}"`` column per (feature, series)
        pair.
    """
    if feature_columns is None:
        feature_columns = [
            c for c in df.columns
            if c not in (group_col, series_col) and pd.api.types.is_numeric_dtype(df[c])
        ]
    if id_vars is None:
        candidates = [
            c for c in df.columns if c not in feature_columns and c not in (group_col, series_col)
        ]
        id_vars = _resolve_id_vars(df, candidates, group_col)
    keys = [group_col, *id_vars]

    wide = df.pivot(index=keys, columns=series_col, values=feature_columns)
    wide.columns = [f"{feature}_{series}" for feature, series in wide.columns]
    return wide.reset_index()
