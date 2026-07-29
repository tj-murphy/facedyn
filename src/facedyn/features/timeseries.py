"""facedyn's own feature set. 31 features per series, in numpy and scipy.

Each feature is implemented from its published definition, cited at the
helper that computes it.

Features are computed on the series exactly as given, with no rescaling.
This differs from R's ``tsfeatures``, which z-scores every series first.
Callers have already made their normalisation decision upstream, so
scale-dependent features are in whatever units they supplied.

Undefined values come back as ``NaN``, never 0. A constant series has no
meaningful autocorrelation, skewness or entropy. Features that stay
well-defined there, such as ``mean`` and ``length``, still return a value.
Constancy is detected with a tolerance rather than ``std(x) == 0``, because
real constant AU series carry about 1e-30 of upstream floating-point noise.

For richer feature sets, or to reproduce prior CMFTS-based work exactly,
use the bridges in :mod:`facedyn.features.external`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from facedyn.features._utils import is_constant as _is_constant
from facedyn.features.reshape import apply_rowwise

FEATURE_NAMES: list[str] = [
    # distribution
    "mean", "sd", "min", "max", "range", "median", "iqr", "skewness", "kurtosis",
    # dynamics
    "acf1", "acf2", "acf10_sumsq", "pacf1", "pacf5_sumsq", "diff1_sd", "mean_abs_diff",
    # trend / stationarity
    "trend_slope", "trend_r2", "mean_crossings", "longest_run_above_mean",
    # complexity
    "sample_entropy", "permutation_entropy", "spectral_entropy",
    # activation
    "prop_active", "n_episodes", "mean_episode_len", "max_episode_len",
    "mean_onset_slope", "mean_offset_slope", "peak_rate",
    # meta
    "length",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "distribution": FEATURE_NAMES[0:9],
    "dynamics": FEATURE_NAMES[9:16],
    "trend": FEATURE_NAMES[16:20],
    "complexity": FEATURE_NAMES[20:23],
    "activation": FEATURE_NAMES[23:30],
    "meta": FEATURE_NAMES[30:31],
}


# --------------------------------------------------------------------------
# distribution
# --------------------------------------------------------------------------


def _distribution(x: np.ndarray) -> dict[str, float]:
    """Location, spread and shape.

    ``skewness`` and ``kurtosis`` use scipy's defaults: the biased
    method-of-moments estimators, with kurtosis in excess form so a normal
    distribution gives 0. Other packages use sample-corrected forms, so
    check the convention before comparing values across tools.
    """
    constant = _is_constant(x)
    q75, q25 = np.percentile(x, [75, 25])
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
        "median": float(np.median(x)),
        "iqr": float(q75 - q25),
        # undefined for a constant series: both are 0/0
        "skewness": np.nan if constant else float(stats.skew(x)),
        "kurtosis": np.nan if constant else float(stats.kurtosis(x)),
    }


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------


def _acf(x: np.ndarray, lag_max: int) -> np.ndarray:
    """Sample autocorrelation at lags ``1..lag_max``.

    Standard biased estimator (Box, Jenkins & Reinsel, eq. 2.1.12). Both
    numerator and denominator divide by `n`. Lags beyond ``n - 1`` are NaN.
    """
    n = len(x)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    out = np.full(lag_max, np.nan)
    for k in range(1, min(lag_max, n - 1) + 1):
        out[k - 1] = float(np.dot(xc[k:], xc[:-k])) / denom
    return out


def _pacf(x: np.ndarray, lag_max: int) -> np.ndarray:
    """Sample partial autocorrelation at lags ``1..lag_max``.

    Uses the Levinson-Durbin recursion on the sample ACF (Brockwell &
    Davis, section 5.2). The lag-`k` value is the correlation between
    ``x[t]`` and ``x[t-k]`` with the intervening lags partialled out.
    """
    r = _acf(x, lag_max)
    out = np.full(lag_max, np.nan)
    phi = np.zeros(lag_max)
    v = 1.0
    for k in range(1, lag_max + 1):
        if not np.isfinite(r[k - 1]):
            break
        num = r[k - 1] - sum(phi[j - 1] * r[k - j - 1] for j in range(1, k))
        if v <= 0:
            break
        phi_kk = num / v
        out[k - 1] = phi_kk
        prev = phi[: k - 1].copy()
        for j in range(1, k):
            phi[j - 1] = prev[j - 1] - phi_kk * prev[k - j - 1]
        phi[k - 1] = phi_kk
        v *= 1.0 - phi_kk**2
    return out


def _dynamics(x: np.ndarray) -> dict[str, float]:
    """Serial dependence and step-to-step volatility.

    ``acf10_sumsq`` and ``pacf5_sumsq`` sum squared coefficients over lags
    1 to 10 and 1 to 5. They summarise total dependence without regard to
    sign, rather than dependence at one chosen lag.
    """
    if _is_constant(x):
        # every quantity here is a ratio with a zero denominator, or a
        # correlation of a constant with itself
        return dict.fromkeys(
            ["acf1", "acf2", "acf10_sumsq", "pacf1", "pacf5_sumsq", "diff1_sd", "mean_abs_diff"],
            np.nan,
        )
    acf = _acf(x, 10)
    pacf = _pacf(x, 5)
    d1 = np.diff(x)
    return {
        "acf1": float(acf[0]),
        "acf2": float(acf[1]),
        "acf10_sumsq": float(np.nansum(acf**2)) if np.isfinite(acf).any() else np.nan,
        "pacf1": float(pacf[0]),
        "pacf5_sumsq": float(np.nansum(pacf**2)) if np.isfinite(pacf).any() else np.nan,
        "diff1_sd": float(np.std(d1, ddof=1)) if len(d1) > 1 else np.nan,
        "mean_abs_diff": float(np.mean(np.abs(d1))) if len(d1) else np.nan,
    }


# --------------------------------------------------------------------------
# trend / stationarity
# --------------------------------------------------------------------------


def _trend(x: np.ndarray) -> dict[str, float]:
    """Linear trend and mean-reversion behaviour.

    ``trend_slope`` and ``trend_r2`` come from an ordinary least-squares fit
    against the frame index. ``mean_crossings`` is how often the series
    crosses its own mean, as a rate. ``longest_run_above_mean`` is the
    longest unbroken stretch above the mean, as a proportion of length.
    """
    n = len(x)
    keys = ["trend_slope", "trend_r2", "mean_crossings", "longest_run_above_mean"]
    if n < 2 or _is_constant(x):
        # a flat line has zero slope but no meaningful R^2, and never
        # crosses its own mean
        return dict.fromkeys(keys, np.nan)

    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    resid = x - (slope * t + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    above = x > x.mean()
    crossings = int(np.sum(above[1:] != above[:-1]))

    longest = best = 0
    for flag in above:
        best = best + 1 if flag else 0
        longest = max(longest, best)

    return {
        "trend_slope": float(slope),
        "trend_r2": float(r2),
        "mean_crossings": crossings / n,
        "longest_run_above_mean": longest / n,
    }


# --------------------------------------------------------------------------
# complexity
# --------------------------------------------------------------------------


def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Sample entropy, ``-log(A/B)`` (Richman & Moorman, 2000).

    `B` counts pairs of length-`m` subsequences within Chebyshev distance
    `r`, and `A` the same for length ``m+1``. Self-matches are excluded.
    `r` defaults to ``0.2 * sd``, the convention from the original paper.
    Returns NaN when either count is zero.
    """
    n = len(x)
    if n < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * float(np.std(x, ddof=1))
    if r <= 0:
        return np.nan

    def _count(k: int) -> int:
        # (n - k + 1, k) matrix of overlapping length-k windows
        windows = np.lib.stride_tricks.sliding_window_view(x, k)
        total = 0
        for i in range(len(windows) - 1):
            dist = np.max(np.abs(windows[i + 1 :] - windows[i]), axis=1)
            total += int(np.sum(dist <= r))
        return total

    b = _count(m)
    a = _count(m + 1)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def _permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalised permutation entropy (Bandt & Pompe, 2002).

    Each length-`order` window is replaced by the rank pattern of its
    values. The Shannon entropy of those patterns is divided by
    ``log(order!)``, giving 0 for a monotone series and 1 when all
    orderings are equally likely.

    This is a working implementation. R's ``tsExpKit`` version returns NA
    where ``combinat`` is not loadable, so values will not match feature
    tables built through that chain.
    """
    n = len(x)
    span = (order - 1) * delay
    if n <= span or _is_constant(x):
        return np.nan
    windows = np.lib.stride_tricks.sliding_window_view(x, span + 1)[:, ::delay]
    # argsort of argsort gives each window's rank pattern
    patterns = np.argsort(np.argsort(windows, axis=1, kind="stable"), axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    from math import factorial

    return float(-np.sum(p * np.log(p)) / np.log(factorial(order)))


def _spectral_entropy(x: np.ndarray) -> float:
    """Normalised spectral entropy.

    Shannon entropy of the power spectral density treated as a distribution
    over frequency, divided by ``log(n_freqs)``. Low values mean power is
    concentrated in a few frequencies. Values near 1 mean it is spread
    evenly. The zero-frequency bin is dropped so the result does not depend
    on the series' mean level.
    """
    if len(x) < 4 or _is_constant(x):
        return np.nan
    _, psd = signal.periodogram(x, detrend="constant")
    psd = psd[1:]
    total = psd.sum()
    if total <= 0 or len(psd) < 2:
        return np.nan
    p = psd / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(psd)))


def _complexity(x: np.ndarray) -> dict[str, float]:
    return {
        "sample_entropy": _sample_entropy(x),
        "permutation_entropy": _permutation_entropy(x),
        "spectral_entropy": _spectral_entropy(x),
    }


# --------------------------------------------------------------------------
# activation
# --------------------------------------------------------------------------


def _resolve_threshold(x: np.ndarray, activation_threshold) -> float:
    if isinstance(activation_threshold, str):
        if activation_threshold == "mean":
            return float(np.mean(x))
        if activation_threshold == "median":
            return float(np.median(x))
        raise ValueError(
            f"activation_threshold must be 'mean', 'median' or a number, "
            f"got {activation_threshold!r}"
        )
    return float(activation_threshold)


def _activation(x: np.ndarray, activation_threshold="mean") -> dict[str, float]:
    """How often, how long and how sharply the AU is engaged.

    An episode is a maximal run of consecutive frames above
    `activation_threshold`. Durations are proportions of series length, so
    they compare across videos of different duration. Onset and offset
    slopes are the per-frame change entering and leaving each episode.

    The threshold defaults to the series' own mean rather than a fixed cut.
    AU values reaching this module have usually been z-scored and shifted
    upstream, so they are no longer on OpenFace's 0-5 scale. Pass a float
    for an absolute cut on raw intensities.
    """
    n = len(x)
    keys = [
        "prop_active", "n_episodes", "mean_episode_len", "max_episode_len",
        "mean_onset_slope", "mean_offset_slope", "peak_rate",
    ]
    if n < 2:
        return dict.fromkeys(keys, np.nan)

    threshold = _resolve_threshold(x, activation_threshold)
    active = x > threshold

    # episode boundaries from the padded transition points
    padded = np.concatenate(([False], active, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])  # exclusive
    lengths = ends - starts

    onsets = [x[s] - x[s - 1] for s in starts if s > 0]
    offsets = [x[e] - x[e - 1] for e in ends if e < n]

    peaks, _ = signal.find_peaks(x)

    return {
        "prop_active": float(np.mean(active)),
        "n_episodes": float(len(starts)),
        "mean_episode_len": float(np.mean(lengths)) / n if len(lengths) else np.nan,
        "max_episode_len": float(np.max(lengths)) / n if len(lengths) else np.nan,
        "mean_onset_slope": float(np.mean(onsets)) if onsets else np.nan,
        "mean_offset_slope": float(np.mean(offsets)) if offsets else np.nan,
        "peak_rate": len(peaks) / n,
    }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def extract_features(series, *, activation_threshold="mean") -> pd.Series:
    """All 31 features for one time series.

    Parameters
    ----------
    series : array-like of float
        A single time series, such as one row of ``fr_1..fr_N``.
    activation_threshold : {"mean", "median"} or float, default "mean"
        Cut above which a frame counts as active.

    Returns
    -------
    pd.Series
        31 entries indexed by feature name, in :data:`FEATURE_NAMES` order.
        Undefined features are ``NaN``.

    Raises
    ------
    ValueError
        If `series` contains NaN.
    """
    x = np.asarray(series, dtype=float)
    if np.isnan(x).any():
        raise ValueError(
            f"Series contains {int(np.isnan(x).sum())} NaN value(s). Impute or "
            f"drop them before extracting features. Silently ignoring them "
            f"would make length-dependent features incomparable across series."
        )

    features: dict[str, float] = {}
    features.update(_distribution(x))
    features.update(_dynamics(x))
    features.update(_trend(x))
    features.update(_complexity(x))
    features.update(_activation(x, activation_threshold))
    features["length"] = float(len(x))
    return pd.Series({name: features[name] for name in FEATURE_NAMES})


def extract_timeseries_features(
    wide_df: pd.DataFrame,
    frame_pattern: str = r"^fr_",
    activation_threshold="mean",
    n_jobs: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """Features for every row of a wide DataFrame.

    Parameters
    ----------
    wide_df : pd.DataFrame
        One row per series. Non-frame columns are carried through unchanged.
    frame_pattern : str, default r"^fr_"
        Regex identifying the frame-value columns.
    activation_threshold : {"mean", "median"} or float, default "mean"
        Forwarded to :func:`extract_features`.
    n_jobs : int, optional
        Parallel worker processes. ``None`` or ``1`` is sequential, ``-1``
        uses all cores.
    verbose : int, default 0
        Forwarded to ``joblib.Parallel``.

    Returns
    -------
    pd.DataFrame
        The non-frame columns, plus the 31 feature columns.
    """
    from functools import partial

    return apply_rowwise(
        wide_df,
        partial(extract_features, activation_threshold=activation_threshold),
        frame_pattern=frame_pattern,
        n_jobs=n_jobs,
        verbose=verbose,
    )


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """scikit-learn transformer wrapping :func:`extract_timeseries_features`.

    Nothing is learned from the training set, since every feature is
    computed within a single series. `fit` only records the feature names.

    Parameters
    ----------
    frame_pattern : str, default r"^fr_"
        Regex identifying the frame-value columns.
    activation_threshold : {"mean", "median"} or float, default "mean"
        Cut above which a frame counts as active.
    n_jobs : int, optional
        Parallel worker processes. ``None`` or ``1`` is sequential, ``-1``
        uses all cores.
    verbose : int, default 0
        Forwarded to ``joblib.Parallel``.

    Examples
    --------
    >>> wide = reshape_to_wide(long_df, value_cols=selector.selected_columns_)
    >>> features = TimeSeriesFeatureExtractor(n_jobs=-1).fit_transform(wide)
    """

    def __init__(
        self,
        frame_pattern: str = r"^fr_",
        activation_threshold="mean",
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        self.frame_pattern = frame_pattern
        self.activation_threshold = activation_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None) -> "TimeSeriesFeatureExtractor":
        self.feature_names_ = list(FEATURE_NAMES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_")
        return extract_timeseries_features(
            X,
            frame_pattern=self.frame_pattern,
            activation_threshold=self.activation_threshold,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)
