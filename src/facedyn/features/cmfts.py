"""CMFTS time-series complexity/statistical feature extraction.

Ports R's `cmfts::cmfts()` (github.com/fjbaldan/CMFTS), which computes 41
named features per time series: 10 measures CMFTS implements itself, and 31
delegated wholesale to the R `tsfeatures` package (Hyndman et al.), which
itself further delegates to `stats`, `urca`, `fracdiff`, `forecast`.

Read directly from the real R source of every package in that chain (not
guessed from documentation), and cross-checked empirically against a live
R installation with `cmfts` and all its dependencies present -- see
`PIPELINE.md` step 6 for the full writeup, including several non-obvious
findings that would otherwise silently produce wrong numbers:

- **`tsfeatures::tsfeatures()` z-scores every series by default**
  (`scale=TRUE`) before computing *any* of its 31 features -- CMFTS never
  overrides this. All 31 "tsfeatures-derived" measures below therefore
  operate on the z-scored series, not the raw one; only the 10
  CMFTS-native measures use the raw series. This one default parameter
  quietly affects every non-scale-invariant feature (e.g. `linearity`,
  `curvature`, `max_level_shift`) -- confirmed empirically: reproducing
  R's real output for `linearity`/`curvature` requires the z-scored
  series; the raw series gives a different, wrong-but-plausible number.
- `permutation_entropy` is **always NaN** in real R output -- not a
  legitimate NaN, but a genuine bug in `tsExpKit::permutationEntropy`
  (calls `permn()` without importing it from `combinat`), confirmed live.
  Reproduced here as an unconditional NaN rather than a "working"
  implementation that wouldn't match real R output.
- `shannon_entropy_CS` (Chao-Shen entropy) is applied directly to the raw
  continuous series as if it were a vector of bin counts -- no
  discretization step exists anywhere in the real call chain. This is
  numerically pathological (confirmed: `Inf` in 824/1110 rows of the real
  training-set fixture) but is what R actually computes, so it is
  reproduced as-is, not "fixed".

Two features have no closed-form R algorithm to transcribe (both
implemented in Fortran, via numerical optimization/smoothing rather than a
formula): `hurst` (exact Gaussian MLE of an ARFIMA(0,d,0) model, via
`fracdiff`) and `stl_features`'s `trend` (Friedman's variable-span
`supsmu` smoother, via `forecast::mstl`) -- see `_hurst`/`_stl_features`
below for the approximation strategy used for each and how closely it
was validated to match real R output.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def reshape_for_cmfts(
    df: pd.DataFrame,
    value_cols: list[str],
    id_vars: list[str] | None = None,
    group_col: str = "video_filename",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """Long-format (one row per video-frame) to CMFTS's wide input shape
    (one row per video x series, columns ``fr_1..fr_N``).

    Replicates `final_analysis_NMF_check.Rmd`'s
    ``pivot_longer(value_cols) %>% pivot_wider(frame_col, names_prefix="fr_")``.
    Directly consumes :class:`~facedyn.representative_aus.RepresentativeAUSelector`'s
    output (long format, one column per representative AU) -- that's the
    intended composition, not a coincidence.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data: one row per (video, frame), with one column per
        series to extract features from (e.g. each representative AU).
    value_cols : list of str
        Columns in `df` to treat as separate series (e.g. the representative
        AU columns). Each becomes its own row per group in the output,
        identified by a new ``"series"`` column holding the original column
        name.
    id_vars : list of str, optional
        Columns identifying a group/row beyond `group_col` (e.g.
        ``isfakeorreal``, ``emotion``). Defaults to every column in `df`
        that isn't `group_col`, `frame_col`, or in `value_cols` **and takes
        exactly one value within every `group_col` group** (e.g.
        ``isfakeorreal``, safe) -- columns that vary *within* a video (raw
        per-frame AU columns, ``timestamp``, ``confidence``, ...) are
        dropped from the output rather than included, with a warning.
        Including a frame-varying column in `id_vars` would silently
        fragment the pivot into one row per (video, frame, series) instead
        of (video, series) -- confirmed to actually happen and produce a
        catastrophically wrong (200x too many rows, ~99.6% NaN) output
        when this wasn't guarded against, since
        :meth:`RepresentativeAUSelector.transform`'s output (this
        function's intended input) passes through *all* non-factorized
        columns, not just genuinely per-video-constant metadata.
    group_col : str, default "video_filename"
        Column identifying which rows belong to which video.
    frame_col : str, default "frame"
        Column giving each row's frame/time index within its video.

    Returns
    -------
    pd.DataFrame
        One row per (group, series) pair, columns: `id_vars`, `group_col`,
        ``"series"`` (the value_cols entry that row came from), then
        ``fr_1, fr_2, ...`` in ascending frame order.
    """
    if id_vars is None:
        candidates = [c for c in df.columns if c not in value_cols and c not in (group_col, frame_col)]
        constant_per_group = df.groupby(group_col, sort=False)[candidates].nunique(dropna=False).max() <= 1
        id_vars = constant_per_group.index[constant_per_group].tolist()
        dropped = [c for c in candidates if c not in id_vars]
        if dropped:
            warnings.warn(
                f"Dropping columns that vary within a {group_col!r} group (not "
                f"safe to carry through as per-video metadata): {dropped}. Pass "
                f"`id_vars` explicitly to control this.",
                stacklevel=2,
            )
    keys = [group_col, *id_vars]

    long = df.melt(
        id_vars=[*keys, frame_col], value_vars=value_cols, var_name="series", value_name="activation"
    )
    wide = long.pivot(index=[*keys, "series"], columns=frame_col, values="activation")
    wide = wide.reindex(sorted(wide.columns), axis=1)
    wide.columns = [f"fr_{c}" for c in wide.columns]
    return wide.reset_index()


_EPS = 1e-10


def _kolmogorov_complexity(x: np.ndarray) -> float:
    """LZ76 parsing complexity of `x` binarized around its own mean.
    Ports `measure.kolmogorov` (`CMFTS/R/lempel_ziv.R`) exactly, including
    its specific (Kaspar-Schuster-style) parsing loop."""
    threshold = x.mean()
    s = (x >= threshold).astype(np.int64)
    n = len(s)
    c = l = k = kmax = 1
    i = 0
    end = False
    while not end:
        if s[i + k - 1] != s[l + k - 1]:
            if k > kmax:
                kmax = k
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    end = True
                else:
                    i = 0
                    k = kmax = 1
            else:
                k = 1
        else:
            k += 1
            if l + k > n:
                c += 1
                end = True
    return float(c)


def _lempel_ziv(x: np.ndarray) -> float:
    """`measure.lempel_ziv`: Kolmogorov complexity normalized by n/log2(n)."""
    n = len(x)
    b = n / np.log2(n)
    return _kolmogorov_complexity(x) / b


def _approximate_entropy(x: np.ndarray, edim: int = 2, r: float | None = None) -> float:
    """`measure.aproximation_entropy` -> `pracma::approx_entropy`: classic
    ApEn (Pincus 1991), Chebyshev distance, self-match included."""
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    n = len(x)
    result = np.zeros(2)
    for j, m in enumerate((edim, edim + 1)):
        n_vec = n - m + 1
        data_mat = np.array([x[i : n - m + i + 1] for i in range(m)])  # (m, n_vec)
        phi = np.empty(n_vec)
        for i in range(n_vec):
            dist = np.max(np.abs(data_mat - data_mat[:, [i]]), axis=0)
            phi[i] = np.sum(dist <= r) / n_vec
        result[j] = np.sum(np.log(phi)) / n_vec
    return float(result[0] - result[1])


def _sample_entropy_cmfts(y: np.ndarray, M: int = 2, r: float | None = None) -> float:
    """`measure.sample_entropy` (CMFTS's own hand-rolled run-length version,
    not `pracma::sample_entropy` -- CMFTS calls it with `package=""`,
    selecting this branch). Direct port of the R loop."""
    if r is None:
        r = 0.2 * np.std(y, ddof=1)
    n = len(y)
    lastrun = np.zeros(n)
    run = np.zeros(n)
    A = np.zeros(M)
    B = np.zeros(M)
    for i in range(n - 1):
        nj = n - i - 1
        y1 = y[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(y[j] - y1) < r:
                run[jj] = lastrun[jj] + 1
                m1 = min(M, int(run[jj]))
                for m in range(1, m1 + 1):
                    A[m - 1] += 1
                    if j < n - 1:
                        B[m - 1] += 1
            else:
                run[jj] = 0
        lastrun[:nj] = run[:nj]
    p = [A[0] / (n * (n - 1) / 2)]
    for m in range(2, M + 1):
        if B[m - 2] == 0:
            continue
        p1 = A[m - 1] / B[m - 2]
        if p1 != 0 and not np.isnan(p1):
            p.append(p1)
    p = np.array(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(-p * np.log(p)))


def _shannon_entropy_cs(y: np.ndarray) -> float:
    """`measure.shannonEntropy(y, "CS")` -> `entropy::entropy.ChaoShen`.
    Applied to the raw vector directly as if it were bin counts -- no
    discretization anywhere in the real call chain (see module docstring).
    """
    yx = y[y > 0]
    n = yx.sum()
    p = yx / n
    f1 = np.sum(yx == 1)
    if f1 == n:
        f1 = n - 1
    C = 1 - f1 / n
    pa = C * p
    la = 1 - (1 - pa) ** n
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(-pa * np.log(pa) / la))


def _shannon_entropy_sg(y: np.ndarray) -> float:
    """`measure.shannonEntropy(y, "SG")` -> `entropy::entropy.Dirichlet(y,
    a=1/length(y))` -- the Schurmann-Grassberger estimator (NOT shrinkage,
    despite the method name's mnemonic similarity), a Dirichlet pseudocount
    of `1/length(y)` added to every raw value, then plug-in entropy."""
    a = 1.0 / len(y)
    p = (y + a) / (y.sum() + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.sum(p * np.log(p)))


def _spectral_entropy_fft(y: np.ndarray) -> float:
    """`measure.spectral_entropy`: Shannon entropy of the normalized FFT
    magnitude spectrum (full complex spectrum, not one-sided)."""
    yf = np.fft.fft(y)
    mag = np.abs(yf)
    mag = mag / mag.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(mag > 0, mag * np.log(1.0 / mag), 0.0)
    return float(np.sum(terms))


def _ordinal_pattern_code(xvec: np.ndarray, ndemb: int) -> int:
    """Keller-coding inversion-table encoding of one length-`ndemb` window
    into its ordinal-pattern index `[0, ndemb!)`. Direct port of
    `statcomp`'s C `ordinal_pattern_loop` (`src/ordinal_patterns.c`),
    including its `1e-10` near-tie tolerance -- needed for exact agreement
    on real (occasionally near-repeated) AU signal values; a plain
    stable-sort argsort encoding measurably disagrees on ~1% of windows in
    real data, propagating into a visibly wrong `nforbiden` value."""
    ipa = np.zeros((ndemb, ndemb), dtype=np.int64)
    for ilag in range(1, ndemb):
        for itime in range(ilag, ndemb):
            ipa[itime, ilag] = ipa[itime - 1, ilag - 1]
            if xvec[itime] <= xvec[itime - ilag] or abs(xvec[itime - ilag] - xvec[itime]) < 1e-10:
                ipa[itime, ilag] += 1
    nd = ipa[ndemb - 1, 1]
    for ilag in range(2, ndemb):
        nd = (ilag + 1) * nd + ipa[ndemb - 1, ilag]
    return int(nd)


def _nforbidden(x: np.ndarray, ndemb: int = 6) -> float:
    """`measure.nforbiden(x, 6)` -> `statcomp::global_complexity(x, ndemb)[3]`:
    fraction of the `ndemb!` possible ordinal patterns that never occur."""
    from math import factorial

    n = len(x)
    total = factorial(ndemb)
    counts = np.zeros(total, dtype=np.int64)
    for j in range(n - ndemb + 1):
        counts[_ordinal_pattern_code(x[j : j + ndemb], ndemb)] += 1
    return float(np.sum(counts == 0) / total)


def _kurtosis_e1071(x: np.ndarray) -> float:
    """`measure.kurtosis` -> `e1071::kurtosis(x)`, default `type=3`."""
    n = len(x)
    xc = x - x.mean()
    m2 = np.mean(xc**2)
    m4 = np.mean(xc**4)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(m4 / m2**2 * (1 - 1 / n) ** 2 - 3)


def _skewness_e1071(x: np.ndarray) -> float:
    """`measure.skewness` -> `e1071::skewness(x)`, default `type=3`."""
    n = len(x)
    xc = x - x.mean()
    m2 = np.mean(xc**2)
    m3 = np.mean(xc**3)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(m3 / m2**1.5 * (1 - 1 / n) ** 1.5)


def _is_constant(x: np.ndarray) -> bool:
    """`forecast::is.constant`: `isTRUE(all.equal(x, rep(x[1], length(x))))`
    -- a *tolerance*-based check (R's `all.equal` default tolerance,
    `sqrt(.Machine$double.eps) ~= 1.49e-8`), not exact equality. Matters
    in practice: real (post-normalisation) "constant" AU series aren't
    bit-identical across frames -- floating-point noise from upstream
    smoothing/shifting leaves a std of ~1e-30, not exactly 0 -- so a plain
    `np.std(x) == 0` check misses them, silently sending garbage through
    `_scalets`'s division instead of leaving the series untouched."""
    scale = np.abs(x).mean()
    tol = 1.49e-8 * scale if scale > 1.49e-8 else 1.49e-8
    return bool(np.mean(np.abs(x - x[0])) < tol)


def _scalets(x: np.ndarray) -> np.ndarray:
    """`tsfeatures:::scalets`: z-score (`sd` with `n-1` divisor), unchanged
    if `x` is constant (`_is_constant`). **Critical, easy-to-miss detail**:
    `tsfeatures::tsfeatures()` applies this to every series by default
    (`scale=TRUE`) before computing *any* of its 31 delegated features --
    confirmed empirically (see module docstring) -- so every function
    below that ports a `tsfeatures`-derived measure must be called on
    `_scalets(x)`, not the raw series."""
    if _is_constant(x):
        return x
    return (x - x.mean()) / np.std(x, ddof=1)


def _burg_ar(x: np.ndarray, order_max: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Burg's method: AR coefficients and innovation variance at every
    order `0..order_max`, via the standard lattice/reflection-coefficient
    recursion (matches R's compiled `ar.burg`/`C_Burg`, verified against
    it directly). Returns `(coefs_by_order, sigma2_by_order)`."""
    n = len(x)
    x = x - x.mean()
    f = x.copy()
    b = x.copy()
    sigma2 = [np.sum(x**2) / n]
    coefs = [np.array([])]
    for m in range(1, order_max + 1):
        num = 2 * np.sum(f[1:] * b[:-1])
        den = np.sum(f[1:] ** 2) + np.sum(b[:-1] ** 2)
        k_m = num / den
        prev = coefs[-1]
        new = np.empty(m)
        new[m - 1] = k_m
        if m > 1:
            new[: m - 1] = prev - k_m * prev[::-1]
        coefs.append(new)
        sigma2.append(sigma2[-1] * (1 - k_m**2))
        f, b = f[1:] - k_m * b[:-1], b[:-1] - k_m * f[1:]
    return coefs, np.array(sigma2)


def _burg_spectral_entropy(x: np.ndarray) -> float:
    """`tsfeatures::entropy` -- a *different* spectral entropy than CMFTS's
    own FFT-based `_spectral_entropy_fft` (name collision, two genuinely
    different algorithms): fits an AR model via Burg's method with order
    chosen by AIC (`order.max = min(n-1, floor(10*log10(n)))`, matching
    `stats::ar.burg`'s default), evaluates the AR spectral density on a
    `ceiling(n/2+1)`-point frequency grid (`stats::spec.ar`), mirrors it
    into a symmetric array, and reports its Shannon entropy in base-`n`
    log after a 0.1% shrinkage toward a uniform prior (a numerical floor
    against log(0)). Returns NaN for an (near-)constant series -- confirmed
    real R output does too (`ar.burg`/`spec.ar` degrade to a meaningless
    answer there; unlike `_acf`, this one doesn't naturally underflow to
    NaN on its own, so it's checked explicitly)."""
    if _is_constant(x):
        return np.nan
    n = len(x)
    order_max = min(n - 1, int(np.floor(10 * np.log10(n))))
    coefs_by_order, sigma2 = _burg_ar(x, order_max)
    aic = n * np.log(sigma2) + 2 * np.arange(order_max + 1) + 2
    order = int(np.argmin(aic))
    coefs, var_pred = coefs_by_order[order], sigma2[order]

    n_freq = int(np.ceil(n / 2 + 1))
    freqs = np.linspace(0, 0.5, n_freq)
    if order == 0:
        spec = np.full(n_freq, var_pred)
    else:
        k = np.arange(1, order + 1)
        angle = 2 * np.pi * freqs[:, None] * k[None, :]
        cos_term = 1 - np.sum(coefs[None, :] * np.cos(angle), axis=1)
        sin_term = np.sum(coefs[None, :] * np.sin(angle), axis=1)
        spec = var_pred / (cos_term**2 + sin_term**2)

    fx = np.concatenate([spec[:0:-1], spec]) / n
    fx = fx / fx.sum()
    fx = 0.999 * fx + 0.001 / len(fx)
    return float(min(1.0, -np.sum(fx * np.log(fx) / np.log(n))))


def _acf(x: np.ndarray, lag_max: int) -> np.ndarray:
    """`stats::acf(x, lag.max, type="correlation")$acf`: biased (n-denominator) ACF, lags 0..lag_max."""
    n = len(x)
    xc = x - x.mean()
    denom = np.sum(xc**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.array([np.sum(xc[: n - k] * xc[k:]) / denom for k in range(lag_max + 1)])


def _pacf(x: np.ndarray, lag_max: int) -> np.ndarray:
    """`stats::pacf(x, lag.max)$acf`: PACF via Durbin-Levinson recursion on `_acf`."""
    rho = _acf(x, lag_max)
    phi_prev = np.zeros(lag_max + 1)
    phi_prev[1] = rho[1]
    out = [phi_prev[1]]
    for k in range(2, lag_max + 1):
        num = rho[k] - np.sum(phi_prev[1:k] * rho[k - 1 : 0 : -1])
        den = 1 - np.sum(phi_prev[1:k] * rho[1:k])
        phi_k = np.zeros(lag_max + 1)
        kk = num / den if den != 0 else np.nan
        phi_k[k] = kk
        phi_k[1:k] = phi_prev[1:k] - kk * phi_prev[k - 1 : 0 : -1]
        out.append(kk)
        phi_prev = phi_k
    return np.array(out)


def _acf_features(x: np.ndarray) -> dict[str, float]:
    """`tsfeatures::acf_features` for a non-seasonal (frequency=1) series:
    `x_acf1`/`x_acf10` (lag.max=10) and the same for the first/second
    differences (lag.max=10, needing length>10/>11). "…10" outputs are the
    sum of squares of *all* lags 1-10, not the single lag-10 value."""
    n = len(x)
    acfx = _acf(x, 10) if n > 1 else np.full(11, np.nan)
    acfd1 = _acf(np.diff(x, 1), 10) if n > 10 else np.full(11, np.nan)
    acfd2 = _acf(np.diff(x, 2), 10) if n > 11 else np.full(11, np.nan)
    return {
        "x_acf1": acfx[1],
        "x_acf10": np.sum(acfx[1:11] ** 2),
        "diff1_acf1": acfd1[1],
        "diff1_acf10": np.sum(acfd1[1:11] ** 2),
        "diff2_acf1": acfd2[1],
        "diff2_acf10": np.sum(acfd2[1:11] ** 2),
    }


def _pacf_features(x: np.ndarray) -> dict[str, float]:
    """`tsfeatures::pacf_features` for a non-seasonal series: sum of
    squares of the first 5 PACF coefficients of `x` and its first/second
    differences."""
    n = len(x)
    pacfx = _pacf(x, 5) if n > 1 else np.full(5, np.nan)
    pacfd1 = _pacf(np.diff(x, 1), 5) if n > 6 else np.full(5, np.nan)
    pacfd2 = _pacf(np.diff(x, 2), 5) if n > 7 else np.full(5, np.nan)
    return {
        "x_pacf5": np.sum(pacfx[:5] ** 2) if n > 5 else np.nan,
        "diff1x_pacf5": np.sum(pacfd1[:5] ** 2),
        "diff2x_pacf5": np.sum(pacfd2[:5] ** 2),
    }


def _arfima_acf(d: float, n: int) -> np.ndarray:
    """`rho[0..n-1]` for an ARFIMA(0,d,0) process: `rho[0]=1`,
    `rho[k] = rho[k-1] * (k-1+d)/(k-d)` (closed-form recursion for the
    normalized autocovariance of fractional differencing)."""
    rho = np.empty(n)
    rho[0] = 1.0
    for k in range(1, n):
        rho[k] = rho[k - 1] * (k - 1 + d) / (k - d)
    return rho


def _arfima_neg_loglik(d: float, x: np.ndarray) -> float:
    """Concentrated (profiled over the innovation variance) negative
    Gaussian log-likelihood of `x` under ARFIMA(0,d,0), via the
    Durbin-Levinson recursion on `_arfima_acf(d, n)`. This is the
    *exact* likelihood (not R's `fracdiff`, which truncates to an AR(1)
    approximation beyond lag 100 for speed on long series -- the source
    of most of the small, consistent residual gap documented in
    `_hurst` below)."""
    n = len(x)
    rho = _arfima_acf(d, n)
    phi_prev = np.array([])
    v = 1.0
    resid = np.empty(n)
    innov_var = np.empty(n)
    resid[0] = x[0]
    innov_var[0] = v
    for t in range(1, n):
        xhat = 0.0 if t == 1 else np.dot(phi_prev, x[t - 1 : 0 : -1])
        resid[t] = x[t] - xhat
        innov_var[t] = v
        k = rho[1] if t == 1 else (rho[t] - np.dot(phi_prev, rho[1:t][::-1])) / v
        phi_new = np.empty(t)
        if t > 1:
            phi_new[:-1] = phi_prev - k * phi_prev[::-1]
        phi_new[-1] = k
        v *= 1 - k**2
        phi_prev = phi_new
    sigma2 = np.mean(resid**2 / innov_var)
    return float(0.5 * (n * np.log(sigma2) + np.sum(np.log(innov_var))))


def _hurst(x: np.ndarray) -> float:
    """`tsfeatures::hurst` -> `fracdiff::fracdiff(x, nar=0, nma=0)$d + 0.5`:
    the Hurst exponent via an ARFIMA(0,d,0) fit, `d` estimated by exact
    Gaussian MLE over `d in [0, 0.5)` (matching `fracdiff`'s default
    `drange`).

    **Approximate, not exact** -- one of the two features with no
    closed-form R algorithm to transcribe (see module docstring). R's real
    `fracdiff` (Haslett & Raftery 1989) truncates its likelihood to an
    AR(1) approximation beyond lag 100 to stay fast on long series; this
    instead computes the *exact* likelihood via Durbin-Levinson over all
    `n` lags. Validated against real R output (40 real training-set rows):
    consistently close but not identical, mean absolute difference
    ~0.0013, max ~0.0022 (both on Hurst's natural `[0, 1]` scale) -- close
    enough to be directionally and quantitatively useful, but genuinely an
    approximation, not a faithful port. Numerical optimization via
    `scipy.optimize.minimize_scalar` cannot close this gap further: the
    remaining difference is a modeling difference (exact vs. truncated
    likelihood), not an optimizer-precision one.
    """
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        _arfima_neg_loglik, bounds=(1e-6, 0.5 - 1e-6), method="bounded", args=(x,), options={"xatol": 1e-8}
    )
    return float(result.x + 0.5)


def _nonlinearity(x: np.ndarray) -> float:
    """`tsfeatures::nonlinearity` -> `tseries::terasvirta.test` (Terasvirta
    neural-network linearity test statistic), simplified: the `length(x)`
    factors in `10*X2/length(x)` cancel exactly against the test
    statistic's own `t*log(...)` factor, leaving `10*log(ssr0/ssr)`."""
    import numpy.linalg as la

    if _is_constant(x):
        return np.nan
    sd = np.std(x, ddof=1)
    if sd == 0:
        return np.nan
    xs = (x - x.mean()) / sd
    y = xs[1:]
    y_lag = xs[:-1]

    def _ols_residuals(design, target):
        coef, *_ = la.lstsq(design, target, rcond=None)
        return target - design @ coef

    design0 = np.column_stack([np.ones_like(y_lag), y_lag])
    u = _ols_residuals(design0, y)
    ssr0 = np.sum(u**2)

    design1 = np.column_stack([np.ones_like(y_lag), y_lag, y_lag**2, y_lag**3])
    v = _ols_residuals(design1, u)
    ssr = np.sum(v**2)

    return float(10 * np.log(ssr0 / ssr))


def _lumpiness_stability(x: np.ndarray, width: int = 10) -> tuple[float, float]:
    """`tsfeatures::lumpiness`/`stability`: split into `width`-length
    windows (a fractional trailing window is silently dropped -- R's
    `seq_len(nr/width)` truncates), variance-of-per-window-variances /
    variance-of-per-window-means."""
    n = len(x)
    if n < 2 * width:
        return 0.0, 0.0
    n_segs = n // width
    windows = x[: n_segs * width].reshape(n_segs, width)
    lumpiness = np.var(windows.var(axis=1, ddof=1), ddof=1)
    stability = np.var(windows.mean(axis=1), ddof=1)
    return float(lumpiness), float(stability)


def _bartlett_long_run_variance(res: np.ndarray, lmax: int) -> float:
    """Shared Bartlett-kernel long-run-variance piece of `urca::ur.kpss`/
    `urca::ur.pp`: `s2 + (2/n) * sum((1 - k/(lmax+1)) * sum(res[k+1:] * res[:-k]))`."""
    n = len(res)
    s2 = np.sum(res**2) / n
    total = 0.0
    for k in range(1, lmax + 1):
        cov = np.sum(res[k:] * res[:-k])
        total += (1 - k / (lmax + 1)) * cov
    return s2 + (2 / n) * total


def _unitroot_kpss(x: np.ndarray) -> float:
    """`tsfeatures::unitroot_kpss` -> `urca::ur.kpss(x)@teststat`, defaults
    `type="mu"` (level-stationarity), `lags="short"`."""
    if _is_constant(x):
        return np.nan
    n = len(x)
    lmax = int(np.trunc(4 * (n / 100) ** 0.25))
    res = x - x.mean()
    s_cum = np.cumsum(res)
    numerator = np.sum(s_cum**2) / n**2
    denominator = _bartlett_long_run_variance(res, lmax)
    return float(numerator / denominator)


def _unitroot_pp(x: np.ndarray) -> float:
    """`tsfeatures::unitroot_pp` -> `urca::ur.pp(x)@teststat`, defaults
    `type="Z-alpha"`, `model="constant"`, `lags="short"`."""
    if _is_constant(x):
        return np.nan
    y = x[1:]
    y_l1 = x[:-1]
    n = len(y)
    lmax = int(np.trunc(4 * (n / 100) ** 0.25))

    design = np.column_stack([np.ones_like(y_l1), y_l1])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    res = y - design @ coef
    alpha = coef[1]

    s = np.sum(res**2) / n
    sig = _bartlett_long_run_variance(res, lmax)
    lam = 0.5 * (sig - s)
    myybar = np.sum((y - y.mean()) ** 2) / n**2
    return float(n * (alpha - 1) - lam / myybar)


def _orthogonal_poly2(n: int) -> tuple[np.ndarray, np.ndarray]:
    """R's `poly(1:n, degree=2)`: an orthonormal (mean-zero, unit-norm)
    linear and quadratic basis over `1..n`, built by classical Gram-Schmidt
    -- matches R's sign convention (increasing linear term, convex
    quadratic), confirmed empirically."""
    x = np.arange(1, n + 1, dtype=float)
    xc = x - x.mean()
    p1 = xc / np.linalg.norm(xc)
    p2 = xc**2
    p2 = p2 - p2.mean()
    p2 = p2 - (p2 @ p1) * p1
    p2 = p2 / np.linalg.norm(p2)
    return p1, p2


def _stl_features(x: np.ndarray) -> dict[str, float]:
    """`tsfeatures::stl_features` for a non-seasonal (frequency=1) series.

    **Approximate, not exact** -- the other of the two features with no
    closed-form R algorithm to transcribe (see module docstring and
    `_hurst`). For frequency=1, `forecast::mstl` (which `stl_features`
    calls) never actually runs `stats::stl` despite the `s.window`/
    `robust` arguments CMFTS passes -- confirmed directly from
    `forecast`'s source -- trend comes entirely from
    `trend <- ts(stats::supsmu(seq_len(n), x)$y)`, Friedman's (1984)
    variable-span smoother. This uses the `supersmoother` package (a
    tested Python port of the same algorithm) in its place. Validated
    against real R output (a handful of real training-set rows):
    consistently close but not identical -- e.g. `trend` strength typically
    within ~0.02-0.05 of R's value, `linearity`/`curvature` within a few
    percent -- close enough to preserve the qualitative pattern (which
    representative AUs have strong/weak/linear/curved trends) but not
    exact, since `supersmoother`'s span selection isn't a bit-exact match
    for `supsmu`'s.

    `spike`, `linearity`, `curvature`, `e_acf1`, `e_acf10` are all
    downstream of this same approximate trend curve, so inherit its
    imprecision; `nperiods=0`/`seasonal_period=1` are exact (no seasonal
    component exists for frequency=1 data, so these are always constant).

    Constant series are special-cased directly (rather than left to
    numerical happenstance): `supersmoother`'s fit to a constant series
    leaves a tiny but non-exactly-zero residual, which -- unlike `_acf`
    applied to the *raw* constant input elsewhere -- doesn't reliably
    underflow to the NaN real R produces for `e_acf1`/`e_acf10` there.
    """
    n = len(x)
    if _is_constant(x):
        return {
            "nperiods": 0.0, "seasonal_period": 1.0, "trend": 0.0, "spike": 0.0,
            "linearity": 0.0, "curvature": 0.0, "e_acf1": np.nan, "e_acf10": np.nan,
        }

    from supersmoother import SuperSmoother

    t = np.arange(1, n + 1, dtype=float)
    trend0 = SuperSmoother().fit(t, x).predict(t)
    remainder = x - trend0

    varx = np.var(x, ddof=1)
    vare = np.var(remainder, ddof=1)
    if varx < np.finfo(float).eps:
        trend_strength = 0.0
    else:
        trend_strength = max(0.0, min(1.0, 1 - vare / varx))

    d = (remainder - remainder.mean()) ** 2
    with np.errstate(invalid="ignore"):
        varloo = (vare * (n - 1) - d) / (n - 2)
    spike = float(np.var(varloo, ddof=1))

    p1, p2 = _orthogonal_poly2(n)
    design = np.column_stack([np.ones(n), p1, p2])
    coef, *_ = np.linalg.lstsq(design, trend0, rcond=None)

    acf_r = _acf_features(remainder)
    return {
        "nperiods": 0.0,
        "seasonal_period": 1.0,
        "trend": float(trend_strength),
        "spike": spike,
        "linearity": float(coef[1]),
        "curvature": float(coef[2]),
        "e_acf1": acf_r["x_acf1"],
        "e_acf10": acf_r["x_acf10"],
    }


def _rolling_window_view(x: np.ndarray, width: int) -> np.ndarray:
    """`(n-width+1, width)` array of consecutive length-`width` windows."""
    n = len(x)
    return np.lib.stride_tricks.sliding_window_view(x, width) if n >= width else np.empty((0, width))


def _trimts(x: np.ndarray, trim: float = 0.1) -> np.ndarray:
    """`tsfeatures:::trimts`: values outside the `[trim, 1-trim]` quantile
    range become NaN (series length unchanged). `max_level_shift`/
    `max_var_shift` are called with `trim=TRUE` (`trim_amount=0.1`
    default) in the `tsfeatures()` wrapper -- easy to miss since it's a
    *second*, separate preprocessing step from the global z-scoring,
    applied only to these two measures."""
    lo, hi = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    out = x.copy()
    out[(out < lo) | (out > hi)] = np.nan
    return out


def _max_level_shift(x: np.ndarray, width: int = 10) -> dict[str, float]:
    """`tsfeatures::max_level_shift` (called with `trim=TRUE`): trim to the
    10th-90th percentile range, `RcppRoll::roll_mean(width, na.rm=TRUE)`,
    then the biggest jump between two adjacent non-overlapping windows."""
    trimmed = _trimts(x)
    if len(trimmed) <= width:
        return {"max_level_shift": np.nan, "time_level_shift": np.nan}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        roll_mean = np.nanmean(_rolling_window_view(trimmed, width), axis=1)
    means = np.abs(roll_mean[width:] - roll_mean[:-width])
    idx = int(np.nanargmax(means))
    return {"max_level_shift": float(means[idx]), "time_level_shift": float(idx + width)}


def _max_var_shift(x: np.ndarray, width: int = 10) -> dict[str, float]:
    """`tsfeatures::max_var_shift` (called with `trim=TRUE`) -- identical
    to `max_level_shift` but on rolling variance (`n-1` divisor) instead
    of rolling mean."""
    trimmed = _trimts(x)
    if len(trimmed) <= width:
        return {"max_var_shift": np.nan, "time_var_shift": np.nan}
    windows = _rolling_window_view(trimmed, width)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        roll_var = np.nanvar(windows, axis=1, ddof=1)
    # RcppRoll::roll_var(..., na.rm=TRUE) returns 0 (not NaN) for a window
    # with zero valid points -- confirmed empirically, a real difference
    # from np.nanvar's NaN there (both agree everywhere else, including
    # returning NaN for exactly-one-valid-point windows).
    roll_var = np.where(np.all(np.isnan(windows), axis=1), 0.0, roll_var)
    diffs = np.abs(roll_var[width:] - roll_var[:-width])
    idx = int(np.nanargmax(diffs))
    return {"max_var_shift": float(diffs[idx]), "time_var_shift": float(idx + width)}


def _max_kl_shift(x: np.ndarray, width: int = 48) -> dict[str, float]:
    """`tsfeatures::max_kl_shift` (called with `width=48`): rolling
    per-point Gaussian-KDE (bandwidth `bw.nrd0`), KL divergence between
    two windows `width` apart, reports the largest *first difference* of
    that KL sequence (not its max value). Early KL entries, where the
    rolling-mean row is entirely undefined (first `width-1` rows), come
    out as `0` (not `NaN`) -- matches R's `sum(..., na.rm=TRUE)` on an
    all-`NA` vector, which `np.nansum` also gives by default."""
    n = len(x)
    if n <= 2 * width:
        return {"max_kl_shift": np.nan, "time_kl_shift": np.nan}

    gw = 100
    xgrid = np.linspace(x.min(), x.max(), gw)
    grid_spacing = xgrid[1] - xgrid[0]

    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sd = np.std(x, ddof=1)
    base = min(sd, iqr / 1.34)
    bw = 0.9 * (base if base > 0 else sd) * n ** (-0.2)

    floor = float(np.exp(-0.5 * 38**2) / np.sqrt(2 * np.pi))
    dens = np.exp(-0.5 * ((xgrid[None, :] - x[:, None]) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    dens = np.maximum(dens, floor)

    roll_mean = np.full((n, gw), np.nan)
    for t in range(width - 1, n):
        roll_mean[t] = dens[t - width + 1 : t + 1].mean(axis=0)

    seq_len = n - width
    lo_rows = roll_mean[:seq_len]
    hi_rows = roll_mean[width : width + seq_len]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = lo_rows * (np.log(lo_rows) - np.log(hi_rows))
    kl = np.nansum(terms, axis=1) * grid_spacing

    diff_kl = np.diff(kl)
    idx = int(np.argmax(diff_kl))
    return {"max_kl_shift": float(diff_kl[idx]), "time_kl_shift": float(idx + width)}


def extract_cmfts_features(series) -> pd.Series:
    """All 41 CMFTS features for one time series.

    Replicates `cmfts::cmfts()`'s per-row computation exactly (see module
    docstring for the two documented approximations, `hurst` and the
    `stl_features` group, and several confirmed-not-fixed quirks:
    `permutation_entropy` always NaN, `shannon_entropy_CS` frequently
    `Inf`).

    Parameters
    ----------
    series : array-like of float
        A single time series (e.g. one row of :func:`reshape_for_cmfts`'s
        ``fr_1..fr_N`` columns).

    Returns
    -------
    pd.Series
        41 entries, indexed by feature name, in the same order as
        `cmfts::cmfts()`'s real output columns.
    """
    x = np.asarray(series, dtype=float)
    z = _scalets(x)

    def _safe(keys: str | tuple[str, ...], fn, *args, **kwargs):
        """Mirrors `cmfts.R`'s `run()`/`measures.Hyndman`'s per-measure
        `tryCatch`/`withTimeout`: any exception (e.g. `lstsq` failing to
        converge on a degenerate/constant series, dividing by a zero
        standard deviation) becomes NaN for just that measure, rather than
        aborting the whole row -- matching real R's actual behavior on
        the 10 exactly-constant representative-AU series in the real
        training set, confirmed to come out all-NaN there too."""
        names = (keys,) if isinstance(keys, str) else keys
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = fn(*args, **kwargs)
        except Exception:
            return dict.fromkeys(names, np.nan)
        if len(names) == 1:
            return {names[0]: result}
        return dict(result)

    features: dict[str, float] = {}
    features.update(_safe("lempel_ziv", _lempel_ziv, x))
    features.update(_safe("aproximation_entropy", _approximate_entropy, x))
    features.update(_safe("sample_entropy", _sample_entropy_cmfts, x))
    features["permutation_entropy"] = np.nan
    features.update(_safe("shannon_entropy_CS", _shannon_entropy_cs, x))
    features.update(_safe("shannon_entropy_SG", _shannon_entropy_sg, x))
    features.update(_safe("spectral_entropy", _spectral_entropy_fft, x))
    features.update(_safe("nforbiden", _nforbidden, x, ndemb=6))
    features.update(_safe("Kurtosis", _kurtosis_e1071, x))
    features.update(_safe("Skewness", _skewness_e1071, x))
    features["length"] = float(len(x))

    features.update(_safe(("x_acf1", "x_acf10", "diff1_acf1", "diff1_acf10", "diff2_acf1", "diff2_acf10"), _acf_features, z))
    features.update(_safe(("x_pacf5", "diff1x_pacf5", "diff2x_pacf5"), _pacf_features, z))
    features.update(_safe("entropy", _burg_spectral_entropy, z))
    features.update(_safe("nonlinearity", _nonlinearity, z))
    features.update(_safe("hurst", _hurst, z))
    def _lumpiness_stability_dict(zz):
        lumpiness, stability = _lumpiness_stability(zz)
        return {"lumpiness": lumpiness, "stability": stability}

    features.update(_safe(("lumpiness", "stability"), _lumpiness_stability_dict, z))
    features.update(_safe("unitroot_kpss", _unitroot_kpss, z))
    features.update(_safe("unitroot_pp", _unitroot_pp, z))
    features.update(
        _safe(
            ("nperiods", "seasonal_period", "trend", "spike", "linearity", "curvature", "e_acf1", "e_acf10"),
            _stl_features,
            z,
        )
    )
    features.update(_safe(("max_level_shift", "time_level_shift"), _max_level_shift, z))
    features.update(_safe(("max_var_shift", "time_var_shift"), _max_var_shift, z))
    features.update(_safe(("max_kl_shift", "time_kl_shift"), _max_kl_shift, z, width=48))

    order = [
        "lempel_ziv", "aproximation_entropy", "sample_entropy", "permutation_entropy",
        "shannon_entropy_CS", "shannon_entropy_SG", "spectral_entropy", "nforbiden",
        "Kurtosis", "Skewness", "length", "x_acf1", "x_acf10", "diff1_acf1", "diff1_acf10",
        "diff2_acf1", "diff2_acf10", "x_pacf5", "diff1x_pacf5", "diff2x_pacf5", "entropy",
        "nonlinearity", "hurst", "stability", "lumpiness", "unitroot_kpss", "unitroot_pp",
        "nperiods", "seasonal_period", "trend", "spike", "linearity", "curvature",
        "e_acf1", "e_acf10", "max_level_shift", "time_level_shift", "max_var_shift",
        "time_var_shift", "max_kl_shift", "time_kl_shift",
    ]
    return pd.Series({name: features[name] for name in order})


def cmfts_features(
    wide_df: pd.DataFrame,
    frame_pattern: str = r"^fr_",
    n_jobs: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """CMFTS features for every row of a :func:`reshape_for_cmfts`-shaped
    DataFrame.

    Replicates `final_analysis_NMF_check.Rmd`'s
    ``apply(dta_cmfts_input[, frame_cols], 1, fn_extract_cmfts_features) %>%
    do.call(rbind, .) %>% cbind(metadata, .)``.

    Parameters
    ----------
    wide_df : pd.DataFrame
        One row per series, with frame-value columns matching
        `frame_pattern` (e.g. ``fr_1, fr_2, ...``) plus any other
        (metadata) columns, which are carried through unchanged --
        typically :func:`reshape_for_cmfts`'s output.
    frame_pattern : str, default r"^fr_"
        Regex identifying the frame-value columns to extract features
        from.
    n_jobs : int, optional
        Number of parallel worker processes (via ``joblib``, matching
        :func:`facedyn.nmf.nmf_cophenetic_correlation`'s convention:
        ``None``/``1`` = sequential, ``-1`` = all cores). Several of the
        per-row computations are O(n^2) or involve a numerical
        optimization, so this is worth setting for anything beyond a
        handful of rows. Each worker's BLAS calls are pinned to a single
        thread for the duration of this call (``threadpoolctl``) --
        without this, `n_jobs` worker *processes* each independently
        trying to multi-thread their own linear-algebra calls oversubscribes
        the machine's cores and measurably slows the whole call down (a
        ~2.8x slowdown confirmed on a real 120-row benchmark).
    verbose : int, default 0
        Forwarded to ``joblib.Parallel``'s own ``verbose`` -- set e.g. to
        ``10`` to print progress as rows complete.

    Returns
    -------
    pd.DataFrame
        `wide_df`'s non-frame columns, plus the 41 CMFTS feature columns.
    """
    from threadpoolctl import threadpool_limits

    frame_cols = [c for c in wide_df.columns if re.search(frame_pattern, c)]
    metadata = wide_df.drop(columns=frame_cols).reset_index(drop=True)
    values = wide_df[frame_cols].to_numpy(dtype=float)

    with threadpool_limits(limits=1):
        rows = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(extract_cmfts_features)(row) for row in values
        )
    features = pd.DataFrame(rows).reset_index(drop=True)
    return pd.concat([metadata, features], axis=1)
