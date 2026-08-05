"""Non-negative matrix factorisation of normalised AU columns."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF
from sklearn.utils.validation import check_is_fitted

from facedyn._plot_utils import save_figure
from facedyn.au_labels import humanise_au_label


def _resolve_columns(
    X: pd.DataFrame, columns: list[str] | None, column_pattern: str
) -> list[str]:
    if columns is not None:
        return list(columns)
    pattern = re.compile(column_pattern)
    return [col for col in X.columns if pattern.search(col)]


def max_normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Min-max scale each column of `matrix` independently to `[0, 1]`.

    Shared by :func:`plot_nmf_basis_heatmap` and
    :func:`facedyn.face_maps.plot_nmf_face_maps`, both of which need to
    remove NMF's per-component scale ambiguity the same way (see
    :class:`NMFDecomposer`'s docstring) before the values are meaningfully
    comparable or displayable. Constant columns (span 0) are left at 0
    rather than raising a divide-by-zero.
    """
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    span = col_max - col_min
    span = np.where(span == 0, 1.0, span)
    return (matrix - col_min) / span


def nmf_rank_mse_sweep(
    X: pd.DataFrame,
    ranks: range | list[int] = range(2, 11),
    columns: list[str] | None = None,
    column_pattern: str = r"^smth_",
    random_state: int | None = None,
    max_iter: int = 750,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Fit NMF at each rank and report reconstruction MSE, for rank selection.

    Measures fit error on the training data only, so error can only improve
    with more components. See :func:`nmf_rank_cv_sweep` for held-out error
    and :func:`nmf_cophenetic_correlation` for stability. Does not pick a
    rank automatically. Inspect or plot the returned table to choose one.

    Parameters
    ----------
    X : pd.DataFrame
        Data containing the columns to factorize (plus any other columns,
        which are ignored).
    ranks : range or list of int, default range(2, 11)
        Candidate values of ``n_components`` to try.
    columns : list of str, optional
        Explicit columns to factorize. If not given, selected via
        ``column_pattern``.
    column_pattern : str, default r"^smth_"
        Regex used to select columns when ``columns`` is not given. Matches
        :class:`ZScoreShiftNormalizer`'s output by default.
    random_state : int, optional
        Seed for NMF initialization.
    max_iter : int, default 750
        Passed to :class:`sklearn.decomposition.NMF`.
    tol : float, default 1e-6
        Passed to :class:`sklearn.decomposition.NMF`.

    Returns
    -------
    pd.DataFrame
        Columns ``rank`` and ``mse``, one row per value in ``ranks``.
    """
    cols = _resolve_columns(X, columns, column_pattern)
    data = X[cols].to_numpy()

    records = []
    for k in ranks:
        model = NMF(
            n_components=k, init="nndsvda", random_state=random_state,
            max_iter=max_iter, tol=tol,
        )
        W = model.fit_transform(data)
        H = model.components_
        mse = ((data - W @ H) ** 2).mean()
        records.append({"rank": k, "mse": mse})

    return pd.DataFrame.from_records(records)


def _masked_nmf(
    X: np.ndarray,
    mask: np.ndarray,
    n_components: int,
    max_iter: int = 750,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative-update NMF with entries excluded from the loss.

    Minimises ``||mask * (X - W @ H)||^2``, so entries where ``mask == 0``
    never influence the fit. Standard Weighted-NMF extension of Lee &
    Seung's multiplicative updates.

    Internal helper for :func:`nmf_rank_cv_sweep`. sklearn's ``NMF`` has no
    way to exclude individual entries from its loss, hence this.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    mask : ndarray of shape (n_samples, n_features)
        1 for entries included in the loss (observed/training), 0 for
        entries excluded (held out).
    n_components : int
    max_iter : int, default 750
    tol : float, default 1e-6
        Stop early once the relative improvement in masked training loss
        drops below this.
    random_state : int, optional
        Seed for the random non-negative initialization of W, H.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
    H : ndarray of shape (n_components, n_features)
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape
    observed = mask.astype(bool)
    scale = np.sqrt(X[observed].mean() / n_components) if observed.any() else 1.0
    scale = max(scale, 1e-8)

    W = rng.uniform(0, scale, size=(n_samples, n_components))
    H = rng.uniform(0, scale, size=(n_components, n_features))

    eps = 1e-10
    n_observed = mask.sum()
    prev_loss = np.inf
    for _ in range(max_iter):
        masked_X = mask * X

        H *= (W.T @ masked_X) / (W.T @ (mask * (W @ H)) + eps)
        W *= (masked_X @ H.T) / ((mask * (W @ H)) @ H.T + eps)

        loss = ((mask * (X - W @ H)) ** 2).sum() / n_observed
        if not np.isfinite(loss):
            break
        if prev_loss - loss < tol * prev_loss:
            break
        prev_loss = loss

    return W, H


_SEED_STRIDE = 1_000_000


def nmf_rank_cv_sweep(
    X: pd.DataFrame,
    ranks: range | list[int] = range(2, 11),
    test_fraction: float = 0.1,
    n_replicates: int = 3,
    n_seeds: int = 1,
    columns: list[str] | None = None,
    column_pattern: str = r"^smth_",
    random_state: int | None = None,
    max_iter: int = 750,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Cross-validated rank selection: held-out reconstruction MSE vs. rank.

    Holds out a random fraction of individual matrix entries, not whole
    rows, then measures error on those entries. This reveals overfitting,
    where held-out error bottoms out and rises with rank while training
    error keeps falling. Row-holdout cannot show this, because a held-out
    row is free to choose new activation weights and so fits about as well
    under an overfit basis. See `PIPELINE.md` step 4.

    One random mask is reused across all `ranks` within a replicate, so
    ranks are compared on the same held-out entries.

    Parameters
    ----------
    X : pd.DataFrame
        Data containing the columns to factorize (plus any other columns,
        which are ignored).
    ranks : range or list of int, default range(2, 11)
        Candidate values of ``n_components`` to try.
    test_fraction : float, default 0.1
        Fraction of matrix entries held out per replicate.
    n_replicates : int, default 3
        Number of independent random holdout masks to average over,
        drawn from a single seeded stream (see ``n_seeds`` for a stronger
        check that also varies that stream itself).
    n_seeds : int, default 1
        Number of top-level seeds to repeat the whole sweep under.
        `n_replicates` alone draws masks from the one stream derived from
        `random_state`, so it cannot test sensitivity to that seed choice.
        Values above 1 vary the seed itself and add a ``seed`` column to
        the output. An opt-in robustness check.
    columns : list of str, optional
        Explicit columns to factorize. If not given, selected via
        ``column_pattern``.
    column_pattern : str, default r"^smth_"
        Regex used to select columns when ``columns`` is not given.
    random_state : int, optional
        Seed for mask generation and NMF initialization.
    max_iter : int, default 750
        Passed to :func:`_masked_nmf`.
    tol : float, default 1e-6
        Passed to :func:`_masked_nmf`.

    Returns
    -------
    pd.DataFrame
        Columns ``rank``, ``rep``, ``train_mse`` and ``test_mse``, one row
        per (rank, replicate), plus a leading ``seed`` column when
        ``n_seeds > 1``. Does not pick a rank automatically. Inspect or plot
        with :func:`plot_nmf_rank_cv` to choose one.
    """
    cols = _resolve_columns(X, columns, column_pattern)
    data = X[cols].to_numpy()

    records = []
    for seed_idx in range(n_seeds):
        seed_state = None if random_state is None else random_state + seed_idx * _SEED_STRIDE
        rng = np.random.default_rng(seed_state)

        for rep in range(n_replicates):
            mask = (rng.random(data.shape) >= test_fraction).astype(float)
            test_mask = 1.0 - mask

            for k in ranks:
                try:
                    init_seed = None if seed_state is None else seed_state + rep * 1000 + k
                    W, H = _masked_nmf(
                        data, mask, n_components=k,
                        max_iter=max_iter, tol=tol, random_state=init_seed,
                    )
                    recon = W @ H
                    train_mse = ((mask * (data - recon)) ** 2).sum() / mask.sum()
                    test_mse = ((test_mask * (data - recon)) ** 2).sum() / test_mask.sum()
                except Exception:
                    train_mse, test_mse = np.nan, np.nan

                row = {"rank": k, "rep": rep, "train_mse": train_mse, "test_mse": test_mse}
                if n_seeds > 1:
                    row = {"seed": seed_idx, **row}
                records.append(row)

    return pd.DataFrame.from_records(records)


def plot_nmf_rank_cv(
    result: pd.DataFrame,
    ax=None,
    robust: bool = True,
    outlier_z: float = 3.5,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot :func:`nmf_rank_cv_sweep` output: train/test MSE vs. rank.

    Requires matplotlib (``pip install facedyn[viz]``).

    Each replicate is a faint line, with a bold summary line on top and a
    vertical marker at the rank with the lowest summary ``test_mse``. Lines
    are grouped per replicate, and per ``(seed, rep)`` when `result` has a
    ``seed`` column, so separate trajectories are not zigzagged together.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nmf_rank_cv_sweep`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    robust : bool, default True
        Summarise replicates by median rather than mean, and scale the
        y-axis to a robust range. Guards against the occasional degenerate
        :func:`_masked_nmf` fit, which can be orders of magnitude larger
        than every other value and would otherwise drag the summary line
        and flatten the axis. Outliers are still drawn, pinned to the top
        of the axis as an annotated ``^`` marker rather than dropped.
    outlier_z : float, default 3.5
        Robust z-score (based on median absolute deviation across all
        train/test MSE values) beyond which a point is treated as an
        outlier for axis-scaling purposes. Only used when ``robust=True``.
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
            "plot_nmf_rank_cv requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    if ax is None:
        _, ax = plt.subplots()

    agg = result.groupby("rank")[["train_mse", "test_mse"]].agg(
        "median" if robust else "mean"
    ).reset_index()
    colors = {"train_mse": "#009E73", "test_mse": "#D55E00"}
    labels = {"train_mse": "Train", "test_mse": "Test"}

    y_top = None
    if robust:
        all_values = pd.concat([result["train_mse"], result["test_mse"]]).dropna()
        if len(all_values) > 0:
            median = all_values.median()
            robust_scale = (all_values - median).abs().median() * 1.4826
            inliers = (
                all_values[(all_values - median).abs() / robust_scale <= outlier_z]
                if robust_scale > 0
                else all_values
            )
            if len(inliers) > 0:
                y_top = inliers.max() * 1.15

    line_group_cols = ["seed", "rep"] if "seed" in result.columns else ["rep"]

    has_outliers = False
    for col, color in colors.items():
        for _, rep_data in result.groupby(line_group_cols):
            rep_data = rep_data.sort_values("rank")
            if y_top is not None:
                is_outlier = rep_data[col] > y_top
                plot_y = rep_data[col].clip(upper=y_top)
            else:
                is_outlier = pd.Series(False, index=rep_data.index)
                plot_y = rep_data[col]
            ax.plot(rep_data["rank"], plot_y, color=color, alpha=0.25, linewidth=0.8)
            ax.scatter(rep_data["rank"], plot_y, color=color, alpha=0.3, s=15)
            if is_outlier.any():
                has_outliers = True
                off_scale = rep_data[is_outlier]
                ax.scatter(
                    off_scale["rank"], [y_top] * len(off_scale), color=color,
                    marker="^", s=70, zorder=4, edgecolor="black", linewidth=0.5,
                )
                for _, row in off_scale.iterrows():
                    ax.annotate(
                        f"{row[col]:.0f}", (row["rank"], y_top),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=color,
                    )
        ax.plot(agg["rank"], agg[col], color=color, linewidth=2.2, label=labels[col])
        ax.scatter(agg["rank"], agg[col], color=color, s=40, zorder=3)

    best_rank = agg.loc[agg["test_mse"].idxmin(), "rank"]
    ax.axvline(best_rank, linestyle="--", color="grey", linewidth=1)

    if y_top is not None:
        ax.set_ylim(top=y_top * 1.05)

    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("MSE")
    title = "Cross-validated NMF rank selection"
    if has_outliers:
        title += "\n(▲ = off-scale point, true value annotated)"
    ax.set_title(title)
    ax.legend()
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def nmf_cophenetic_correlation(
    X: pd.DataFrame,
    ranks: range | list[int] = range(2, 11),
    n_runs: int = 10,
    n_jobs: int | None = None,
    columns: list[str] | None = None,
    column_pattern: str = r"^smth_",
    random_state: int | None = None,
    max_iter: int = 750,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Cophenetic correlation per rank, an NMF stability diagnostic.

    A stability criterion rather than an accuracy one, so it complements
    :func:`nmf_rank_cv_sweep` rather than replacing it. Fits NMF `n_runs`
    times per rank from random initialisations, assigns each row to its
    dominant component, and measures how cleanly block-structured the
    resulting consensus matrix is. Stable ranks score near 1. Scores drop
    once rank exceeds the number of well-separated components, which
    reconstruction error cannot show. From Brunet et al. (2004).

    Uses ``init="random"`` rather than this module's usual ``"nndsvda"``,
    which is near-deterministic and would make every rank look stable.

    Cost warning: ``O(n_samples^2)`` in memory and time. Subsample `X` to a
    few hundred or a couple of thousand rows before calling.

    Parameters
    ----------
    X : pd.DataFrame
        Data containing the columns to factorize (plus any other columns,
        which are ignored). Subsample before calling - see cost warning
        above.
    ranks : range or list of int, default range(2, 11)
        Candidate values of ``n_components`` to try.
    n_runs : int, default 10
        Number of independent random-init NMF fits per rank used to build
        the consensus matrix. More runs give a more stable estimate at
        proportionally higher cost.
    n_jobs : int, optional
        Parallel worker processes for the NMF fits. ``None`` or ``1`` is
        sequential, ``-1`` uses all cores. Every fit is seeded
        independently of execution order, so results are identical for any
        value. Start-up overhead can make small workloads slower in
        parallel than sequentially.
    columns : list of str, optional
        Explicit columns to factorize. If not given, selected via
        ``column_pattern``.
    column_pattern : str, default r"^smth_"
        Regex used to select columns when ``columns`` is not given.
    random_state : int, optional
        Seed for the NMF random initializations.
    max_iter : int, default 750
        Passed to :class:`sklearn.decomposition.NMF`.
    tol : float, default 1e-6
        Passed to :class:`sklearn.decomposition.NMF`.

    Returns
    -------
    pd.DataFrame
        Columns ``rank`` and ``cophenetic_correlation`` - one row per
        value in ``ranks``. Does not pick a rank automatically; inspect or
        plot (see :func:`plot_nmf_cophenetic_correlation`) alongside
        :func:`nmf_rank_cv_sweep`'s reconstruction-error evidence, not in
        place of it.
    """
    cols = _resolve_columns(X, columns, column_pattern)
    data = X[cols].to_numpy()
    n_samples = data.shape[0]

    def _dominant_labels(k: int, run: int) -> np.ndarray:
        seed = None if random_state is None else random_state + run * 1000 + k
        model = NMF(
            n_components=k, init="random", random_state=seed,
            max_iter=max_iter, tol=tol,
        )
        W = model.fit_transform(data)
        return W.argmax(axis=1)

    tasks = [(k, run) for k in ranks for run in range(n_runs)]
    all_labels = Parallel(n_jobs=n_jobs)(delayed(_dominant_labels)(k, run) for k, run in tasks)

    labels_by_rank: dict[int, list[np.ndarray]] = {k: [] for k in ranks}
    for (k, _run), dominant in zip(tasks, all_labels):
        labels_by_rank[k].append(dominant)

    records = []
    for k in ranks:
        consensus = np.zeros((n_samples, n_samples))
        for dominant in labels_by_rank[k]:
            consensus += (dominant[:, None] == dominant[None, :])
        consensus /= n_runs

        condensed = squareform(1.0 - consensus, checks=False)
        coph_corr, _ = cophenet(linkage(condensed, method="average"), condensed)

        records.append({"rank": k, "cophenetic_correlation": coph_corr})

    return pd.DataFrame.from_records(records)


def plot_nmf_cophenetic_correlation(
    result: pd.DataFrame,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot :func:`nmf_cophenetic_correlation` output against rank.

    Requires matplotlib (``pip install facedyn[viz]``).

    Complements :func:`plot_nmf_rank_cv`, which measures accuracy rather
    than stability. Look for a rank still close to 1 but about to drop at
    the next rank tried.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nmf_cophenetic_correlation`.
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
            "plot_nmf_cophenetic_correlation requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    if ax is None:
        _, ax = plt.subplots()

    result = result.sort_values("rank")
    ax.plot(
        result["rank"], result["cophenetic_correlation"],
        color="#0072B2", marker="o", linewidth=2,
    )
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Cophenetic correlation")
    ax.set_ylim(0, 1.05)
    ax.set_title("NMF clustering stability vs. rank")
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


class NMFDecomposer(BaseEstimator, TransformerMixin):
    """Non-negative matrix factorisation of AU columns via sklearn's NMF.

    Fits on the resolved numeric columns. ``transform`` returns all other
    columns unchanged, plus one activation column per component.

    Activation magnitudes are only defined up to an arbitrary positive
    scale per component, since ``(W, H)`` and ``(W·S, S⁻¹·H)`` reconstruct
    the same data for any positive diagonal ``S``. Do not compare raw
    magnitudes across fits or against R without rescaling. See
    `PIPELINE.md` step 4 for the validation against real R output.

    Parameters
    ----------
    n_components : int, default 3
        Number of NMF components. Default matches the rank chosen in the
        original analysis.
    columns : list of str, optional
        Explicit columns to factorize. If not given, selected via
        ``column_pattern``.
    column_pattern : str, default r"^smth_"
        Regex used to select columns when ``columns`` is not given.
    prefix : str, default "nmf"
        Prefix for the output activation columns (``nmf1``, ``nmf2``, ...),
        matching the R pipeline's naming.
    random_state : int, optional
        Seed for NMF initialization. Not expected to reproduce R's exact
        output (different RNG), only comparable statistically.
    max_iter : int, default 750
        Passed to :class:`sklearn.decomposition.NMF`.
    tol : float, default 1e-6
        Passed to :class:`sklearn.decomposition.NMF`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The basis matrix (R's ``W``, transposed), exposed under sklearn's
        own attribute name since this class already wraps sklearn.
    """

    def __init__(
        self,
        n_components: int = 3,
        columns: list[str] | None = None,
        column_pattern: str = r"^smth_",
        prefix: str = "nmf",
        random_state: int | None = None,
        max_iter: int = 750,
        tol: float = 1e-6,
    ):
        self.n_components = n_components
        self.columns = columns
        self.column_pattern = column_pattern
        self.prefix = prefix
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: pd.DataFrame, y=None) -> "NMFDecomposer":
        self.columns_ = _resolve_columns(X, self.columns, self.column_pattern)
        self.model_ = NMF(
            n_components=self.n_components,
            init="nndsvda",
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.model_.fit(X[self.columns_].to_numpy())
        self.components_ = self.model_.components_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "model_")
        activations = self.model_.transform(X[self.columns_].to_numpy())
        metadata = X.drop(columns=self.columns_).reset_index(drop=True)
        activation_cols = pd.DataFrame(
            activations,
            columns=[f"{self.prefix}{i + 1}" for i in range(self.n_components)],
        )
        return pd.concat([metadata, activation_cols], axis=1)


def plot_nmf_basis_heatmap(
    decomposer: NMFDecomposer,
    normalize: bool = True,
    labels: list[str] | None = None,
    ax=None,
    cmap: str = "Blues",
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot a fitted :class:`NMFDecomposer`'s basis matrix as a heatmap.

    Features on rows, components on columns, with no clustering or
    reordering of either axis.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` or ``fit_transform`` already
        called).
    normalize : bool, default True
        Min-max scale each component's column to ``[0, 1]`` before
        plotting. This removes NMF's arbitrary per-component scale (see
        :class:`NMFDecomposer`), without which two correct fits need not
        land on the same colour scale. Set to False to see raw
        ``components_`` values, which are only comparable to another fit
        of the same model.
    labels : list of str, optional
        Row labels, one per factorized column, in the same order as
        ``decomposer.columns_``. Defaults to ``decomposer.columns_``
        itself; pass ``facedyn.humanise_au_labels(decomposer.columns_)``
        for readable AU names instead of raw column names.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    cmap : str, default "Blues"
        Matplotlib colormap name. The default is a sequential blue scale,
        matching the R figure's ``colorRampPalette(brewer.pal(6, "Blues"))``.
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
            "plot_nmf_basis_heatmap requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    check_is_fitted(decomposer, "components_")
    basis = decomposer.components_.T  # (n_features, n_components)

    if normalize:
        basis = max_normalize_columns(basis)

    row_labels = labels if labels is not None else decomposer.columns_
    col_labels = [f"{decomposer.prefix}{i + 1}" for i in range(decomposer.n_components)]

    if ax is None:
        _, ax = plt.subplots(figsize=(4 + 0.4 * len(col_labels), 0.35 * len(row_labels) + 1.5))

    im = ax.imshow(basis, aspect="auto", cmap=cmap, vmin=0 if normalize else None)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    cbar_label = "loading (min-max normalised per component)" if normalize else "loading"
    ax.figure.colorbar(im, ax=ax, label=cbar_label)

    ax.set_title("Basis Matrix (W)" + (" - Normalised" if normalize else ""))
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def _reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """RMSE, NRMSE, MAE and R2 of `reconstructed` against `original`.

    NRMSE divides by the original data's own range. R2 uses a single scalar
    mean over the whole matrix as its baseline, not a per-column mean, so
    this is not `sklearn.metrics.r2_score`, which defaults to per-column
    baselines for 2D input.
    """
    error = original - reconstructed
    rmse = np.sqrt(np.mean(error**2))
    return {
        "RMSE": rmse,
        "NRMSE": rmse / (original.max() - original.min()),
        "MAE": np.mean(np.abs(error)),
        "R2": 1 - np.sum(error**2) / np.sum((original - original.mean()) ** 2),
    }


def _reconstruct(decomposer: NMFDecomposer, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(original, reconstructed)` arrays for `X` under a fitted `decomposer`.

    Shared by :func:`nmf_reconstruction_error` and
    :func:`nmf_reconstruction_r2_per_au`. Projects `X` onto the decomposer's
    fixed basis via ``decomposer.transform``, then reconstructs via
    ``activations @ components_``. The same path serves in-sample and
    out-of-sample data.
    """
    check_is_fitted(decomposer, "components_")
    original = X[decomposer.columns_].to_numpy()
    activation_cols = [f"{decomposer.prefix}{i + 1}" for i in range(decomposer.n_components)]
    activations = decomposer.transform(X)[activation_cols].to_numpy()
    reconstructed = activations @ decomposer.components_
    return original, reconstructed


def nmf_reconstruction_error(decomposer: NMFDecomposer, X: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reconstruction-quality metrics for a fitted decomposer.

    Reconstructs `X` from its NMF activations and reports how much of the
    original AU signal survives compression to ``decomposer.n_components``.
    R2 is the proportion of AU signal variance retained.

    Works on any `X` containing ``decomposer.columns_``. Pass the training
    data for in-sample quality, or held-out data for generalisation. If R2
    is low, consider
    :func:`~facedyn.representative_aus.select_representative_aus` instead
    of NMF activations. See `PIPELINE.md` step 4 for the validation against
    real R output, including a discrepancy between R's prose and its own
    per-AU numbers.

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` already called).
    X : pd.DataFrame
        Data containing ``decomposer.columns_`` (plus any other columns,
        which are ignored).

    Returns
    -------
    pd.DataFrame
        Columns ``metric`` and ``value``, one row each for ``"RMSE"``,
        ``"NRMSE"``, ``"MAE"`` and ``"R2"``.
    """
    original, reconstructed = _reconstruct(decomposer, X)
    metrics = _reconstruction_metrics(original, reconstructed)
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def nmf_reconstruction_r2_per_au(
    decomposer: NMFDecomposer, X: pd.DataFrame, labels: list[str] | None = None
) -> pd.DataFrame:
    """Per-AU reconstruction R2 for a fitted decomposer.

    The same R2 formula as :func:`nmf_reconstruction_error`, computed per AU
    column rather than over the whole matrix. Shows which AUs the NMF
    compression preserves and which it loses.

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` already called).
    X : pd.DataFrame
        Data containing ``decomposer.columns_`` (plus any other columns,
        which are ignored).
    labels : list of str, optional
        Row labels, one per factorized column, in ``decomposer.columns_``
        order. Pass ``facedyn.humanise_au_labels(decomposer.columns_)`` for
        readable AU names.

    Returns
    -------
    pd.DataFrame
        Columns ``au`` and ``r2``, one row per factorized column, sorted
        descending by ``r2``.
    """
    original, reconstructed = _reconstruct(decomposer, X)
    error = original - reconstructed
    col_mean = original.mean(axis=0)
    r2 = 1 - np.sum(error**2, axis=0) / np.sum((original - col_mean) ** 2, axis=0)
    au_labels = labels if labels is not None else decomposer.columns_
    return (
        pd.DataFrame({"au": au_labels, "r2": r2})
        .sort_values("r2", ascending=False)
        .reset_index(drop=True)
    )


def plot_nmf_reconstruction(
    decomposer: NMFDecomposer,
    X: pd.DataFrame,
    au: str | None = None,
    video_id=None,
    group_col: str = "video_filename",
    frame_col: str = "frame",
    au_label: str | None = None,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot original against reconstructed AU activation for one AU and video.

    A visual check of what :func:`nmf_reconstruction_error`'s R2 means for a
    single signal.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` already called).
    X : pd.DataFrame
        Data containing ``decomposer.columns_``, ``group_col`` and
        ``frame_col``.
    au : str, optional
        Which factorized column to plot. Defaults to
        ``decomposer.columns_[0]``.
    video_id : optional
        Value of ``group_col`` identifying which video's rows to plot.
        Defaults to the first unique value in ``group_col``.
    group_col : str, default "video_filename"
        Column identifying which rows belong to which video.
    frame_col : str, default "frame"
        Column used for the x-axis.
    au_label : str, optional
        Display name for `au` in the y-axis label and title. `au` itself
        still selects the column, so passing a humanised label (e.g. via
        :func:`facedyn.humanise_au_label`) doesn't affect what's plotted,
        only how it's captioned. Defaults to `au` unchanged.
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
            "plot_nmf_reconstruction requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    if au is None:
        au = decomposer.columns_[0]
    if video_id is None:
        video_id = X[group_col].iloc[0]
    if au_label is None:
        au_label = au

    original, reconstructed = _reconstruct(decomposer, X)
    au_idx = decomposer.columns_.index(au)
    rows = (X[group_col] == video_id).to_numpy()
    frames = X.loc[rows, frame_col]
    order = np.argsort(frames.to_numpy())

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(
        frames.to_numpy()[order], original[rows, au_idx][order],
        color="#009E73", alpha=0.6, linewidth=1, label="Original",
    )
    ax.plot(
        frames.to_numpy()[order], reconstructed[rows, au_idx][order],
        color="#D55E00", linewidth=1.5, label="Reconstructed",
    )
    ax.set_xlabel(frame_col)
    ax.set_ylabel(au_label)
    # Two lines plus `wrap`: a long AU label and/or video_id easily overflow
    # a single line, and `wrap` catches whatever the explicit split doesn't.
    ax.set_title(
        f"{au_label}\nOriginal vs. Reconstructed ({video_id})", fontsize=9, wrap=True,
    )
    ax.legend()
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax


def plot_nmf_reconstruction_extremes(
    decomposer: NMFDecomposer,
    X: pd.DataFrame,
    video_id=None,
    group_col: str = "video_filename",
    frame_col: str = "frame",
    r2_table: pd.DataFrame | None = None,
    humanise: bool = False,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot the best and worst reconstructed AUs side by side, for one video.

    Finds the highest- and lowest-R2 AUs via
    :func:`nmf_reconstruction_r2_per_au` and plots each as
    :func:`plot_nmf_reconstruction` would.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` already called).
    X : pd.DataFrame
        Data containing ``decomposer.columns_``, ``group_col`` and
        ``frame_col``.
    video_id : optional
        Value of ``group_col`` identifying which video's rows to plot.
        Defaults to the first unique value in ``group_col``.
    group_col : str, default "video_filename"
        Column identifying which rows belong to which video.
    frame_col : str, default "frame"
        Column used for the x-axis.
    r2_table : pd.DataFrame, optional
        Output of :func:`nmf_reconstruction_r2_per_au` for `decomposer`/`X`,
        to reuse instead of recomputing it here.
    humanise : bool, default False
        Caption each subplot with :func:`facedyn.humanise_au_label` instead
        of the raw column name (e.g. ``"AU07 - Lid Tightener"`` instead of
        ``"smth_AU07_r"``). Only affects the label; which AU is selected as
        best/worst is unchanged.
    ax : sequence of matplotlib.axes.Axes, optional
        Two Axes (best, worst) to draw on. A new ``1 x 2`` grid is created
        if not given.
    save_path : str or pathlib.Path, optional
        Filename to save the figure to. Format is inferred from the
        extension. Not saved if ``None``.
    output_dir : str or pathlib.Path, default "."
        Directory `save_path` is written into, created if needed.
    dpi : int, default 300
        Resolution for raster formats. Ignored for vector formats.

    Returns
    -------
    list of matplotlib.axes.Axes
        ``[best_ax, worst_ax]``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_nmf_reconstruction_extremes requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    if r2_table is None:
        r2_table = nmf_reconstruction_r2_per_au(decomposer, X)
    best_au = r2_table.loc[r2_table["r2"].idxmax(), "au"]
    worst_au = r2_table.loc[r2_table["r2"].idxmin(), "au"]

    if video_id is None:
        video_id = X[group_col].iloc[0]

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        axes = np.atleast_1d(ax)
        if len(axes) != 2:
            raise ValueError(f"ax must have 2 entries, got {len(axes)}.")

    for target_ax, au, role in zip(axes, [best_au, worst_au], ["Best", "Worst"]):
        au_label = humanise_au_label(au) if humanise else au
        plot_nmf_reconstruction(
            decomposer, X, au=au, video_id=video_id,
            group_col=group_col, frame_col=frame_col, au_label=au_label, ax=target_ax,
        )
        title = target_ax.get_title()
        target_ax.set_title(f"{role}: {title}", fontsize=9, wrap=True)

    save_figure(axes[0].figure, save_path, output_dir, dpi)
    return list(axes)


def plot_nmf_reconstruction_r2_bar(
    r2_table: pd.DataFrame,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Bar chart of per-AU reconstruction R2.

    A sorted view of which AUs the NMF compression retains and which it
    loses. Takes :func:`nmf_reconstruction_r2_per_au`'s output.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    r2_table : pd.DataFrame
        Output of :func:`nmf_reconstruction_r2_per_au`.
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
            "plot_nmf_reconstruction_r2_bar requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    r2_table = r2_table.sort_values("r2", ascending=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.35 * len(r2_table) + 1.5))

    ax.barh(r2_table["au"], r2_table["r2"], color="#0072B2")
    ax.set_xlabel("R² (reconstruction)")
    ax.set_title("Per-AU reconstruction R² - signal retained by NMF")
    save_figure(ax.figure, save_path, output_dir, dpi)
    return ax
