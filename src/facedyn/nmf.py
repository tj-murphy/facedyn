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

    Mirrors the R pipeline's "Optimal K" reconstruction-error sweep. Does
    not pick a rank automatically — the paper's own rank choice combined
    this curve with interpretability and alignment with prior work, not a
    hard rule. Inspect/plot the returned table to choose one, the same way
    the original analysis did.

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

    Minimizes ``||mask * (X - W @ H)||^2`` instead of the ordinary
    ``||X - W @ H||^2`` — entries where ``mask == 0`` never influence the
    fit. Standard "Weighted NMF" extension of Lee & Seung's multiplicative
    update algorithm (the unweighted updates ``H *= (WᵀX)/(WᵀWH)``,
    ``W *= (XHᵀ)/(WHHᵀ)`` generalize directly by inserting the mask).

    Internal helper — not part of the public API. Used by
    :func:`nmf_rank_cv_sweep` to measure genuine held-out reconstruction
    error, which requires entries invisible to fitting (unlike sklearn's
    ``NMF``, which has no notion of excluding individual entries).

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

    Unlike :func:`nmf_rank_mse_sweep` (which only measures fit error on the
    data the model was trained on, so error can only ever improve with more
    components), this measures generalization: a random fraction of
    individual matrix *entries* (not whole rows) is held out from fitting
    per replicate, using :func:`_masked_nmf`, and reconstruction error is
    measured on those held-out entries. This can reveal overfitting — held-
    out error bottoming out and then rising with rank, even as training
    error keeps falling — which a train-only sweep cannot show.

    Note on mechanism: entries are masked, not rows. An earlier version of
    this function held out whole rows and projected them onto a fixed
    learned basis (as ``sklearn.decomposition.NMF.transform`` does) —
    that approach turned out *not* to reveal overfitting in practice: a
    held-out row can always be re-fit about as well as a training row by a
    sufficiently flexible basis, regardless of whether that basis is
    itself overfit, since the row is free to choose new activation weights
    for it. Entry-masking avoids this: a component that merely memorizes
    one training row's idiosyncrasies can't help reconstruct a masked
    entry *within that same row*, because the row's other visible entries
    constrain the fit. This is also why sklearn's off-the-shelf ``NMF``
    isn't used here — it has no notion of excluding individual entries
    from its loss, hence :func:`_masked_nmf`.

    The same random mask is reused across all ``ranks`` within a given
    replicate, so ranks are compared on the same held-out entries within
    that replicate (a fixed, fair split, not an independent one per rank).

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
        Number of independent top-level seeds to repeat the whole
        ``n_replicates``-run sweep under. ``n_replicates`` alone only
        draws multiple masks/initializations from the *one* stream
        derived from ``random_state``; it doesn't test whether the
        result is sensitive to that particular seed choice. Setting
        ``n_seeds > 1`` additionally varies the top-level seed itself
        (each seed offset from ``random_state`` by a large stride, so
        their streams don't overlap) and adds a ``seed`` column to the
        output identifying which top-level seed produced each row. The
        default of 1 reproduces the original single-seed output exactly
        (no ``seed`` column) — this is an opt-in robustness check, not a
        replacement for choosing ``n_replicates``.
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
        Columns ``rank``, ``rep``, ``train_mse``, ``test_mse`` — one row
        per (rank, replicate) combination — plus a leading ``seed``
        column when ``n_seeds > 1``. Does not pick a rank automatically;
        inspect or plot (see :func:`plot_nmf_rank_cv`) to choose one.
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
    """Plot :func:`nmf_rank_cv_sweep` output: per-fold train/test MSE vs. rank.

    Requires matplotlib (``pip install facedyn[viz]``).

    Each replicate is drawn as a faint individual line; a bold summary line
    across replicates is drawn on top, with a vertical marker at the rank
    with the lowest summary ``test_mse``. Replicates are plotted as separate
    line groups deliberately — an earlier version of this same plot (in this
    project's R exploration) grouped only by train/test color and not by
    replicate, which zigzagged between replicates' values at each rank and
    produced confusing breaks wherever a replicate had a missing value. When
    ``result`` has a ``seed`` column (see :func:`nmf_rank_cv_sweep`'s
    ``n_seeds``), lines are grouped by ``(seed, rep)`` for the same reason —
    otherwise two different seeds' ``rep=0`` rows would be zigzagged
    together as if they were one trajectory.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nmf_rank_cv_sweep`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    robust : bool, default True
        :func:`_masked_nmf`'s multiplicative updates can occasionally settle
        on a degenerate fit — most often at higher ranks with limited data —
        where one component's weight and loading blow up in a way that
        cancels out on the entries used for fitting but explodes on the
        held-out ones. That single replicate can then be orders of
        magnitude larger than every other value, which (a) drags a plain
        mean summary line far from where most replicates actually sit, and
        (b) forces a linear y-axis that compresses all the genuinely
        informative variation near zero. When ``True``, the summary line
        uses the **median** across replicates instead of the mean, and the
        y-axis is scaled to a robust range (see ``outlier_z``); any point
        outside that range is still drawn — pinned to the top of the axis
        as a ``^`` marker, annotated with its true value — rather than
        silently dropped. Set to ``False`` to fall back to a plain mean and
        full autoscaling.
    outlier_z : float, default 3.5
        Robust z-score (based on median absolute deviation across all
        train/test MSE values) beyond which a point is treated as an
        outlier for axis-scaling purposes. Only used when ``robust=True``.
    save_path : str or pathlib.Path, optional
        If given, save the figure to this filename (e.g. ``"rank_cv.pdf"``
        or ``"rank_cv.png"``) -- format is inferred from the extension.
        Not saved if left as ``None`` (the default).
    output_dir : str or pathlib.Path, default "."
        Directory ``save_path`` is written into (created if it doesn't
        already exist). Ignored if ``save_path`` is None.
    dpi : int, default 300
        Resolution used when saving to a raster format (e.g. PNG); ignored
        for vector formats (e.g. PDF) and if ``save_path`` is None.

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
    """Cophenetic correlation per rank - an NMF stability diagnostic for
    rank selection, independent of reconstruction error.

    Not part of the original R pipeline (no equivalent computation exists
    there - see ``PIPELINE.md``); a new addition, since the original
    analysis's rank choice combined a single train-only MSE-vs-rank curve
    with alignment to prior work, and this package's own
    :func:`nmf_rank_cv_sweep` re-checked that with held-out reconstruction
    error but is still fundamentally an accuracy-based criterion.
    Implements the complementary, stability-based criterion from Brunet et
    al. (2004), "Metagenes and molecular pattern discovery using matrix
    factorization": for a given rank, fit NMF ``n_runs`` times from
    independent random initializations, and for each run assign every row
    to its *dominant component* (the column with the largest weight in
    that row's slice of ``W``). Average, across runs, how often each pair
    of rows landed in the same dominant component into a single
    "consensus matrix" - entries near 1 mean that pair was assigned
    together almost every run, entries near 0 mean almost never. The
    cophenetic correlation coefficient then measures how cleanly
    block-structured that consensus matrix is (hierarchical clustering on
    ``1 - consensus`` as a distance, then correlating the dendrogram's
    implied distances against the actual ones): a rank whose row-clustering
    is genuinely stable across random restarts scores close to 1; one
    where restarts disagree scores lower. This typically starts dropping
    once rank exceeds the number of genuinely distinct, well-separated
    components in the data — a signal reconstruction error alone can miss,
    since more components can always reduce reconstruction error even as
    their row-assignments become unstable.

    Deliberately uses ``init="random"`` rather than the rest of this
    module's usual ``"nndsvda"`` (see :func:`nmf_rank_mse_sweep`,
    :class:`NMFDecomposer`): ``nndsvda`` is a near-deterministic SVD-based
    init, so repeated runs would mostly converge to the same solution and
    the consensus matrix would trivially look stable regardless of rank —
    genuinely random restarts are the point here.

    **Cost warning**: builds an ``n_samples x n_samples`` consensus matrix
    and runs hierarchical clustering on it — both ``O(n_samples^2)`` in
    memory/time. This is meant for a manageable subsample (a few hundred
    to a couple of thousand rows), not the full ~90k-row training set;
    subsample ``X`` yourself before calling this, the same way the
    ``facedyn`` demo notebook subsamples for :func:`nmf_rank_cv_sweep`.

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
        Number of parallel worker processes for the ``len(ranks) *
        n_runs`` independent NMF fits (via ``joblib``, following
        scikit-learn's own convention: ``None``/``1`` = sequential, ``-1``
        = all cores). Purely a wall-clock optimization — every fit is
        seeded independently of the others regardless of execution order,
        so results are identical to the sequential default for any value
        of ``n_jobs``. The O(n_samples^2) consensus/linkage cost (see
        below) is unaffected either way — this only parallelizes the NMF
        refitting, which is what actually dominates runtime at the
        subsample sizes this function is meant for. Only worth setting
        for larger workloads: process start-up overhead can make small
        ones (a handful of ranks/runs on a few hundred rows) *slower*
        under parallel execution than sequential.
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
    """Plot :func:`nmf_cophenetic_correlation` output: cophenetic
    correlation vs. rank.

    Requires matplotlib (``pip install facedyn[viz]``).

    A complementary view to :func:`plot_nmf_rank_cv`: that plot measures
    fit accuracy, this measures how *stable* each rank's row-clustering is
    across random restarts. Look for a rank that's still close to 1 but
    about to drop at the next rank tried - that's the point beyond which
    additional components stop corresponding to a distinct, reproducible
    pattern rather than an unstable split of an existing one.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nmf_cophenetic_correlation`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    save_path : str or pathlib.Path, optional
        If given, save the figure to this filename (e.g.
        ``"cophenetic.pdf"`` or ``"cophenetic.png"``) -- format is inferred
        from the extension. Not saved if left as ``None`` (the default).
    output_dir : str or pathlib.Path, default "."
        Directory ``save_path`` is written into (created if it doesn't
        already exist). Ignored if ``save_path`` is None.
    dpi : int, default 300
        Resolution used when saving to a raster format (e.g. PNG); ignored
        for vector formats (e.g. PDF) and if ``save_path`` is None.

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

    Fits on the resolved numeric columns; ``transform`` returns all other
    (metadata) columns unchanged plus new per-row component-activation
    columns. Mirrors R's ``dta_nmf_output`` shape: R's factorisation
    ``A ≈ W · diag(d) · H`` separates shape (unit-normalized W, H) from
    scale (d) for interpretability, which sklearn's NMF doesn't do — its
    solver absorbs scale directly into W/H instead.

    **Validated against real R output** (see
    ``tests/validation/test_nmf_decomposer_validation.py``, comparing
    against ``r_NMF.csv`` / R's real ``dta_nmf_output``): after matching
    components between the two fits (unconstrained NMF only identifies
    components up to a permutation and an arbitrary positive rescaling of
    each component, since ``(W, H)`` and ``(W·S, S⁻¹·H)`` reconstruct the
    same data for any positive diagonal ``S`` — components are matched via
    Hungarian assignment on cross-correlation, the standard technique for
    this ambiguity), each matched pair correlates at >0.999 and is related
    by a clean per-component proportional scale (least-squares slope
    through the origin, <1% relative residual) — strong evidence sklearn
    finds essentially the same 3 components RcppML did. This *contradicts*
    an earlier, untested assumption recorded here that sklearn's
    ``fit_transform`` output would already numerically equal R's
    ``diag(d)``-scaled H with no rescaling needed: empirically, each
    component ends up on its own arbitrary scale (R's real output values
    are ~60-130x larger than sklearn's per component, and that ratio
    differs by component) — proportional per component, not identical.
    Not corrected here since nothing downstream yet depends on matching
    R's absolute scale (representative-AU selection, the next pipeline
    step, only needs each component's *argmax* AU, which is scale-invariant
    per column) — but worth knowing before relying on these activation
    values' absolute magnitude for anything new.

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

    Replicates the original R analysis's basis-matrix figure
    (``NMF::aheatmap`` on ``model_nmf$w`` in ``final_analysis.Rmd``):
    features (rows) against components (columns), no clustering/reordering
    of either axis, sequential color scale.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` or ``fit_transform`` already
        called).
    normalize : bool, default True
        If True (matching the R analysis's *published* figure, built from
        ``apply(model_nmf$w, 2, fn_maxnormalise)``), each component's
        column is independently min-max scaled to ``[0, 1]`` before
        plotting. This isn't just cosmetic: unconstrained NMF only
        identifies components up to an arbitrary positive per-component
        scale (see :class:`NMFDecomposer`'s docstring), so the *raw* basis
        values from two different NMF fits -- even a correct one -- aren't
        expected to land on the same color scale; min-max normalizing each
        column independently removes that ambiguity, which is why it's
        what R's own published figure actually plots, not the raw matrix.
        Set to False to see the untransformed ``components_`` values (only
        meaningfully comparable to another fit of the *same* model, not
        across libraries/re-fits).
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
        If given, save the figure to this filename (e.g. ``"basis.pdf"`` or
        ``"basis.png"``) -- format is inferred from the extension. Not
        saved if left as ``None`` (the default).
    output_dir : str or pathlib.Path, default "."
        Directory ``save_path`` is written into (created if it doesn't
        already exist). Ignored if ``save_path`` is None.
    dpi : int, default 300
        Resolution used when saving to a raster format (e.g. PNG); ignored
        for vector formats (e.g. PDF) and if ``save_path`` is None.

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
    """RMSE/NRMSE/MAE/R² of `reconstructed` vs. `original`, matching
    `final_analysis.Rmd`'s reconstruction-validation formulas exactly
    (~L695-711): NRMSE divides by the original data's own range, and R² is
    ``1 - sum(error**2) / sum((original - mean(original))**2)`` using
    ``original``'s single *scalar* mean over the whole matrix -- R's
    ``mean()`` on a matrix, not a per-column mean -- so this is not simply
    `sklearn.metrics.r2_score` (which defaults to per-column baselines for
    2D input).
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
    *fixed* basis via ``decomposer.transform`` (sklearn's non-negative
    least-squares-equivalent solve for activations given frozen
    ``components_``) and reconstructs via ``activations @ components_``.
    This one code path covers both in-sample data (training reconstruction
    quality) and out-of-sample data (held-out generalisation) -- unlike
    `final_analysis.Rmd`, which hand-rolls a separate per-frame NNLS
    projection loop (~L1128-1198) for the latter since base R has no
    fixed-basis projection primitive; sklearn's `NMF.transform` already
    solves exactly that problem.
    """
    check_is_fitted(decomposer, "components_")
    original = X[decomposer.columns_].to_numpy()
    activation_cols = [f"{decomposer.prefix}{i + 1}" for i in range(decomposer.n_components)]
    activations = decomposer.transform(X)[activation_cols].to_numpy()
    reconstructed = activations @ decomposer.components_
    return original, reconstructed


def nmf_reconstruction_error(decomposer: NMFDecomposer, X: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reconstruction-quality metrics for a fitted decomposer.

    Replicates `final_analysis.Rmd`'s reconstruction-validation section
    (~L695-711): reconstructs `X` from its NMF activations and reports how
    much of the original AU signal survives the compression down to
    ``decomposer.n_components`` components. R² here is exactly "proportion
    of AU signal variance retained by NMF" -- the quantity the original
    analysis found surprisingly low (~0.42 in-sample), motivating the
    later move to representative-AU selection instead of NMF activations
    for downstream feature extraction.

    Works on *any* `X` containing ``decomposer.columns_``: pass the
    training data for in-sample fit quality, or held-out data (a test
    split, or any other video set) for out-of-sample generalisation --
    R's separate "test set" and "40 held-out videos" sections are both
    just this same call with different data, not separate code paths.

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
        ``"NRMSE"``, ``"MAE"``, ``"R2"`` -- matching
        `final_analysis.Rmd`'s ``dta_nmf_recon_err`` table shape.
    """
    original, reconstructed = _reconstruct(decomposer, X)
    metrics = _reconstruction_metrics(original, reconstructed)
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def nmf_reconstruction_r2_per_au(
    decomposer: NMFDecomposer, X: pd.DataFrame, labels: list[str] | None = None
) -> pd.DataFrame:
    """Per-AU reconstruction R² for a fitted decomposer.

    Replicates `final_analysis.Rmd`'s per-AU breakdown (~L4225-4260,
    ``r2_vec``/``dta_r2``): the same R² formula as
    :func:`nmf_reconstruction_error`, computed independently per AU column
    rather than aggregated over the whole matrix -- shows *which* AUs the
    NMF compression preserves well vs. poorly (the original analysis found
    AU07 best, ~0.887-0.899, and AU23 worst, ~0.010-0.011).

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` already called).
    X : pd.DataFrame
        Data containing ``decomposer.columns_`` (plus any other columns,
        which are ignored).
    labels : list of str, optional
        Row labels, one per factorized column, in the same order as
        ``decomposer.columns_``. Defaults to ``decomposer.columns_``
        itself; pass ``facedyn.humanise_au_labels(decomposer.columns_)``
        for readable AU names instead of raw column names.

    Returns
    -------
    pd.DataFrame
        Columns ``au`` and ``r2``, one row per factorized column, sorted
        descending by ``r2`` -- matching
        `final_analysis.Rmd`'s ``dplyr::arrange(desc(...))``.
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
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot original vs. reconstructed AU activation over time, for one AU
    and one video.

    Replicates `final_analysis.Rmd`'s "Visualisation of Reconstruction
    Quality" plot (~L738-780): a quick visual sanity check of what the R²
    numbers from :func:`nmf_reconstruction_error` mean in practice for a
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
        ``decomposer.columns_[0]``. (`final_analysis.Rmd` hardcoded AU07
        for this plot with no stated reason -- any factorized AU can be
        requested here instead.)
    video_id : optional
        Value of ``group_col`` identifying which video's rows to plot.
        Defaults to the first unique value in ``group_col``.
    group_col : str, default "video_filename"
        Column identifying which rows belong to which video.
    frame_col : str, default "frame"
        Column used for the x-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    save_path : str or pathlib.Path, optional
        If given, save the figure to this filename -- format is inferred
        from the extension. Not saved if left as ``None`` (the default).
    output_dir : str or pathlib.Path, default "."
        Directory ``save_path`` is written into (created if it doesn't
        already exist). Ignored if ``save_path`` is None.
    dpi : int, default 300
        Resolution used when saving to a raster format (e.g. PNG); ignored
        for vector formats (e.g. PDF) and if ``save_path`` is None.

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
    ax.set_ylabel(au)
    ax.set_title(f"{au} - Original vs. Reconstructed ({video_id})")
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
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot original vs. reconstructed activation for the best- and
    worst-reconstructed AUs, side by side, for one video.

    Replicates `final_analysis.Rmd`'s "Reconstruction Pt2" plot
    (~L4262-4310): identifies the highest- and lowest-R² AUs (via
    :func:`nmf_reconstruction_r2_per_au`) and plots each as
    :func:`plot_nmf_reconstruction` would, so the best- and worst-case
    reconstructions can be compared directly.

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
    ax : sequence of matplotlib.axes.Axes, optional
        Two Axes (best, worst) to draw on. A new ``1 x 2`` grid is created
        if not given.
    save_path : str or pathlib.Path, optional
        If given, save the figure to this filename -- format is inferred
        from the extension. Not saved if left as ``None`` (the default).
    output_dir : str or pathlib.Path, default "."
        Directory ``save_path`` is written into (created if it doesn't
        already exist). Ignored if ``save_path`` is None.
    dpi : int, default 300
        Resolution used when saving to a raster format (e.g. PNG); ignored
        for vector formats (e.g. PDF) and if ``save_path`` is None.

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
        _, axes = plt.subplots(1, 2, figsize=(9, 4))
    else:
        axes = np.atleast_1d(ax)
        if len(axes) != 2:
            raise ValueError(f"ax must have 2 entries, got {len(axes)}.")

    for target_ax, au, role in zip(axes, [best_au, worst_au], ["Best", "Worst"]):
        plot_nmf_reconstruction(
            decomposer, X, au=au, video_id=video_id,
            group_col=group_col, frame_col=frame_col, ax=target_ax,
        )
        target_ax.set_title(f"{role}: {target_ax.get_title()}")

    save_figure(axes[0].figure, save_path, output_dir, dpi)
    return list(axes)


def plot_nmf_reconstruction_r2_bar(
    r2_table: pd.DataFrame,
    ax=None,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Bar chart of per-AU reconstruction R² (:func:`nmf_reconstruction_r2_per_au`'s output).

    Not part of `final_analysis.Rmd` (it only tabulates ``dta_r2``, never
    plots it) -- added since a sorted bar chart is the most direct single
    view of which AUs the NMF compression retains vs. loses, the exact
    question the original analysis's reconstruction check was trying to
    answer.

    Requires matplotlib (``pip install facedyn[viz]``).

    Parameters
    ----------
    r2_table : pd.DataFrame
        Output of :func:`nmf_reconstruction_r2_per_au`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    save_path : str or pathlib.Path, optional
        If given, save the figure to this filename -- format is inferred
        from the extension. Not saved if left as ``None`` (the default).
    output_dir : str or pathlib.Path, default "."
        Directory ``save_path`` is written into (created if it doesn't
        already exist). Ignored if ``save_path`` is None.
    dpi : int, default 300
        Resolution used when saving to a raster format (e.g. PNG); ignored
        for vector formats (e.g. PDF) and if ``save_path`` is None.

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
