"""Non-negative matrix factorisation of normalised AU columns."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF
from sklearn.utils.validation import check_is_fitted


def _resolve_columns(
    X: pd.DataFrame, columns: list[str] | None, column_pattern: str
) -> list[str]:
    if columns is not None:
        return list(columns)
    pattern = re.compile(column_pattern)
    return [col for col in X.columns if pattern.search(col)]


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


def nmf_rank_cv_sweep(
    X: pd.DataFrame,
    ranks: range | list[int] = range(2, 11),
    test_fraction: float = 0.1,
    n_replicates: int = 3,
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
        Number of independent random holdout masks to average over.
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
        per (rank, replicate) combination. Does not pick a rank
        automatically; inspect or plot (see :func:`plot_nmf_rank_cv`) to
        choose one.
    """
    cols = _resolve_columns(X, columns, column_pattern)
    data = X[cols].to_numpy()
    rng = np.random.default_rng(random_state)

    records = []
    for rep in range(n_replicates):
        mask = (rng.random(data.shape) >= test_fraction).astype(float)
        test_mask = 1.0 - mask

        for k in ranks:
            try:
                seed = None if random_state is None else random_state + rep * 1000 + k
                W, H = _masked_nmf(
                    data, mask, n_components=k,
                    max_iter=max_iter, tol=tol, random_state=seed,
                )
                recon = W @ H
                train_mse = ((mask * (data - recon)) ** 2).sum() / mask.sum()
                test_mse = ((test_mask * (data - recon)) ** 2).sum() / test_mask.sum()
            except Exception:
                train_mse, test_mse = np.nan, np.nan

            records.append({
                "rank": k, "rep": rep,
                "train_mse": train_mse, "test_mse": test_mse,
            })

    return pd.DataFrame.from_records(records)


def plot_nmf_rank_cv(result: pd.DataFrame, ax=None):
    """Plot :func:`nmf_rank_cv_sweep` output: per-fold train/test MSE vs. rank.

    Requires matplotlib (``pip install facedyn[viz]``).

    Each replicate is drawn as a faint individual line; the mean across
    replicates is drawn bold, with a vertical marker at the rank with the
    lowest mean ``test_mse``. Replicates are plotted as separate line
    groups deliberately — an earlier version of this same plot (in this
    project's R exploration) grouped only by train/test color and not by
    replicate, which zigzagged between replicates' values at each rank and
    produced confusing breaks wherever a replicate had a missing value.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nmf_rank_cv_sweep`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.

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

    agg = result.groupby("rank")[["train_mse", "test_mse"]].mean().reset_index()
    colors = {"train_mse": "#009E73", "test_mse": "#D55E00"}
    labels = {"train_mse": "Train", "test_mse": "Test"}

    for col, color in colors.items():
        for _, rep_data in result.groupby("rep"):
            rep_data = rep_data.sort_values("rank")
            ax.plot(rep_data["rank"], rep_data[col], color=color, alpha=0.25, linewidth=0.8)
            ax.scatter(rep_data["rank"], rep_data[col], color=color, alpha=0.3, s=15)
        ax.plot(agg["rank"], agg[col], color=color, linewidth=2.2, label=labels[col])
        ax.scatter(agg["rank"], agg[col], color=color, s=40, zorder=3)

    best_rank = agg.loc[agg["test_mse"].idxmin(), "rank"]
    ax.axvline(best_rank, linestyle="--", color="grey", linewidth=1)

    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("MSE")
    ax.set_title("Cross-validated NMF rank selection")
    ax.legend()
    return ax


class NMFDecomposer(BaseEstimator, TransformerMixin):
    """Non-negative matrix factorisation of AU columns via sklearn's NMF.

    Fits on the resolved numeric columns; ``transform`` returns all other
    (metadata) columns unchanged plus new per-row component-activation
    columns. Mirrors R's ``dta_nmf_output`` shape: R's factorisation
    ``A ≈ W · diag(d) · H`` separates shape (unit-normalized W, H) from
    scale (d) for interpretability, which sklearn's NMF doesn't do — its
    solver absorbs scale directly into W/H. That actually simplifies
    things here: sklearn's ``fit_transform`` output already *is* the
    equivalent of R's scaled, transposed H (real activation magnitudes,
    not unit-normalized), so no separate rescaling step is needed.

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
