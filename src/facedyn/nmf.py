"""Non-negative matrix factorisation of normalised AU columns."""

from __future__ import annotations

import re

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
