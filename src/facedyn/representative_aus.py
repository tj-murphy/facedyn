"""Representative-AU selection: NMF's basis matrix without the NMF step.

An alternative to NMF activations when reconstruction R2 is low (see
:func:`facedyn.nmf.nmf_reconstruction_error`). Takes the highest-loading
AU per component from a fitted :class:`~facedyn.nmf.NMFDecomposer`'s basis
matrix and keeps that AU's raw time series as the component's
representative feature.

The original R analysis hardcoded the chosen AUs by eye from the
basis-matrix heatmap. This replaces that step with a tested argmax. See
`PIPELINE.md` step 5.
"""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from facedyn.nmf import NMFDecomposer


def select_representative_aus(
    decomposer: NMFDecomposer, labels: list[str] | None = None
) -> pd.DataFrame:
    """Per NMF component, the single AU with the highest basis-matrix loading.

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` already called).
    labels : list of str, optional
        Row labels, one per factorized column, in ``decomposer.columns_``
        order. Adds a ``label`` column of readable names. Pass
        ``facedyn.humanise_au_labels(decomposer.columns_)`` for FACS-style
        names. The ``au`` column is always returned either way.

    Returns
    -------
    pd.DataFrame
        Columns ``component`` (``nmf1``, ``nmf2``, ... using
        ``decomposer.prefix``) and ``au``, one row per component, plus
        ``label`` if ``labels`` was given.
    """
    check_is_fitted(decomposer, "components_")
    # Argmax per component (row) -- invariant to NMF's per-component
    # positive-rescaling ambiguity (see NMFDecomposer's docstring), so it
    # doesn't matter whether components_ is normalized first.
    best_idx = decomposer.components_.argmax(axis=1)
    selected = [decomposer.columns_[i] for i in best_idx]

    if len(set(selected)) < len(selected):
        warnings.warn(
            f"The same AU was selected as representative for more than one "
            f"component: {selected}. Each component will still get its own "
            f"row below, but downstream code selecting these columns will "
            f"see fewer than {decomposer.n_components} distinct AUs.",
            stacklevel=2,
        )

    result = {
        "component": [f"{decomposer.prefix}{i + 1}" for i in range(decomposer.n_components)],
        "au": selected,
    }
    if labels is not None:
        result["label"] = [labels[i] for i in best_idx]
    return pd.DataFrame(result)


class RepresentativeAUSelector(BaseEstimator, TransformerMixin):
    """Replace NMF activation columns with each component's representative raw AU.

    Takes an already-fitted decomposer rather than fitting its own, so the
    AUs selected here are the ones from the fit you inspected. A second
    independent fit could select different AUs.

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` already called). Its basis matrix
        determines which AU is representative for each component.

    Attributes
    ----------
    selection_ : pd.DataFrame
        Output of :func:`select_representative_aus` for ``decomposer``.
    selected_columns_ : list of str
        ``selection_``'s ``au`` column as a plain list. These are the
        columns ``transform`` keeps.
    """

    def __init__(self, decomposer: NMFDecomposer):
        self.decomposer = decomposer

    def fit(self, X: pd.DataFrame, y=None) -> "RepresentativeAUSelector":
        self.selection_ = select_representative_aus(self.decomposer)
        self.selected_columns_ = list(self.selection_["au"])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_columns_")
        metadata = X.drop(columns=self.decomposer.columns_).reset_index(drop=True)
        representative = X[self.selected_columns_].reset_index(drop=True)
        return pd.concat([metadata, representative], axis=1)
