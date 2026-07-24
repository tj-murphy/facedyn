"""Representative-AU selection: NMF's basis matrix without the NMF step.

Replicates the original analysis's response to NMF's reconstruction R²
being surprisingly low (see :func:`facedyn.nmf.nmf_reconstruction_error`):
instead of trusting NMF's compressed 3-component activations, take the
single highest-loading AU per component from an already-fitted
:class:`~facedyn.nmf.NMFDecomposer`'s basis matrix, and keep that AU's
*raw* time series as the representative feature for that component.

`final_analysis_NMF_check.Rmd` has no actual argmax code for this -- it
hardcodes the three chosen column names directly into a ``select()``, based
on the researcher eyeballing the basis-matrix heatmap and writing the AUs
down by hand (confirmed by its own intro text: ``Component 1 = AU12,
Component 2 = AU17, Component 3 = AU01``). This module replaces that
eyeball-and-transcribe step with real, tested code.
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
        Row labels, one per factorized column, in the same order as
        ``decomposer.columns_``. If given, an extra ``label`` column is
        added with the readable name for each selected AU; pass
        ``facedyn.humanise_au_labels(decomposer.columns_)`` for FACS-style
        names. The ``au`` column (the actual column name) is always
        returned regardless, since that's what :class:`RepresentativeAUSelector`
        needs to select real data.

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

    Unlike :class:`~facedyn.nmf.NMFDecomposer`, this does not fit its own
    NMF model -- it takes an **already-fitted** decomposer, matching the
    actual workflow this replicates: fit :class:`NMFDecomposer` once,
    inspect its reconstruction quality (:func:`facedyn.nmf.nmf_reconstruction_error`,
    :func:`facedyn.nmf.nmf_reconstruction_r2_per_au`), decide representative-AU
    selection is warranted, then reuse that same fit here -- not a second,
    independent NMF fit that could select different AUs than what was
    actually inspected.

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
        ``selection_``'s ``au`` column, as a plain list -- the columns
        ``transform`` keeps.
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
