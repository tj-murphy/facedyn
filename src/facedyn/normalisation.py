"""Z-score normalisation with a non-negativity shift, for NMF input."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ZScoreShiftNormalizer(BaseEstimator, TransformerMixin):
    """Z-score columns using train-set-learned mean/SD, then shift to non-negative.

    Replicates the R pipeline's ``fn_z_score_shift``. Two behaviors here are
    less obvious than they look, confirmed against the R source (both
    differ from how the originating paper's prose describes this step,
    which is imprecise — this replicates the code, since that's what
    produced the published results):

    - Mean and SD are pooled across *all* rows passed to ``fit`` (typically
      the whole training set: every video, every frame) rather than
      computed per video. One pair of numbers per column.
    - The non-negativity shift (subtracting a column's minimum) is **not**
      a frozen fit-time parameter. It's recomputed fresh every time
      ``transform`` is called, on whatever data is passed in — so
      transforming the training set, the test set, and any later new data
      each get their own independently-anchored shift. Only the mean/SD
      used for the z-score itself are frozen from ``fit``. This guarantees
      the output is non-negative regardless of how extreme new data is
      compared to what was seen during training.

    Parameters
    ----------
    columns : list of str, optional
        Explicit columns to normalize. If not given, columns are selected
        via ``column_pattern``.
    column_pattern : str, default r"^smth_"
        Regex used to select columns when ``columns`` is not given. The
        default matches any column produced by :class:`RollingSmoother`
        (its default ``prefix``), not just AU-named ones.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        column_pattern: str = r"^smth_",
    ):
        self.columns = columns
        self.column_pattern = column_pattern

    def fit(self, X: pd.DataFrame, y=None) -> "ZScoreShiftNormalizer":
        self.columns_ = self._resolve_columns(X)
        self.means_ = X[self.columns_].mean()
        sds = X[self.columns_].std()  # ddof=1, matches R's sd()
        self.sds_ = sds.mask(sds.isna() | (sds == 0), 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "means_")
        result = X.copy()
        for col in self.columns_:
            z = (X[col] - self.means_[col]) / self.sds_[col]
            shifted = z - z.min(skipna=True)
            result[col] = shifted.fillna(0)
        return result

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return list(self.columns)
        pattern = re.compile(self.column_pattern)
        return [col for col in X.columns if pattern.search(col)]
