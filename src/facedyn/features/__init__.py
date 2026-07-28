"""Time-series feature extraction.

Three interchangeable routes from the same wide input shape (see
:func:`~facedyn.features.reshape.reshape_to_wide`):

- :class:`TimeSeriesFeatureExtractor`. facedyn's own 31 features, in pure
  numpy and scipy, with no extra dependencies.
- :class:`RFeatureExtractor` / :func:`cmfts_r_features`. Any R feature
  package via ``rpy2`` (optional extra: ``pip install facedyn[r]``).
- :class:`CallableFeatureExtractor`. Any Python feature function
  (pycatch22, tsfresh, etc).
"""

from facedyn.features.external import (
    CallableFeatureExtractor,
    RFeatureExtractor,
    cmfts_r_features,
)
from facedyn.features.reshape import apply_rowwise, reshape_to_wide
from facedyn.features.timeseries import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    TimeSeriesFeatureExtractor,
    extract_features,
    extract_timeseries_features,
)

__all__ = [
    "reshape_to_wide",
    "apply_rowwise",
    "TimeSeriesFeatureExtractor",
    "extract_features",
    "extract_timeseries_features",
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "RFeatureExtractor",
    "CallableFeatureExtractor",
    "cmfts_r_features",
]
