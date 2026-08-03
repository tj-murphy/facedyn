"""Unit tests for the external-extractor bridges.

The R side is exercised against real R in
`tests/validation/test_cmfts_r_bridge_validation.py`; here it is only
checked that a missing rpy2 or a missing R package fails with an
actionable message rather than a bare ImportError.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from facedyn.features.external import (
    CallableFeatureExtractor,
    RFeatureExtractor,
    _as_feature_series,
)


def make_wide(n_rows: int = 3, n_frames: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "video_filename": [f"v{i}" for i in range(n_rows)],
        "series": ["smth_AU01_r"] * n_rows,
        **{f"fr_{j + 1}": rng.random(n_rows) for j in range(n_frames)},
    })


# --------------------------------------------------------------------------
# return-value normalisation
# --------------------------------------------------------------------------


def test_as_feature_series_accepts_a_plain_mapping():
    result = _as_feature_series({"a": 1.0, "b": 2.0})
    pd.testing.assert_series_equal(result, pd.Series({"a": 1.0, "b": 2.0}))


def test_as_feature_series_accepts_pycatch22_style_names_values_dict():
    """``pycatch22.catch22_all`` returns exactly this shape."""
    result = _as_feature_series({"names": ["a", "b"], "values": [1.0, 2.0]})
    assert list(result.index) == ["a", "b"]
    assert list(result) == [1.0, 2.0]


def test_as_feature_series_accepts_a_names_values_tuple():
    result = _as_feature_series((["a", "b"], [1.0, 2.0]))
    assert list(result.index) == ["a", "b"]


def test_as_feature_series_passes_a_series_through_unchanged():
    series = pd.Series({"a": 1.0})
    assert _as_feature_series(series) is series


def test_as_feature_series_labels_a_bare_sequence_with_feature_names():
    result = _as_feature_series([1.0, 2.0], feature_names=["a", "b"])
    assert list(result.index) == ["a", "b"]


def test_as_feature_series_rejects_an_unlabelled_bare_sequence():
    with pytest.raises(TypeError, match="no\n?\\s*names|bare sequence"):
        _as_feature_series([1.0, 2.0])


def test_as_feature_series_rejects_mismatched_feature_names_length():
    with pytest.raises(ValueError, match="3 entries but the extractor returned 2"):
        _as_feature_series([1.0, 2.0], feature_names=["a", "b", "c"])


# --------------------------------------------------------------------------
# CallableFeatureExtractor
# --------------------------------------------------------------------------


def _stub_features(row):
    return {"total": float(np.sum(row)), "n": float(len(row))}


def _stub_bare(row):
    return [float(np.sum(row)), float(len(row))]


def test_callable_extractor_carries_metadata_and_appends_features():
    wide = make_wide()

    result = CallableFeatureExtractor(_stub_features).fit_transform(wide)

    assert list(result.columns) == ["video_filename", "series", "total", "n"]
    assert (result["n"] == 10).all()
    np.testing.assert_allclose(
        result["total"].to_numpy(),
        wide[[c for c in wide.columns if c.startswith("fr_")]].to_numpy().sum(axis=1),
    )


def test_callable_extractor_labels_unnamed_output_via_feature_names():
    wide = make_wide()

    result = CallableFeatureExtractor(
        _stub_bare, feature_names=["total", "n"]
    ).fit_transform(wide)

    assert list(result.columns) == ["video_filename", "series", "total", "n"]


def test_callable_extractor_reports_feature_names_out():
    extractor = CallableFeatureExtractor(_stub_features)
    extractor.fit_transform(make_wide())

    assert list(extractor.get_feature_names_out()) == ["total", "n"]


@pytest.mark.slow  # 6 s: starts a joblib pool
def test_callable_extractor_parallel_matches_serial():
    wide = make_wide(n_rows=4)

    pd.testing.assert_frame_equal(
        CallableFeatureExtractor(_stub_features, n_jobs=1).fit_transform(wide),
        CallableFeatureExtractor(_stub_features, n_jobs=2).fit_transform(wide),
    )


def test_callable_extractor_requires_fit_before_transform():
    with pytest.raises(NotFittedError):
        CallableFeatureExtractor(_stub_features).transform(make_wide())


def test_callable_extractor_output_matches_builtin_extractor_shape():
    """The whole point of the bridge: same input, same output contract as
    facedyn's own extractor, so the two are interchangeable downstream."""
    from facedyn.features.timeseries import TimeSeriesFeatureExtractor

    wide = make_wide(n_frames=60)

    builtin = TimeSeriesFeatureExtractor(n_jobs=1).fit_transform(wide)
    bridged = CallableFeatureExtractor(_stub_features).fit_transform(wide)

    metadata = ["video_filename", "series"]
    assert list(builtin.columns[: len(metadata)]) == metadata
    assert list(bridged.columns[: len(metadata)]) == metadata
    pd.testing.assert_frame_equal(builtin[metadata], bridged[metadata])


# --------------------------------------------------------------------------
# RFeatureExtractor
# --------------------------------------------------------------------------


HAS_RPY2 = importlib.util.find_spec("rpy2") is not None


@pytest.mark.skipif(HAS_RPY2, reason="rpy2 is installed, so the guidance path is unreachable")
def test_r_extractor_missing_rpy2_message_names_both_requirements():
    """Two independent things can be missing -- the Python package and the
    R package -- so the error has to say which, and how to fix it."""
    with pytest.raises(ImportError) as exc:
        RFeatureExtractor("cmfts").fit_transform(make_wide())

    message = str(exc.value)
    assert "facedyn[r]" in message
    assert "R installation" in message
    assert "cmfts" in message


@pytest.mark.skipif(not HAS_RPY2, reason="needs rpy2")
def test_r_extractor_missing_r_package_message_is_actionable():
    with pytest.raises(ImportError, match="could not be loaded"):
        RFeatureExtractor("definitelynotarealrpackage").fit_transform(make_wide())


def test_r_extractor_requires_fit_before_transform():
    with pytest.raises(NotFittedError):
        RFeatureExtractor("cmfts").transform(make_wide())


def test_r_extractor_stores_r_kwargs_for_the_r_call():
    extractor = RFeatureExtractor("cmfts", "cmfts", n_cores=1, na=True)
    assert extractor.r_kwargs == {"n_cores": 1, "na": True}
