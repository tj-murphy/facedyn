"""Accumulated local effects (ALE): feature_effect_ale and friends.

The estimator is checked against cases where ALE has a closed form:

- a hand-computed curve for f(x) = x^2 on five points;
- for a linear model f(x) = b * x_j + ..., the uncentred curve is exactly
  b * (z - z_0), and the centred curve is close to b * (z - mean(x_j));
- a model that ignores x_j has an ALE of exactly zero;
- centring makes the curve's data-weighted mean exactly zero;
- per-class probability curves of a multiclass model sum to zero.

The last test shows why ALE and not partial dependence: with two strongly
correlated features and a model that uses only one of them, ALE of the
unused feature stays near zero.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from facedyn.au_labels import humanise_feature_label
from facedyn.effects import (
    ale_grid,
    feature_effect_ale,
    feature_effect_ale_cv,
    plot_feature_effect,
    summarise_feature_effects,
)


class FunctionModel(BaseEstimator, RegressorMixin):
    """A 'fitted' regressor wrapping a fixed function of a DataFrame."""

    def __init__(self, fn=None):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.fn(pd.DataFrame(X).reset_index(drop=True))


def centred_mean(curve: pd.DataFrame) -> float:
    """Data-weighted mean of a centred curve: midpoint height per interval."""
    effect, n = curve["effect"].to_numpy(), curve["n"].to_numpy()
    midpoints = (effect[:-1] + effect[1:]) / 2
    return float((n[1:] * midpoints).sum() / n[1:].sum())


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 10, size=(2000, 3)), columns=["a", "b", "c"])
    y = 2.0 * X["a"] - 3.0 * X["b"] + 0.5 * X["c"]
    return X, y


# --------------------------------------------------------------------------
# Estimator correctness
# --------------------------------------------------------------------------

def test_hand_computed_curve_for_square():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    model = FunctionModel(lambda d: d["x"].to_numpy() ** 2)
    curve = feature_effect_ale(model, X, "x", grid_size=4)
    # Edges 0..4; intervals hold {0, 1}, {2}, {3}, {4}; local effects
    # 1, 3, 5, 7 -> g = 0, 1, 4, 9, 16; centre = (2*.5 + 2.5 + 6.5 + 12.5)/5.
    np.testing.assert_allclose(curve["value"], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(curve["effect"], [-4.5, -3.5, -0.5, 4.5, 11.5])
    np.testing.assert_array_equal(curve["n"], [0, 2, 1, 1, 1])
    assert curve["class"].isna().all()


def test_linear_model_has_closed_form_effect(linear_data):
    X, y = linear_data
    model = LinearRegression().fit(X, y)
    curves = feature_effect_ale(model, X, ["a", "b"], grid_size=25)
    for feature, slope in [("a", 2.0), ("b", -3.0)]:
        curve = curves[curves["feature"] == feature]
        z, effect = curve["value"].to_numpy(), curve["effect"].to_numpy()
        # Exact: differences between grid values follow the slope.
        np.testing.assert_allclose(effect - effect[0], slope * (z - z[0]), atol=1e-8)
        # Approximately b * (x - mean): centring uses interval midpoints.
        np.testing.assert_allclose(effect, slope * (z - X[feature].mean()), atol=0.05 * abs(slope))


def test_ignored_feature_has_zero_effect(linear_data):
    X, _ = linear_data
    model = FunctionModel(lambda d: 3.0 * d["b"].to_numpy() + np.sin(d["c"].to_numpy()))
    curve = feature_effect_ale(model, X, "a")
    np.testing.assert_allclose(curve["effect"], 0.0, atol=1e-12)


@pytest.mark.parametrize("grid_size", [1, 5, 20, 60])
def test_centred_curve_has_zero_mean_over_data(linear_data, grid_size):
    X, _ = linear_data
    model = FunctionModel(lambda d: np.exp(d["a"].to_numpy() / 4) * d["b"].to_numpy())
    curve = feature_effect_ale(model, X, "a", grid_size=grid_size)
    assert abs(centred_mean(curve)) < 1e-9
    assert curve["n"].sum() == len(X)


def test_ale_is_not_fooled_by_correlated_features():
    # b is almost a copy of a; the model uses only a. Partial dependence of
    # b would be flat too here, but ALE of a must recover the slope while
    # only evaluating the model near the observed (a, b) pairs.
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 3000)
    X = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.01, size=a.size)})
    model = FunctionModel(lambda d: 4.0 * d["a"].to_numpy())
    curves = feature_effect_ale(model, X, ["a", "b"])
    a_curve = curves[curves["feature"] == "a"]
    assert np.ptp(a_curve["effect"]) == pytest.approx(4.0 * np.ptp(a_curve["value"]))
    np.testing.assert_allclose(curves.loc[curves["feature"] == "b", "effect"], 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------

def test_binary_classifier_traces_positive_class_by_default(linear_data):
    X, y = linear_data
    labels = np.where(y > y.median(), "high", "low")
    model = LogisticRegression(max_iter=1000).fit(X, labels)
    curve = feature_effect_ale(model, X, "a")
    # classes_ sorted: ["high", "low"], so the default is "low".
    assert curve["class"].unique().tolist() == ["low"]
    # a raises y, so it lowers P(low).
    assert curve["effect"].iloc[-1] < curve["effect"].iloc[0]
    both = feature_effect_ale(model, X, "a", classes=["high", "low"])
    high = both.loc[both["class"] == "high", "effect"].to_numpy()
    low = both.loc[both["class"] == "low", "effect"].to_numpy()
    np.testing.assert_allclose(high, -low, atol=1e-10)


def test_multiclass_curves_sum_to_zero(linear_data):
    X, y = linear_data
    labels = pd.cut(y, 3, labels=["angry", "happy", "sad"]).astype(str)
    model = LogisticRegression(max_iter=2000).fit(X, labels)
    curves = feature_effect_ale(model, X, ["a", "b"])
    assert sorted(curves["class"].unique()) == ["angry", "happy", "sad"]
    totals = curves.groupby(["feature", "value"])["effect"].sum()
    np.testing.assert_allclose(totals, 0.0, atol=1e-10)
    only = feature_effect_ale(model, X, "a", classes="sad")
    assert only["class"].unique().tolist() == ["sad"]


def test_numpy_input_passes_arrays_to_the_model(linear_data):
    X, y = linear_data
    model = LinearRegression().fit(X.to_numpy(), y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "X has feature names" warning
        curve = feature_effect_ale(model, X.to_numpy(), 0)
    np.testing.assert_allclose(np.diff(curve["effect"]), 2.0 * np.diff(curve["value"]))


# --------------------------------------------------------------------------
# Grid and errors
# --------------------------------------------------------------------------

def test_ale_grid_merges_ties_and_ignores_nan():
    edges = ale_grid([0, 0, 0, 1, 1, np.nan], grid_size=10)
    np.testing.assert_array_equal(edges, [0.0, 1.0])
    with pytest.raises(ValueError, match="two distinct"):
        ale_grid([2.0, 2.0, np.nan])
    with pytest.raises(ValueError, match="grid_size"):
        ale_grid([0, 1], grid_size=0)


def test_explicit_grid_drops_outside_points_and_is_flat_over_empty_intervals():
    X = pd.DataFrame({"x": [0.0, 0.5, 1.0, 3.5, 4.0, 9.0]})
    model = FunctionModel(lambda d: 2.0 * d["x"].to_numpy())
    curve = feature_effect_ale(model, X, "x", grid={"x": [0, 1, 2, 3, 4]})
    np.testing.assert_array_equal(curve["n"], [0, 3, 0, 0, 2])  # 9.0 dropped
    steps = np.diff(curve["effect"].to_numpy())
    np.testing.assert_allclose(steps, [2.0, 0.0, 0.0, 2.0])
    assert abs(centred_mean(curve)) < 1e-12


def test_errors():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "k": [1.0, 1.0, 1.0]})
    model = FunctionModel(lambda d: d["x"].to_numpy())
    with pytest.raises(KeyError, match="nope"):
        feature_effect_ale(model, X, "nope")
    with pytest.raises(ValueError, match="two distinct"):
        feature_effect_ale(model, X, "k")
    with pytest.raises(ValueError, match="predict_proba"):
        feature_effect_ale(model, X, "x", classes="a")
    clf = LogisticRegression().fit(X[["x"]], ["a", "b", "b"])
    with pytest.raises(ValueError, match="not among"):
        feature_effect_ale(clf, X[["x"]], "x", classes="c")


def test_nan_in_feature_is_skipped():
    X = pd.DataFrame({"x": [0.0, 1.0, np.nan, 2.0, 3.0]})
    model = FunctionModel(lambda d: d["x"].fillna(0).to_numpy())
    curve = feature_effect_ale(model, X, "x", grid_size=3)
    assert curve["n"].sum() == 4
    np.testing.assert_allclose(np.diff(curve["effect"]), np.diff(curve["value"]))


# --------------------------------------------------------------------------
# Feature labels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("column, expected", [
    ("kurtosis_smth_AU12_r", "AU12 - Lip Corner Puller — kurtosis"),
    ("acf10_sumsq_AU06_r", "AU06 - Cheek Raiser — acf10_sumsq"),
    ("acf1_AU06_r", "AU06 - Cheek Raiser — acf1"),
    ("mean_abs_diff_AU01_r", "AU01 - Inner Brow Raiser — mean_abs_diff"),
    ("AU12_r__kurtosis", "AU12 - Lip Corner Puller — kurtosis"),
    ("sd_nmf2", "nmf2 — sd"),
    ("video_filename", "video_filename"),
    ("smth_AU12_r", "AU12 - Lip Corner Puller"),
])
def test_humanise_feature_label(column, expected):
    assert humanise_feature_label(column) == expected


# --------------------------------------------------------------------------
# Across CV folds
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cv_result():
    from test_crossval import fast_kwargs, make_frames

    from facedyn.crossval import cross_validate_pipeline

    frames = make_frames(classes=("a", "b", "c"), n_participants=15, videos_per_participant=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_validate_pipeline(frames, "label", "participant", keep_models=True,
                                       **fast_kwargs())


def _a_feature(result):
    return result.feature_frequency().index[0]


def test_cv_curves_share_one_grid_across_folds(cv_result):
    feature = _a_feature(cv_result)
    curves = feature_effect_ale_cv(cv_result, feature, grid_size=10)
    assert list(curves.columns[:2]) == ["repeat", "fold"]
    assert sorted(curves["class"].unique()) == cv_result.classes
    grids = curves.groupby(["repeat", "fold", "class"])["value"].apply(tuple)
    assert grids.nunique() == 1
    # Each fold's curve is estimated on its own test videos.
    for (rep, fold), part in curves.groupby(["repeat", "fold"]):
        n_test = len(cv_result.fold_models[(rep, fold)]["X_test"])
        assert part.groupby("class")["n"].sum().eq(n_test).all()
    totals = curves.groupby(["repeat", "fold", "value"])["effect"].sum()
    np.testing.assert_allclose(totals, 0.0, atol=1e-10)


def test_cv_matches_single_model_on_the_shared_grid(cv_result):
    feature = _a_feature(cv_result)
    curves = feature_effect_ale_cv(cv_result, feature, grid_size=8, repeat=0)
    (rep, fold), part = next(iter(curves.groupby(["repeat", "fold"])))
    model_info = cv_result.fold_models[(rep, fold)]
    edges = np.sort(part["value"].unique())
    single = feature_effect_ale(model_info["model"], model_info["X_test"], feature,
                                grid={feature: edges})
    merged = part.merge(single, on=["feature", "class", "value"], suffixes=("_cv", ""))
    np.testing.assert_allclose(merged["effect_cv"], merged["effect"])
    assert curves["repeat"].unique().tolist() == [0]


def test_cv_skips_folds_that_did_not_use_the_feature(cv_result):
    feature = _a_feature(cv_result)
    trimmed = type(cv_result)(**{**cv_result.__dict__})
    key = next(iter(cv_result.fold_models))
    info = cv_result.fold_models[key]
    kept = [f for f in info["features"] if f != feature]
    trimmed.fold_models = {**cv_result.fold_models,
                           key: {**info, "features": kept, "X_test": info["X_test"][kept]}}
    curves = feature_effect_ale_cv(trimmed, feature)
    used = set(map(tuple, curves[["repeat", "fold"]].drop_duplicates().to_numpy()))
    assert key not in used
    assert len(used) == len(cv_result.fold_models) - 1

    summary = summarise_feature_effects(curves)
    assert (summary["n_folds"] == len(used)).all()
    assert {"mean", "sd", "min", "max"} <= set(summary.columns)


def test_cv_with_shared_table_and_errors(cv_result):
    feature = _a_feature(cv_result)
    # Folds choose different AUs, so a shared table covers only some of them;
    # use one fold's table and check the folds it doesn't cover are skipped.
    key, info = next((k, v) for k, v in cv_result.fold_models.items()
                     if feature in v["features"])
    X = info["X_test"]
    curves = feature_effect_ale_cv(cv_result, feature, X=X)
    used = curves[["repeat", "fold"]].drop_duplicates()
    covered = [k for k, v in cv_result.fold_models.items()
               if feature in v["features"] and set(v["features"]) <= set(X.columns)]
    assert sorted(map(tuple, used.to_numpy())) == sorted(covered)
    assert key in covered
    # With X, every fold is estimated over all of X.
    assert curves.groupby(["repeat", "fold", "class"])["n"].sum().eq(len(X)).all()
    # Folds whose features X lacks are skipped; none left -> error.
    with pytest.raises(ValueError, match="No fold model"):
        feature_effect_ale_cv(cv_result, feature, X=X[[feature]].rename(columns=str.upper))
    with pytest.raises(ValueError, match="No fold model"):
        feature_effect_ale_cv(cv_result, "not_a_feature")
    no_models = type(cv_result)(**{**cv_result.__dict__, "fold_models": {}})
    with pytest.raises(ValueError, match="keep_models=True"):
        feature_effect_ale_cv(no_models, feature)
    with pytest.raises(ValueError, match="No folds for repeat"):
        feature_effect_ale_cv(cv_result, feature, repeat=99)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib", reason="plots need the viz extra")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


def test_plot_single_model(plt, linear_data, tmp_path):
    X, y = linear_data
    model = LinearRegression().fit(X, y)
    fig = plot_feature_effect(model, X, ["a", "b", "c"], ncols=2, save_path="ale.png",
                              output_dir=tmp_path)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    assert (tmp_path / "ale.png").exists()
    with pytest.raises(ValueError, match="X is required"):
        plot_feature_effect(model, features=["a"])
    with pytest.raises(ValueError, match="features="):
        plot_feature_effect(model, X)


@pytest.mark.parametrize("band, show_folds", [("sd", False), ("range", True), (None, False)])
def test_plot_cv_result(plt, cv_result, band, show_folds):
    fig = plot_feature_effect(cv_result, band=band, show_folds=show_folds, grid_size=8)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert 1 <= len(axes) <= 6
    title = axes[0].get_title()
    assert "folds" in title
    # One mean line per class, plus fold lines when asked for.
    n_lines = len(axes[0].get_lines())
    assert n_lines >= len(cv_result.classes)
    if show_folds:
        assert n_lines > len(cv_result.classes) * 2


def test_plot_rejects_bad_band_and_too_few_axes(plt, cv_result):
    with pytest.raises(ValueError, match="band"):
        plot_feature_effect(cv_result, band="iqr")
    _, ax = plt.subplots()
    features = cv_result.feature_frequency().index[:2].tolist()
    with pytest.raises(ValueError, match="Need 2 axes"):
        plot_feature_effect(cv_result, features=features, axes=[ax])
