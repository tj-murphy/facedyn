"""Unit tests for the classifier registry.

These check the wiring -- that each model is built with the parameters the
R analysis used, that scaling happens inside the pipeline (so it is refit
per CV fold rather than leaking test-fold statistics), and that the models
behave like ordinary sklearn estimators. Whether the numbers match R is a
separate question, answered in `tests/validation/`.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from facedyn.classifiers import (
    CLASSIFIER_NAMES,
    PAPER1_PARAM_GRIDS,
    PAPER1_PARAMS,
    fit_classifiers,
    make_classifier,
    make_classifiers,
)


def make_xy(n: int = 60, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # A learnable signal, so `predict` is not degenerate.
    y = np.where(X["f0"] + 0.5 * rng.normal(size=n) > 0, "fake", "real")
    return X, y


@pytest.mark.parametrize("name", CLASSIFIER_NAMES)
def test_every_classifier_scales_before_the_model(name):
    pipeline = make_classifier(name)
    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("random_forest", RandomForestClassifier),
        ("c50_substitute", AdaBoostClassifier),
        ("svm_rbf", SVC),
        ("logistic_regression", LogisticRegression),
    ],
)
def test_registry_builds_the_expected_estimator(name, expected_type):
    assert isinstance(make_classifier(name).named_steps["classifier"], expected_type)


def test_paper1_preset_applies_the_r_hyperparameters():
    forest = make_classifier("random_forest").named_steps["classifier"]
    assert forest.n_estimators == 500
    assert forest.max_features == 4  # R's mtry = 4

    boosted = make_classifier("c50_substitute").named_steps["classifier"]
    assert boosted.n_estimators == 20  # R's trials = 20

    svm = make_classifier("svm_rbf").named_steps["classifier"]
    assert svm.C == 0.25  # caret's svmRadial grid at tuneLength = 1
    assert svm.probability is True


def test_default_preset_leaves_sklearn_defaults_alone():
    forest = make_classifier("random_forest", preset="default").named_steps["classifier"]
    assert forest.n_estimators == RandomForestClassifier().n_estimators
    assert forest.max_features == RandomForestClassifier().max_features


def test_logistic_regression_is_unregularised():
    model = make_classifier("logistic_regression").named_steps["classifier"]
    # Expressed as `penalty=None` up to scikit-learn 1.7 and as `C=inf`
    # from 1.8; either way there must be no shrinkage.
    assert getattr(model, "penalty", None) is None or model.C == np.inf


def test_overrides_win_over_the_preset():
    forest = make_classifier("random_forest", n_estimators=7).named_steps["classifier"]
    assert forest.n_estimators == 7
    assert forest.max_features == 4  # preset still applied elsewhere


def test_unknown_classifier_names_are_rejected():
    with pytest.raises(ValueError, match="Unknown classifier"):
        make_classifier("xgboost")


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="preset must be"):
        make_classifier("random_forest", preset="tuned")


def test_param_grids_address_the_pipeline_step():
    for name, grid in PAPER1_PARAM_GRIDS.items():
        model = make_classifier(name)
        for key in grid:
            assert key.startswith("classifier__")
            model.set_params(**{key: grid[key][0]})  # settable, not just a string


def test_make_classifiers_returns_all_names_by_default():
    models = make_classifiers()
    assert list(models) == list(CLASSIFIER_NAMES)
    assert set(PAPER1_PARAMS) == set(CLASSIFIER_NAMES)


@pytest.mark.filterwarnings("ignore:Setting penalty=None")
def test_fit_classifiers_fits_every_model_on_string_labels():
    X, y = make_xy()
    models = fit_classifiers(make_classifiers(random_state=0), X, y)

    for name, model in models.items():
        predictions = model.predict(X)
        assert set(predictions) <= {"fake", "real"}
        assert set(model.classes_) == {"fake", "real"}
        assert model.predict_proba(X).shape == (len(X), 2)


def test_seeded_models_are_reproducible():
    X, y = make_xy()
    a = make_classifier("random_forest", random_state=3, n_estimators=25).fit(X, y)
    b = make_classifier("random_forest", random_state=3, n_estimators=25).fit(X, y)

    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))


def test_mtry_of_four_still_works_with_fewer_features():
    # R's mtry=4 was chosen for an 8-feature table; scikit-learn clamps it
    # rather than erroring, so the preset stays usable on smaller inputs.
    X, y = make_xy(n_features=2)
    model = make_classifier("random_forest", random_state=0, n_estimators=10).fit(X, y)
    assert model.predict(X).shape == (len(X),)
