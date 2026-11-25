"""Classifiers for the video-level decision.

Binary or multiclass: every model here accepts any number of classes.

Four models, matching the R analysis of the deepfake study (Murphy, Cook &
Cuve, in prep) that facedyn was first built for: random forest, a documented
C5.0 substitute, an RBF-kernel SVM, and unpenalised logistic regression.
Each is an ordinary :class:`sklearn.pipeline.Pipeline` -- nothing here
wraps or hides scikit-learn, so anything that works with an sklearn
estimator works with these.

Every pipeline standardises its features first, because the R code z-scored
the feature table with ``caret::preProcess(method=c("center","scale"))``
before fitting *any* model, trees included. Keeping the scaler inside the
pipeline also means it is refitted within each cross-validation fold rather
than leaking test-fold statistics into training.

When cross-validating these on repeated-measures or matched-pairs data,
pass ``groups`` (from :func:`facedyn.splitting.groups_from_column` for one
group per participant, or :func:`facedyn.splitting.pair_groups` for matched
pairs) and use :class:`facedyn.splitting.RepeatedStratifiedGroupKFold`.
Better still, start from frame-level data with
:func:`facedyn.crossval.cross_validate_pipeline`, which refits every step
inside each fold.
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

__all__ = [
    "CLASSIFIER_LABELS",
    "CLASSIFIER_NAMES",
    "PAPER1_PARAMS",
    "PAPER1_PARAM_GRIDS",
    "fit_classifiers",
    "make_classifier",
    "make_classifiers",
]

CLASSIFIER_NAMES = (
    "random_forest",
    "c50_substitute",
    "svm_rbf",
    "logistic_regression",
)

#: Human-readable labels, used by the plotting helpers in
#: :mod:`facedyn.evaluation`.
CLASSIFIER_LABELS = {
    "random_forest": "Random Forest",
    "c50_substitute": "C5.0 substitute (AdaBoost)",
    "svm_rbf": "SVM (RBF)",
    "logistic_regression": "Logistic Regression",
}

#: Hyperparameters replicating the deepfake study's fixed-parameter R fits
#: (``caret::trainControl(method="none")``). See each entry's note in
#: :func:`make_classifier`.
PAPER1_PARAMS = {
    "random_forest": {"n_estimators": 500, "max_features": 4},
    "c50_substitute": {"n_estimators": 20},
    "svm_rbf": {"estimator__C": 0.25},
    "logistic_regression": {},
}

#: Grids replicating R's ``repeatedcv`` tuning, keyed for
#: :class:`~sklearn.model_selection.GridSearchCV` on the pipelines this
#: module builds. R tuned with ``tuneLength=5`` on ROC; caret's own
#: ``svmRadial`` grid at that length is ``C = 2^(-2:2)`` with sigma fixed,
#: which is reproduced here. The SVM's ``C`` sits one level down
#: (``estimator__C``) because the SVC is wrapped for Platt scaling.
PAPER1_PARAM_GRIDS = {
    "random_forest": {"classifier__max_features": [1, 2, 4, 6, 8]},
    "c50_substitute": {"classifier__n_estimators": [1, 10, 20, 30, 40]},
    "svm_rbf": {"classifier__estimator__C": [0.25, 0.5, 1.0, 2.0, 4.0]},
    "logistic_regression": {},
}


def _unpenalised_logistic_params() -> dict:
    """How to ask the installed scikit-learn for an unregularised fit.

    R's ``glm(family = binomial)`` is plain maximum likelihood, whereas
    scikit-learn regularises by default. ``penalty=None`` said that up to
    scikit-learn 1.7 and is deprecated from 1.8 in favour of ``C=inf``, so
    pick whichever the installed version wants rather than emitting a
    `FutureWarning` on new versions or a `TypeError` on old ones.

    On scikit-learn 1.8 the ``C=inf`` route still prints "Setting
    penalty=None will ignore the C and l1_ratio parameters" when fitting.
    That warning is spurious -- it comes from scikit-learn translating
    ``C=inf`` into its own internal ``penalty=None`` and then warning as
    though the caller had set it. The fit really is unregularised
    (coefficients match ``penalty=None`` exactly, and are about twice the
    norm of the regularised default).
    """
    version = tuple(int(part) for part in sklearn.__version__.split(".")[:2] if part.isdigit())
    return {"C": np.inf} if version >= (1, 8) else {"penalty": None}


def _build_estimator(name: str, random_state, n_jobs) -> BaseEstimator:
    if name == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    if name == "c50_substitute":
        # AdaBoost's default base learner (a depth-1 tree) is kept, having
        # been measured as the closest stand-in for C5.0 -- see
        # `make_classifier`.
        return AdaBoostClassifier(random_state=random_state)
    if name == "svm_rbf":
        # SVC has no probability of its own; R's `caret` fits Platt scaling
        # for `classProbs = TRUE`. scikit-learn deprecated doing this through
        # `SVC(probability=True)` in 1.9 (removal in 1.11) in favour of this
        # wrapper, which runs the same procedure: a sigmoid fitted to 5-fold
        # cross-validated decision values, with the SVC itself refit on all
        # the data. Used on every scikit-learn version so results do not
        # depend on which one is installed. Deterministic: the folds are
        # unshuffled, so `random_state` is not needed.
        return CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=5, ensemble=False)
    if name == "logistic_regression":
        # Unregularised, matching R's `glm(family = binomial)`; sklearn's
        # default applies L2, which would not be the same model.
        return LogisticRegression(max_iter=1000, **_unpenalised_logistic_params())
    raise ValueError(f"Unknown classifier {name!r}. Choose from {CLASSIFIER_NAMES}.")


def make_classifier(
    name: str,
    preset: str = "paper1",
    random_state: int | None = None,
    n_jobs: int | None = None,
    **overrides,
) -> Pipeline:
    """Build one classifier as a scaler + estimator pipeline.

    Parameters
    ----------
    name : {"random_forest", "c50_substitute", "svm_rbf", "logistic_regression"}
        Which model to build.

        ``"c50_substitute"`` **is a substitution, not a port.** C5.0 has no
        Python implementation, and bridging to R's would make R a runtime
        dependency, which this package deliberately avoids (rpy2 is only
        ever an optional extra). C5.0's boosting is adaptive reweighting of decision
        trees, so :class:`~sklearn.ensemble.AdaBoostClassifier` is the
        closest same-family stand-in, and R's ``trials = 20`` maps directly
        onto ``n_estimators=20``.

        Its base learner is AdaBoost's own default, a depth-1 tree, chosen
        by measurement rather than assumption. Trained on the deepfake
        study's 370 videos with its eight published features and scored on
        the held-out 94, depth-1 reaches ROC-AUC **0.660** against R's real C5.0
        (``model="tree", trials=20, winnow=FALSE``) at **0.651**; depth-2
        reaches 0.697 and depth-3 falls to 0.582. Deeper trees are not a
        better C5.0, they are a different model. That is one dataset's
        worth of evidence, so treat it as a sensible default rather than a
        settled equivalence -- and report these numbers as coming from an
        AdaBoost substitute, not from C5.0.
    preset : {"paper1", "default"}, default "paper1"
        ``"paper1"`` applies :data:`PAPER1_PARAMS`, the fixed
        hyperparameters the deepfake study's R analysis used with
        ``trainControl(method="none")``: 500 trees at ``mtry=4`` for the
        forest (a value chosen for its 8-feature table -- scikit-learn
        clamps it if you have fewer), 20 boosting iterations, and
        ``C=0.25``, which is the single row of `caret`'s own ``svmRadial``
        grid at ``tuneLength=1``. ``"default"`` leaves scikit-learn's
        defaults in place, which is the better starting point on any other
        dataset. ``"paper1"`` stays the default only so that existing code
        keeps reproducing the published numbers.
    random_state : int, optional
        Seeds the estimator. Logistic regression and the SVM are
        deterministic and ignore it.
    n_jobs : int, optional
        Passed to the random forest; the other estimators are
        single-threaded.
    **overrides
        Estimator parameters applied last, overriding the preset --
        ``make_classifier("random_forest", n_estimators=100)``. For
        ``"svm_rbf"``, the SVC's own parameters take an ``estimator__``
        prefix (``estimator__C=1.0``), because the SVC is wrapped in
        :class:`~sklearn.calibration.CalibratedClassifierCV`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps ``"scaler"`` (:class:`~sklearn.preprocessing.StandardScaler`)
        and ``"classifier"``. Set parameters through the pipeline with the
        ``classifier__`` prefix, as :data:`PAPER1_PARAM_GRIDS` does.

    Notes
    -----
    ``"svm_rbf"`` is an RBF :class:`~sklearn.svm.SVC` inside
    :class:`~sklearn.calibration.CalibratedClassifierCV` (sigmoid, 5-fold,
    ``ensemble=False``): Platt scaling, as `caret` uses for class
    probabilities. ``predict`` takes the most probable class, so it always
    agrees with ``predict_proba`` (``SVC(probability=True)`` predicted from
    the raw decision function and could disagree).

    One deliberate difference from R for the SVM: `caret` takes the RBF
    width from ``kernlab::sigest``, a data-driven heuristic with no
    scikit-learn equivalent. These pipelines use ``gamma="scale"``, which
    is also data-driven (``1 / (n_features * X.var())``) but not the same
    estimator, so SVM numbers will differ from R's more than the other
    three models do.
    """
    if preset not in ("paper1", "default"):
        raise ValueError(f"preset must be 'paper1' or 'default', got {preset!r}")

    estimator = _build_estimator(name, random_state, n_jobs)
    if preset == "paper1":
        estimator.set_params(**PAPER1_PARAMS[name])
    if overrides:
        estimator.set_params(**overrides)

    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def make_classifiers(
    names: list[str] | None = None,
    preset: str = "paper1",
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Pipeline]:
    """Build several classifiers at once, keyed by name.

    Parameters
    ----------
    names : list of str, optional
        Defaults to all of :data:`CLASSIFIER_NAMES`.
    preset, random_state, n_jobs
        As :func:`make_classifier`.

    Returns
    -------
    dict of str to Pipeline
        Unfitted pipelines, in the order given.
    """
    names = list(CLASSIFIER_NAMES) if names is None else list(names)
    return {
        name: make_classifier(name, preset=preset, random_state=random_state, n_jobs=n_jobs)
        for name in names
    }


def fit_classifiers(models: dict[str, Pipeline], X, y) -> dict[str, Pipeline]:
    """Fit each model on the same training data.

    Parameters
    ----------
    models : dict of str to estimator
        As returned by :func:`make_classifiers`.
    X : pd.DataFrame or array-like
        Training features -- the selected feature columns only, with no
        metadata columns.
    y : array-like
        Training labels, two classes or more. Strings such as
        ``"happy"``/``"sad"``/``"angry"`` are fine and are carried through to
        :func:`facedyn.evaluation.evaluate_models`.

    Returns
    -------
    dict of str to Pipeline
        The same models, fitted in place and returned for chaining.
    """
    for model in models.values():
        model.fit(X, y)
    return models
