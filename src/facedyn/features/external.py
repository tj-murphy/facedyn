"""Bridges to feature extractors outside facedyn.

Two adapters push a wide frame through another package and return the same
shape facedyn's own extractor does, so the routes are interchangeable.

- :class:`RFeatureExtractor`. Any R feature package via ``rpy2``, such as
  ``cmfts`` or ``tsfeatures``. Needs ``pip install facedyn[r]`` and an R
  installation with that package.
- :class:`CallableFeatureExtractor`. Any Python function mapping a 1-D
  series to features, such as ``pycatch22`` or ``tsfresh``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from facedyn.features.reshape import apply_rowwise, split_wide


def _as_feature_series(result, feature_names=None) -> pd.Series:
    """Normalise an extractor's return value into a ``pd.Series``.

    Accepts a ``pd.Series``, a mapping, a ``(names, values)`` pair, a dict
    with ``"names"`` and ``"values"`` keys as ``pycatch22`` returns, or a
    bare sequence when `feature_names` is given.
    """
    if isinstance(result, pd.Series):
        return result
    if isinstance(result, dict):
        # pycatch22 returns {"names": [...], "values": [...]}
        if set(result) == {"names", "values"}:
            return pd.Series(list(result["values"]), index=list(result["names"]))
        return pd.Series(result)
    if isinstance(result, tuple) and len(result) == 2:
        names, values = result
        return pd.Series(list(values), index=list(names))
    values = np.asarray(result).ravel()
    if feature_names is None:
        raise TypeError(
            f"Extractor returned a bare sequence of {len(values)} values with no "
            f"names. Pass `feature_names` to label them, or return a dict/Series."
        )
    if len(feature_names) != len(values):
        raise ValueError(
            f"`feature_names` has {len(feature_names)} entries but the extractor "
            f"returned {len(values)} values."
        )
    return pd.Series(values, index=list(feature_names))


class CallableFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wrap a Python per-series feature function as a facedyn transformer.

    Parameters
    ----------
    fn : callable
        Maps a 1-D float array to features. May return a ``pd.Series``, a
        mapping, a ``(names, values)`` pair, or a bare sequence if
        `feature_names` is supplied.
    feature_names : sequence of str, optional
        Names for `fn`'s output when it returns an unlabelled sequence.
    frame_pattern : str, default r"^fr_"
        Regex identifying the frame-value columns.
    n_jobs : int, optional
        Parallel worker processes. ``None`` or ``1`` is sequential, ``-1``
        uses all cores. `fn` must be picklable to run in parallel, so use a
        module-level function rather than a lambda.
    verbose : int, default 0
        Forwarded to ``joblib.Parallel``.

    Examples
    --------
    >>> import pycatch22
    >>> extractor = CallableFeatureExtractor(
    ...     lambda x: pycatch22.catch22_all(list(x))
    ... )
    >>> features = extractor.fit_transform(wide)
    """

    def __init__(
        self,
        fn,
        feature_names=None,
        frame_pattern: str = r"^fr_",
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        self.fn = fn
        self.feature_names = feature_names
        self.frame_pattern = frame_pattern
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None) -> "CallableFeatureExtractor":
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_")
        fn, names = self.fn, self.feature_names

        def _call(row):
            return _as_feature_series(fn(row), names)

        out = apply_rowwise(
            X,
            _call,
            frame_pattern=self.frame_pattern,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.feature_names_ = [c for c in out.columns if c not in X.columns]
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)


def _require_rpy2(package: str):
    """Import rpy2 and the named R package, or raise with advice.

    rpy2 and the R package fail independently, so each gets its own message.
    """
    try:
        from rpy2.robjects.packages import importr
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RFeatureExtractor needs rpy2, which is an optional extra: "
            "`pip install facedyn[r]`. You also need a working R installation "
            f"with the `{package}` package available to it. facedyn's core "
            "never imports rpy2, so this is the only place it is required."
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on environment
        # rpy2 links against libR at import time, so an rpy2 built for a
        # newer R than the one installed fails here with a cffi symbol
        # error rather than an ImportError. Encountered for real: rpy2
        # 3.6.x needs R >= 4.5 (`R_getVar`) and dies on R 4.4.
        raise ImportError(
            f"rpy2 is installed but failed to load against your R "
            f"installation ({type(exc).__name__}: {exc}). This is usually an "
            f"rpy2/R version mismatch. rpy2 3.6+ requires R 4.5 or newer. "
            f"On R 4.4 or older pin `pip install 'rpy2<3.6'`. Check your R "
            f"version with `R --version`."
        ) from exc

    try:
        return importr(package)
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            f"rpy2 is installed but R package `{package}` could not be loaded. "
            f"Install it in R, e.g. `install.packages('{package}')`, or for "
            f"GitHub-only packages such as cmfts: "
            f"`remotes::install_github('fjbaldan/CMFTS')`."
        ) from exc


class RFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features by calling an R package's function through ``rpy2``.

    Parameters
    ----------
    package : str
        R package name to load, such as ``"cmfts"``.
    function : str, optional
        Function within `package` to call. Defaults to `package` itself,
        matching the usual convention.
    per_row : bool, default False
        ``False`` passes the whole value block as one R matrix of
        one-series-per-row, which is what dataset-oriented functions expect
        and is much faster. ``True`` calls R once per series and stacks the
        results, for functions taking a single series.
    frame_pattern : str, default r"^fr_"
        Regex identifying the frame-value columns.
    n_jobs : int, optional
        Python-side parallelism, used only when ``per_row=True``. Each
        worker gets its own embedded R.
    verbose : int, default 0
        Forwarded to ``joblib.Parallel`` when ``per_row=True``.
    **r_kwargs
        Passed straight to the R function. R argument names containing a
        dot are not valid Python identifiers, so pass those as
        ``**{"na.rm": True}``.

    Notes
    -----
    Do not let an R function fork. ``cmfts::cmfts`` defaults to
    ``n_cores = parallel::detectCores() - 1``, and that fork aborts the R
    process even in plain ``Rscript``. Pass the equivalent argument through
    `r_kwargs` to disable it. Parallelise with `n_jobs` instead.

    Examples
    --------
    >>> extractor = RFeatureExtractor("cmfts", "cmfts", n_cores=1)
    >>> features = extractor.fit_transform(wide)
    """

    def __init__(
        self,
        package: str,
        function: str | None = None,
        per_row: bool = False,
        frame_pattern: str = r"^fr_",
        n_jobs: int | None = None,
        verbose: int = 0,
        **r_kwargs,
    ):
        self.package = package
        self.function = function
        self.per_row = per_row
        self.frame_pattern = frame_pattern
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.r_kwargs = r_kwargs

    def fit(self, X: pd.DataFrame, y=None) -> "RFeatureExtractor":
        self.fitted_ = True
        return self

    def _r_function(self):
        r_package = _require_rpy2(self.package)
        name = self.function or self.package
        try:
            return getattr(r_package, name.replace(".", "_"))
        except AttributeError as exc:
            raise AttributeError(
                f"R package `{self.package}` has no exported function `{name}`."
            ) from exc

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_")
        # resolve the R function first: it raises the actionable
        # "install rpy2 / install the R package" error, which a bare
        # `import rpy2.robjects` above would pre-empt with a raw
        # ModuleNotFoundError
        r_fn = self._r_function()

        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter

        metadata, values = split_wide(X, self.frame_pattern)
        converter = ro.default_converter + numpy2ri.converter + pandas2ri.converter

        if self.per_row:
            def _call(row):
                with localconverter(converter):
                    result = r_fn(row.reshape(1, -1), **self.r_kwargs)
                    return pd.DataFrame(result).iloc[0]

            out = apply_rowwise(
                X, _call, frame_pattern=self.frame_pattern,
                n_jobs=self.n_jobs, verbose=self.verbose,
            )
        else:
            with localconverter(converter):
                result = r_fn(values, **self.r_kwargs)
                features = pd.DataFrame(result).reset_index(drop=True)
            if len(features) != len(metadata):
                raise ValueError(
                    f"R function `{self.function or self.package}` returned "
                    f"{len(features)} rows for {len(metadata)} input series. It "
                    f"may not accept a matrix of one-series-per-row, try "
                    f"`per_row=True`."
                )
            out = pd.concat([metadata, features], axis=1)

        self.feature_names_ = [c for c in out.columns if c not in X.columns]
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)


def cmfts_r_features(
    wide_df: pd.DataFrame,
    frame_pattern: str = r"^fr_",
    n_cores: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """41 CMFTS features (Báldan & Benítez, 2023) via the R package.

    Wraps :class:`RFeatureExtractor` for ``cmfts::cmfts``. Requires
    ``pip install facedyn[r]`` and an R installation with `cmfts`
    (``remotes::install_github('fjbaldan/CMFTS')``).

    Parameters
    ----------
    wide_df : pd.DataFrame
        :func:`~facedyn.features.reshape.reshape_to_wide`'s output.
    frame_pattern : str, default r"^fr_"
        Regex identifying the frame-value columns.
    n_cores : int, default 1
        Passed to ``cmfts()``. Leave at 1. Its own default forks and aborts
        the R process.
    **kwargs
        Further arguments for ``cmfts()``, such as ``na`` and ``timeout``.

    Returns
    -------
    pd.DataFrame
        The non-frame columns, plus the 41 CMFTS feature columns.

    Notes
    -----
    ``permutation_entropy`` is environment-dependent. CMFTS reaches it
    through ``tsExpKit::permutationEntropy``, which calls ``permn()``
    without importing it from ``combinat``. Where ``combinat`` is not
    loadable the feature is ``NA`` for every series. Feature tables built on
    different machines can therefore differ in this one column.
    """
    return RFeatureExtractor(
        "cmfts", "cmfts", frame_pattern=frame_pattern, n_cores=n_cores, **kwargs
    ).fit_transform(wide_df)
