# Contributing to facedyn

Bug reports, questions and pull requests are welcome. Please open an
[issue](https://github.com/tj-murphy/facedyn/issues) first for anything
larger than a small fix, so we can agree on the approach.

## Reporting a bug

Include:
- the facedyn version (`python -c "import facedyn; print(facedyn.__version__)"`),
  plus your Python, pandas and scikit-learn versions;
- a minimal example that reproduces the problem, ideally on synthetic data;
- what you expected and what happened, with the full traceback.

## Setting up

```bash
git clone https://github.com/tj-murphy/facedyn.git
cd facedyn
python -m pip install -U pip           # dependency groups need pip >= 25.1
pip install --group dev -e ".[viz]"   # tests, ruff, mypy, pre-commit
pre-commit install                    # lint and format on every commit
```

For the R bridge (`RFeatureExtractor`), also install the `r` extra and a
working R: `pip install -e ".[viz,r]"`. On R 4.4 or older, pin `rpy2<3.6`.

## Checks

| Command | What it does |
|---|---|
| `pytest` | Fast tests, about a minute. Tests marked `slow` are skipped. |
| `pytest --runslow` | Everything, including Boruta fits and the R-validation tests. CI runs this. |
| `pre-commit run --all-files` | Ruff lint and format, plus file hygiene. |
| `mypy` | Type-check `src/`. |

The test configuration treats warnings as errors. If your change makes
facedyn warn on purpose, assert it in the test with `pytest.warns`.

The suite leaves two cores free on machines with more than four, since
several tests ask for `n_jobs=-1`. Add `nice -n 19` for a full run you want
to forget about.

## Conventions

- **Code style**: ruff, configured in `pyproject.toml`. Line length 100.
- **Docstrings**: numpy style for every public function, class and method.
- **Estimators**: follow scikit-learn's rules. `__init__` only stores its
  arguments; validation and fitting happen in `fit`; fitted attributes end
  in `_`; randomness comes from a `random_state` argument; list the mixins
  before `BaseEstimator` (`class X(TransformerMixin, BaseEstimator)`).
- **Exports**: add each new public name to its module's `__all__`, and to
  `facedyn/__init__.py` if it belongs in the top-level API.
- **Tests**: every new function gets unit tests in `tests/unit/`. Mark
  anything that takes more than a few seconds `@pytest.mark.slow`.
- **Validation against R**: tests in `tests/validation/` compare facedyn
  with the original R analysis's output. A change that breaks one needs a
  reason, not a new tolerance.
- **Licences**: implement methods from their published description and cite
  it in the docstring. Do not copy code from GPL-licensed packages.
- **Changelog**: add a line under `[Unreleased]` in `CHANGELOG.md`.

## Releasing (maintainer)

1. Move the `[Unreleased]` entries in `CHANGELOG.md` under the new version.
2. Set `__version__` in `src/facedyn/__init__.py`.
3. Tag `vX.Y.Z` and publish a GitHub release. Zenodo archives the release
   and mints a DOI (once the repository is connected to Zenodo).
