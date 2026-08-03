"""Shared pytest configuration.

**`pytest` runs the fast tests only. `pytest --runslow` runs everything.**

The suite is dominated by a handful of genuinely expensive tests: Boruta
fits (hundreds of random forests each), `missForest`-equivalent iterative
imputation, and the rpy2 bridge. Together they take tens of minutes and
saturate every core, which is a poor default for the edit-run loop even
though every one of them earns its place in CI.

Marking them `slow` and skipping by default keeps the common case in
seconds. Nothing is deleted or weakened -- CI passes `--runslow`, so the
full suite still gates every push.

Two habits worth keeping:

- **Mark a test slow for what it costs, not for whether it matters.** The
  R-validation tests are the most valuable in the repo and most of them
  are slow; that is exactly why they belong in CI rather than in every
  local run.
- **`nice` is not enough on its own.** Several tests ask for
  ``n_jobs=-1`` by design, so the full suite will start a worker per core
  and drive load average well past the core count; ``nice`` fixes who
  wins the contention but not how much of it there is. Hence the core cap
  below, and ``nice -n 19 pytest --runslow`` on top of it for a run you
  want to ignore completely.
"""

import os
import pytest

#: Cores left free for whatever else the machine is doing. `n_jobs=-1`
#: inside a test then means "all but these" rather than "all", which is
#: the difference between a suite you can work through and one you cannot.
#: Set ``LOKY_MAX_CPU_COUNT`` yourself to override entirely.
RESERVED_CORES = 2

#: Below this, reserving anything does more harm than good -- a 2-core CI
#: runner would be left with a single worker, which costs far more than the
#: interactivity it buys on a machine with no interactive user.
MIN_CORES_TO_RESERVE = 5


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked `slow` (the full suite; tens of minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: expensive test (Boruta fits, imputation, rpy2). Skipped unless "
        "--runslow is passed; always run in CI.",
    )

    # Set before any test imports joblib and builds a pool, which is why
    # this lives in `pytest_configure` rather than in a fixture.
    cores = os.cpu_count() or 1
    if "LOKY_MAX_CPU_COUNT" not in os.environ and cores >= MIN_CORES_TO_RESERVE:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(cores - RESERVED_CORES)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test -- pass --runslow to include")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
