"""End-to-end classifier checks against Paper 1's real data.

Two different things are validated here, and they need different standards
of evidence.

**The port reproduces the published result.** Training a random forest on
the 370 training videos with Paper 1's eight published features and
scoring the genuinely held-out 94 gives a ROC-AUC around 0.70-0.72,
against the 0.694 the paper reports and the 0.6926 a live `caret`/
`randomForest` run produces on the same inputs. Random forests are
stochastic and the two implementations differ, so this asserts a band
rather than a number -- the project's standing rule for stochastic steps
(`PIPELINE.md`, "Validation protocol").

**Grouped cross-validation is not optional on this data.** Every fake
video is generated from a specific real one, so the 370 rows are 185
matched pairs carrying *opposite* labels. Splitting a pair across folds
trains on one twin and tests on the other, which inverts predictions
rather than merely weakening them: measured here, plain `StratifiedKFold`
on all 108 features scores *below chance*. That is the bug
`facedyn.splitting.pair_groups` exists to prevent, and these tests pin it
so it cannot come back silently.

Skipped when `R Validation Data/` is absent, so CI runs without the
private dataset. The metric formulas themselves are validated separately,
and unconditionally, in `test_classifier_metrics_validation.py`.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from facedyn.classifiers import make_classifier
from facedyn.evaluation import cross_validate_grouped, evaluate_models
from facedyn.splitting import pair_group_report, pair_groups_from_filenames

DATA_DIR = Path(__file__).parents[2] / "R Validation Data"
TRAIN_PATH = DATA_DIR / "r_cmfts_output_imputed_zerosd.csv"
TEST_PATH = DATA_DIR / "r_cmfts_output_imputed_zerosd_test.csv"

pytestmark = pytest.mark.skipif(
    not (TRAIN_PATH.exists() and TEST_PATH.exists()),
    reason="R CMFTS train/test exports not available locally",
)

METADATA_COLUMNS = ["video_filename", "isfakeorreal", "emotion", "valence"]

# The eight features the paper published, hardcoded downstream in the Rmd.
PAPER_8 = [
    "diff1x_pacf5_AU17_chin_raiser",
    "diff1x_pacf5_AU01_inner_brow_raiser",
    "diff2_acf1_AU01_inner_brow_raiser",
    "diff2_acf10_AU12_lip_corner_puller",
    "max_kl_shift_AU12_lip_corner_puller",
    "diff1_acf1_AU12_lip_corner_puller",
    "lumpiness_AU12_lip_corner_puller",
    "diff2_acf1_AU12_lip_corner_puller",
]


@pytest.fixture(scope="module")
def data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = [c for c in train.columns if c not in METADATA_COLUMNS]
    return {
        "train": train,
        "test": test,
        "features": features,
        "groups": pair_groups_from_filenames(train),
    }


def test_the_exports_are_the_sets_the_paper_describes(data):
    assert data["train"].shape == (370, 112)
    assert data["test"].shape == (94, 112)
    assert len(data["features"]) == 108
    assert not set(data["test"]["video_filename"]) & set(data["train"]["video_filename"])


def test_pair_groups_recover_the_matched_pairs_from_filenames(data):
    for frame, expected_pairs in [(data["train"], 185), (data["test"], 47)]:
        groups = pair_groups_from_filenames(frame)
        report = pair_group_report(frame, groups)

        # Every video is in a group of exactly two: the pairing is complete.
        assert report.to_dict("records") == [
            {"group_size": 2, "n_groups": expected_pairs, "n_videos": 2 * expected_pairs}
        ]


def test_paper_eight_reproduce_the_published_held_out_result(data):
    """The end-to-end check on the port: ~0.71 against the paper's 0.694."""
    model = make_classifier("random_forest", random_state=0, n_jobs=-1)
    model.fit(data["train"][PAPER_8], data["train"]["isfakeorreal"])

    results = evaluate_models(
        {"random_forest": model},
        data["test"][PAPER_8],
        data["test"]["isfakeorreal"],
        positive_label="fake",
    )
    auc = results.loc["random_forest", "roc_auc"]

    # R's own run of the same procedure scores 0.6926; the paper reports
    # 0.694. Different forests, so a band, not a number.
    assert 0.65 <= auc <= 0.77
    # The DeLong interval R reports for its run is (0.584, 0.802); ours
    # should overlap it substantially rather than sit somewhere else.
    assert results.loc["random_forest", "roc_auc_ci_lower"] < 0.65
    assert results.loc["random_forest", "roc_auc_ci_upper"] > 0.75


def test_random_features_do_not_reach_the_selected_features_score(data):
    """The control that makes the number above interpretable."""
    rng = np.random.default_rng(0)
    selected = make_classifier("random_forest", random_state=0, n_jobs=-1)
    selected.fit(data["train"][PAPER_8], data["train"]["isfakeorreal"])
    selected_auc = evaluate_models(
        {"m": selected}, data["test"][PAPER_8], data["test"]["isfakeorreal"],
        positive_label="fake",
    ).loc["m", "roc_auc"]

    random_aucs = []
    for _ in range(5):
        columns = list(rng.choice(data["features"], 8, replace=False))
        model = make_classifier("random_forest", random_state=0, n_jobs=-1)
        model.fit(data["train"][columns], data["train"]["isfakeorreal"])
        random_aucs.append(
            evaluate_models(
                {"m": model}, data["test"][columns], data["test"]["isfakeorreal"],
                positive_label="fake",
            ).loc["m", "roc_auc"]
        )

    assert selected_auc > np.mean(random_aucs)


def test_ungrouped_cross_validation_falls_below_chance_on_all_features(data):
    """The measurement that exposed the bug: nothing is below chance by luck."""
    model = make_classifier("random_forest", random_state=0, n_jobs=-1)
    y = data["train"]["isfakeorreal"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the missing-groups warning
        ungrouped = cross_validate_grouped(
            model, data["train"][data["features"]], y,
            cv=StratifiedKFold(5, shuffle=True, random_state=0),
        )["roc_auc"].mean()

    assert ungrouped < 0.5


def test_grouping_by_pair_recovers_the_lost_signal(data):
    """Grouped CV should beat ungrouped by a wide margin (~0.18 measured)."""
    model = make_classifier("random_forest", random_state=0, n_jobs=-1)
    y = data["train"]["isfakeorreal"]

    grouped = cross_validate_grouped(
        model, data["train"][data["features"]], y, groups=data["groups"],
        n_splits=5, n_repeats=1, random_state=0,
    )["roc_auc"].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ungrouped = cross_validate_grouped(
            model, data["train"][data["features"]], y,
            cv=StratifiedKFold(5, shuffle=True, random_state=0),
        )["roc_auc"].mean()

    assert grouped > 0.55
    assert grouped - ungrouped > 0.10


def test_the_default_cross_validator_is_group_aware(data):
    """A regression guard: the default path must not split pairs."""
    from facedyn.splitting import RepeatedStratifiedGroupKFold

    cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=3, random_state=0)
    y = (data["train"]["isfakeorreal"] == "fake").to_numpy()
    groups = data["groups"]

    for train_idx, test_idx in cv.split(data["train"][data["features"]], y, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])
