import numpy as np
import pandas as pd
import pytest

from facedyn.nmf import NMFDecomposer, nmf_rank_mse_sweep

# The paper's actual max_iter/tol settings (kept as this module's defaults
# for fidelity) don't always fully converge on tiny synthetic test
# matrices — expected and harmless, not a correctness issue.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def make_low_rank_df(n_samples: int = 200, n_features: int = 8, rank: int = 3, seed: int = 0) -> pd.DataFrame:
    """Synthetic non-negative data with a known low-rank structure plus noise."""
    rng = np.random.default_rng(seed)
    W_true = rng.uniform(0, 1, size=(n_samples, rank))
    H_true = rng.uniform(0, 1, size=(rank, n_features))
    noise = rng.normal(0, 0.01, size=(n_samples, n_features))
    data = np.clip(W_true @ H_true + noise, 0, None)

    df = pd.DataFrame(data, columns=[f"smth_AU{i:02d}_r" for i in range(n_features)])
    df.insert(0, "video_filename", [f"vid_{i % 10}" for i in range(n_samples)])
    df.insert(1, "frame", list(range(n_samples)))
    return df


def test_decomposer_output_is_non_negative_and_correct_shape():
    df = make_low_rank_df()
    decomposer = NMFDecomposer(n_components=3, random_state=0)
    out = decomposer.fit_transform(df)

    assert (out[["nmf1", "nmf2", "nmf3"]] >= 0).all().all()
    assert len(out) == len(df)
    assert "video_filename" in out.columns
    assert "frame" in out.columns
    # AU columns should have been replaced by nmf activation columns
    assert not any(c.startswith("smth_") for c in out.columns)


def test_components_shape():
    df = make_low_rank_df(n_features=8)
    decomposer = NMFDecomposer(n_components=3, random_state=0).fit(df)

    assert decomposer.components_.shape == (3, 8)


def test_same_random_state_is_deterministic():
    df = make_low_rank_df()
    out_a = NMFDecomposer(n_components=3, random_state=42).fit_transform(df)
    out_b = NMFDecomposer(n_components=3, random_state=42).fit_transform(df)

    pd.testing.assert_frame_equal(out_a, out_b)


def test_rank_sweep_returns_one_row_per_rank():
    df = make_low_rank_df()
    result = nmf_rank_mse_sweep(df, ranks=range(2, 6), random_state=0)

    assert list(result["rank"]) == [2, 3, 4, 5]
    assert (result["mse"] >= 0).all()


def test_rank_sweep_mse_drops_sharply_near_true_rank():
    df = make_low_rank_df(rank=3)
    result = nmf_rank_mse_sweep(df, ranks=range(2, 6), random_state=0)
    mse_by_rank = dict(zip(result["rank"], result["mse"]))

    # Should improve markedly approaching the true rank, then flatten out.
    assert mse_by_rank[3] < mse_by_rank[2]
    assert mse_by_rank[3] == pytest.approx(mse_by_rank[5], abs=mse_by_rank[2] * 0.1)


def test_column_selection_by_pattern():
    df = make_low_rank_df(n_features=4)
    decomposer = NMFDecomposer(n_components=2, random_state=0).fit(df)

    assert set(decomposer.columns_) == {"smth_AU00_r", "smth_AU01_r", "smth_AU02_r", "smth_AU03_r"}


def test_explicit_columns_override_pattern():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "a": [0.1, 0.2, 0.3, 0.4],
        "b": [0.4, 0.3, 0.2, 0.1],
    })
    decomposer = NMFDecomposer(n_components=1, columns=["a", "b"], random_state=0).fit(df)
    assert decomposer.columns_ == ["a", "b"]
