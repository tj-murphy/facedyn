import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import NMF as SklearnNMF

import facedyn.nmf as facedyn_nmf
from facedyn.nmf import (
    NMFDecomposer,
    _masked_nmf,
    nmf_cophenetic_correlation,
    nmf_rank_cv_sweep,
    nmf_rank_mse_sweep,
)

# The paper's actual max_iter/tol settings (kept as this module's defaults
# for fidelity) don't always fully converge on tiny synthetic test
# matrices — expected and harmless, not a correctness issue.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def make_low_rank_df(
    n_samples: int = 200, n_features: int = 8, rank: int = 3, noise: float = 0.01, seed: int = 0
) -> pd.DataFrame:
    """Synthetic non-negative data with a known low-rank structure plus noise."""
    rng = np.random.default_rng(seed)
    W_true = rng.uniform(0, 1, size=(n_samples, rank))
    H_true = rng.uniform(0, 1, size=(rank, n_features))
    noise_matrix = rng.normal(0, noise, size=(n_samples, n_features))
    data = np.clip(W_true @ H_true + noise_matrix, 0, None)

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


def test_masked_nmf_fully_observed_is_comparable_to_sklearn():
    df = make_low_rank_df()
    data = df.filter(like="smth_").to_numpy()
    mask = np.ones_like(data)

    W, H = _masked_nmf(data, mask, n_components=3, random_state=0)
    masked_mse = ((data - W @ H) ** 2).mean()

    sklearn_model = SklearnNMF(n_components=3, init="nndsvda", random_state=0, max_iter=750, tol=1e-6)
    W_sk = sklearn_model.fit_transform(data)
    sklearn_mse = ((data - W_sk @ sklearn_model.components_) ** 2).mean()

    # Different algorithm/init -- not bit-for-bit, just comparably good.
    assert masked_mse == pytest.approx(sklearn_mse, abs=0.05)


def test_masked_nmf_output_is_non_negative_and_fits_observed_entries_well():
    df = make_low_rank_df(noise=0.01)
    data = df.filter(like="smth_").to_numpy()
    rng = np.random.default_rng(0)
    mask = (rng.random(data.shape) >= 0.1).astype(float)

    W, H = _masked_nmf(data, mask, n_components=3, random_state=0)

    assert (W >= 0).all()
    assert (H >= 0).all()
    train_mse = ((mask * (data - W @ H)) ** 2).sum() / mask.sum()
    assert train_mse < 0.05  # low-noise low-rank data should fit tightly


def test_masked_nmf_same_random_state_is_deterministic():
    df = make_low_rank_df()
    data = df.filter(like="smth_").to_numpy()
    mask = np.ones_like(data)

    W_a, H_a = _masked_nmf(data, mask, n_components=3, random_state=5)
    W_b, H_b = _masked_nmf(data, mask, n_components=3, random_state=5)

    np.testing.assert_array_equal(W_a, W_b)
    np.testing.assert_array_equal(H_a, H_b)


def test_cv_sweep_returns_one_row_per_rank_and_replicate():
    df = make_low_rank_df()
    result = nmf_rank_cv_sweep(df, ranks=range(2, 5), n_replicates=4, random_state=0)

    assert len(result) == 3 * 4  # 3 ranks x 4 replicates
    assert set(result.columns) == {"rank", "rep", "train_mse", "test_mse"}
    assert result["test_mse"].dropna().ge(0).all()


def test_cv_sweep_reveals_overfitting_past_the_true_rank():
    # Small sample count and larger noise, relative to nmf_rank_mse_sweep's
    # tests, so overfitting is actually inducible within a compact rank
    # range -- this is the exact property that failed silently with the
    # discarded row-holdout mechanism, so it must be a real, non-trivial
    # check (a too-easy case would pass regardless of whether the CV
    # mechanism works).
    df = make_low_rank_df(n_samples=60, n_features=10, rank=3, noise=0.15, seed=1)
    result = nmf_rank_cv_sweep(
        df, ranks=range(2, 9), test_fraction=0.2, n_replicates=4, random_state=1
    )
    agg = result.groupby("rank")[["train_mse", "test_mse"]].mean()

    best_rank = agg["test_mse"].idxmin()
    assert best_rank <= 5  # near the true rank=3, allowing some slack

    # Training error keeps improving well past where test error is best.
    assert agg["train_mse"].iloc[-1] < agg["train_mse"].loc[best_rank]


def test_cv_sweep_same_random_state_is_deterministic():
    df = make_low_rank_df()
    result_a = nmf_rank_cv_sweep(df, ranks=range(2, 4), n_replicates=3, random_state=7)
    result_b = nmf_rank_cv_sweep(df, ranks=range(2, 4), n_replicates=3, random_state=7)

    pd.testing.assert_frame_equal(result_a, result_b)


def test_cv_sweep_default_n_seeds_has_no_seed_column():
    df = make_low_rank_df()
    result = nmf_rank_cv_sweep(df, ranks=range(2, 4), n_replicates=2, random_state=0)

    assert set(result.columns) == {"rank", "rep", "train_mse", "test_mse"}


def test_cv_sweep_n_seeds_1_matches_default_behaviour_exactly():
    df = make_low_rank_df()
    result_default = nmf_rank_cv_sweep(df, ranks=range(2, 4), n_replicates=2, random_state=9)
    result_explicit = nmf_rank_cv_sweep(
        df, ranks=range(2, 4), n_replicates=2, n_seeds=1, random_state=9
    )

    pd.testing.assert_frame_equal(result_default, result_explicit)


def test_cv_sweep_multiple_seeds_adds_seed_column_and_varies_results():
    df = make_low_rank_df(n_samples=60, n_features=10, rank=3, noise=0.15, seed=1)
    result = nmf_rank_cv_sweep(
        df, ranks=range(2, 5), n_replicates=2, n_seeds=3, random_state=0
    )

    assert set(result.columns) == {"seed", "rank", "rep", "train_mse", "test_mse"}
    assert set(result["seed"]) == {0, 1, 2}
    assert len(result) == 3 * 2 * 3  # ranks x reps x seeds

    # Different top-level seeds should draw different masks/inits, so
    # results shouldn't be identical across seeds.
    by_seed = result.pivot_table(index=["rank", "rep"], columns="seed", values="test_mse")
    assert not by_seed[0].equals(by_seed[1])


def test_plot_nmf_rank_cv_handles_multiple_seeds():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.nmf import plot_nmf_rank_cv

    df = make_low_rank_df()
    result = nmf_rank_cv_sweep(df, ranks=range(2, 4), n_replicates=2, n_seeds=2, random_state=0)

    ax = plot_nmf_rank_cv(result)
    assert ax is not None
    assert len(ax.lines) > 0


def test_cv_sweep_records_nan_instead_of_raising(monkeypatch):
    real_masked_nmf = facedyn_nmf._masked_nmf

    def flaky_masked_nmf(X, mask, n_components, **kwargs):
        if n_components == 3:
            raise RuntimeError("simulated NMF failure")
        return real_masked_nmf(X, mask, n_components, **kwargs)

    monkeypatch.setattr(facedyn_nmf, "_masked_nmf", flaky_masked_nmf)

    df = make_low_rank_df()
    result = nmf_rank_cv_sweep(df, ranks=range(2, 5), n_replicates=2, random_state=0)

    failed_rows = result[result["rank"] == 3]
    ok_rows = result[result["rank"] != 3]
    assert failed_rows["train_mse"].isna().all()
    assert failed_rows["test_mse"].isna().all()
    assert ok_rows["train_mse"].notna().all()


def test_plot_nmf_rank_cv_runs_and_returns_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.nmf import plot_nmf_rank_cv

    df = make_low_rank_df()
    result = nmf_rank_cv_sweep(df, ranks=range(2, 5), n_replicates=3, random_state=0)

    ax = plot_nmf_rank_cv(result)
    assert ax is not None
    assert len(ax.lines) > 0


def _result_with_outlier() -> pd.DataFrame:
    """Synthetic sweep output mimicking a degenerate masked-NMF replicate:
    one (rank, rep) cell's test_mse is orders of magnitude larger than
    every other value, the failure mode plot_nmf_rank_cv's robust mode
    defends against."""
    records = []
    for rank in range(2, 6):
        for rep in range(3):
            records.append({
                "rank": rank, "rep": rep,
                "train_mse": 1.0 - 0.1 * rank,
                "test_mse": 0.8 + 0.05 * rank,
            })
    records.append({"rank": 5, "rep": 0, "train_mse": 0.5, "test_mse": 500.0})
    return pd.DataFrame.from_records(records)


def test_plot_nmf_rank_cv_robust_mode_clips_outlier_and_uses_median():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.nmf import plot_nmf_rank_cv

    result = _result_with_outlier()
    ax = plot_nmf_rank_cv(result, robust=True)

    # The y-axis should be scaled to the well-behaved values, not the outlier.
    assert ax.get_ylim()[1] < 50
    # A marker (the outlier triangle) should be drawn beyond the ordinary line/scatter artists.
    assert any(coll.get_paths() for coll in ax.collections)
    # The outlier should be called out in the title rather than silently dropped.
    assert "off-scale" in ax.get_title()


def test_plot_nmf_rank_cv_non_robust_mode_is_not_clipped():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.nmf import plot_nmf_rank_cv

    result = _result_with_outlier()
    ax = plot_nmf_rank_cv(result, robust=False)

    # Without robust scaling, the axis autoscales to include the outlier.
    assert ax.get_ylim()[1] > 100
    assert "off-scale" not in ax.get_title()


def test_plot_nmf_basis_heatmap_runs_and_returns_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.nmf import plot_nmf_basis_heatmap

    df = make_low_rank_df(n_features=6)
    decomposer = NMFDecomposer(n_components=3, random_state=0).fit(df)

    ax = plot_nmf_basis_heatmap(decomposer)
    assert ax is not None
    assert len(ax.get_yticklabels()) == 6
    assert len(ax.get_xticklabels()) == 3
    assert [t.get_text() for t in ax.get_xticklabels()] == ["nmf1", "nmf2", "nmf3"]


def test_plot_nmf_basis_heatmap_normalize_scales_each_column_to_unit_range():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.nmf import plot_nmf_basis_heatmap

    df = make_low_rank_df(n_features=6)
    decomposer = NMFDecomposer(n_components=3, random_state=0).fit(df)

    ax = plot_nmf_basis_heatmap(decomposer, normalize=True)
    plotted = ax.images[0].get_array()
    assert np.isclose(plotted.max(axis=0), 1.0).all()
    assert np.isclose(plotted.min(axis=0), 0.0).all()


def test_plot_nmf_basis_heatmap_accepts_custom_labels():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.nmf import plot_nmf_basis_heatmap

    df = make_low_rank_df(n_features=4)
    decomposer = NMFDecomposer(n_components=2, random_state=0).fit(df)

    custom_labels = [f"custom_{i}" for i in range(4)]
    ax = plot_nmf_basis_heatmap(decomposer, labels=custom_labels)
    assert [t.get_text() for t in ax.get_yticklabels()] == custom_labels


def make_blocky_df(
    n_per_block: int = 30, n_blocks: int = 3, n_features: int = 9,
    noise: float = 0.01, seed: int = 0,
) -> pd.DataFrame:
    """Synthetic data with `n_blocks` well-separated, non-overlapping row
    groups, each dominated by its own distinct slice of features. Unlike
    `make_low_rank_df` (continuous, overlapping activations across all
    components), this is built so NMF's row-cluster assignment at
    `n_blocks` components should be highly reproducible across random
    restarts -- the property `nmf_cophenetic_correlation` is meant to
    detect."""
    rng = np.random.default_rng(seed)
    n_samples = n_per_block * n_blocks
    feats_per_block = n_features // n_blocks
    data = np.zeros((n_samples, n_features))
    for b in range(n_blocks):
        rows = slice(b * n_per_block, (b + 1) * n_per_block)
        feats = slice(b * feats_per_block, (b + 1) * feats_per_block)
        data[rows, feats] = rng.uniform(0.5, 1.0, size=(n_per_block, feats_per_block))
    data = np.clip(data + rng.normal(0, noise, size=data.shape), 0, None)

    df = pd.DataFrame(data, columns=[f"smth_AU{i:02d}_r" for i in range(n_features)])
    df.insert(0, "video_filename", [f"vid_{i}" for i in range(n_samples)])
    df.insert(1, "frame", list(range(n_samples)))
    return df


def test_cophenetic_correlation_returns_one_row_per_rank():
    df = make_low_rank_df(n_samples=60, n_features=8, rank=3, seed=0)
    result = nmf_cophenetic_correlation(df, ranks=range(2, 5), n_runs=3, random_state=0)

    assert list(result["rank"]) == [2, 3, 4]
    assert result["cophenetic_correlation"].between(-1.0, 1.0).all()


def test_cophenetic_correlation_deterministic_with_same_seed():
    df = make_low_rank_df(n_samples=60, n_features=8, rank=3, seed=0)
    result_a = nmf_cophenetic_correlation(df, ranks=range(2, 4), n_runs=3, random_state=5)
    result_b = nmf_cophenetic_correlation(df, ranks=range(2, 4), n_runs=3, random_state=5)

    pd.testing.assert_frame_equal(result_a, result_b)


def test_cophenetic_correlation_n_jobs_matches_serial_exactly():
    # n_jobs is purely a wall-clock optimization -- every fit is seeded
    # independently of execution order, so parallel must reproduce the
    # sequential default bit-for-bit, not just approximately.
    df = make_low_rank_df(n_samples=60, n_features=8, rank=3, seed=0)
    serial = nmf_cophenetic_correlation(df, ranks=range(2, 4), n_runs=3, random_state=0)
    parallel = nmf_cophenetic_correlation(
        df, ranks=range(2, 4), n_runs=3, random_state=0, n_jobs=2
    )

    pd.testing.assert_frame_equal(serial, parallel)


def test_cophenetic_correlation_high_at_true_block_count_lower_when_overspecified():
    df = make_blocky_df(n_per_block=30, n_blocks=3, n_features=9, noise=0.01, seed=2)
    result = nmf_cophenetic_correlation(df, ranks=[3, 9], n_runs=8, random_state=0)
    by_rank = dict(zip(result["rank"], result["cophenetic_correlation"]))

    # At the true number of well-separated blocks, row-cluster assignment
    # should be highly stable across random restarts.
    assert by_rank[3] > 0.9
    # Overspecifying the rank should make assignments less stable, not more.
    assert by_rank[9] < by_rank[3]


def test_plot_nmf_cophenetic_correlation_runs_and_returns_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.nmf import plot_nmf_cophenetic_correlation

    df = make_low_rank_df(n_samples=60, n_features=8, rank=3, seed=0)
    result = nmf_cophenetic_correlation(df, ranks=range(2, 5), n_runs=3, random_state=0)

    ax = plot_nmf_cophenetic_correlation(result)
    assert ax is not None
    assert len(ax.lines) > 0
