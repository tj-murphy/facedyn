import numpy as np
import pandas as pd
import pytest

from facedyn.nmf import NMFDecomposer
from facedyn.representative_aus import RepresentativeAUSelector, select_representative_aus


def make_fitted_decomposer(columns: list[str], components: np.ndarray) -> NMFDecomposer:
    """Hand-set a decomposer's fitted attributes without a real NMF fit,
    so tests can use an exact, unambiguous basis matrix."""
    decomposer = NMFDecomposer(n_components=components.shape[0])
    decomposer.columns_ = columns
    decomposer.components_ = components
    return decomposer


def test_select_representative_aus_picks_argmax_per_component():
    columns = ["smth_AU01_r", "smth_AU06_r", "smth_AU12_r", "smth_AU17_r"]
    components = np.array([
        [0.1, 0.2, 0.9, 0.3],  # component 1 -> AU12 (index 2)
        [0.1, 0.2, 0.1, 0.8],  # component 2 -> AU17 (index 3)
    ])
    decomposer = make_fitted_decomposer(columns, components)

    result = select_representative_aus(decomposer)

    assert list(result.columns) == ["component", "au"]
    assert list(result["component"]) == ["nmf1", "nmf2"]
    assert list(result["au"]) == ["smth_AU12_r", "smth_AU17_r"]


def test_select_representative_aus_adds_label_column_when_given():
    columns = ["smth_AU01_r", "smth_AU06_r"]
    components = np.array([[0.9, 0.1]])
    decomposer = make_fitted_decomposer(columns, components)

    result = select_representative_aus(decomposer, labels=["Inner Brow Raiser", "Cheek Raiser"])

    assert list(result.columns) == ["component", "au", "label"]
    assert result["label"].iloc[0] == "Inner Brow Raiser"


def test_select_representative_aus_warns_on_duplicate_selection():
    columns = ["smth_AU01_r", "smth_AU06_r"]
    # Both components load most heavily on the same AU (index 0).
    components = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
    ])
    decomposer = make_fitted_decomposer(columns, components)

    with pytest.warns(UserWarning, match="more than one component"):
        result = select_representative_aus(decomposer)

    assert list(result["au"]) == ["smth_AU01_r", "smth_AU01_r"]


def test_select_representative_aus_unaffected_by_column_rescaling():
    # NMF's per-component positive-rescaling ambiguity shouldn't change
    # which AU is picked -- argmax is scale-invariant per row regardless.
    columns = ["smth_AU01_r", "smth_AU06_r", "smth_AU12_r"]
    components = np.array([[0.1, 0.9, 0.3]])
    decomposer = make_fitted_decomposer(columns, components)
    scaled_decomposer = make_fitted_decomposer(columns, components * 1000)

    assert list(select_representative_aus(decomposer)["au"]) == list(
        select_representative_aus(scaled_decomposer)["au"]
    )


def _sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame({
        "video_filename": [f"vid_{i % 4}" for i in range(n)],
        "frame": list(range(n)),
        "isfakeorreal": ["real"] * n,
        "smth_AU01_r": rng.random(n),
        "smth_AU06_r": rng.random(n),
        "smth_AU12_r": rng.random(n),
        "smth_AU17_r": rng.random(n),
    })


def test_representative_au_selector_transform_keeps_metadata_and_selected_columns():
    df = _sample_data()
    columns = ["smth_AU01_r", "smth_AU06_r", "smth_AU12_r", "smth_AU17_r"]
    components = np.array([
        [0.1, 0.2, 0.9, 0.3],  # -> AU12
        [0.1, 0.2, 0.1, 0.8],  # -> AU17
    ])
    decomposer = make_fitted_decomposer(columns, components)

    selector = RepresentativeAUSelector(decomposer)
    out = selector.fit_transform(df)

    assert set(out.columns) == {"video_filename", "frame", "isfakeorreal", "smth_AU12_r", "smth_AU17_r"}
    assert "smth_AU01_r" not in out.columns
    assert "smth_AU06_r" not in out.columns
    assert len(out) == len(df)


def test_representative_au_selector_passes_through_raw_values_unchanged():
    """The whole point of representative-AU selection is that the kept
    columns are the original raw signal, not an NMF reconstruction/
    activation -- verify the values are literally identical, not just
    correlated."""
    df = _sample_data()
    columns = ["smth_AU01_r", "smth_AU06_r", "smth_AU12_r", "smth_AU17_r"]
    components = np.array([[0.1, 0.2, 0.9, 0.3]])
    decomposer = make_fitted_decomposer(columns, components)

    selector = RepresentativeAUSelector(decomposer)
    out = selector.fit_transform(df)

    pd.testing.assert_series_equal(
        out["smth_AU12_r"].reset_index(drop=True),
        df["smth_AU12_r"].reset_index(drop=True),
    )


def test_representative_au_selector_selected_columns_match_selection_table():
    df = _sample_data()
    columns = ["smth_AU01_r", "smth_AU06_r", "smth_AU12_r", "smth_AU17_r"]
    components = np.array([
        [0.1, 0.2, 0.9, 0.3],
        [0.1, 0.2, 0.1, 0.8],
    ])
    decomposer = make_fitted_decomposer(columns, components)

    selector = RepresentativeAUSelector(decomposer).fit(df)

    assert selector.selected_columns_ == ["smth_AU12_r", "smth_AU17_r"]
    pd.testing.assert_frame_equal(selector.selection_, select_representative_aus(decomposer))
