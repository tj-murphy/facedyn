import re

import numpy as np
import pytest

from facedyn.au_labels import AU_DESCRIPTIONS
from facedyn.face_maps import (
    _FEAT_AU_ORDER,
    _NEUTRAL_LANDMARKS,
    _MUSCLE_TO_AU,
    _face_outline_paths,
    _muscle_polygons,
    _predict_landmarks,
)
from facedyn.nmf import NMFDecomposer

_NEUTRAL_X = [p[0] for p in _NEUTRAL_LANDMARKS]
_NEUTRAL_Y = [p[1] for p in _NEUTRAL_LANDMARKS]


def make_fitted_decomposer(columns: list[str], components: np.ndarray) -> NMFDecomposer:
    """A decomposer with hand-set attributes, bypassing an actual NMF fit --
    lets tests target specific AU/component combinations exactly."""
    decomposer = NMFDecomposer(n_components=components.shape[0])
    decomposer.columns_ = columns
    decomposer.components_ = components
    return decomposer


def test_muscle_polygons_are_well_formed():
    polygons = _muscle_polygons(_NEUTRAL_X, _NEUTRAL_Y)
    assert len(polygons) > 0
    for name, vertices in polygons.items():
        assert len(vertices) >= 3, name
        for x, y in vertices:
            assert np.isfinite(x) and np.isfinite(y), name


def test_face_outline_paths_are_non_empty():
    paths = _face_outline_paths(_NEUTRAL_X, _NEUTRAL_Y)
    assert len(paths) > 0
    for x, y in paths:
        assert len(x) == len(y) > 0


def test_muscle_to_au_values_are_well_formed_codes():
    for muscle, au_code in _MUSCLE_TO_AU.items():
        assert re.fullmatch(r"AU\d{2}", au_code), (muscle, au_code)


def test_muscle_to_au_keys_are_drawable_polygons():
    from facedyn.face_maps import _ALIAS_MUSCLES

    polygons = _muscle_polygons(_NEUTRAL_X, _NEUTRAL_Y)
    drawable = set(polygons) | set(_ALIAS_MUSCLES)
    for muscle in _MUSCLE_TO_AU:
        assert muscle in drawable, muscle


def test_predict_landmarks_at_zero_roughly_matches_neutral_template():
    """au=0 should reproduce close to the static neutral template -- not
    exact (the model is a statistical fit, not an identity function at 0),
    but close enough to confirm the embedded coef/intercept/x_mean weren't
    corrupted or misaligned during extraction from py-feat's model file."""
    x, y = _predict_landmarks(np.zeros(len(_FEAT_AU_ORDER)))
    assert max(abs(a - b) for a, b in zip(x, _NEUTRAL_X)) < 5
    assert max(abs(a - b) for a, b in zip(y, _NEUTRAL_Y)) < 5


def test_predict_landmarks_returns_68_points():
    x, y = _predict_landmarks(np.zeros(len(_FEAT_AU_ORDER)))
    assert len(x) == len(y) == 68


def test_au12_deformation_raises_mouth_corners_like_a_smile():
    """Behavioral check of the deformation model itself: AU12 (lip corner
    puller) should visibly raise the mouth corners (landmarks 48, 54) --
    y decreases in this image-coordinate scheme, since y increases
    downward -- not just recolor a region, confirming this module actually
    replicates py-feat's shape-changing behavior, not just static shading."""
    au = np.zeros(len(_FEAT_AU_ORDER))
    au[_FEAT_AU_ORDER.index("AU12")] = 1.0
    x, y = _predict_landmarks(au)

    neutral_x, neutral_y = _predict_landmarks(np.zeros(len(_FEAT_AU_ORDER)))
    assert y[48] < neutral_y[48]  # left mouth corner rises
    assert y[54] < neutral_y[54]  # right mouth corner rises


def test_plot_nmf_face_maps_returns_one_axes_per_component():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.face_maps import plot_nmf_face_maps

    columns = [f"smth_{au}_r" for au in ["AU01", "AU06", "AU12"]]
    decomposer = make_fitted_decomposer(columns, np.abs(np.random.default_rng(0).random((2, 3))))

    axes = plot_nmf_face_maps(decomposer)
    assert len(axes) == 2
    assert [ax.get_title() for ax in axes] == ["Component 1", "Component 2"]


def test_plot_nmf_face_maps_rejects_wrong_number_of_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from facedyn.face_maps import plot_nmf_face_maps

    columns = [f"smth_{au}_r" for au in ["AU01", "AU06"]]
    decomposer = make_fitted_decomposer(columns, np.abs(np.random.default_rng(0).random((3, 2))))

    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="3 entries"):
        plot_nmf_face_maps(decomposer, ax=axes)


def test_plot_nmf_face_maps_warns_about_unmapped_aus():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from facedyn.face_maps import plot_nmf_face_maps

    columns = [f"smth_{au}_r" for au in ["AU05", "AU25", "AU06"]]
    decomposer = make_fitted_decomposer(columns, np.abs(np.random.default_rng(0).random((1, 3))))

    with pytest.warns(UserWarning, match="AU05, AU25"):
        plot_nmf_face_maps(decomposer)


def test_plot_nmf_face_maps_does_not_warn_when_all_aus_are_mapped():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import warnings
    from facedyn.face_maps import plot_nmf_face_maps

    columns = [f"smth_{au}_r" for au in ["AU01", "AU06", "AU12"]]
    decomposer = make_fitted_decomposer(columns, np.abs(np.random.default_rng(0).random((1, 3))))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_nmf_face_maps(decomposer)  # would raise if any warning fired


def test_face_map_shades_the_region_matching_the_dominant_au(monkeypatch):
    """Behavioral check, not just "it runs": a component driven entirely by
    AU01 should shade AU01's mapped region (frontalis_inner) at the top of
    the colormap, and an AU20-mapped region (orb_oris, unrelated) at the
    bottom -- catching AU/region-index mistakes a smoke test would miss.
    Deformation is stubbed to the fixed neutral template so drawn polygon
    vertices are predictable and matchable by position -- deformation
    itself is covered separately above."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import facedyn.face_maps as face_maps

    monkeypatch.setattr(
        face_maps, "_predict_landmarks",
        lambda au: (_NEUTRAL_X, _NEUTRAL_Y),
    )

    columns = [f"smth_{au}_r" for au in AU_DESCRIPTIONS if au not in {"AU05", "AU25"}]
    components = np.zeros((1, len(columns)))
    components[0, columns.index("smth_AU01_r")] = 1.0
    decomposer = make_fitted_decomposer(columns, components)

    axes = face_maps.plot_nmf_face_maps(decomposer, warn_unmapped=False, alpha=1.0)
    polygons = _muscle_polygons(_NEUTRAL_X, _NEUTRAL_Y)
    target = np.array(polygons["frontalis_inner_l"])
    unrelated = np.array(polygons["orb_oris_u"])

    colormap = plt.get_cmap("Blues")
    target_patch = unrelated_patch = None
    for patch in axes[0].patches:
        verts = patch.get_xy()[:-1] if len(patch.get_xy()) > len(target) else patch.get_xy()
        if verts.shape == target.shape and np.allclose(verts, target):
            target_patch = patch
        if verts.shape == unrelated.shape and np.allclose(verts, unrelated):
            unrelated_patch = patch

    assert target_patch is not None
    assert unrelated_patch is not None
    assert np.allclose(target_patch.get_facecolor(), colormap(1.0), atol=1e-6)
    assert np.allclose(unrelated_patch.get_facecolor(), colormap(0.0), atol=1e-6)


def test_alias_muscle_uses_its_own_au_mapping_not_its_source_polygons(monkeypatch):
    """Regression test: orb_oc_l/orb_oc_r (AU07) reuse orb_oc_l_outer/
    orb_oc_r_outer's polygon shape (AU06) purely for geometry -- drawing
    must color that shape by AU07's value, not silently fall back to
    AU06's, which a naive `_MUSCLE_TO_AU[source]` lookup would do."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import facedyn.face_maps as face_maps
    from facedyn.face_maps import _ALIAS_MUSCLES

    monkeypatch.setattr(
        face_maps, "_predict_landmarks",
        lambda au: (_NEUTRAL_X, _NEUTRAL_Y),
    )

    columns = [f"smth_{au}_r" for au in AU_DESCRIPTIONS if au not in {"AU05", "AU25"}]
    components = np.zeros((1, len(columns)))
    components[0, columns.index("smth_AU07_r")] = 1.0  # AU07 only, AU06 = 0
    decomposer = make_fitted_decomposer(columns, components)

    axes = face_maps.plot_nmf_face_maps(decomposer, warn_unmapped=False, alpha=1.0)
    shape = np.array(_muscle_polygons(_NEUTRAL_X, _NEUTRAL_Y)[_ALIAS_MUSCLES["orb_oc_l"]])

    colormap = plt.get_cmap("Blues")
    alias_patch = None
    for patch in axes[0].patches:
        verts = patch.get_xy()[:-1] if len(patch.get_xy()) > len(shape) else patch.get_xy()
        if verts.shape == shape.shape and np.allclose(verts, shape):
            alias_patch = patch

    assert alias_patch is not None
    assert np.allclose(alias_patch.get_facecolor(), colormap(1.0), atol=1e-6)


def test_normalize_false_uses_raw_component_values(monkeypatch):
    """normalize=False should skip max_normalize_columns -- a component
    with all-zero raw values except one small positive entry should shade
    that entry's region at a middling, not maximal, colormap value (since
    without normalization there's no guarantee the max is 1)."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import facedyn.face_maps as face_maps

    monkeypatch.setattr(
        face_maps, "_predict_landmarks",
        lambda au: (_NEUTRAL_X, _NEUTRAL_Y),
    )

    columns = [f"smth_{au}_r" for au in AU_DESCRIPTIONS if au not in {"AU05", "AU25"}]
    components = np.zeros((1, len(columns)))
    components[0, columns.index("smth_AU01_r")] = 0.3
    decomposer = make_fitted_decomposer(columns, components)

    axes_normalized = face_maps.plot_nmf_face_maps(decomposer, normalize=True, warn_unmapped=False)
    axes_raw = face_maps.plot_nmf_face_maps(decomposer, normalize=False, warn_unmapped=False)

    polygons = _muscle_polygons(_NEUTRAL_X, _NEUTRAL_Y)
    target = np.array(polygons["frontalis_inner_l"])

    def facecolor_for(axes):
        for patch in axes[0].patches:
            verts = patch.get_xy()[:-1] if len(patch.get_xy()) > len(target) else patch.get_xy()
            if verts.shape == target.shape:
                return patch.get_facecolor()
        return None

    # Only one nonzero entry -> normalized view maxes it out to the top of
    # the colormap regardless of its raw magnitude; raw view doesn't.
    import matplotlib.pyplot as plt
    colormap = plt.get_cmap("Blues")
    assert facecolor_for(axes_normalized) is not None
    assert not np.allclose(facecolor_for(axes_raw), colormap(1.0), atol=1e-6)
