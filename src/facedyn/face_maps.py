"""Face-map visualisation of NMF component AU loadings.

Replicates the original R analysis's Figure 1B/C/D-style face maps
("visualizing the facial AU groups" for each NMF component,
`Murphy, Cook & Cuve, in prep`), which were originally produced with the
`py-feat <https://py-feat.org>`_ library's ``plot_face(muscles={"all":
"heatmap"})``.

This module is a self-contained, matplotlib-only reimplementation -- not a
`py-feat` wrapper. Two reasons:

1. `py-feat` (as of 2.0) pulls in a large ML dependency stack (``torch``,
   ``torchvision``, ``timm``, ``xgboost``, ...) for its face-detection
   models, none of which this plot needs, and its video-decoding submodule
   (unconditionally imported by ``feat/__init__.py``) depends on an
   exact-matching system FFmpeg build -- a fragile native dependency this
   package's "easy to install" goal can't afford. Confirmed empirically:
   `import feat` hard-crashes in this project's own dev environment over
   exactly this.
2. `py-feat`'s ``plot_face(muscles=...)`` turns out to only need two small,
   *static* pieces from its trained model, not `py-feat` itself: (a) a PLS
   regression that maps a 20-AU vector to 68 deformed face-landmark
   points (this *does* meaningfully change the face's shape -- e.g. high
   AU06/AU12 visibly produces a smiling mouth -- unlike an earlier version
   of this module, which incorrectly assumed the shape was AU-invariant),
   and (b) ~19 named facial-muscle-shaped regions, defined purely as index
   references into that same 68-point landmark array, whose *fill color*
   (on top of the deformed shape) depends on AU values. Both are small,
   fixed, numeric artifacts -- extracted once from `py-feat`'s own
   openly-hosted files and embedded below -- not a live dependency.

Adapted from `py-feat` v0.6.2 (MIT licensed, github.com/cosanlab/py-feat)
-- credit to its authors (Cheong, Xie, Byrne, Jolly & Chang; py-feat is
cited as reference 28 in the paper this module's output replicates):

- The 68-point neutral landmark template and muscle-polygon geometry, from
  ``feat/plotting.py``'s ``draw_lineface``/``draw_muscles``/``get_heat``.
- The AU-to-landmark deformation model itself (``coef``, ``intercept``,
  ``x_mean`` below) -- the fitted parameters of `py-feat`'s
  ``pyfeat_aus_to_landmarks`` PLS regression, downloaded from its public
  GitHub release asset (github.com/cosanlab/py-feat/releases/tag/v0.1) and
  re-embedded here (float32, gzip+base64, ~15KB) as three small arrays --
  not the original 20MB file, which also bundles training data this
  module has no use for.

**AU coverage caveat**: `py-feat`'s original design maps its own 20-AU
model's codes onto muscle regions, some of which (AU11, AU24, AU28, AU43)
have no OpenFace counterpart -- `facedyn`'s AU set (see
:data:`facedyn.au_labels.AU_DESCRIPTIONS`). Two adaptations were made
deliberately, not silently: `py-feat` itself defines an alternate
OpenFace-appropriate mapping for the masseter/temporalis regions (its own
``"_rel"`` muscle variants) mapped to AU26 instead of AU24 -- used here.
And `py-feat`'s AU43 (its own code for eye closure) and OpenFace's AU45
(``Blink``) denote the same physical action under each toolkit's own
numbering, so the palpebral-orbicularis-oculi region is driven by AU45
here. There is no such equivalent for AU11 (zygomaticus major has no
OpenFace-trackable driver in this design) -- that region is left
undriven (colored neutral). AU05 (upper lid raiser) and AU25 (lips part)
have no dedicated *region* in this face-map style at all -- a limitation
inherited from `py-feat`'s own original design -- but, unlike a region's
color, face *shape* is driven by the full 20-AU deformation model
directly, so AU05 in particular still visibly affects the eyes even
without its own shaded region. :func:`plot_nmf_face_maps` warns once,
listing exactly which AUs have no facial *region* (not "no effect at
all"), rather than silently dropping them.
"""

from __future__ import annotations

import base64
import gzip
import io
import warnings
from pathlib import Path

import numpy as np
from sklearn.utils.validation import check_is_fitted

from facedyn._plot_utils import save_figure
from facedyn.au_labels import extract_au_code
from facedyn.nmf import NMFDecomposer, max_normalize_columns

# 68-point neutral face landmark template (iBUG/dlib scheme: 0-16 jaw,
# 17-21/22-26 eyebrows, 27-35 nose, 36-41/42-47 eyes, 48-67 mouth).
# Adapted from py-feat v0.6.2's feat/resources/neutral_face_coordinates.csv
# (MIT licensed, github.com/cosanlab/py-feat). Used as a fallback when no
# AU values are given; normally superseded by _predict_landmarks's output.
_NEUTRAL_LANDMARKS: tuple[tuple[float, float], ...] = (
    (37.5150, 118.9955), (38.3475, 135.9312), (40.7755, 152.8328), (44.1093, 169.1279),
    (49.9828, 184.5333), (59.1889, 198.0161), (70.4151, 209.2830), (83.6596, 217.8258),
    (98.6748, 220.0064), (113.3650, 217.3562), (126.0972, 208.6155), (137.3728, 197.2664),
    (146.1511, 183.9505), (151.7203, 168.7033), (154.9017, 152.5496), (157.0171, 136.0792),
    (157.8124, 119.2871), (45.8734, 109.0519), (53.8370, 101.4328), (65.6123, 99.4465),
    (77.4900, 101.3463), (88.3183, 105.6623), (108.8051, 105.1858), (120.1805, 100.8485),
    (131.6712, 99.2243), (142.8041, 101.3981), (150.0927, 108.7464), (98.9396, 117.1664),
    (99.0114, 128.4488), (99.0906, 139.7134), (99.2241, 151.3220), (85.9724, 158.1914),
    (92.2064, 160.6166), (98.6786, 162.5644), (105.2685, 160.6251), (111.1423, 158.3269),
    (59.2283, 118.6319), (66.0875, 114.3926), (74.6689, 114.5992), (81.8068, 120.0082),
    (74.3443, 121.7055), (65.7238, 121.8222), (114.7523, 119.9065), (122.2983, 114.2635),
    (130.6195, 114.3840), (137.0371, 118.4849), (131.2152, 121.5122), (122.9746, 121.5653),
    (75.3983, 179.4071), (84.5599, 176.2146), (92.9024, 174.4243), (98.5653, 176.0654),
    (104.9778, 174.4577), (113.1126, 176.3997), (121.1997, 179.1979), (113.1631, 185.6905),
    (105.2637, 188.3144), (98.4177, 188.9656), (92.2240, 188.3854), (84.0511, 185.7495),
    (79.1842, 179.8066), (92.7172, 179.5202), (98.5297, 180.1630), (105.0593, 179.4237),
    (117.4371, 179.7109), (104.9087, 180.3298), (98.3593, 181.1598), (92.4949, 180.4899),
)

# Axis bounds matching py-feat's own plot_face (note the inverted y-limits --
# these landmarks use image-style coordinates, y increasing downward).
_XLIM = (25, 172)
_YLIM = (240, 50)

# The 20 AU codes py-feat's deformation model and muscle-color mapping were
# both trained/defined against, in the exact order both expect (from
# py-feat's AU_LANDMARK_MAP["Feat"]). Four of these (AU11, AU24, AU28,
# AU43) have no OpenFace counterpart and are always fed as 0.
_FEAT_AU_ORDER = (
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
)

# py-feat's pyfeat_aus_to_landmarks PLS model (coef, intercept, x_mean only
# -- see module docstring), float32, gzip-compressed, base64-encoded.
_PLS_MODEL_B64 = (
    "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"
)


def _load_pls_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = gzip.decompress(base64.b64decode(_PLS_MODEL_B64))
    with np.load(io.BytesIO(raw)) as data:
        return data["coef"], data["intercept"], data["x_mean"]


_PLS_COEF, _PLS_INTERCEPT, _PLS_X_MEAN = _load_pls_model()


def _predict_landmarks(au_vector: np.ndarray) -> tuple[list[float], list[float]]:
    """68 deformed (x, y) landmark points for a 20-AU vector (in
    `_FEAT_AU_ORDER`), replicating sklearn's `PLSRegression.predict`:
    `(au - x_mean) @ coef + intercept`, reshaped to (2, 68). Matches
    py-feat's own `predict()` to float32 precision."""
    flat = (au_vector - _PLS_X_MEAN) @ _PLS_COEF + _PLS_INTERCEPT
    x, y = flat.reshape(2, 68)
    return list(x), list(y)


def _pt(x, y, x_idx: int, y_idx: int | None = None) -> tuple[float, float]:
    """A landmark point, or a synthetic point mixing one landmark's x with
    another's y (py-feat's originals do this for a few muscle vertices)."""
    if y_idx is None:
        y_idx = x_idx
    return (x[x_idx], y[y_idx])


def _face_outline_paths(x, y) -> list[tuple[list[float], list[float]]]:
    """(x, y) coordinate lists for the face's line-drawn features: jaw,
    eyebrows, eyes, nose, lips. Adapted from py-feat's draw_lineface."""
    index_groups = [
        list(range(0, 17)),  # jaw
        [17, 18, 19, 20, 21],  # left eyebrow
        [22, 23, 24, 25, 26],  # right eyebrow
        [36, 37, 38, 39, 40, 41, 36],  # left eye
        [42, 43, 44, 45, 46, 47, 42],  # right eye
        [27, 28, 29, 30],  # nose bridge
        [31, 32, 33, 34, 35],  # nose base
        [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60, 48],  # upper lip loop
        [48, 60, 67, 66, 65, 64, 54, 55, 56, 57, 58, 59, 48],  # lower lip loop
    ]
    return [([x[i] for i in idx], [y[i] for i in idx]) for idx in index_groups]


def _muscle_polygons(x, y) -> dict[str, list[tuple[float, float]]]:
    """Facial-muscle-region polygons (vertex lists) for the given (deformed
    or neutral) landmark arrays. Adapted from py-feat's draw_muscles."""
    pt = lambda i, j=None: _pt(x, y, i, j)  # noqa: E731
    bottom = (y[8] - y[57]) / 2
    eye_l_width = (y[21] - y[39]) / 2
    eye_l_width2 = (y[38] - y[2]) / 1.5
    eye_r_width = (y[23] - y[43]) / 2
    eye_r_width2 = (y[44] - y[14]) / 1.5

    return {
        "masseter_l": [pt(2), pt(3), pt(4), pt(5), pt(6), pt(5, 33)],
        "masseter_r": [pt(14), pt(13), pt(12), pt(11), pt(10), pt(11, 33)],
        "temporalis_l": [pt(2), pt(1), pt(0), pt(17), pt(36)],
        "temporalis_r": [pt(14), pt(15), pt(16), pt(26), pt(45)],
        "dep_lab_inf_l": [pt(57), pt(58), pt(59), pt(6), pt(7)],
        "dep_lab_inf_r": [pt(57), pt(56), pt(55), pt(10), pt(9)],
        "dep_ang_or_l": [pt(48), pt(7), pt(6)],
        "dep_ang_or_r": [pt(54), pt(9), pt(10)],
        "mentalis_l": [pt(58), pt(7), pt(8)],
        "mentalis_r": [pt(56), pt(9), pt(8)],
        "risorius_l": [pt(4), pt(5), pt(48)],
        "risorius_r": [pt(11), pt(12), pt(54)],
        "orb_oris_l": [
            pt(48), pt(59), pt(58), pt(57), pt(56), pt(55), pt(54),
            (x[55], y[55] + bottom), (x[56], y[56] + bottom),
            (x[57], y[57] + bottom), (x[58], y[58] + bottom), (x[59], y[59] + bottom),
        ],
        "orb_oris_u": [pt(48), pt(49), pt(50), pt(51), pt(52), pt(53), pt(54), pt(33)],
        "frontalis_l": [pt(27), pt(39), pt(38), pt(37), pt(36), pt(17), pt(18), pt(19), pt(20), pt(21)],
        "frontalis_r": [pt(27), pt(22), pt(23), pt(24), pt(25), pt(26), pt(45), pt(44), pt(43), pt(42)],
        "frontalis_inner_l": [pt(27), pt(39), pt(21)],
        "frontalis_inner_r": [pt(27), pt(42), pt(22)],
        "cor_sup_l": [pt(28), pt(19), pt(20)],
        "cor_sup_r": [pt(28), pt(23), pt(24)],
        "lev_lab_sup_l": [pt(41), pt(40), pt(49)],
        "lev_lab_sup_r": [pt(47), pt(46), pt(53)],
        "lev_lab_sup_an_l": [pt(39), pt(49), pt(31)],
        "lev_lab_sup_an_r": [pt(35), pt(42), pt(53)],
        "zyg_maj_l": [pt(48), pt(3), pt(2)],
        "zyg_maj_r": [pt(54), pt(13), pt(14)],
        "bucc_l": [pt(48), pt(5, 50), pt(5, 57)],
        "bucc_r": [pt(54), pt(11, 52), pt(11, 57)],
        "orb_oc_l_inner": [
            (x[36] - eye_l_width / 6, y[36] + eye_l_width / 5),
            (x[36], y[36] + eye_l_width / 2),
            (x[37], y[37] + eye_l_width / 2),
            (x[38], y[38] + eye_l_width / 2),
            (x[39], y[39] + eye_l_width / 2),
            (x[39] + eye_l_width / 6, y[39] + eye_l_width / 5),
            (x[39] + eye_l_width / 5, y[39]),
            (x[39] + eye_l_width / 6, y[39] - eye_l_width / 5),
            (x[39], y[39] - eye_l_width / 2),
            (x[40], y[40] - eye_l_width / 2),
            (x[41], y[41] - eye_l_width / 2),
            (x[36], y[36] - eye_l_width / 2),
            (x[36] - eye_l_width / 6, y[36] - eye_l_width / 5),
            (x[36] - eye_l_width / 5, y[36]),
        ],
        "orb_oc_r_inner": [
            (x[42] - eye_r_width / 6, y[42] + eye_r_width / 5),
            (x[42], y[42] + eye_r_width / 2),
            (x[43], y[43] + eye_r_width / 2),
            (x[44], y[44] + eye_r_width / 2),
            (x[45], y[45] + eye_r_width / 2),
            (x[45] + eye_r_width / 6, y[45] + eye_r_width / 5),
            (x[45] + eye_r_width / 5, y[45]),
            (x[45] + eye_r_width / 6, y[45] - eye_r_width / 5),
            (x[45], y[45] - eye_r_width / 2),
            (x[46], y[46] - eye_r_width / 2),
            (x[47], y[47] - eye_r_width / 2),
            (x[42], y[42] - eye_r_width / 2),
            (x[42] - eye_r_width / 6, y[42] - eye_r_width / 5),
            (x[42] - eye_r_width / 5, y[42]),
        ],
        "orb_oc_l_outer": [
            (x[39] + eye_l_width / 2, y[39]),
            (x[39], y[39] - eye_l_width),
            (x[40], y[40] - eye_l_width2),
            (x[41], y[41] - eye_l_width2),
            (x[36], y[36] - eye_l_width2),
            (x[36] - eye_l_width2 / 3, y[36] - eye_l_width2 / 2),
            (x[36] - eye_l_width / 2, y[36]),
        ],
        "orb_oc_r_outer": [
            (x[42] - eye_r_width / 2, y[42]),
            (x[47], y[47] - eye_r_width2),
            (x[46], y[46] - eye_r_width2),
            (x[45], y[45] - eye_r_width2),
            (x[45] + eye_r_width2 / 3, y[45] - eye_r_width2 / 2),
            (x[45] + eye_r_width / 2, y[45]),
        ],
        "orb_oc_l": [
            (x[36] - eye_l_width / 3, y[36] + eye_l_width / 2),
            (x[36], y[36] + eye_l_width),
            (x[37], y[37] + eye_l_width),
            (x[38], y[38] + eye_l_width),
            (x[39], y[39] + eye_l_width),
            (x[39] + eye_l_width / 3, y[39] + eye_l_width / 2),
            (x[39] + eye_l_width / 2, y[39]),
            (x[39] + eye_l_width / 3, y[39] - eye_l_width / 2),
            (x[39], y[39] - eye_l_width),
            (x[40], y[40] - eye_l_width),
            (x[41], y[41] - eye_l_width),
            (x[36], y[36] - eye_l_width),
            (x[36] - eye_l_width / 3, y[36] - eye_l_width / 2),
            (x[36] - eye_l_width / 2, y[36]),
        ],
        "orb_oc_r": [
            (x[42] - eye_r_width / 3, y[42] + eye_r_width / 2),
            (x[42], y[42] + eye_r_width),
            (x[43], y[43] + eye_r_width),
            (x[44], y[44] + eye_r_width),
            (x[45], y[45] + eye_r_width),
            (x[45] + eye_r_width / 3, y[45] + eye_r_width / 2),
            (x[45] + eye_r_width / 2, y[45]),
            (x[45] + eye_r_width / 3, y[45] - eye_r_width / 2),
            (x[45], y[45] - eye_r_width),
            (x[46], y[46] - eye_r_width),
            (x[47], y[47] - eye_r_width),
            (x[42], y[42] - eye_r_width),
            (x[42] - eye_r_width / 3, y[42] - eye_r_width / 2),
            (x[42] - eye_r_width / 2, y[42]),
        ],
    }


# Muscle region -> the OpenFace AU code that drives its shading. Adapted
# from py-feat's get_heat -- see module docstring for the two deliberate
# adaptations (masseter/temporalis -> AU26, orb_oc_*_inner -> AU45) and the
# known gap (zyg_maj_* has no OpenFace-trackable driver, left unmapped).
_MUSCLE_TO_AU: dict[str, str] = {
    "frontalis_inner_l": "AU01", "frontalis_inner_r": "AU01",
    "frontalis_l": "AU02", "frontalis_r": "AU02",
    "cor_sup_l": "AU04", "cor_sup_r": "AU04",
    "orb_oc_l_outer": "AU06", "orb_oc_r_outer": "AU06",
    "orb_oc_l": "AU07", "orb_oc_r": "AU07",
    "lev_lab_sup_an_l": "AU09", "lev_lab_sup_an_r": "AU09",
    "lev_lab_sup_l": "AU10", "lev_lab_sup_r": "AU10",
    "bucc_l": "AU12", "bucc_r": "AU12",
    "dep_ang_or_l": "AU14", "dep_ang_or_r": "AU14",
    "mentalis_l": "AU15", "mentalis_r": "AU15",
    "risorius_l": "AU17", "risorius_r": "AU17",
    "orb_oris_l": "AU20", "orb_oris_u": "AU20",
    "dep_lab_inf_l": "AU23", "dep_lab_inf_r": "AU23",
    "masseter_l": "AU26", "masseter_r": "AU26",
    "temporalis_l": "AU26", "temporalis_r": "AU26",
    "orb_oc_l_inner": "AU45", "orb_oc_r_inner": "AU45",
}


def plot_nmf_face_maps(
    decomposer: NMFDecomposer,
    ax=None,
    normalize: bool = True,
    cmap: str = "Blues",
    alpha: float = 1.0,
    warn_unmapped: bool = True,
    save_path: str | Path | None = None,
    output_dir: str | Path = ".",
    dpi: int = 300,
):
    """Plot a schematic face map per NMF component, deformed and shaded by
    AU loading.

    Replicates the original analysis's Figure 1B/C/D face maps: a
    line-drawn face, its shape deformed by the component's AU values
    (e.g. high AU06/AU12 visibly produces a smiling mouth) via py-feat's
    original PLS deformation model, with named muscle regions layered on
    top and shaded by how strongly each region's associated AU loads onto
    that component. See the module docstring for what this is adapted
    from and its known AU-coverage limitations.

    Requires matplotlib (``pip install facedyn[viz]``) -- nothing else.

    Parameters
    ----------
    decomposer : NMFDecomposer
        A fitted decomposer (i.e. ``fit`` or ``fit_transform`` already
        called).
    ax : sequence of matplotlib.axes.Axes, optional
        One Axes per component, in order. A new ``1 x n_components`` grid
        is created if not given.
    normalize : bool, default True
        If True (the default, matching :func:`facedyn.nmf.plot_nmf_basis_heatmap`),
        each component's AU loadings are independently min-max scaled to
        ``[0, 1]`` before use -- both for region shading *and* as the input
        to the deformation model, which was itself trained on AU values in
        that same ``[0, 1]`` range (see module docstring). This isn't just
        cosmetic: it's what removes NMF's per-component scale ambiguity
        (see :class:`NMFDecomposer`'s docstring) so different components'
        face maps are comparable at all. Set to False to use
        ``decomposer.components_`` unmodified -- values far outside
        ``[0, 1]`` will likely push the deformation model outside the
        range it was trained on, and will saturate every region to the
        same color.
    cmap : str, default "Blues"
        Matplotlib colormap name, applied to each region's AU value. To
        match py-feat's own ``get_heat`` exactly, a region's color is looked
        up at ``int(value * 100) / 150`` rather than at ``value`` directly
        -- py-feat quantizes into a 151-level discrete palette and only
        ever indexes the bottom 101 of those levels (``au`` is scaled to
        ``[0, 100]``, then truncated to an int), so even a region at full
        loading renders at ~67% of the colormap's range, not its darkest
        end -- and this reproduces that quantization and ceiling faithfully
        rather than using the colormap's full continuous range.
    alpha : float, default 1.0
        Multiplier on a shaded muscle region's opacity. A region's actual
        opacity is ``value * alpha``, matching py-feat's own ``get_heat``
        (whose opacity is exactly ``au_value / 100``, i.e. ``alpha=1.0``
        here) -- a region with little to no loading fades toward fully
        transparent rather than staying visible at a flat shade. Set below
        1.0 to additionally dim every region uniformly (e.g. if regions
        overlapping substantially in this design make a fully-opaque
        region's edge, drawn on top, hide too much of an earlier region
        underneath it).
    warn_unmapped : bool, default True
        Warn once, listing any of ``decomposer``'s AU columns that have no
        facial *region* in this face-map style (their loading still
        affects face *shape* via the deformation model, just isn't given
        its own shaded region), rather than silently dropping them.
    save_path : str or pathlib.Path, optional
        If given, save the figure to this filename (e.g. ``"face_maps.pdf"``
        or ``"face_maps.png"``) -- the format is inferred from the
        extension, so both a print-quality PDF and a raster PNG (see
        ``dpi``) are supported, as well as any other format matplotlib's
        ``savefig`` recognises. The figure is *not* saved if this is left
        as ``None`` (the default).
    output_dir : str or pathlib.Path, default "."
        Directory ``save_path`` is written into (created if it doesn't
        already exist). Defaults to the current directory. Ignored if
        ``save_path`` is None.
    dpi : int, default 300
        Resolution used when saving to a raster format (e.g. PNG);
        ignored for vector formats (e.g. PDF) and if ``save_path`` is
        None.

    Returns
    -------
    list of matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError as e:
        raise ImportError(
            "plot_nmf_face_maps requires matplotlib. Install with: "
            "pip install facedyn[viz]"
        ) from e

    check_is_fitted(decomposer, "components_")
    n_components = decomposer.n_components
    basis = decomposer.components_.T  # (n_features, n_components)
    if normalize:
        basis = max_normalize_columns(basis)

    au_codes = [extract_au_code(col) for col in decomposer.columns_]
    if warn_unmapped:
        mapped_aus = set(_MUSCLE_TO_AU.values())
        unmapped = sorted({code for code in au_codes if code and code not in mapped_aus})
        if unmapped:
            warnings.warn(
                f"{', '.join(unmapped)} have no facial region in this face-map "
                "style, so their loading isn't shown as shading (it may still "
                "affect face shape) -- see plot_nmf_face_maps's module "
                "docstring for why.",
                stacklevel=2,
            )

    if ax is None:
        _, axes = plt.subplots(1, n_components, figsize=(2.2 * n_components, 2.6))
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(ax)
        if len(axes) != n_components:
            raise ValueError(f"ax must have {n_components} entries, got {len(axes)}.")

    colormap = plt.get_cmap(cmap)

    for component_idx, component_ax in enumerate(axes):
        deform_au = np.zeros(len(_FEAT_AU_ORDER))
        for position, feat_code in enumerate(_FEAT_AU_ORDER):
            if feat_code in au_codes:
                deform_au[position] = basis[au_codes.index(feat_code), component_idx]
        x, y = _predict_landmarks(deform_au)

        for path_x, path_y in _face_outline_paths(x, y):
            component_ax.plot(path_x, path_y, color="k", linewidth=1, zorder=2)

        polygons = _muscle_polygons(x, y)
        for muscle, vertices in polygons.items():
            au_code = _MUSCLE_TO_AU.get(muscle)
            value = 0.0
            if au_code is not None and au_code in au_codes:
                value = basis[au_codes.index(au_code), component_idx]
            component_ax.add_patch(
                Polygon(
                    vertices, facecolor=colormap(int(value * 100) / 150), edgecolor="none",
                    alpha=value * alpha, zorder=1,
                )
            )

        for eye_idx in ([36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47]):
            component_ax.add_patch(
                Polygon([(x[i], y[i]) for i in eye_idx], facecolor="white", edgecolor="k", zorder=3)
            )
        component_ax.add_patch(
            Polygon(
                [(x[i], y[i]) for i in [60, 61, 62, 63, 64, 65, 66, 67]],
                facecolor="white", edgecolor="k", zorder=3,
            )
        )

        component_ax.set_xlim(_XLIM)
        component_ax.set_ylim(_YLIM)
        component_ax.set_aspect("equal")
        component_ax.set_xticks([])
        component_ax.set_yticks([])
        component_ax.set_title(f"Component {component_idx + 1}")

    save_figure(axes[0].figure, save_path, output_dir, dpi)

    return list(axes)
