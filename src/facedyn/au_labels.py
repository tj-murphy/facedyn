"""Human-readable names for OpenFace Action Unit (AU) intensity columns."""

from __future__ import annotations

import re

#: FACS action names for OpenFace's 17 AU intensity ("_r") columns, keyed by
#: AU code. Taken from the original R analysis (``final_analysis.Rmd``,
#: ``au_names``), not re-derived, since the exact wording there is what the
#: published figures use.
AU_DESCRIPTIONS: dict[str, str] = {
    "AU01": "Inner Brow Raiser",
    "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",
    "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",
    "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",
    "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser",
    "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener",
    "AU25": "Lips Part",
    "AU26": "Jaw Drop",
    "AU45": "Blink",
}

_AU_CODE_PATTERN = re.compile(r"AU\d{2}", re.IGNORECASE)


def extract_au_code(column: str) -> str | None:
    """Extract the ``AU##`` code from a column name, or ``None`` if absent.

    Shared by :func:`humanise_au_label` and
    :func:`facedyn.face_maps.plot_nmf_face_maps`, both of which need to
    identify which AU a column refers to regardless of surrounding naming
    convention (``AU01_r``, ``smth_AU01_r``, ``AU01_inner_brow_raiser``, ...).
    Unlike :func:`humanise_au_label`, this doesn't check the code against
    :data:`AU_DESCRIPTIONS` -- it returns any ``AU##``-shaped code found,
    known or not, since callers may want to detect and report unrecognized
    codes themselves rather than have them silently swallowed.

    Examples
    --------
    >>> extract_au_code("smth_AU01_r")
    'AU01'
    >>> extract_au_code("video_filename")
    """
    match = _AU_CODE_PATTERN.search(column)
    return match.group(0).upper() if match else None


def humanise_au_label(column: str) -> str:
    """Convert an AU column name to a human-readable ``"AU## - Description"`` label.

    Looks for an ``AU##`` code anywhere in the name, so it works regardless
    of surrounding convention: OpenFace's raw (``AU01_r``), `facedyn`'s
    smoothed (``smth_AU01_r``), or the original R analysis's already
    human-readable (``AU01_inner_brow_raiser``) columns all normalize to the
    same label. Names with no recognisable AU code, or an AU code outside
    OpenFace's 17-column intensity set (see :data:`AU_DESCRIPTIONS`), are
    returned unchanged rather than raising.

    Parameters
    ----------
    column : str
        A column name to humanise.

    Returns
    -------
    str
        ``"AU## - Description"`` if an AU code was recognized, otherwise
        ``column`` unchanged.

    Examples
    --------
    >>> humanise_au_label("smth_AU01_r")
    'AU01 - Inner Brow Raiser'
    >>> humanise_au_label("AU01_inner_brow_raiser")
    'AU01 - Inner Brow Raiser'
    >>> humanise_au_label("video_filename")
    'video_filename'
    """
    code = extract_au_code(column)
    if code is None:
        return column
    description = AU_DESCRIPTIONS.get(code)
    if description is None:
        return column
    return f"{code} - {description}"


def humanise_au_labels(columns: list[str]) -> list[str]:
    """Apply :func:`humanise_au_label` to each name in a list of columns."""
    return [humanise_au_label(column) for column in columns]
