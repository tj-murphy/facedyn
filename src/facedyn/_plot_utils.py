"""Shared helper for facedyn's matplotlib-based ``plot_*`` functions."""

from __future__ import annotations

from pathlib import Path


def save_figure(fig, save_path: str | Path | None, output_dir: str | Path, dpi: int) -> None:
    """Save `fig` to `output_dir / save_path`, if `save_path` is given.

    Saving a finished figure doesn't depend on what was plotted, so every
    ``plot_*`` function's ``save_path``/``output_dir``/``dpi`` parameters
    share this one implementation rather than each re-deriving it. No-op
    when ``save_path`` is None (the default for all of them), so a plot
    call never saves unless a filename is explicitly given.
    """
    if save_path is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / save_path, dpi=dpi, bbox_inches="tight")
