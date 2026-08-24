"""Shared matplotlib styling for the article figures.

Every figure script in ``figures/`` calls :func:`use_article_style` once,
then :func:`style_axes` on each axis.  Keeping the numbers here is what
guarantees that all panels of the manuscript share the same tick weight,
font scale and spine width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

__all__ = [
    "BASE_FONT",
    "LINE_WIDTH",
    "TICK_SCALE",
    "font",
    "panel_title",
    "save",
    "style_axes",
    "style_colorbar",
    "use_article_style",
]

# The manuscript is typeset at a size where the scienceplots defaults read
# too light; every font and rule is scaled by these two numbers.
BASE_FONT = 12.0
TICK_SCALE = 1.2
LINE_WIDTH = 1.5


def use_article_style() -> None:
    """Activate the scienceplots theme used throughout the manuscript."""
    plt.style.use(["science", "nature"])


def font(scale: float = 1.0) -> float:
    """Return a font size on the article scale.

    Parameters
    ----------
    scale : float
        Multiplier applied to :data:`BASE_FONT`.  Use ``1.0`` for titles
        and axis labels, ``10 / 12`` for major ticks.
    """
    return BASE_FONT * scale * TICK_SCALE


def style_axes(ax: Any) -> None:
    """Apply the article tick and spine weights to *ax*."""
    ax.tick_params(
        axis="both",
        which="major",
        width=LINE_WIDTH,
        length=6,
        labelsize=font(10 / BASE_FONT),
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=LINE_WIDTH,
        length=3,
        labelsize=font(8 / BASE_FONT),
    )
    for spine in ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)


def style_colorbar(cbar: Any, label: str | None = None) -> None:
    """Apply the article styling to a colorbar, optionally labelling it."""
    if label is not None:
        cbar.set_label(
            label,
            rotation=270,
            labelpad=15,
            fontsize=font(),
        )
    cbar.ax.tick_params(
        axis="y",
        which="major",
        width=LINE_WIDTH,
        length=6,
        labelsize=font(10 / BASE_FONT),
    )
    cbar.ax.tick_params(
        axis="y",
        which="minor",
        width=LINE_WIDTH,
        length=3,
        labelsize=font(8 / BASE_FONT),
    )
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)


def panel_title(ax: Any, index: int, text: str, pad: float = 5.0) -> None:
    """Set a ``(a) text`` panel title on *ax* from a zero-based *index*."""
    ax.set_title(
        f"({'abcdefghij'[index]}) {text}",
        pad=pad,
        fontsize=font(),
    )


def save(fig: Any, name: str, figs_dir: Path, dpi: int = 600) -> Path:
    """Write *fig* to ``figs_dir/name.png`` and return the path."""
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi)
    return path
