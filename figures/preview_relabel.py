"""Preview the reviewer-driven relabelling of figures 7-17, without any run.

The four curves are extracted pixel by pixel from an already rendered figure in
``output/``, so no SMART-G run and no GPU are involved.  Only the ``r < 2 km``
core of panel (a), which is too steep to be resolved by pixel extraction, is
rebuilt from a King profile fitted on the extracted tail.

This script is a throw-away preview: it is not wired into ``makefig`` and can be
deleted once the labelling is settled.

Usage
-----
python figures/preview_relabel.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from PIL import Image
from scipy.optimize import least_squares

plt.style.use(["science", "nature"])

REPO = Path(__file__).parent.parent
SRC_PNG = REPO / "output" / "figure8.png"
OUT_PNG = REPO / "output" / "figure8_relabel_preview.png"

# Grid and axis limits of the figure being read back, as set by figure7_17.py.
RES_KM = 0.12
N = 1999
X_MAX = 160.0
Y_MIN, Y_MAX = 1e-10, 1e-1
CORE_KM = 2.0

# science.mplstyle colour cycle, in the order the AOT values are plotted.
CURVES = {
    "0.1": (0x0C, 0x5D, 0xA5),
    "0.3": (0x00, 0xB9, 0x45),
    "0.5": (0xFF, 0x95, 0x00),
    "0.7": (0xFF, 0x2C, 0x00),
}

# Header content, i.e. what the reviewer asked to see inside the graphic.
SPECIES = "Sulphate"


def group(idx: np.ndarray, gap: int = 3) -> list[tuple[int, int, float]]:
    """Group sorted indices into runs of consecutive values.

    Parameters
    ----------
    idx : np.ndarray
        Sorted integer indices.
    gap : int, optional
        Largest step still considered contiguous (default 3).

    Returns
    -------
    list of (int, int, float)
        ``(first, last, centre)`` of every run.
    """
    out: list[list[int]] = [[int(idx[0])]]
    for prev, cur in zip(idx[:-1], idx[1:]):
        if cur - prev <= gap:
            out[-1].append(int(cur))
        else:
            out.append([int(cur)])
    return [(g[0], g[-1], 0.5 * (g[0] + g[-1])) for g in out]


def calibrate(dark: np.ndarray) -> dict[str, float]:
    """Locate the two panels and the log decade pitch of panel (b).

    Parameters
    ----------
    dark : np.ndarray
        Boolean mask of near-black pixels.

    Returns
    -------
    dict of str to float
        Pixel positions of the spines plus the decade pitch of panel (b).
    """
    vert = group(np.where(dark.sum(axis=0) > 500)[0])
    horiz = group(np.where(dark.sum(axis=1) > 800)[0])
    if len(vert) < 4 or len(horiz) < 2:
        raise RuntimeError("could not locate the two panel frames")

    b_left, b_right = vert[2][2], vert[3][2]
    # Inward major x ticks of panel (b) give the decade pitch; the two outermost
    # detections are biased by the spines and are dropped.
    strip = dark[int(horiz[1][2]) - 44 : int(horiz[1][2]) - 8, :]
    ticks = np.where(strip.mean(axis=0) > 0.9)[0]
    ticks = ticks[(ticks > b_left + 20) & (ticks < b_right - 20)]
    centres = [c for *_, c in group(ticks)]
    pitch = float(np.mean(np.diff(centres)))

    return {
        "a_left": vert[0][2],
        "a_right": vert[1][2],
        "b_left": b_left,
        "b_right": b_right,
        "top": horiz[0][2],
        "bottom": horiz[1][2],
        "b_decade": pitch,
        "b_ref": centres[0],
    }


def track(
    rgb: np.ndarray, colour: tuple[int, int, int], left: float, right: float,
    top: float, bottom: float, tol: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow one coloured curve column by column, left to right.

    Parameters
    ----------
    rgb : np.ndarray
        Image as ``(h, w, 3)`` int16.
    colour : tuple of int
        Target RGB triplet.
    left, right, top, bottom : float
        Panel frame in pixels.
    tol : int, optional
        Maximum summed channel distance to the target colour (default 40).

    Returns
    -------
    tuple of np.ndarray
        Column indices and the matching row centres.
    """
    mask = np.abs(rgb - np.array(colour, dtype=np.int16)).sum(axis=2) < tol
    cols, rows, prev = [], [], None
    for x in range(int(left) + 1, int(right)):
        hit = np.where(mask[int(top) : int(bottom), x])[0]
        if not len(hit):
            continue
        runs = group(hit)
        if prev is None:
            best = runs[0]
        else:
            best = min(runs, key=lambda g: abs(g[2] - prev))
            if abs(best[2] - prev) > 60:  # legend swatch, not the curve
                continue
        prev = best[2]
        cols.append(x)
        rows.append(best[2] + int(top))
    return np.asarray(cols, float), np.asarray(rows, float)


def king(
    r: np.ndarray, sigma: float, gamma: float, grid_r: np.ndarray
) -> np.ndarray:
    """Evaluate the King profile, normalised as adjeff normalises its kernel.

    Parameters
    ----------
    r : np.ndarray
        Radii [km].
    sigma : float
        Core width [km].
    gamma : float
        Power-law index.
    grid_r : np.ndarray
        Per-pixel radii of the full 2-D kernel grid, used for the sum the
        profile is divided by.

    Returns
    -------
    np.ndarray
        Normalised profile at *r*.
    """
    shape = lambda x: (1.0 + x**2 / (2.0 * sigma**2 * gamma)) ** (-gamma)
    return shape(r) / shape(grid_r).sum()


def extract() -> tuple[dict[str, tuple], dict[str, tuple], float]:
    """Read the source figure and return the PSF and CDF curves.

    Returns
    -------
    tuple
        Per-AOT ``(r, value)`` pairs for panel (a), the same for panel (b), and
        the left limit of the log radius axis of panel (b) [km].
    """
    rgb = np.asarray(Image.open(SRC_PNG).convert("RGB")).astype(np.int16)
    cal = calibrate(rgb.max(axis=2) < 110)
    span_y = cal["bottom"] - cal["top"]
    n_dec = np.log10(Y_MAX / Y_MIN)

    axis = (np.arange(N) - (N - 1) // 2) * RES_KM
    grid_r = np.hypot(*np.meshgrid(axis, axis))

    psf, cdf = {}, {}
    for label, colour in CURVES.items():
        px, py = track(
            rgb, colour, cal["a_left"], cal["a_right"], cal["top"], cal["bottom"]
        )
        r = (px - cal["a_left"]) / (cal["a_right"] - cal["a_left"]) * X_MAX
        p = 10 ** (np.log10(Y_MAX) - (py - cal["top"]) / span_y * n_dec)

        # Fit on the reliable tail, then use the fit only to fill the core.
        tail = r >= CORE_KM
        fit = least_squares(
            lambda q: np.log10(king(r[tail], *np.exp(q), grid_r))
            - np.log10(p[tail]),
            x0=[0.0, 0.0],
        )
        sigma, gamma = np.exp(fit.x)
        core = np.linspace(0.0, CORE_KM, 200)[:-1]
        psf[label] = (
            np.concatenate([core, r[tail]]),
            np.concatenate([king(core, sigma, gamma, grid_r), p[tail]]),
        )
        print(
            f"AOT={label}: sigma={sigma:.4f} km  gamma={gamma:.4f}  "
            f"rms={np.sqrt(np.mean(fit.fun**2)):.4f} dex"
        )

        px, py = track(
            rgb, colour, cal["b_left"], cal["b_right"], cal["top"], cal["bottom"]
        )
        cdf[label] = (
            10 ** ((px - cal["b_ref"]) / cal["b_decade"]),
            1.0 - (py - cal["top"]) / span_y,
        )

    b_min = 10 ** ((cal["b_left"] - cal["b_ref"]) / cal["b_decade"])
    return psf, cdf, b_min


def draw(psf: dict, cdf: dict, b_min: float) -> None:
    """Render the relabelled figure and write it to ``output/``.

    Parameters
    ----------
    psf, cdf : dict
        Curves returned by :func:`extract`.
    b_min : float
        Left limit of the log radius axis of panel (b) [km].
    """
    tick_factor = 1.2
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.25), layout="constrained")

    for label in CURVES:
        axes[0].plot(*psf[label], linewidth=1.3, label=f"AOT $= {label}$")
        axes[1].plot(*cdf[label], linewidth=1.3, label=f"AOT $= {label}$")

    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylim(Y_MIN, Y_MAX)
    axes[1].set_ylim(0.0, 1.0)

    n_decades = round(np.log10(Y_MAX / Y_MIN))
    axes[0].set_yticks(
        [Y_MAX * 10.0 ** (-2 * k) for k in range(n_decades // 2 + 1)]
    )

    axes[0].set_title(r"(a) PSF", pad=5, fontsize=12 * tick_factor)
    axes[1].set_title(r"(b) Encircled Energy", pad=5, fontsize=12 * tick_factor)
    axes[0].set_ylabel(r"$P_{5S}(r)$", fontsize=12 * tick_factor)
    axes[1].set_ylabel(r"$\mathrm{CDF}[P_{5S}](r)$", fontsize=12 * tick_factor)

    axes[0].set_xlim(0.0, X_MAX)
    axes[1].set_xlim(b_min, X_MAX)
    for ax in axes:
        ax.set_xlabel(r"Radius $r$ [km]", fontsize=12 * tick_factor)
        ax.tick_params(
            axis="both",
            which="major",
            width=1.5,
            length=6,
            labelsize=10 * tick_factor,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            width=1.5,
            length=3,
            labelsize=8 * tick_factor,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    axes[0].legend(loc="upper right", fontsize=10 * tick_factor)
    axes[1].legend(loc="lower right", fontsize=10 * tick_factor)

    fig.suptitle("\\textbf{" + SPECIES + "} aerosol", fontsize=13 * tick_factor)

    fig.savefig(OUT_PNG, dpi=600)
    print(f"wrote {OUT_PNG}")


def main() -> None:
    """Extract the curves and render the preview."""
    if not SRC_PNG.exists():
        raise SystemExit(f"{SRC_PNG} not found; render figure8 first.")
    psf, cdf, b_min = extract()
    draw(psf, cdf, b_min)


if __name__ == "__main__":
    main()
