"""The three BRDF shapes at three sun zenith angles, on one figure.

Draws nothing but the angular shapes: the coefficients come from the
cached MODIS ensemble and no simulation is run.  It is the companion
figure to the Lambertian-assumption table of answer T1, and it exists
separately from `hotspot.py` so that redrawing it costs seconds.

Each curve is divided by its own value in the viewing direction, which is
what the simulation applies: the landscapes are rescaled so that the
reflectance the sensor *observes* is the same for every surface, and only
what leaves towards the hemisphere differs.  All four curves therefore
pass through one at nadir, and their spread away from it is the whole
effect under study.

Usage
-----
python hotspot_brdf.py
python hotspot_brdf.py --sza 10 40 60 --no-clamp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hotspot import CHOSEN

from adjeff_article_1.rossli import shape_of
from adjeff_article_1.runconfig import REPO_ROOT
from adjeff_article_1.style import font, style_axes, use_article_style

OUTPUT = REPO_ROOT / "output"

#: Shown in this order, darkest surface first, so the legend reads as a
#: gradient of anisotropy rather than as the order of a CSV.
ORDER = ("libya4-desert", "konza-grassland", "skukuza-savanna")

#: Names as the answer spells them.
LABELS = {
    "libya4-desert": "Libya-4, desert",
    "konza-grassland": "Konza, grassland",
    "skukuza-savanna": "Skukuza, savanna",
}


def load(path: Path) -> pd.DataFrame:
    """Return the three sites with their relative Ross-Li weights."""
    frame = pd.read_csv(path)
    frame = frame[frame["site"].isin(CHOSEN)].copy()
    if "k1p" not in frame:
        frame["k1p"] = frame["f_geo"] / frame["f_iso"]
    if "k2p" not in frame:
        frame["k2p"] = frame["f_vol"] / frame["f_iso"]
    return frame.set_index("site").loc[list(ORDER)].reset_index()


def draw(frame: pd.DataFrame, angles: list[float], clamp: bool, path: Path) -> None:
    """Draw one panel per sun zenith angle, with a legend shared by all."""
    use_article_style()
    vza = np.linspace(-80.0, 80.0, 321)
    # A negative view zenith is the far side of the principal plane.
    raa = np.where(vza < 0.0, 180.0, 0.0)

    fig, axes = plt.subplots(
        1, len(angles), figsize=(2.5 * len(angles) + 0.6, 2.9),
        sharey=True, layout="constrained",
    )
    axes = np.atleast_1d(axes)

    handles: list = []
    for index, (ax, sza) in enumerate(zip(axes, angles, strict=True)):
        ax.axhline(1.0, color="0.45", linestyle="--", linewidth=1.0, zorder=1)
        ax.axvline(sza, color="0.8", linewidth=0.8, zorder=0)
        for row in frame.itertuples():
            at_sensor = float(shape_of(sza, 0.0, 0.0, row.k1p, row.k2p, clamp))
            shape = (
                shape_of(sza, np.abs(vza), raa, row.k1p, row.k2p, clamp) / at_sensor
            )
            (line,) = ax.plot(vza, shape, linewidth=1.3, zorder=2)
            if index == 0:
                handles.append(line)

        ax.set_title(rf"$\theta_s = {sza:g}^\circ$", fontsize=font(), pad=4)
        ax.set_xlabel(r"$\theta_v$ [$^\circ$]", fontsize=font(11 / 12))
        ax.set_xticks([-80, -40, 0, 40, 80])
        style_axes(ax)

    axes[0].set_ylabel("BRDF shape, relative\nto the viewing direction",
                       fontsize=font(11 / 12))

    lambertian = plt.Line2D(
        [], [], color="0.45", linestyle="--", linewidth=1.0, label="Lambertian"
    )
    fig.legend(
        handles=[*handles, lambertian],
        labels=[*(LABELS[s] for s in ORDER), "Lambertian"],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=font(10 / 12),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f">>> wrote {path}", flush=True)


def main() -> None:
    """Read the coefficients and draw the panels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sza", type=float, nargs="+", default=[10.0, 40.0, 60.0],
        help="sun zenith angles, one panel each",
    )
    parser.add_argument(
        "--sites-csv", type=Path,
        default=OUTPUT / "hotspot_mcd43a1_sites.csv",
        help="cached MODIS ensemble",
    )
    parser.add_argument(
        "--no-clamp", action="store_true",
        help="draw the raw Ross-Li fit, negative tail included",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT / "hotspot_brdf.png")
    args = parser.parse_args()

    draw(load(args.sites_csv), args.sza, not args.no_clamp, args.out)


if __name__ == "__main__":
    main()
