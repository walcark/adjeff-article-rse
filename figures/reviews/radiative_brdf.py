"""What the Lambertian assumption costs on ``tdif_up`` and ``sph_alb``.

The study of answer T1 applies a King kernel fitted on Lambertian
landscapes to scenes simulated with a Ross-Li BRDF.  Two of the six
radiative quantities are surface-dependent, and they were kept
Lambertian throughout: the four others never see the ground.  This
script measures how wrong that is, before any kernel enters the picture.

For every sun zenith angle and every surface it draws ``tdif_up`` and
``sph_alb`` over an aerosol sweep, once with the Lambertian samplers and
once with their BRDF variants.  Each surface is rescaled by
``k0 = 1 / shape_of(...)`` so that the reflectance the sensor observes is
the same for all four, which is the rescaling the landscapes already get:
what differs is only what leaves towards the hemisphere.

The Lambertian case is run **twice**, through the Lambertian samplers and
through the BRDF ones at ``k1p = k2p = 0``.  Those two must agree, and
their gap is the Monte-Carlo floor every other number has to clear.

Usage
-----
python radiative_brdf.py --smoke                # a few minutes
python radiative_brdf.py                        # sza 20, 40, 60
python radiative_brdf.py --sza 40 --no-clamp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import ImageDict, S2Band, SensorBand
from adjeff.modules.samplers import (
    SphAlbBrdfSampler,
    SphAlbSampler,
    TdifDownSampler,
    TdifUpBrdfSampler,
    TdifUpSampler,
    TdirDownSampler,
    TdirUpSampler,
)
from adjeff.utils import CacheStore

from adjeff_article_1.rossli import shape_of
from adjeff_article_1.runconfig import REPO_ROOT
from adjeff_article_1.style import font, style_axes, use_article_style
from hotspot import CHOSEN
from hotspot_brdf import LABELS, ORDER, load
OUTPUT = REPO_ROOT / "output"

#: The two quantities that depend on the surface, and how they read.
QUANTITIES = {
    "tdif_up": r"$t^{\uparrow}_{\mathrm{dif}}$",
    "sph_alb": r"$s$",
}

#: Drawn on top of the three surfaces, as in ``hotspot_brdf.py``.
LAMBERTIAN = "lambertian"


def geometry(sza: float, args: argparse.Namespace) -> GeoConfig:
    """Return the observation geometry for one sun zenith angle."""
    return GeoConfig(
        sza=sza, vza=args.vza, saa=args.saa, vaa=args.vaa, sat_height=args.sat_height
    )


def atmosphere(aot: np.ndarray, args: argparse.Namespace) -> AtmoConfig:
    """Return the swept atmospheric state."""
    return AtmoConfig(
        aot=aot, rh=args.rh, h=args.h, href=args.href, species={args.species: 1.0}
    )


def weights(row: pd.Series | None, sza: float, args: argparse.Namespace) -> dict:
    """Return the RTLS weights, rescaled to a unit BRF in the view direction.

    ``k0`` carries the rescaling the landscapes already get, so that the
    four surfaces send the same reflectance to the sensor and differ only
    in what they send to the hemisphere.  A Lambertian surface is the
    degenerate case with no kernel at all.
    """
    if row is None:
        return {"k0": 1.0, "k1p": 0.0, "k2p": 0.0}
    raa = args.saa - args.vaa
    shape = float(shape_of(sza, args.vza, raa, row.k1p, row.k2p, args.clamp))
    return {"k0": 1.0 / shape, "k1p": float(row.k1p), "k2p": float(row.k2p)}


def sample(
    band: SensorBand,
    sza: float,
    aot: np.ndarray,
    kernel: dict,
    args: argparse.Namespace,
    cache: CacheStore,
) -> dict[str, np.ndarray]:
    """Return ``tdif_up`` and ``sph_alb`` over the aerosol sweep, under a BRDF."""
    common = dict(
        atmo_config=atmosphere(aot, args),
        geo_config=geometry(sza, args),
        spectral_config=SpectralConfig.from_bands([band]),
        remove_rayleigh=False,
        cache=cache,
    )
    scene = SphAlbBrdfSampler(**common, **kernel, n_ph=args.n_ph)(ImageDict({}))
    scene = TdifUpBrdfSampler(
        **common, **kernel, n_ph=args.n_ph, n_ph_tdif_down=args.n_ph
    )(scene)
    return {name: _flat(scene[band][name]) for name in QUANTITIES}


def sample_lambertian(
    band: SensorBand,
    sza: float,
    aot: np.ndarray,
    args: argparse.Namespace,
    cache: CacheStore,
) -> dict[str, np.ndarray]:
    """Return the same two quantities from the Lambertian samplers.

    The reference the study has always used, and the one the BRDF path at
    ``k1p = k2p = 0`` has to reproduce.
    """
    common = dict(
        atmo_config=atmosphere(aot, args),
        geo_config=geometry(sza, args),
        spectral_config=SpectralConfig.from_bands([band]),
        remove_rayleigh=False,
        cache=cache,
    )
    scene = TdirDownSampler(**common)(ImageDict({}))
    scene = TdifDownSampler(**common, n_ph=args.n_ph)(scene)
    scene = TdirUpSampler(**common)(scene)
    scene = TdifUpSampler(**common, n_ph=args.n_ph)(scene)
    scene = SphAlbSampler(
        atmo_config=atmosphere(aot, args),
        spectral_config=SpectralConfig.from_bands([band]),
        remove_rayleigh=False,
        n_ph=args.n_ph,
        cache=cache,
    )(scene)
    return {name: _flat(scene[band][name]) for name in QUANTITIES}


def _flat(array) -> np.ndarray:
    """Return the sweep as a 1-D array over the aerosol axis."""
    return np.asarray(array.squeeze(drop=True).values, dtype=float).ravel()


def collect(args: argparse.Namespace, frame: pd.DataFrame) -> pd.DataFrame:
    """Run every ``(sza, surface)`` pair and return one tidy table."""
    band = S2Band.from_wl(args.wl)
    aot = np.linspace(args.aot_min, args.aot_max, args.n_aot)
    cache = CacheStore(args.cache_dir)
    rows: list[dict] = []

    for sza in args.sza:
        print(f">>> sza = {sza:g} deg, lambertian samplers", flush=True)
        reference = sample_lambertian(band, sza, aot, args, cache)
        for name, values in reference.items():
            rows += _rows(sza, aot, "lambertian-reference", name, values)

        for site in [None, *ORDER]:
            row = None if site is None else frame.set_index("site").loc[site]
            label = LAMBERTIAN if site is None else site
            print(f">>> sza = {sza:g} deg, {label}", flush=True)
            got = sample(band, sza, aot, weights(row, sza, args), args, cache)
            for name, values in got.items():
                rows += _rows(sza, aot, label, name, values)

    return pd.DataFrame(rows)


def _rows(sza, aot, surface, quantity, values) -> list[dict]:
    """Return one record per aerosol optical thickness."""
    return [
        {
            "sza": sza,
            "aot": float(a),
            "surface": surface,
            "quantity": quantity,
            "value": float(v),
        }
        for a, v in zip(aot, values, strict=True)
    ]


def draw(table: pd.DataFrame, args: argparse.Namespace, path: Path) -> None:
    """Draw one column per quantity, one row per sun zenith angle.

    The two quantities never share an axis: they are different scales, so
    they get their own column rather than a second y-axis.  Rows share
    the aerosol axis, columns share their own scale, so a surface can be
    followed down a column and across a row.
    """
    use_article_style()
    angles = sorted(table["sza"].unique())
    fig, axes = plt.subplots(
        len(angles),
        len(QUANTITIES),
        figsize=(2.9 * len(QUANTITIES) + 0.6, 2.3 * len(angles) + 0.6),
        sharex=True,
        sharey="col",
        layout="constrained",
        squeeze=False,
    )

    handles: list = []
    for row, sza in enumerate(angles):
        for column, (quantity, symbol) in enumerate(QUANTITIES.items()):
            ax = axes[row, column]
            here = table[(table["sza"] == sza) & (table["quantity"] == quantity)]

            lam = here[here["surface"] == LAMBERTIAN].sort_values("aot")
            ax.plot(
                lam["aot"],
                lam["value"],
                color="0.45",
                linestyle="--",
                linewidth=1.0,
                zorder=3,
            )
            for site in ORDER:
                part = here[here["surface"] == site].sort_values("aot")
                (line,) = ax.plot(part["aot"], part["value"], linewidth=1.3, zorder=2)
                if row == 0 and column == 0:
                    handles.append(line)

            if row == 0:
                ax.set_title(symbol, fontsize=font(), pad=4)
            if row == len(angles) - 1:
                ax.set_xlabel(r"$\tau_a$", fontsize=font(11 / 12))
            if column == 0:
                ax.set_ylabel(
                    rf"$\theta_s = {sza:g}^\circ$", fontsize=font(11 / 12)
                )
            style_axes(ax)

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


def report(table: pd.DataFrame) -> None:
    """Print the Monte-Carlo floor, then the departure of each surface.

    The floor is the gap between the two ways of computing the Lambertian
    case.  A surface whose departure does not clear it has not been shown
    to differ from Lambertian at this photon count.
    """
    wide = table.pivot_table(
        index=["sza", "quantity", "aot"], columns="surface", values="value"
    )
    floor = (
        (wide[LAMBERTIAN] - wide["lambertian-reference"]).abs()
        / wide["lambertian-reference"]
    )
    print("\n=== Monte-Carlo floor, lambertian samplers vs BRDF path at k1p=k2p=0")
    print((100.0 * floor).groupby(level=["sza", "quantity"]).max().round(2))

    print("\n=== departure from Lambertian [%], largest over the aerosol sweep")
    for site in ORDER:
        gap = 100.0 * (wide[site] - wide[LAMBERTIAN]).abs() / wide[LAMBERTIAN]
        print(f"\n{LABELS[site]}")
        print(gap.groupby(level=["sza", "quantity"]).max().round(2))


def main() -> None:
    """Run the sweep, write the table and draw the panels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sza", type=float, nargs="+", default=[20.0, 40.0, 60.0])
    parser.add_argument("--vza", type=float, default=0.0)
    parser.add_argument("--saa", type=float, default=0.0)
    parser.add_argument("--vaa", type=float, default=0.0)
    parser.add_argument("--sat-height", type=float, default=786.0)
    parser.add_argument("--wl", type=float, default=665.0)
    parser.add_argument("--rh", type=float, default=50.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--href", type=float, default=2.0)
    parser.add_argument("--species", type=str, default="sulphate")
    parser.add_argument("--aot-min", type=float, default=0.05)
    parser.add_argument("--aot-max", type=float, default=0.8)
    parser.add_argument("--n-aot", type=int, default=8)
    parser.add_argument("--n-ph", type=int, default=int(3e7))
    parser.add_argument(
        "--clamp", action=argparse.BooleanOptionalAction, default=True,
        help="clamp the Ross-Li shape at zero, as hotspot.py does",
    )
    parser.add_argument("--sites-csv", type=Path,
                        default=OUTPUT / "hotspot_mcd43a1_sites.csv")
    parser.add_argument("--cache-dir", type=str, default=str(OUTPUT / "cache_review"))
    parser.add_argument(
        "--out", type=Path, default=None,
        help="PNG path; defaults to output/radiative_brdf_<clamp>.png",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="a coarse sweep at a low photon count, to check the chain",
    )
    args = parser.parse_args()
    if args.smoke:
        args.n_aot, args.n_ph = 3, int(1e6)

    frame = load(args.sites_csv)
    missing = set(ORDER) - set(frame["site"])
    if missing:
        raise RuntimeError(f"{args.sites_csv} lacks {sorted(missing)}; CHOSEN={CHOSEN}")

    table = collect(args, frame)
    tag = "clamped" if args.clamp else "unclamped"
    csv = OUTPUT / f"radiative_brdf_{tag}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv, index=False)
    print(f">>> wrote {csv}", flush=True)

    report(table)
    draw(table, args, args.out or OUTPUT / f"radiative_brdf_{tag}.png")


if __name__ == "__main__":
    main()
