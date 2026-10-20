"""Reflectance transect across a real MAJA scene (figure 17).

Vertical transect at x=101 km of the Cairo Sentinel-2 tile (2023-06-02),
comparing the surface reflectance without adjacency correction, with the
fitted King model, and the resulting forward TOA estimate.

Reads the reduced Cairo product of ``inputs/maja/`` by default; point it
at a full MAJA product and its DEM to work on another scene.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from adjeff import FullConfig, apply_psf, fit_psf, load_config, load_maja
from adjeff.core import KingPSF, S2Band
from adjeff.modules.classic import Unif2Toa
from adjeff.modules.models import Unif2Surface
from adjeff.utils import CacheStore

from adjeff_article_1.runconfig import parse_run
from adjeff_article_1.style import font, save, style_axes, use_article_style
from adjeff_article_1.utils import get_inputs_path

PRODUCT = "maja/SENTINEL2A_20230602-084138-448_L2A_T36RVV_C_V1-0"
MNT = "maja/DEM"
BAND = S2Band.B8A
RES_KM = 0.12
N_TRAIN = 1999
TRANSECT_X_KM = 101.0
N_BINS = 10
BATCH_SIZE = 8


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for this script's own options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--product-path",
        type=Path,
        default=get_inputs_path() / PRODUCT,
        help="MAJA L2A product",
    )
    parser.add_argument(
        "--mnt-path",
        type=Path,
        default=get_inputs_path() / MNT,
        help="DEM directory",
    )
    parser.add_argument("--x-km", type=float, default=TRANSECT_X_KM)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=N_BINS,
        help=(
            "aot/h discretisation levels: more of them is a finer "
            "atmosphere and more Smart-G calls."
        ),
    )
    return parser


def link_fre_probe(product_path: Path) -> None:
    """Point a FRE B2 at the SRE one, which sensorsio insists on opening.

    Its ``Sentinel2`` constructor reads the bounds, transform and CRS off a
    FRE B2 whatever band is asked for.  An SRE-only product has none, and
    the two carry the same grid, so a link is enough.
    """
    sre = next(product_path.glob("*_SRE_B2.tif"), None)
    if sre is None:
        return
    fre = product_path / sre.name.replace("_SRE_", "_FRE_")
    if not fre.exists():
        fre.symlink_to(sre.name)


def main() -> None:
    """Fit a King PSF on the scene's own atmosphere, and transect the result."""
    run, args = parse_run(__doc__.splitlines()[0], build_parser())
    run = run.resolve(n=N_TRAIN, res_km=RES_KM)

    link_fre_probe(args.product_path)
    cache = CacheStore(run.cache_dir)

    scene = load_maja(
        product_path=args.product_path,
        bands=[BAND],
        res=RES_KM,
        mnt_path=args.mnt_path,
        as_map=True,
        cache=cache,
        n_bins=args.n_bins,
        compute_radiatives=True,
        dedup=True,
        batch_size=BATCH_SIZE,
    )

    # The MAJA product carries rho_unif (atmospheric correction only, no
    # adjacency correction), stored under the rho_s slot by the loader.
    scene[BAND]["rho_unif"] = scene[BAND]["rho_s"]
    n_eval = scene[BAND].sizes["x"]

    cfg: FullConfig = load_config(scene, BAND, aggregate=True)

    psf_king = fit_psf(
        scene=scene,
        bands=[BAND],
        psf_type=KingPSF,
        init_parameters={"sigma": 0.5, "gamma": 2.0},
        model_cls=Unif2Surface,
        res_km=RES_KM,
        n_train=run.n,
        cache=cache,
        device=run.device,
    )

    scene = apply_psf(
        scene,
        psf_king,
        BAND,
        psf_type=KingPSF,
        n=n_eval,
        res_km=RES_KM,
        device=run.device,
    )
    scene = Unif2Toa()(scene)

    use_article_style()
    fig, ax = plt.subplots(figsize=(13, 4))

    # Both axes are distances from the tile's own origin, as the caption
    # reads them; the scene carries UTM coordinates in km.
    ds = scene[BAND]
    x_utm = float(ds.x.min()) + args.x_km
    dist = ds.y - ds.y.min()

    def transect(name: str) -> tuple:
        da = ds[name].sel(x=x_utm, method="nearest").squeeze(drop=True)
        return dist, da

    profile = transect("rho_unif")
    ax.plot(
        *profile,
        color="green",
        marker="x",
        markersize=7,
        markevery=20,
        label=r"$\rho_{unif}$, no adjacency effects correction",
        linewidth=2.0,
    )
    ax.plot(
        *transect("rho_s_est"),
        label=r"$\rho_{s}$, adjacency effects correction (King optimized)",
        color="orange",
        linewidth=2.0,
    )
    ax.plot(
        *transect("rho_toa"),
        label=r"$\rho_{toa}$",
        color="red",
        linewidth=2.0,
    )
    ax.legend(loc="upper right", fontsize=font())
    ax.grid(alpha=0.7, linestyle="-")
    ax.set_xlabel(r"Distance $[km]$", fontsize=font())
    ax.set_ylabel(r"Reflectance", fontsize=font())
    ax.set_xlim(0.0, float(dist.max()))
    style_axes(ax)
    fig.tight_layout()
    path = save(fig, "figure17", run.figs_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
