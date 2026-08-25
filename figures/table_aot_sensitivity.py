r"""Surface reflectance error induced by an AOT misestimation.

Answers reviewer question T9: give a first-order quantitative estimate of
the surface-reflectance error caused by a +/-0.2 deviation between the
tile-mean AOT used for the correction and the true local AOT.

A single AOT estimate feeds both the 5S scalar terms (rho_atm, T_up,
T_down, s) and the shape of the PSF, so a wrong AOT corrupts both.  The
resulting error is therefore reported as three contributions, all derived
from the very same simulations:

  - ``eps_total``   both the scalar terms and the PSF use the (wrong)
                    tile-mean AOT.  This is the operational case.
  - ``eps_scalar``  only the scalar terms use the tile-mean AOT, the PSF
                    matches the true AOT.  Classic atmospheric correction
                    error, outside the scope of this paper.
  - ``eps_psf``     only the PSF uses the tile-mean AOT, the scalar terms
                    match the true AOT.  Error attributable to the single
                    tile-averaged kernel, i.e. the reviewer's concern.

Two reference columns are reported alongside: ``no_adj_corr`` (rho_unif
compared to rho_s, i.e. no adjacency correction at all) and ``matched``
(both the scalar terms and the PSF at the true AOT, i.e. the best
achievable accuracy).  Reading eps_psf against these two is what answers
the question, since the raw RMSE also grows with the aerosol load on its
own.

Errors are radial RMSE (Metric.RMSE_RAD, masked on rho_unif), averaged
over the 6 training landscapes of Section 2.3.1, exactly as in Tables 2
and 3.  Since the landscapes span rho_s in [0, 1], the normalisation of
the metric is close to unity and the values read directly as surface
reflectance.

Usage
-----
python table_aot_sensitivity.py \\
    --aot-ref 0.4 --aot-delta 0.2 \\
    --wl 490 560 665 865 1610 2190 \\
    --rh 50.0 --h 0.0 --href 2.0 \\
    --sza 40.0 --vza 0.0 --saa 0.0 --vaa 0.0 \\
    --species sulphate
"""

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from adjeff.api import (
    make_full_config,
    make_model,
    run_forward_pipeline,
)
from adjeff.core import (
    ImageDict,
    KingPSF,
    SensorBand,
    disk_image_dict,
    gaussian_image_dict,
)
from adjeff.modules.models import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages, fit
from adjeff.utils import CacheStore
from adjeff_article_1.runconfig import RunConfig, parse_run
from adjeff.core import psf_kernel
from adjeff_article_1.shim import (
    RADIATIVE_VARS,
    correct,
    radial_rmse,
    select_scalar,
    wl_to_band,
)

RES_KM = 0.05
N = 3999

SCALES_KM = [1.0, 5.0, 50.0]

# name -> (AOT feeding the scalar terms, AOT feeding the PSF), as a
# function of (true AOT, tile-mean AOT assumed by the correction).
CASES = {
    "matched": lambda true, ref: (true, true),
    "eps_total": lambda true, ref: (ref, ref),
    "eps_scalar": lambda true, ref: (ref, true),
    "eps_psf": lambda true, ref: (true, ref),
}

COLUMNS = [
    "wl_nm",
    "aot_true",
    "no_adj_corr",
    "matched",
    "eps_total",
    "eps_scalar",
    "eps_psf",
]


def parse_species(text: str) -> dict[str, float]:
    """Parse ``"sulphate"`` or ``"sulphate:0.7,dust:0.3"`` into a dict."""
    mix: dict[str, float] = {}
    for item in text.split(","):
        name, _, frac = item.partition(":")
        mix[name.strip()] = float(frac) if frac else 1.0
    total = sum(mix.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Species fractions must sum to 1.0, got {total}.")
    return mix


def build_landscapes(
    band: SensorBand, res_km: float, n: int
) -> list[ImageDict]:
    """Return the 6 training landscapes of Section 2.3.1.

    Three Gaussian landscapes with sigma = 1, 5, 50 km followed by three
    disk landscapes with r = 1, 5, 50 km, all spanning rho_s in [0, 1].
    """
    common = dict(res_km=res_km, rho_min=0.0, rho_max=1.0, bands=[band], n=n)
    gauss = [gaussian_image_dict(sigma=s, **common) for s in SCALES_KM]
    disks = [disk_image_dict(radius=r, **common) for r in SCALES_KM]
    return gauss + disks






def evaluate_band(
    scenes: list[ImageDict],
    psf_tree: xr.DataTree,
    band: SensorBand,
    aots: list[float],
    aot_ref: float,
    device: str,
) -> list[dict[str, float]]:
    """Return one result row per true AOT for a single band."""
    rows: list[dict[str, float]] = []

    for aot_true in aots:
        acc = {name: 0.0 for name in CASES}
        acc["no_adj_corr"] = 0.0

        for scene in scenes:
            ds = scene[band]
            truth = ds["rho_s"]

            for name, pick in CASES.items():
                aot_scalar, aot_psf = pick(aot_true, aot_ref)
                # The TOA reflectance is the one actually measured, at
                # aot_true, while the 5S scalar terms are taken at
                # aot_scalar and the kernel at aot_psf.  Mixing those
                # three is what isolates the error contributions.
                est, unif = correct(
                    ds=select_scalar(ds, aot=aot_scalar),
                    band=band,
                    kernel=select_scalar(psf_kernel(psf_tree, band), aot=aot_psf),
                    device=device,
                    rho_toa=select_scalar(ds["rho_toa"], aot=aot_true),
                )
                acc[name] += radial_rmse(est, truth, unif, device)
                if name == "matched":
                    acc["no_adj_corr"] += radial_rmse(
                        unif, truth, unif, device
                    )

        rows.append(
            {
                "wl_nm": band.wl_nm,
                "aot_true": aot_true,
                **{k: v / len(scenes) for k, v in acc.items()},
            }
        )
        print(f"    {rows[-1]}", flush=True)

    return rows


def run_band(
    wl: float,
    args: argparse.Namespace,
    run: RunConfig,
    cache: CacheStore,
) -> list[dict[str, float]]:
    """Simulate, optimise and evaluate every AOT case for one wavelength."""
    band = wl_to_band(wl)
    aots = [
        args.aot_ref - args.aot_delta,
        args.aot_ref,
        args.aot_ref + args.aot_delta,
    ]

    print(f">>> {band} ({wl:.0f} nm), AOT = {aots}", flush=True)

    cfg = make_full_config(
        bands=[band],
        aot=aots,
        rh=args.rh,
        h=args.h,
        href=args.href,
        sza=args.sza,
        vza=args.vza,
        saa=args.saa,
        vaa=args.vaa,
        species=parse_species(args.species),
    )

    scenes = build_landscapes(band, run.res_km, run.n)
    scenes = run_forward_pipeline(
        scenes, **cfg, nr=args.nr, n_ph=run.n_ph, cache=cache
    )

    model = make_model(
        Unif2Surface,
        KingPSF,
        [band],
        res_km=run.res_km,
        n=run.n,
        init_parameters={"sigma": 0.1, "gamma": 1.0},
        device=run.device,
    )
    tree = fit(
        model,
        TrainingImages(images=scenes, weights=[1.0] * len(scenes)),
        loss=Loss(Metric.RMSE_RAD),
        device=run.device,
    )

    return evaluate_band(
        scenes=scenes,
        psf_tree=tree,
        band=band,
        aots=aots,
        aot_ref=args.aot_ref,
        device=run.device,
    )


def main() -> None:
    """Build the AOT sensitivity table over all requested wavelengths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aot-ref", type=float, default=0.4)
    parser.add_argument("--aot-delta", type=float, default=0.2)
    parser.add_argument(
        "--wl",
        nargs="+",
        type=float,
        default=[490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0],
    )
    parser.add_argument("--rh", type=float, default=50.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--href", type=float, default=2.0)
    parser.add_argument("--sza", type=float, default=40.0)
    parser.add_argument("--vza", type=float, default=0.0)
    parser.add_argument("--saa", type=float, default=0.0)
    parser.add_argument("--vaa", type=float, default=0.0)
    parser.add_argument(
        "--species",
        type=str,
        default="sulphate",
        help='Aerosol mix, e.g. "sulphate" or "sulphate:0.7,dust:0.3".',
    )
    parser.add_argument("--nr", type=int, default=500)
    parser.add_argument("--name", type=str, default="table_aot_sensitivity")
    run, args = parse_run(__doc__.splitlines()[0], parser)
    run = run.resolve(n=N, res_km=RES_KM)
    cache = CacheStore(run.cache_dir)

    wavelengths = args.wl[:1] if run.smoke else args.wl
    rows: list[dict[str, float]] = []
    for wl in wavelengths:
        rows += run_band(wl, args, run, cache)

    df = pd.DataFrame(rows)[COLUMNS]

    run.figs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run.figs_dir / f"{args.name}.csv"
    tex_path = run.figs_dir / f"{args.name}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format="%.4f")

    print()
    print(df.to_string(index=False, float_format="%.5f"))
    print()
    print(f"Wrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
