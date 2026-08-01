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
    optimize_adam_lbfgs,
    run_forward_pipeline,
)
from adjeff.core import (
    ImageDict,
    KingPSF,
    PSFDict,
    S2Band,
    SensorBand,
    disk_image_dict,
    gaussian_image_dict,
)
from adjeff.modules.classic import Toa2Unif
from adjeff.modules.models import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages
from adjeff.utils import CacheStore

RES_KM = 0.05
N = 3999
FIGS_DIR = Path(__file__).parent.parent / "output"

WL_TO_BAND = {
    443.0: S2Band.B01,
    490.0: S2Band.B02,
    560.0: S2Band.B03,
    665.0: S2Band.B04,
    705.0: S2Band.B05,
    740.0: S2Band.B06,
    783.0: S2Band.B07,
    842.0: S2Band.B08,
    865.0: S2Band.B8A,
    945.0: S2Band.B09,
    1610.0: S2Band.B11,
    2190.0: S2Band.B12,
}

SCALES_KM = [1.0, 5.0, 50.0]

RADIATIVE_VARS = [
    "tdir_up",
    "tdif_up",
    "tdir_down",
    "tdif_down",
    "rho_atm",
    "sph_alb",
]

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


def at_aot(ds: xr.Dataset, aot: float) -> xr.Dataset:
    """Return *ds* at a single AOT, with all remaining singleton dims gone."""
    return ds.sel(aot=aot, method="nearest", drop=True).squeeze(drop=True)


def kernel_at(psf_dict: PSFDict, band: SensorBand, aot: float) -> xr.DataArray:
    """Return the kernel optimised for *aot*, without its combo dims."""
    kernel = psf_dict.kernel(band)
    return kernel.sel(aot=aot, method="nearest", drop=True).squeeze(drop=True)


def correct(
    ds: xr.Dataset,
    band: SensorBand,
    kernel: xr.DataArray,
    aot_true: float,
    aot_scalar: float,
    device: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Correct one landscape and return ``(rho_s_est, rho_unif)``.

    The TOA reflectance is the one actually measured, i.e. simulated at
    *aot_true*, while the 5S scalar terms are taken at *aot_scalar* and
    the convolution uses *kernel*.  Mixing those three is what isolates
    the error contributions.

    Note that the Dataset handed to the modules deliberately carries no
    ``rho_s``, since :class:`Unif2Surface` writes its output under that
    name and would otherwise overwrite the ground truth.
    """
    measured = at_aot(ds, aot_true)
    assumed = at_aot(ds, aot_scalar)
    mixed = xr.Dataset(
        {
            "rho_toa": measured["rho_toa"],
            **{var: assumed[var] for var in RADIATIVE_VARS},
        }
    )

    scene = Toa2Unif()(ImageDict({band: mixed}))
    model = Unif2Surface(
        psf_dict=PSFDict.from_kernels({band: kernel}), device=device
    )
    model.eval()
    scene = model(scene)
    return scene[band]["rho_s"], scene[band]["rho_unif"]


def radial_rmse(
    pred: xr.DataArray,
    truth: xr.DataArray,
    mask_on: xr.DataArray,
    device: str,
) -> float:
    """Radial RMSE between *pred* and *truth*, masked on *mask_on*.

    The shape guard matters here: a leftover singleton dimension on one
    of the operands would be silently broadcast by the metric and yield
    a meaningless value rather than an error.
    """
    if not pred.shape == truth.shape == mask_on.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape}, truth {truth.shape}, "
            f"mask {mask_on.shape}. Extra dimensions were not squeezed."
        )
    return float(
        Metric.RMSE_RAD(
            pred.adjeff.to_tensor().to(device),
            truth.adjeff.to_tensor().to(device),
            truth.adjeff.dists.to(device),
            mask_on.adjeff.to_tensor().to(device),
        )
    )


def evaluate_band(
    scenes: list[ImageDict],
    psf_dict: PSFDict,
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
                est, unif = correct(
                    ds=ds,
                    band=band,
                    kernel=kernel_at(psf_dict, band, aot_psf),
                    aot_true=aot_true,
                    aot_scalar=aot_scalar,
                    device=device,
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
    wl: float, args: argparse.Namespace, cache: CacheStore
) -> list[dict[str, float]]:
    """Simulate, optimise and evaluate every AOT case for one wavelength."""
    band = WL_TO_BAND[wl]
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

    scenes = build_landscapes(band, args.res_km, args.n)
    scenes = run_forward_pipeline(
        scenes, **cfg, nr=args.nr, n_ph=args.n_ph, cache=cache
    )

    model = make_model(
        Unif2Surface,
        KingPSF,
        [band],
        res_km=args.res_km,
        n=args.n,
        init_parameters={"sigma": 0.1, "gamma": 1.0},
        device=args.device,
    )
    psf_dict = optimize_adam_lbfgs(
        model,
        TrainingImages(images=scenes, weights=[1.0] * len(scenes)),
        Loss(Metric.RMSE_RAD),
        device=args.device,
    )

    return evaluate_band(
        scenes=scenes,
        psf_dict=psf_dict,
        band=band,
        aots=aots,
        aot_ref=args.aot_ref,
        device=args.device,
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
    parser.add_argument("--res-km", type=float, default=RES_KM)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--nr", type=int, default=500)
    parser.add_argument("--n-ph", type=int, default=int(1e5))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-dir", type=str, default="/tmp/adjeff-figures")
    parser.add_argument("--name", type=str, default="table_aot_sensitivity")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)
    cache = CacheStore(args.cache_dir)

    unknown = [wl for wl in args.wl if wl not in WL_TO_BAND]
    if unknown:
        raise ValueError(f"No Sentinel-2 band for wavelengths {unknown}.")

    rows: list[dict[str, float]] = []
    for wl in args.wl:
        rows += run_band(wl, args, cache)

    df = pd.DataFrame(rows)[COLUMNS]

    csv_path = FIGS_DIR / f"{args.name}.csv"
    tex_path = FIGS_DIR / f"{args.name}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format="%.4f")

    print()
    print(df.to_string(index=False, float_format="%.5f"))
    print()
    print(f"Wrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
