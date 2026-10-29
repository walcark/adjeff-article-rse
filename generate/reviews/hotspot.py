"""The cost of a Lambertian surface assumption on three non-Lambertian sites.

The King kernel of the article is fitted on Lambertian landscapes, then applied
unchanged to scenes simulated with a Ross-Li BRDF from MODIS MCD43A1. The three
sites sit on either side of Lambertian, and all three keep under 1 % of their
upward flux where the Ross-Li fit turns negative:

    site               a       BRF negative   negative flux
    skukuza-savanna    0.985   83 deg         0.79 %
    libya4-desert      1.015   90 deg         0.012 %
    konza-grassland    1.056   86 deg         0.36 %

``a = albedo / BRF(view)`` is one for a Lambertian surface, below one
when the surface is brighter towards the sensor than towards the sky.

Usage
-----
python hotspot.py --smoke              # a few minutes, checks the chain
python hotspot.py --sza 40             # the manuscript's geometry
python hotspot.py --sza 20 --clamp     # all surfaces below Lambertian
python hotspot.py --sza 60 --clamp     # all above
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from adjeff.analysis import rmse
from adjeff.api import make_full_config, make_model, run_forward_pipeline
from adjeff.core import (
    ImageDict,
    KingPSF,
    SensorBand,
    disk_image_dict,
    gaussian_image_dict,
    psf_kernel,
)
from adjeff.modules.models import Unif2Surface
from adjeff.modules.samplers import RADIATIVE_VARS, RadiativePipeline
from adjeff.optim import Loss, Metric, TrainingImages, fit
from adjeff.utils import CacheStore

from adjeff_article_1.appeears import fetch_mcd43a1 as appeears_fetch
from adjeff_article_1.correction import correct
from adjeff_article_1.credentials import (
    add_credentials_arguments,
    load_credentials,
)
from adjeff_article_1.rossli import (
    bihemispheric,
    brf_onset,
    hemispheric,
    negative_flux,
    shape_of,
)
from adjeff_article_1.runconfig import (
    REPO_ROOT,
    RunConfig,
    band_from_wl,
    parse_run,
)
from adjeff_article_1.skylight import MODIS_BANDS, blue_sky, skylight_fraction
from adjeff_article_1.utils import get_cache_path

SITES: tuple[tuple[str, float, float], ...] = (
    ("konza-grassland", 39.0824, -96.5603),
    ("skukuza-savanna", -25.0197, 31.4969),
    ("libya4-desert", 28.5500, 23.3900),
)

CHOSEN = ("skukuza-savanna", "libya4-desert", "konza-grassland")

UNINFORMATIVE = ("disk1", "disk5")

OUTPUT = REPO_ROOT / "results" / "tables"


def load_ensemble(args: argparse.Namespace) -> pd.DataFrame:
    """Return the MCD43A1 coefficients, from the CSV or from AppEEARS.

    `inputs/modis/mcd43a1_sites.csv` is versioned, so a fresh clone needs no
    Earthdata account.  Any CSV with `site`, `f_iso`, `f_geo` and `f_vol`
    does; `--refresh-sites` fetches and overwrites it.
    """
    path = Path(args.sites_csv)
    if path.exists() and not args.refresh_sites:
        frame = pd.read_csv(path)
        missing = {"site", "f_iso", "f_geo", "f_vol"} - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path} lacks the columns {sorted(missing)}")
        if "k1p" not in frame:
            frame["k1p"] = frame["f_geo"] / frame["f_iso"]
        if "k2p" not in frame:
            frame["k2p"] = frame["f_vol"] / frame["f_iso"]
        print(f">>> MCD43A1 ensemble read from {path}", flush=True)
        return frame

    frame = _fetch_ensemble(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"    cached to {path}", flush=True)
    return frame


def _fetch_ensemble(args: argparse.Namespace) -> pd.DataFrame:
    """Fetch the ensemble from AppEEARS.

    It needs an Earthdata account and works by submitting a task, which
    is slower than a plain query, but it is the path that stays up.
    """
    print(">>> MCD43A1 ensemble, from AppEEARS", flush=True)
    rows = appeears_fetch(
        load_credentials(args),
        SITES,
        args.modis_band,
        args.start_date,
        args.end_date,
        timeout_s=args.appeears_timeout,
    )
    frame = pd.DataFrame(rows)
    # `device.cu` wants the weights relative to the isotropic one, and
    # AppEEARS returns the three absolute ones.
    frame["k1p"] = frame["f_geo"] / frame["f_iso"]
    frame["k2p"] = frame["f_vol"] / frame["f_iso"]
    return frame


@contextlib.contextmanager
def rtls_surface(k1p: float, k2p: float) -> Iterator[None]:
    """Make ``SurfaceFactory.surface`` return an ``RTLSSurface``.

    ``k0`` comes from the landscape rather than being fixed, because Smart-G
    applies the BRDF factor to the whole ``ENV=2`` expression::

        weight *= BRDF * (gauss * (alb_surface - alb_env) + alb_env)

    Both albedos must therefore already carry the ``1 / rtls_shape``
    normalisation, which :func:`scaled_landscape` does, or a flat field
    picks up a spurious Gaussian modulation.
    """
    from adjeff.atmosphere import SurfaceFactory
    from smartg.albedo import AlbedoCst
    from smartg.surface import RTLSSurface

    original = SurfaceFactory.surface

    def patched(self, arr: xr.Dataset):
        params = arr["rho_s"].adjeff.params() or {}
        return RTLSSurface(
            k0=AlbedoCst(float(params["rho_max"])),
            k1p=AlbedoCst(k1p),
            k2p=AlbedoCst(k2p),
        )

    SurfaceFactory.surface = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        SurfaceFactory.surface = original  # type: ignore[method-assign]


def train(
    scenes: list[tuple[str, ImageDict]],
    band: SensorBand,
    run: RunConfig,
) -> tuple[xr.DataArray, dict[str, float]]:
    """Fit one King PSF over the whole training set.

    One kernel for the six landscapes, not one each: the point is to apply
    it unchanged to surfaces it was never trained for.
    """
    images = [img for _, img in scenes]

    model = make_model(
        Unif2Surface,
        KingPSF,
        [band],
        res_km=run.res_km,
        n=run.n,
        init_parameters={"sigma": 0.1, "gamma": 2.0},
        device=run.device,
    )
    tree = fit(
        model,
        TrainingImages(images=images),
        loss=Loss(Metric.RMSE_RAD),
        device=run.device,
    )
    kernel = psf_kernel(tree, band).squeeze(drop=True)
    params = model.psf_params(band)

    del model, tree
    if run.device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()

    return kernel, params


def selected_surfaces(args: argparse.Namespace) -> pd.DataFrame:
    """Return the three sites, ordered by anisotropy, with their diagnostics."""
    frame = load_ensemble(args)
    frame = frame[frame["site"].isin(CHOSEN)].copy()
    missing = set(CHOSEN) - set(frame["site"])
    if missing:
        raise RuntimeError(
            f"the ensemble does not hold {sorted(missing)}; it has "
            f"{sorted(load_ensemble(args)['site'])}"
        )

    if "k1p" not in frame:
        frame["k1p"] = frame["f_geo"] / frame["f_iso"]
    if "k2p" not in frame:
        frame["k2p"] = frame["f_vol"] / frame["f_iso"]

    raa = args.saa - args.vaa
    frame["shape_view"] = [
        float(shape_of(args.sza, args.vza, raa, k1p, k2p, args.clamp))
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]
    frame["dhr"] = [
        hemispheric(args.sza, k1p, k2p, args.clamp)
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]
    frame["bhr"] = [
        bihemispheric(k1p, k2p, args.clamp)
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]
    frame["skyl"] = skylight_fraction(
        args.sza, args.aot, band=args.modis_band, aerosol=args.aerosol
    )
    frame["albedo"] = [
        blue_sky(
            dhr,
            bhr,
            args.sza,
            args.aot,
            band=args.modis_band,
            aerosol=args.aerosol,
        )
        for dhr, bhr in zip(frame["dhr"], frame["bhr"], strict=True)
    ]
    frame["a"] = frame["albedo"] / frame["shape_view"]
    frame["a_black"] = frame["dhr"] / frame["shape_view"]
    frame["brf_neg_vza"] = [
        brf_onset(args.sza, k1p, k2p)
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]
    frame["negative_flux_pct"] = [
        negative_flux(args.sza, k1p, k2p)
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]
    return frame.sort_values("a").reset_index(drop=True)


def write_coefficients(frame: pd.DataFrame, path: Path) -> None:
    """Write the coefficients of the three surfaces, and print them."""
    columns = [
        "site",
        "lat",
        "lon",
        "f_iso",
        "f_vol",
        "f_geo",
        "k1p",
        "k2p",
        "shape_view",
        "dhr",
        "bhr",
        "skyl",
        "albedo",
        "a",
        "a_black",
        "brf_neg_vza",
        "negative_flux_pct",
    ]
    kept = [c for c in columns if c in frame.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[kept].to_csv(path, index=False)
    print(f">>> coefficients written to {path}", flush=True)
    print(frame[kept].to_string(index=False), flush=True)


def landscapes(
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
    scale: float = 1.0,
) -> list[tuple[str, ImageDict]]:
    """Return the six training landscapes, optionally rescaled."""
    common = {
        "res_km": run.res_km,
        "rho_min": 0.0,
        "rho_max": args.rho_max * scale,
        "bands": [band],
        "n": run.n,
    }
    out: list[tuple[str, ImageDict]] = []
    for value in args.scales:
        out.append((f"gauss{value:g}", gaussian_image_dict(sigma=value, **common)))
    for value in args.scales:
        out.append((f"disk{value:g}", disk_image_dict(radius=value, **common)))
    return out


def truth_of(band: SensorBand, args: argparse.Namespace, run: RunConfig) -> dict:
    """Return the unscaled surface reflectance of every landscape."""
    return {
        name: image[band]["rho_s"].squeeze(drop=True)
        for name, image in landscapes(band, args, run)
    }


def simulate(
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
    frame: pd.DataFrame,
) -> tuple[list[tuple[str, ImageDict]], dict[str, list[tuple[str, ImageDict]]]]:
    """Return the Lambertian scenes, and the RTLS scenes keyed by site."""
    cfg = make_full_config(
        bands=[band],
        aot=args.aot,
        rh=args.rh,
        h=args.h,
        href=args.href,
        sza=args.sza,
        vza=args.vza,
        saa=args.saa,
        vaa=args.vaa,
        species={args.species: 1.0},
    )
    pipeline = {"nr": args.nr, "n_ph": run.n_ph}

    # Shared between clamped and unclamped modes
    print(">>> forward pipeline, Lambertian", flush=True)
    store = CacheStore(lambertian_cache(run))
    lambertian = [
        (name, run_forward_pipeline(image, **cfg, cache=store, **pipeline))
        for name, image in landscapes(band, args, run)
    ]

    # The cache key ignores the surface patch, so each surface needs a
    # store of its own or it reads another's result back.
    rtls: dict[str, list[tuple[str, ImageDict]]] = {}
    for row in frame.itertuples():
        print(f">>> forward pipeline, RTLS {row.site}", flush=True)
        with rtls_surface(row.k1p, row.k2p):
            rtls[row.site] = [
                (
                    name,
                    run_forward_pipeline(
                        image,
                        **cfg,
                        cache=CacheStore(run.cache_dir + f"/rtls_{row.site}"),
                        **pipeline,
                    ),
                )
                for name, image in landscapes(band, args, run, 1.0 / row.shape_view)
            ]
    return lambertian, rtls


def lambertian_cache(run: RunConfig) -> str:
    """Return the Lambertian store, shared across clamp modes."""
    return run.cache_dir.removesuffix("-clamped") + "/lambertian"


def brdf_scalars(
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
    row: pd.Series,
) -> xr.Dataset:
    """Return the six radiative quantities computed over the RTLS surface."""
    cfg = make_full_config(
        bands=[band],
        aot=args.aot,
        rh=args.rh,
        h=args.h,
        href=args.href,
        sza=args.sza,
        vza=args.vza,
        saa=args.saa,
        vaa=args.vaa,
        species={args.species: 1.0},
    )
    pipeline = RadiativePipeline(
        atmo_config=cfg["atmo_config"],
        geo_config=cfg["geo_config"],
        spectral_config=cfg["spectral_config"],
        remove_rayleigh=False,
        rtls=(1.0 / row.shape_view, row.k1p, row.k2p),
        cache=CacheStore(run.cache_dir + f"/scalars_{row.site}"),
    )
    return pipeline(ImageDict({band: xr.Dataset()}))[band]


def _like(new: xr.DataArray, old: xr.DataArray) -> xr.DataArray:
    """Return *new* carrying the dimensions *old* has, and no others."""
    extra = [d for d in new.dims if d not in old.dims]
    sizes = {d: new.sizes[d] for d in extra}
    if any(n != 1 for n in sizes.values()):
        raise RuntimeError(
            f"cannot drop {sizes} from {new.name}: the scene's own copy has "
            f"{dict(old.sizes)} and the swept axes are not singletons."
        )
    return new.squeeze(extra, drop=True) if extra else new


def errors(
    scenes: list[tuple[str, ImageDict]],
    truth: dict[str, xr.DataArray],
    kernel: xr.DataArray,
    band: SensorBand,
    device: str,
    *,
    scalars: xr.Dataset | None = None,
    deconvolve: bool = True,
) -> dict[str, float]:
    """Return the radial RMSE of the retrieval, per landscape."""
    out: dict[str, float] = {}
    for name, scene in scenes:
        ds = scene[band]
        if scalars is not None:
            ds = ds.assign({v: _like(scalars[v], ds[v]) for v in RADIATIVE_VARS})
        estimate, uniform = correct(ds, band, kernel, device)
        got = estimate if deconvolve else uniform
        out[name] = float(
            rmse(got, truth[name], mask=truth[name], radial=True, device=device)
        )
    return out


ARMS = ("lam_nopsf", "lam_psf", "brdf_nopsf", "brdf_psf")


def report(
    lambertian: dict[str, dict[str, float]],
    rtls: dict[str, dict[str, dict[str, float]]],
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Assemble, print and write the error table.

    The Lambertian scene has no BRDF arm: assuming the surface it actually
    has *is* the Lambertian assumption.
    """
    names = list(lambertian["lam_psf"])
    rows = []
    for name in names:
        row = {"landscape": name}
        row.update({f"lambertian_{arm}": lambertian[arm][name] for arm in ARMS[:2]})
        for site, arms in rtls.items():
            row.update({f"{site}_{arm}": arms[arm][name] for arm in ARMS})
        rows.append(row)

    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

    print("\n>>> radial RMSE of the retrieved surface reflectance\n", flush=True)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    informative = [n for n in names if n not in UNINFORMATIVE]

    # A mean of ratios, not a ratio of means: the errors span an order
    # of magnitude, so each landscape counts once.
    def mean(numerator: dict[str, float], reference: dict[str, float]) -> float:
        """Return the mean ratio over the informative landscapes."""
        return float(np.mean([numerator[n] / reference[n] for n in informative]))

    print(
        f"\n>>> where the error of the operational correction comes from"
        f"\n    ({', '.join(UNINFORMATIVE)} set aside: edge-dominated)\n",
        flush=True,
    )
    print(f"{'surface':<24} {'a':>7} {'PSF gain':>10} {'radiative':>11} {'rest':>7}")
    print(
        f"{'lambertian':<24} {1.0:7.3f} "
        f"{mean(lambertian['lam_psf'], lambertian['lam_nopsf']):9.2f}x "
        f"{'-':>11} {'100%':>7}"
    )
    for row in frame.itertuples():
        arms = rtls[row.site]
        rest = mean(arms["brdf_psf"], arms["lam_psf"])
        print(
            f"{row.site:<24} {row.a:7.3f} "
            f"{mean(arms['lam_psf'], arms['lam_nopsf']):9.2f}x "
            f"{1.0 - rest:10.0%} {rest:7.0%}"
        )

    print(f"\n>>> table written to {path}", flush=True)
    return table


def build_parser() -> argparse.ArgumentParser:
    """Return the parser, with the geometry and the ensemble options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wl", type=float, default=665.0)
    parser.add_argument("--modis-band", type=int, default=1)
    parser.add_argument("--aot", type=float, default=0.4)
    parser.add_argument("--rh", type=float, default=50.0)
    parser.add_argument(
        "--aerosol",
        choices=("Continental", "Maritime"),
        default="Continental",
        help="atmosphere the skylight fraction is read for",
    )
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--href", type=float, default=2.0)
    parser.add_argument("--sza", type=float, default=40.0)
    parser.add_argument("--vza", type=float, default=0.0)
    parser.add_argument("--saa", type=float, default=0.0)
    parser.add_argument("--vaa", type=float, default=0.0)
    parser.add_argument("--species", type=str, default="sulphate")
    parser.add_argument("--rho-max", type=float, default=0.3)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 50.0],
        help="landscape sizes in km, three Gaussians and three disks",
    )
    parser.add_argument("--nr", type=int, default=500)
    parser.add_argument("--start-date", type=str, default="A2020169")
    parser.add_argument("--end-date", type=str, default="A2020217")
    parser.add_argument(
        "--sites-csv",
        type=Path,
        default=REPO_ROOT / "inputs" / "modis" / "mcd43a1_sites.csv",
        help="cached MODIS ensemble; reused when present",
    )
    parser.add_argument("--refresh-sites", action="store_true")
    parser.add_argument("--appeears-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--clamp",
        action="store_true",
        help="hold the Ross-Li BRDF at zero where the fit goes negative; "
        "requires the same clamp in Smart-G, see tools/clamp_brdf.py",
    )
    add_credentials_arguments(parser)
    return parser


def check_band_agrees(wl_nm: float, modis_band: int) -> None:
    """Refuse to run when the simulation and the BRDF fit disagree on colour.

    The Ross-Li weights are per band, and so is the skylight fraction.
    Reading them on one band and simulating on another is silent: a desert
    is far brighter at 858 than at 665 nm, and nothing downstream says so.
    """
    low, high = MODIS_BANDS[modis_band]
    if low <= wl_nm <= high:
        print(
            f">>> band: {wl_nm:g} nm inside MODIS band {modis_band} "
            f"({low:g}-{high:g} nm)",
            flush=True,
        )
        return
    raise ValueError(
        f"the simulation runs at {wl_nm:g} nm but the BRDF coefficients are "
        f"read on MODIS band {modis_band}, which covers {low:g} to {high:g} "
        "nm. Set --wl and --modis-band on the same colour."
    )


def check_clamp_agrees(requested: bool) -> None:
    """Refuse to run when Python and the CUDA kernel disagree.

    A mismatch gives no error and a plausible number, the correction being
    judged against a truth simulated for another surface.  It is the one
    failure this study cannot see in its own output.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from clamp_brdf import clamp_state

    kernel = clamp_state()
    if kernel == requested:
        print(f">>> BRDF clamp: {'ON' if kernel else 'OFF'} on both sides", flush=True)
        return
    raise RuntimeError(
        f"the BRDF clamp is {'ON' if kernel else 'OFF'} in Smart-G and "
        f"{'ON' if requested else 'OFF'} here. Run "
        f"`python tools/clamp_brdf.py --{'on' if requested else 'off'}` "
        "first, or drop --clamp."
    )


def main() -> None:
    """Measure what the Lambertian assumption costs on three surfaces."""
    run, args = parse_run(__doc__.splitlines()[0], build_parser())
    run = run.resolve(n=1999, res_km=0.10, cache_dir=str(get_cache_path() / "hotspot"))
    if run.smoke:
        args.scales = [1.0, 5.0]
        args.nr = 20

    check_clamp_agrees(args.clamp)
    check_band_agrees(args.wl, args.modis_band)
    if args.clamp:
        run = run.resolve(cache_dir=run.cache_dir + "-clamped")
    # The kernel is fitted per geometry, so a sun zenith is a whole run
    # of its own and must not read another one's cache back.
    run = run.resolve(cache_dir=f"{run.cache_dir}/sza{args.sza:g}")

    # Named per geometry: the anisotropy on show depends on it, 0.03
    # from Lambertian at a sun zenith of 40 degrees against 0.19 at 10.
    suffix = f"_sza{args.sza:g}" + ("_clamped" if args.clamp else "")
    band = band_from_wl(args.wl)
    frame = selected_surfaces(args)
    write_coefficients(frame, OUTPUT / f"hotspot_coefficients{suffix}.csv")

    lambertian_scenes, rtls_scenes = simulate(band, args, run, frame)

    print(">>> fitting the operational kernel on the Lambertian set", flush=True)
    kernel, params = train(lambertian_scenes, band, run)
    print(f"    {params}", flush=True)

    truth = truth_of(band, args, run)
    common = (truth, kernel, band, run.device)
    lambertian_arms = {
        "lam_nopsf": errors(lambertian_scenes, *common, deconvolve=False),
        "lam_psf": errors(lambertian_scenes, *common),
    }
    # Four arms per surface: kernel or not, own scalar terms or
    # Lambertian ones.  Only the second axis needs a new simulation.
    rtls_arms = {}
    for row in frame.itertuples():
        scenes = rtls_scenes[row.site]
        print(f">>> radiative terms over the RTLS surface, {row.site}", flush=True)
        scalars = brdf_scalars(band, args, run, row)
        rtls_arms[row.site] = {
            "lam_nopsf": errors(scenes, *common, deconvolve=False),
            "lam_psf": errors(scenes, *common),
            "brdf_nopsf": errors(scenes, *common, scalars=scalars, deconvolve=False),
            "brdf_psf": errors(scenes, *common, scalars=scalars),
        }

    report(lambertian_arms, rtls_arms, frame, OUTPUT / f"hotspot_rmse{suffix}.csv")


if __name__ == "__main__":
    main()
