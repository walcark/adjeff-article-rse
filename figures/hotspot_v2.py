"""What the Lambertian assumption costs, on three surfaces that simulate cleanly.

`hotspot.py` samples eight MODIS sites, ranks them by anisotropy and keeps
the extremes.  It also inherits a problem: the Ross-Li fit is not
constrained positive, and the surfaces that are most anisotropic are the
ones whose BRF goes negative earliest at grazing view.  Smart-G does not
clip that, so those photons carry a negative weight in exactly the
angular range the far-field adjacency signal leaves the ground through.

This is the same study on three surfaces chosen so that question does not
arise, and reduced to what it takes to answer one question: how wrong is
the Lambertian assumption, in reflectance, on the landscapes the kernel
was trained on.

The three
---------
Not the three safest, which all sit on the same side of Lambertian and
would only test half of the problem.  These bracket it, and all three
keep under 1 % of their upward flux in the negative region:

===================== ======= ============== ===================
site                  a       BRF negative   negative flux
===================== ======= ============== ===================
skukuza-savanna       0.964   83 deg         0.79 %
libya4-desert         1.010   90 deg         0.011 %
konza-grassland       1.028   86 deg         0.36 %
===================== ======= ============== ===================

``a = DHR / BRF(view)`` is one when the surface is Lambertian, below one
when it is brighter towards the sensor than towards the hemisphere, and
above one in the other direction.  The sign matters: it decides which way
the retrieval is biased.

The geometry is the other variable
----------------------------------
``a`` is not a property of a surface alone: the hemispheric integral
depends on the sun, the reference reflectance on the sensor.  Measured
across sun zenith, at nadir view:

=========== ============ ============ ============ =================
sun zenith  skukuza      libya4       konza        sign
=========== ============ ============ ============ =================
10 deg      0.808        0.973        0.845        all below one
40 deg      0.972        1.010        1.032        two cross over
60 deg      1.158        1.050        1.262        all above one
=========== ============ ============ ============ =================

The manuscript's 40 degrees sits almost exactly where the effect
vanishes, which is why two of the three surfaces cost nothing there.
Running 10 and 60 as well separates the two readings left open: whether
the cost follows the *sign* of ``a - 1``, one geometry having them all
below and the other all above, or its magnitude.

What it produces
----------------
Named by geometry and clamp mode, so runs can be compared rather than
overwrite each other:

1. ``hotspot_v2_coefficients_szaN.csv``: the MODIS coefficients of the
   three, with their derived weights and diagnostics.
2. ``hotspot_v2_brdf_szaN.png``: the angular shape of the three against a
   Lambertian, each divided by its own value in the viewing direction,
   which is what the simulation applies.
3. A table of retrieval error per landscape, and the ratio to the
   Lambertian surface.  Numbers, not a figure.

``disk1`` and ``disk5`` are reported and set aside: their error is the
edge the deconvolution cannot resolve, twenty times the adjacency error,
and they discriminate no surface.  See ``UNINFORMATIVE``.

Usage
-----
python hotspot_v2.py --smoke              # a few minutes, checks the chain
python hotspot_v2.py --sza 40             # the manuscript's geometry
python hotspot_v2.py --sza 10 --clamp     # all surfaces below Lambertian
python hotspot_v2.py --sza 60 --clamp     # all above
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from adjeff.analysis import rmse
from adjeff.api import make_full_config, run_forward_pipeline
from adjeff.core import ImageDict, S2Band, SensorBand, disk_image_dict
from adjeff.core import gaussian_image_dict
from adjeff.utils import CacheStore

from adjeff_article_1.correction import correct
from adjeff_article_1.credentials import add_credentials_arguments
from adjeff_article_1.runconfig import RunConfig, parse_run
from adjeff_article_1.style import font, style_axes, use_article_style
from hotspot import (
    brf_onset,
    load_ensemble,
    rtls_shape,
    rtls_surface,
    train,
)

#: The three surfaces, chosen to bracket Lambertian while keeping under
#: 1 % of their upward flux where the Ross-Li fit turns negative.
CHOSEN = ("skukuza-savanna", "libya4-desert", "konza-grassland")

#: Landscapes whose retrieval error is dominated by an edge the
#: deconvolution cannot resolve rather than by adjacency, and which
#: therefore discriminate no surface.  A disk of 1 km on a 50 m grid has
#: a one-pixel edge: its Lambertian error is 0.018 against 0.0009 for a
#: Gaussian of the same size, twenty times larger, and it swings by 15 %
#: between two runs that differ only in their random draws.  Measured,
#: the most anisotropic surface scores 1.08 and 1.21 times the Lambertian
#: one there, against 5 to 8 everywhere else.
#:
#: They stay in the training set, which needs their high frequencies and
#: must remain the article's.  They are reported and then set aside, on a
#: criterion that is prior to the result and checkable: whether the
#: landscape is retrievable at all.
UNINFORMATIVE = ("disk1", "disk5")

OUTPUT = Path(__file__).resolve().parent.parent / "output"


def selected_surfaces(args: argparse.Namespace) -> pd.DataFrame:
    """Return the three sites, with their weights and diagnostics.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line, carrying the geometry and the ensemble
        location.

    Returns
    -------
    pd.DataFrame
        One row per site, ordered by anisotropy, with ``k1p``, ``k2p``,
        ``shape_view``, ``dhr``, ``a`` and ``brf_neg_vza``.
    """
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
    frame["a"] = frame["dhr"] / frame["shape_view"]
    frame["brf_neg_vza"] = [
        brf_onset(args.sza, k1p, k2p)
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]
    frame["negative_flux_pct"] = [
        negative_flux(args.sza, k1p, k2p)
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]
    return frame.sort_values("a").reset_index(drop=True)


def shape_of(
    sza_deg: float | np.ndarray,
    vza_deg: float | np.ndarray,
    raa_deg: float | np.ndarray,
    k1p: float,
    k2p: float,
    clamp: bool,
) -> np.ndarray:
    """Return the angular factor, clamped at zero or not.

    Clamping has to happen on both sides or neither.  `device.cu` decides
    what the photons do; this decides what `a`, the hemispheric integral
    and the rescaling describe.  If the two disagree, the study compares
    a correction against a truth simulated for another surface.
    """
    shape = rtls_shape(sza_deg, vza_deg, raa_deg, k1p, k2p)
    return np.maximum(shape, 0.0) if clamp else shape


def hemispheric(sza_deg: float, k1p: float, k2p: float, clamp: bool) -> float:
    """Return the black-sky albedo of the shape, one for a Lambertian.

    Reimplemented here rather than taken from `hotspot.rtls_dhr` so that
    the clamp can be applied inside the integrand: clamping afterwards
    would not remove the negative contribution, which is the point.
    """
    theta = np.linspace(1e-4, np.pi / 2.0 - 1e-4, 400)
    phi = np.linspace(0.0, 2.0 * np.pi, 241)
    grid_t, grid_p = np.meshgrid(theta, phi, indexing="ij")
    shape = shape_of(
        sza_deg, np.degrees(grid_t), np.degrees(grid_p), k1p, k2p, clamp
    )
    weighted = shape * np.cos(grid_t) * np.sin(grid_t)
    return float(
        np.trapezoid(np.trapezoid(weighted, phi, axis=1), theta) / np.pi
    )


def negative_flux(sza_deg: float, k1p: float, k2p: float, n_theta: int = 200) -> float:
    """Return the share of upward flux leaving where the BRF is negative.

    Weighted as the hemispheric integral is, by ``cos(theta) sin(theta)``,
    because that is the flux that feeds the adjacency term.  The solid
    angle alone overstates it: the grazing directions where the fit
    misbehaves are the ones the cosine suppresses.

    Parameters
    ----------
    sza_deg : float
        Sun zenith angle [deg].
    k1p, k2p : float
        Relative geometric and volumetric weights.
    n_theta : int
        Quadrature resolution in zenith.

    Returns
    -------
    float
        Percentage of the positive flux carried by the negative region.
    """
    theta = np.linspace(1e-4, np.pi / 2.0 - 1e-4, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, 121)
    grid_t, grid_p = np.meshgrid(theta, phi, indexing="ij")
    shape = rtls_shape(sza_deg, np.degrees(grid_t), np.degrees(grid_p), k1p, k2p)
    weight = np.cos(grid_t) * np.sin(grid_t)

    def integrate(values: np.ndarray) -> float:
        return float(np.trapezoid(np.trapezoid(values, phi, axis=1), theta))

    negative = integrate(np.where(shape < 0.0, shape, 0.0) * weight)
    positive = integrate(np.where(shape < 0.0, 0.0, shape) * weight)
    return 100.0 * abs(negative) / positive


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
        "a",
        "brf_neg_vza",
        "negative_flux_pct",
    ]
    kept = [c for c in columns if c in frame.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[kept].to_csv(path, index=False)
    print(f">>> coefficients written to {path}", flush=True)
    print(frame[kept].to_string(index=False), flush=True)


def plot_brdf(frame: pd.DataFrame, args: argparse.Namespace, path: Path) -> None:
    """Draw the angular shape of each surface through the principal plane.

    The view zenith runs from one side of the principal plane to the
    other, negative angles meaning a relative azimuth of 180 deg.  The
    hotspot sits where the sensor looks straight down the sun's own
    direction, which for a sun at *sza* is the positive side.

    Parameters
    ----------
    frame : pd.DataFrame
        The three surfaces, with ``k1p`` and ``k2p``.
    args : argparse.Namespace
        Carries ``sza``.
    path : Path
        Where to write the figure.
    """
    use_article_style()
    vza = np.linspace(-80.0, 80.0, 321)
    # A negative view zenith is the far side of the principal plane.
    raa = np.where(vza < 0.0, 180.0, 0.0)

    fig, ax = plt.subplots(figsize=(5.4, 3.4), layout="constrained")
    ax.axhline(
        1.0,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label="Lambertian",
        zorder=1,
    )
    for row in frame.itertuples():
        # Divided by its own value in the viewing direction, which is
        # what the simulation applies: `scaled_landscape` scales each
        # surface so the *observed* reflectance matches the Lambertian
        # one.  Only what leaves towards the hemisphere differs, and the
        # curves therefore all pass through one at the sensor's angle.
        shape = (
            shape_of(args.sza, np.abs(vza), raa, row.k1p, row.k2p, args.clamp)
            / row.shape_view
        )
        ax.plot(
            vza,
            shape,
            linewidth=1.4,
            label=f"{row.site}  ($a$ = {row.a:.3f})",
            zorder=2,
        )

    ax.axvline(args.vza, color="0.75", linewidth=0.8, linestyle=":", zorder=0)
    ax.axvline(args.sza, color="0.6", linewidth=0.8, zorder=0)
    ax.annotate(
        "sensor", xy=(args.vza, 1.0), xytext=(4, 4),
        textcoords="offset points", fontsize=font(10 / 12), color="0.4",
    )
    ax.annotate(
        "hotspot",
        xy=(args.sza, ax.get_ylim()[1]),
        xytext=(3, -10),
        textcoords="offset points",
        fontsize=font(10 / 12),
        color="0.4",
    )
    ax.set_xlabel(r"View zenith angle [$^\circ$]", fontsize=font())
    ax.set_ylabel(
        r"$\left[1 + k_1' F_1 + k_2' F_2\right] \,/\, $ value at sensor",
        fontsize=font(),
    )
    ax.set_title(
        rf"BRDF shape, normalised at the sensor, $\theta_s$ = {args.sza:g}$^\circ$, "
        rf"$\theta_v$ = {args.vza:g}$^\circ$",
        fontsize=font(),
        pad=6,
    )
    ax.legend(fontsize=font(10 / 12), loc="upper left")
    style_axes(ax)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f">>> wrote {path}", flush=True)


def landscapes(
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
    scale: float = 1.0,
) -> list[tuple[str, ImageDict]]:
    """Return the six training landscapes, optionally rescaled.

    Three Gaussians and three disks, as in Section 2.3.1.  *scale*
    multiplies both reflectance levels so that the field the sensor
    *observes* is the same whatever the surface's angular shape: only
    what leaves towards the hemisphere differs, which is the one thing
    under test.  The truth to compare against therefore stays the
    unscaled landscape.
    """
    common = dict(
        res_km=run.res_km,
        rho_min=0.0,
        rho_max=args.rho_max * scale,
        bands=[band],
        n=run.n,
    )
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
    """Run the forward pipeline once Lambertian and once per surface.

    Returns
    -------
    tuple
        The Lambertian scenes, and the RTLS scenes keyed by site name.
    """
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
    pipeline = dict(nr=args.nr, n_ph=run.n_ph)

    # The Lambertian surface has no BRDF, so the clamp cannot change it:
    # its store is deliberately shared between the two modes.  Recomputing
    # it per mode gave the reference different random draws, and the two
    # runs disagreed by 5.6 % on a quantity that is identical by
    # construction.  That noise was larger than what libya4 and konza
    # were being measured at.
    print(">>> forward pipeline, Lambertian", flush=True)
    store = CacheStore(lambertian_cache(run))
    lambertian = [
        (name, run_forward_pipeline(image, **cfg, cache=store, **pipeline))
        for name, image in landscapes(band, args, run)
    ]

    # The cache is keyed on the scene configuration, which the surface
    # patch does not alter, so each surface needs a store of its own or
    # it would silently read another surface's result back.
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
    """Return the Lambertian store, shared across clamp modes.

    The cache directory carries a `-clamped` suffix so that a clamped run
    cannot read an unclamped surface back.  The Lambertian scenes are the
    one exception: nothing about them depends on the clamp, and giving
    them one store makes the two runs share a reference rather than two
    noisy estimates of it.
    """
    return run.cache_dir.removesuffix("-clamped") + "/lambertian"


def retrieval_error(
    scenes: list[tuple[str, ImageDict]],
    truth: dict[str, xr.DataArray],
    kernel: xr.DataArray,
    band: SensorBand,
    device: str,
) -> dict[str, float]:
    """Return the radial RMSE of the retrieved reflectance, per landscape.

    The kernel is the operational one, fitted on the Lambertian training
    set and applied unchanged: that is the whole question, how it fares
    on a surface it was not trained for.
    """
    errors: dict[str, float] = {}
    for name, scene in scenes:
        estimate, uniform = correct(scene[band], band, kernel, device)
        errors[name] = float(
            rmse(estimate, truth[name], mask=uniform, radial=True, device=device)
        )
    return errors


def uncorrected_error(
    scenes: list[tuple[str, ImageDict]],
    truth: dict[str, xr.DataArray],
    kernel: xr.DataArray,
    band: SensorBand,
    device: str,
) -> dict[str, float]:
    """Return the error of doing no adjacency correction at all.

    The reference every corrected number should be read against: a
    correction that costs more than it saves is not one.
    """
    errors: dict[str, float] = {}
    for name, scene in scenes:
        _, uniform = correct(scene[band], band, kernel, device)
        errors[name] = float(
            rmse(uniform, truth[name], mask=uniform, radial=True, device=device)
        )
    return errors


def report(
    lambertian: dict[str, float],
    rtls: dict[str, dict[str, float]],
    no_correction: dict[str, float],
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Assemble, print and write the error table."""
    names = list(lambertian)
    rows = []
    for name in names:
        row = {
            "landscape": name,
            "no_correction": no_correction[name],
            "lambertian": lambertian[name],
        }
        row.update({site: errors[name] for site, errors in rtls.items()})
        rows.append(row)

    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

    print("\n>>> radial RMSE of the retrieved surface reflectance\n", flush=True)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    informative = [n for n in names if n not in UNINFORMATIVE]
    surfaces = list(rtls)

    # A mean of ratios rather than a ratio of means: the errors span an
    # order of magnitude across landscapes, so an arithmetic mean is the
    # worst landscape and little else.  Each landscape then counts once.
    print(
        f"\n>>> ratio to the Lambertian surface, per landscape"
        f"\n    ({', '.join(UNINFORMATIVE)} set aside: edge-dominated)\n",
        flush=True,
    )
    header = f"{'landscape':<12}" + "".join(f"{s:>24}" for s in surfaces)
    print(header)
    for name in names:
        mark = "  " if name in informative else " *"
        line = f"{name:<12}"
        for site in surfaces:
            line += f"{rtls[site][name] / lambertian[name]:23.2f}x"
        print(line + mark)

    print(f"\n>>> cost of the Lambertian assumption\n", flush=True)
    print(
        f"{'surface':<24} {'a':>7} {'|a-1|':>7} "
        f"{'mean ratio':>12} {'all six':>10}"
    )
    for row in frame.itertuples():
        useful = np.mean([rtls[row.site][n] / lambertian[n] for n in informative])
        every = np.mean([rtls[row.site][n] / lambertian[n] for n in names])
        print(
            f"{row.site:<24} {row.a:7.3f} {abs(row.a - 1):7.3f} "
            f"{useful:11.2f}x {every:9.2f}x"
        )

    gain = np.mean([no_correction[n] / lambertian[n] for n in informative])
    print(f"\n{'no correction at all':<24} {'-':>7} {'-':>7} {gain:11.2f}x")
    print(f">>> table written to {path}", flush=True)
    return table


def build_parser() -> argparse.ArgumentParser:
    """Return the parser, with the geometry and the ensemble options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wl", type=float, default=665.0)
    parser.add_argument("--modis-band", type=int, default=1)
    parser.add_argument("--aot", type=float, default=0.4)
    parser.add_argument("--rh", type=float, default=50.0)
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
        default=OUTPUT / "hotspot_mcd43a1_sites.csv",
        help="cached MODIS ensemble; reused when present",
    )
    parser.add_argument("--refresh-sites", action="store_true")
    parser.add_argument(
        "--brdf-source", choices=("auto", "ornl", "appeears"), default="auto"
    )
    parser.add_argument("--appeears-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--clamp",
        action="store_true",
        help="hold the Ross-Li BRDF at zero where the fit goes negative; "
        "requires the same clamp in Smart-G, see scripts/clamp_brdf.py",
    )
    add_credentials_arguments(parser)
    return parser


def check_clamp_agrees(requested: bool) -> None:
    """Refuse to run when Python and the CUDA kernel disagree.

    A mismatch produces no error and a plausible number: the correction
    would be judged against a truth simulated for a different surface.
    It is the one failure this study cannot detect from its own output,
    so it is checked before anything runs.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from clamp_brdf import clamp_state

    kernel = clamp_state()
    if kernel == requested:
        print(f">>> BRDF clamp: {'ON' if kernel else 'OFF'} on both sides",
              flush=True)
        return
    raise RuntimeError(
        f"the BRDF clamp is {'ON' if kernel else 'OFF'} in Smart-G and "
        f"{'ON' if requested else 'OFF'} here. Run "
        f"`python scripts/clamp_brdf.py --{'on' if requested else 'off'}` "
        "first, or drop --clamp."
    )


def main() -> None:
    """Measure what the Lambertian assumption costs on three surfaces."""
    run, args = parse_run(__doc__.splitlines()[0], build_parser())
    run = run.resolve(n=1999, res_km=0.10, cache_dir="/tmp/adjeff-hotspot-v2")
    if run.smoke:
        args.scales = [1.0, 5.0]
        args.nr = 20

    check_clamp_agrees(args.clamp)
    if args.clamp:
        run = run.resolve(cache_dir=run.cache_dir + "-clamped")
    # The kernel is fitted per geometry, so a sun zenith is a whole run
    # of its own and must not read another one's cache back.
    run = run.resolve(cache_dir=f"{run.cache_dir}/sza{args.sza:g}")

    # The geometry decides how much anisotropy there is to see: at a sun
    # zenith of 40 degrees the three surfaces sit within 0.03 of
    # Lambertian and two of them cross it, while at 10 degrees they reach
    # 0.19 and all sit below.  Runs are named so that the geometries can
    # be compared rather than overwrite each other.
    suffix = f"_sza{args.sza:g}" + ("_clamped" if args.clamp else "")
    band = S2Band.from_wl(args.wl)
    frame = selected_surfaces(args)
    write_coefficients(frame, OUTPUT / f"hotspot_v2_coefficients{suffix}.csv")
    plot_brdf(frame, args, run.figs_dir / f"hotspot_v2_brdf{suffix}.png")

    lambertian_scenes, rtls_scenes = simulate(band, args, run, frame)

    print(">>> fitting the operational kernel on the Lambertian set", flush=True)
    kernel, params = train(lambertian_scenes, band, run)
    print(f"    {params}", flush=True)

    truth = truth_of(band, args, run)
    report(
        retrieval_error(lambertian_scenes, truth, kernel, band, run.device),
        {
            site: retrieval_error(scenes, truth, kernel, band, run.device)
            for site, scenes in rtls_scenes.items()
        },
        uncorrected_error(lambertian_scenes, truth, kernel, band, run.device),
        frame,
        OUTPUT / f"hotspot_v2_rmse{suffix}.csv",
    )


if __name__ == "__main__":
    main()
