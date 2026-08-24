"""PSF sensitivity to atmospheric and geometric parameters (figures 7-17).

For each value of the swept parameter, a KingPSF is optimised on three
disk training fields (radii 1, 5, 50 km).  The resulting kernels are
compared on two subplots: radial profile (log scale) and encircled energy.

Usage
-----
python figure7_17.py --figure figure7 \\
    --aot 0.1 0.3 0.5 0.7 \\
    --rh 50.0 --h 0.0 --href 2.0 --wl 560.0 \\
    --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \\
    --species blackcar --remove_rayleigh
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from adjeff.api import make_full_config, make_model, optimize_adam_lbfgs
from adjeff.core import ImageDict, KingPSF, SensorBand, psf_kernel
from adjeff.modules.models import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages
from adjeff_article_1.runconfig import RunConfig, parse_run
from adjeff_article_1.scenes import disk_scenes
from adjeff_article_1.shim import wl_to_band
from adjeff_article_1.style import (
    font,
    save,
    style_axes,
    use_article_style,
)

RES_KM = 0.12
N = 1999

# Legend entry for each value of the swept parameter.
SWEEP_LABEL = {
    "aot": lambda v: f"AOT $= {v}$",
    "h": lambda v: f"$h = {v}~km$",
    "href": lambda v: f"$h_{{ref}} = {v}~km$",
    "wl": lambda v: f"$\\lambda = {v:.0f}~nm$",
    "sza": lambda v: f"SZA $= {v}°$",
    "vza": lambda v: f"VZA $= {v}°$",
}

# Name of the swept parameter, as printed in the figure header.  Reviewer 3
# asked for the aerosol type to be readable from the graphic alone; the header
# also carries the relative humidity, which drives the particle size, and the
# parameter being swept, without which the legend values are ambiguous.
HEADER_SWEEP_LABEL = {
    "aot": "AOT",
    "h": "$h$ [km]",
    "href": "$h_\\mathrm{ref}$ [km]",
    "wl": "$\\lambda$ [nm]",
    "sza": "$\\theta_s$ [$^\\circ$]",
    "vza": "$\\theta_v$ [$^\\circ$]",
}

# Displayed name of each aerosol species. Figures sweeping the same variable
# for different species are otherwise indistinguishable from the graphic alone.
SPECIES_LABEL = {
    "blackcar": "Black carbon",
    "sulphate": "Sulphate",
    "seasalt": "Sea salt",
}

SWEEPABLE = list(SWEEP_LABEL)


def detect_sweep(args: argparse.Namespace) -> tuple[str, list[float]]:
    """Return ``(sweep_var, sweep_vals)``, the one argument given several values.

    Raises
    ------
    ValueError
        If no argument or more than one argument carries several values.
    """
    multi = [
        (name, getattr(args, name))
        for name in SWEEPABLE
        if len(getattr(args, name)) > 1
    ]
    if not multi:
        raise ValueError("At least one argument must have multiple values.")
    if len(multi) > 1:
        raise ValueError(
            f"Only one sweep variable allowed; got {[k for k, _ in multi]}."
        )
    return multi[0]


def optimised_kernel(
    band: SensorBand,
    run: RunConfig,
    scenes: list[ImageDict],
) -> xr.DataArray:
    """Optimise a KingPSF on *scenes* and return its frozen kernel."""
    model = make_model(
        Unif2Surface,
        KingPSF,
        [band],
        res_km=run.res_km,
        n=run.n,
        init_parameters={"sigma": 0.1, "gamma": 1.0},
        device=run.device,
    )
    tree = optimize_adam_lbfgs(
        model,
        TrainingImages(images=scenes, weights=[1.0] * len(scenes)),
        Loss(Metric.RMSE_RAD),
        device=run.device,
    )
    return psf_kernel(tree, band).squeeze()


def run_one(
    value: float,
    sweep_var: str,
    args: argparse.Namespace,
    run: RunConfig,
) -> xr.DataArray:
    """Return the kernel optimised for one point of the sweep.

    Every sweepable argument is read at its first value except the one
    being swept, which takes *value*.
    """

    def at(name: str) -> float:
        return value if sweep_var == name else getattr(args, name)[0]

    band = wl_to_band(at("wl"))
    cfg = make_full_config(
        bands=[band],
        aot=[at("aot")],
        h=at("h"),
        rh=args.rh,
        href=at("href"),
        sza=at("sza"),
        vza=at("vza"),
        saa=args.saa,
        vaa=args.vaa,
        species={args.species: 1.0},
    )
    scenes = disk_scenes(
        band, run, cfg=cfg, remove_rayleigh=args.remove_rayleigh
    )
    return optimised_kernel(band, run, scenes)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the sweep options of this figure family."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--figure",
        default="figure7_17",
        help="Output figure name (default: the script name).",
    )
    parser.add_argument("--aot", nargs="+", type=float, default=[0.4])
    parser.add_argument("--rh", type=float, default=50.0)
    parser.add_argument("--h", nargs="+", type=float, default=[0.0])
    parser.add_argument("--href", nargs="+", type=float, default=[2.0])
    parser.add_argument("--wl", nargs="+", type=float, default=[560.0])
    parser.add_argument("--sza", nargs="+", type=float, default=[40.0])
    parser.add_argument("--vza", nargs="+", type=float, default=[8.0])
    parser.add_argument("--saa", type=float, default=0.0)
    parser.add_argument("--vaa", type=float, default=0.0)
    parser.add_argument("--species", type=str, default="sulphate")
    parser.add_argument("--remove_rayleigh", action="store_true")
    return parser


def plot(
    kernels: list[xr.DataArray],
    labels: list[str],
    sweep_var: str,
    args: argparse.Namespace,
    run: RunConfig,
) -> None:
    """Draw the radial profile and the encircled energy of every kernel."""
    all_vals = np.concatenate([k.values.ravel() for k in kernels])
    all_pos = all_vals[all_vals > 0]
    y_min = 10 ** np.floor(np.log10(all_pos.min()))
    y_max = 10 ** np.ceil(np.log10(all_pos.max()))

    # Height grown from 3.0 to leave room for the header without shrinking the
    # plotting area.
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.25), layout="constrained")

    for kernel, label in zip(kernels, labels):
        prof = kernel.adjeff.radial()
        cdf = kernel.adjeff.radial(stat="cdf")
        opts = dict(label=label, linewidth=1.3)
        axes[0].plot(prof.coords["r"].values, prof.values, **opts)
        axes[1].plot(cdf.coords["r"].values, cdf.values, **opts)

    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(0.0, 1.0)

    # Anchor the decade ticks on y_max: the log locator would otherwise pick a
    # different decade parity from one figure of the series to the next.
    n_decades = round(np.log10(y_max / y_min))
    axes[0].set_yticks(
        [y_max * 10.0 ** (-2 * k) for k in range(n_decades // 2 + 1)]
    )

    axes[0].set_title(r"(a) PSF", pad=5, fontsize=font())
    axes[1].set_title(r"(b) Encircled Energy", pad=5, fontsize=font())
    axes[0].set_ylabel(r"$P_{5S}(r)$", fontsize=font())
    axes[1].set_ylabel(r"$\mathrm{CDF}[P_{5S}](r)$", fontsize=font())

    for ax in axes:
        ax.set_xlim(0, 160)
        ax.set_xlabel(r"Radius $r$ [km]", fontsize=font())
        style_axes(ax)

    axes[0].legend(loc="upper right", fontsize=font(10 / 12))
    axes[1].legend(loc="lower right", fontsize=font(10 / 12))

    species = SPECIES_LABEL.get(args.species, args.species)
    fig.suptitle(
        "\\textbf{" + species + "} aerosol, "
        f"$\\mathrm{{RH}} = {args.rh:.0f}\\%$"
        "$\\;|\\;$"
        f"varying {HEADER_SWEEP_LABEL[sweep_var]}",
        fontsize=font(13 / 12),
    )

    save(fig, args.figure, run.figs_dir)
    if not run.smoke:
        plt.show()


def main() -> None:
    run, args = parse_run(__doc__.splitlines()[0], build_parser())
    run = run.resolve(n=N, res_km=RES_KM)
    use_article_style()

    # A sweep is what this script draws, so a bare `--smoke` has nothing
    # to plot.  Give it the smallest one rather than making the smoke
    # runner carry a table of per-script arguments.
    if run.smoke and not any(
        len(getattr(args, name)) > 1 for name in SWEEPABLE
    ):
        args.aot = [0.1, 0.5]

    sweep_var, sweep_vals = detect_sweep(args)

    kernels = [run_one(v, sweep_var, args, run) for v in sweep_vals]
    labels = [SWEEP_LABEL[sweep_var](v) for v in sweep_vals]

    plot(kernels, labels, sweep_var, args, run)


if __name__ == "__main__":
    main()
