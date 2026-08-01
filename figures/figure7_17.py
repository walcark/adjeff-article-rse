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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from adjeff.api import (
    make_full_config,
    make_model,
    optimize_adam_lbfgs,
    run_forward_pipeline,
)
from adjeff.core import KingPSF, S2Band, disk_image_dict
from adjeff.modules.models.unif2surface import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages
from adjeff.utils import CacheStore

plt.style.use(["science", "nature"])

cache = CacheStore("/tmp/adjeff-figures")
RES_KM = 0.05
N = 3999
N_PH = int(1e5)
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

SWEEP_LABEL = {
    "aot": lambda v: f"AOT $= {v}$",
    "h": lambda v: f"$h = {v}$ km",
    "href": lambda v: f"$h_{{ref}} = {v}$ km",
    "wl": lambda v: f"$\\lambda = {v:.0f}$ nm",
    "sza": lambda v: f"SZA $= {v}°$",
    "vza": lambda v: f"VZA $= {v}°$",
}


def detect_sweep(args):
    """Return (sweep_var, sweep_vals) from parsed args."""
    candidates = {
        "aot": args.aot,
        "h": args.h,
        "href": args.href,
        "wl": args.wl,
        "sza": args.sza,
        "vza": args.vza,
    }
    multi = [(k, v) for k, v in candidates.items() if len(v) > 1]
    if not multi:
        raise ValueError("At least one argument must have multiple values.")
    if len(multi) > 1:
        raise ValueError(f"Only one sweep variable allowed; got {[k for k, _ in multi]}.")
    return multi[0]


def run_one(
    band,
    aot,
    h,
    rh,
    href,
    sza,
    vza,
    saa,
    vaa,
    species,
    remove_rayleigh,
):
    """Optimise a KingPSF for one parameter configuration.

    Returns the frozen kernel DataArray.
    """
    cfg = make_full_config(
        bands=[band],
        aot=[aot],
        h=h,
        rh=rh,
        href=href,
        sza=sza,
        vza=vza,
        saa=saa,
        vaa=vaa,
        species={species: 1.0},
    )

    scenes = []
    for radius in [1.0, 5.0, 50.0]:
        scene = disk_image_dict(
            radius=radius, res_km=RES_KM, bands=[band], n=N
        )
        scene = run_forward_pipeline(
            scene, **cfg, remove_rayleigh=remove_rayleigh, n_ph=N_PH, cache=cache,
        )
        scenes.append(scene)

    train_images = TrainingImages(images=scenes, weights=[1.0, 1.0, 1.0])

    model = make_model(
        Unif2Surface,
        KingPSF,
        [band],
        res_km=RES_KM,
        n=N,
        init_parameters={"sigma": 0.1, "gamma": 1.0},
    )
    psf_dict = optimize_adam_lbfgs(
        model, train_images, Loss(Metric.RMSE_RAD)
    )
    return psf_dict.kernel(band).squeeze()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", required=True, help="Output figure name.")
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
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    sweep_var, sweep_vals = detect_sweep(args)

    kernels = []
    labels = []
    for val in sweep_vals:
        band = WL_TO_BAND[val] if sweep_var == "wl" else WL_TO_BAND[args.wl[0]]
        kernel = run_one(
            band=band,
            aot=val if sweep_var == "aot" else args.aot[0],
            h=val if sweep_var == "h" else args.h[0],
            rh=args.rh,
            href=val if sweep_var == "href" else args.href[0],
            sza=val if sweep_var == "sza" else args.sza[0],
            vza=val if sweep_var == "vza" else args.vza[0],
            saa=args.saa,
            vaa=args.vaa,
            species=args.species,
            remove_rayleigh=args.remove_rayleigh,
        )
        kernels.append(kernel)
        labels.append(SWEEP_LABEL[sweep_var](val))

    # Build log-scale min/max across all kernels for consistent y-axis
    all_vals = np.concatenate([k.values.ravel() for k in kernels])
    all_pos = all_vals[all_vals > 0]
    y_min = 10 ** np.floor(np.log10(all_pos.min()))
    y_max = 10 ** np.ceil(np.log10(all_pos.max()))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    tick_factor = 1.2

    for kernel, label in zip(kernels, labels):
        prof = kernel.adjeff.radial()
        r = prof.coords["r"].values
        v = prof.values
        cdf = kernel.adjeff.radial_cdf()
        v_cdf = cdf.values

        opts = dict(linewidth=1.3)
        axes[0].plot(r, v, label=label, **opts)
        axes[1].plot(r, v_cdf, label=label, **opts)

    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(0.0, 1.0)

    axes[0].set_title(r"(a) PSF", pad=5, fontsize=12 * tick_factor)
    axes[1].set_title(
        r"(b) Encircled Energy", pad=5, fontsize=12 * tick_factor
    )

    axes[0].set_ylabel(r"$P_{5S}(r)$", fontsize=12 * tick_factor)
    axes[1].set_ylabel(
        r"$\mathrm{CDF}[P_{5S}](r)$", fontsize=12 * tick_factor
    )

    for ax in axes:
        ax.set_xlim(0, 160)
        ax.set_xlabel(r"Radius $r$ [km]", fontsize=12 * tick_factor)
        ax.tick_params(
            axis="both", which="major",
            width=1.5, length=6, labelsize=10 * tick_factor,
        )
        ax.tick_params(
            axis="both", which="minor",
            width=1.5, length=3, labelsize=8 * tick_factor,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    axes[0].legend(loc="upper right", fontsize=10 * tick_factor)
    axes[1].legend(loc="lower right", fontsize=10 * tick_factor)

    fig.tight_layout()
    plt.savefig(FIGS_DIR / f"{args.figure}.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
