"""Scatter plot: radial RMSE loss vs encircled-energy radius for GG kernels.

An 80×80 grid over the (n, sigma) parameter space is evaluated.
For each point the loss (trained on three Gaussian fields) and three
encircled-energy radii (EE10%, EE50%, EE99%) are computed.  The scatter
reveals which kernel shapes are physically plausible vs. which suffer from
a high loss.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from adjeff.api import make_full_config, run_forward_pipeline
from adjeff.core import GeneralizedGaussianPSF, PSFGrid, S2Band, gaussian_image_dict
from adjeff.optim import (
    Loss,
    Metric,
    TrainingImages,
    energy_radius_landscape,
    loss_landscape,
)
from adjeff.utils import CacheStore

plt.style.use(["science", "nature"])

BAND = S2Band.B03
RES_KM = 0.05
N = 3999
N_PH = int(1e5)
N_SAMPLES = 8
FIGS_DIR = Path(__file__).parent.parent / "output"
cache = CacheStore("/tmp/adjeff-figures")


def main() -> None:
    FIGS_DIR.mkdir(exist_ok=True)

    cfg = make_full_config(
        bands=[BAND],
        aot=0.4,
        h=0.0,
        rh=50.0,
        href=2.0,
        sza=45.0,
        vza=8.0,
        saa=0.0,
        vaa=0.0,
        species={"sulphate": 1.0},
    )

    # Three Gaussian training scenes
    scenes = [
        run_forward_pipeline(
            gaussian_image_dict(sigma=s, res_km=RES_KM, bands=[BAND], n=N),
            **cfg,
            n_ph=N_PH,
            cache=cache,
        )
        for s in (1.0, 5.0, 50.0)
    ]
    train_images = TrainingImages(images=scenes, weights=[1.0, 1.0, 1.0])

    # Parameter grid — sigma log-spaced in [1e-6, 1] km, n linear in [0.1, 0.4]
    grid = PSFGrid(res=RES_KM, n=N)
    sigma_vals = np.logspace(-6, 0, N_SAMPLES).astype(np.float32)
    n_vals = np.linspace(0.1, 0.4, N_SAMPLES).astype(np.float32)
    psf_modules = [
        GeneralizedGaussianPSF(grid, BAND, sigma=float(s), n=float(n))
        for s in sigma_vals
        for n in n_vals
    ]
    loss_fn = Loss(Metric.RMSE_RAD)

    losses = loss_landscape(
        train_images=train_images,
        band=BAND,
        psf_modules=psf_modules,
        loss=loss_fn,
        device="cpu",
    )

    metrics = energy_radius_landscape(psf_modules=psf_modules)

    tick_factor = 1.2
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        sharey=True,
        figsize=(8, 3),
    )

    ee_max = max(float(np.max(v)) for v in metrics.values())

    for idx, (ax, (label, radii)) in enumerate(
        zip(axes.flat, metrics.items())
    ):
        ax.scatter(losses, radii, s=2, alpha=0.4)
        ax.set_title(
            f"({'abc'[idx]}) {label}",
            pad=5,
            fontsize=12 * tick_factor,
        )
        ax.set_xlim(1e-3, 1e0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.1, ee_max)
        ax.set_xlabel("Radial RMSE", fontsize=12 * tick_factor)
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

    axes[0].set_ylabel(r"Radius $[km]$", fontsize=12 * tick_factor)

    fig.tight_layout()
    plt.savefig(FIGS_DIR / "figure3.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
