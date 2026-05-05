"""Loss landscape for the GG PSF model under four training configurations.

Four 50×50 grids over the (n, sigma) parameter space are shown, each
built from a different set of training scenes:

  (a) Single Gaussian sigma = 1.0 km
  (b) Single Gaussian sigma = 50.0 km
  (c) Three Gaussians (sigma = 1, 5, 50 km)
  (d) Three Gaussians + three disks (radius = 1, 5, 50 km)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from adjeff.api import make_full_config, run_forward_pipeline
from adjeff.core import (
    GaussGeneralPSF,
    PSFGrid,
    S2Band,
    disk_image_dict,
    gaussian_image_dict,
)
from adjeff.optim import Loss, TrainingImages, loss_landscape
from adjeff.optim.metrics import Metric
from adjeff.utils import CacheStore

plt.style.use(["science", "nature"])

BAND = S2Band.B03
RES_KM = 0.12
N = 1999
N_PH = int(1e5)
N_SAMPLES = 5
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

    # Build all six training scenes once and reuse across configs
    gauss_scenes = [
        run_forward_pipeline(
            gaussian_image_dict(sigma=s, res_km=RES_KM, bands=[BAND], n=N),
            **cfg,
            n_ph=N_PH,
            cache=cache,
        )
        for s in (1.0, 5.0, 50.0)
    ]
    disk_scenes = [
        run_forward_pipeline(
            disk_image_dict(radius=r, res_km=RES_KM, bands=[BAND], n=N),
            **cfg,
            n_ph=N_PH,
            cache=cache,
        )
        for r in (1.0, 5.0, 50.0)
    ]

    train_configs = [
        (
            r"(a) Gauss $1.0~km$",
            TrainingImages(images=[gauss_scenes[0]], weights=[1.0]),
        ),
        (
            r"(b) Gauss $50.0~km$",
            TrainingImages(images=[gauss_scenes[2]], weights=[1.0]),
        ),
        (
            r"(c) Full Gauss",
            TrainingImages(images=gauss_scenes, weights=[1.0, 1.0, 1.0]),
        ),
        (
            r"(d) Full Gauss + Disk",
            TrainingImages(
                images=gauss_scenes + disk_scenes,
                weights=[1.0] * 6,
            ),
        ),
    ]

    # Parameter grid — sigma log-spaced in [1e-6, 1] km, n linear in [0.1, 0.4]
    grid = PSFGrid(res=RES_KM, n=N)
    sigma_vals = np.logspace(-6, 0, N_SAMPLES).astype(np.float32)
    n_vals = np.linspace(0.1, 0.4, N_SAMPLES).astype(np.float32)
    psf_modules = [
        GaussGeneralPSF(grid, BAND, sigma=float(s), n=float(n))
        for s in sigma_vals
        for n in n_vals
    ]
    loss_fn = Loss(Metric.RMSE_RAD)

    # Axis extent: x = n, y = sigmoid(sigma) in [1e-6, 1]
    # Row 0 (sigma_vals[0]=1e-6) maps to y_top=1.0 with origin="upper"
    extent = (0.1, 0.4, 1e-6, 1.0)
    aspect_ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])

    tick_factor = 1.2
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(7.4, 6),
    )

    im = None
    for ax, (title, train_images) in zip(axes.flat, train_configs):
        losses = loss_landscape(
            train_images=train_images,
            band=BAND,
            psf_modules=psf_modules,
            loss=loss_fn,
            device="cpu",
        ).reshape(N_SAMPLES, N_SAMPLES)

        im = ax.imshow(losses, extent=extent, origin="upper")
        ax.set_aspect(aspect_ratio)
        ax.set_title(title, pad=10.0, fontsize=12 * tick_factor)
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

    for ax in axes[:, 0]:
        ax.set_yticks([1e-6, 0.333, 0.666, 1.0])
        ax.set_yticklabels(
            [r"$10^{0}$", r"$10^{-2}$", r"$10^{-4}$", r"$10^{-6}$"]
        )
        ax.set_ylabel(r"$\sigma~[km]$", fontsize=12 * tick_factor)

    for ax in axes[-1, :]:
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4"])
        ax.set_xlabel(r"$n$", fontsize=12 * tick_factor)

    assert im is not None
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_label(
        r"Radial $\mathrm{RMSE}$",
        rotation=270,
        labelpad=15,
        fontsize=12 * tick_factor,
    )
    cbar.ax.tick_params(
        axis="y",
        which="major",
        width=1.5,
        length=6,
        labelsize=10 * tick_factor,
    )
    cbar.ax.tick_params(
        axis="y",
        which="minor",
        width=1.5,
        length=3,
        labelsize=8 * tick_factor,
    )
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.5)

    plt.savefig(FIGS_DIR / "figure2.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
