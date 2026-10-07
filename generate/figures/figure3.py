"""Scatter plot: radial RMSE loss vs encircled-energy radius for GG kernels.

For a Gaussian Generalized PSF model, the (n, sigma) space is explored and
for each couple (n, sigma), the radial RMSE loss and encircled-energy are
computed.

The radial RMSE loss expresses the ability of the PSF model to link TOA
reflectance to surface reflactance in the context of adjacency effects
correction.

The scatter reveals whether close-to-optimal kernels are close to each
other, which is in favour the the optimal PSF model is representative of
the true physics, and will not vary due to non-optimal optimisation.
"""

import matplotlib.pyplot as plt
import numpy as np
from adjeff.core import S2Band
from adjeff.optim import (
    Loss,
    Metric,
    TrainingImages,
    energy_radius_landscape,
    loss_landscape,
)

from adjeff_article_1.runconfig import parse_run
from adjeff_article_1.scenes import gauss_scenes, gg_parameter_grid
from adjeff_article_1.style import (
    font,
    panel_title,
    save,
    style_axes,
    use_article_style,
)

# Global parameters
BAND = S2Band.B03
RES_KM = 0.05
N = 3999
N_SAMPLES = 8


def main() -> None:

    # Parse input parameters
    run, _ = parse_run(__doc__.splitlines()[0])
    run = run.resolve(n=N, res_km=RES_KM, n_samples=N_SAMPLES)
    use_article_style()

    # Build the surface and compute rho_toa with Smart-G
    scenes = gauss_scenes(BAND, run)
    train_images = TrainingImages(images=scenes)

    # Generate of grid of GG PSF models
    psf_modules = gg_parameter_grid(BAND, run, run.n_samples)

    # Compute the radial RMSE loss and for each grid point
    losses = loss_landscape(
        train_images=train_images,
        band=BAND,
        psf_modules=psf_modules,
        loss=Loss(Metric.RMSE_RAD),
        device="cpu",
    )

    # Compute the energy radius for each grid point
    metrics = energy_radius_landscape(psf_modules=psf_modules)
    ee_max = max(float(np.max(v)) for v in metrics.values())

    # Plot the results
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 3))

    for idx, (ax, (label, radii)) in enumerate(zip(axes, metrics.items())):
        ax.scatter(losses, radii, s=2, alpha=0.4)
        panel_title(ax, idx, label)
        ax.set_xlim(1e-3, 1e0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.1, ee_max)
        ax.set_xlabel("Radial RMSE", fontsize=font())
        style_axes(ax)

    axes[0].set_ylabel(r"Radius $[km]$", fontsize=font())

    fig.tight_layout()
    save(fig, "figure3", run.figs_dir)
    if not run.smoke:
        plt.show()


if __name__ == "__main__":
    main()
