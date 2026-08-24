"""Scatter plot: radial RMSE loss vs encircled-energy radius for GG kernels.

A grid over the (n, sigma) parameter space is evaluated.  For each point
the loss (trained on three Gaussian fields) and three encircled-energy
radii (EE10%, EE50%, EE99%) are computed.  The scatter reveals which
kernel shapes are physically plausible vs. which suffer from a high loss.
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

BAND = S2Band.B03
N_SAMPLES = 8


def main() -> None:
    run, _ = parse_run(__doc__.splitlines()[0])
    use_article_style()

    scenes = gauss_scenes(BAND, run)
    train_images = TrainingImages(images=scenes, weights=[1.0] * len(scenes))

    psf_modules = gg_parameter_grid(BAND, run, run.n_samples or N_SAMPLES)

    losses = loss_landscape(
        train_images=train_images,
        band=BAND,
        psf_modules=psf_modules,
        loss=Loss(Metric.RMSE_RAD),
        device="cpu",
    )
    metrics = energy_radius_landscape(psf_modules=psf_modules)
    ee_max = max(float(np.max(v)) for v in metrics.values())

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
