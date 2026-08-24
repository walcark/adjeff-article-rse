"""Loss landscape for the GG PSF model under four training configurations.

Four grids over the (n, sigma) parameter space are shown, each built from
a different set of training scenes:

  (a) Single Gaussian sigma = 1.0 km
  (b) Single Gaussian sigma = 50.0 km
  (c) Three Gaussians (sigma = 1, 5, 50 km)
  (d) Three Gaussians + three disks (radius = 1, 5, 50 km)
"""

import matplotlib.pyplot as plt

from adjeff.core import S2Band
from adjeff.optim import Loss, Metric, TrainingImages, loss_landscape
from adjeff_article_1.runconfig import parse_run
from adjeff_article_1.scenes import (
    disk_scenes,
    gauss_scenes,
    gg_parameter_grid,
)
from adjeff_article_1.style import (
    font,
    save,
    style_axes,
    style_colorbar,
    use_article_style,
)

BAND = S2Band.B03
RES_KM = 0.05
N = 3999
N_SAMPLES = 5

# x = n in [0.1, 0.4], y = sigma mapped to [1e-6, 1].  Row 0 of the
# landscape (sigma = 1e-6) sits at y_top with origin="upper".
EXTENT = (0.1, 0.4, 1e-6, 1.0)


def main() -> None:
    run, _ = parse_run(__doc__.splitlines()[0])
    run = run.resolve(n=N, res_km=RES_KM, n_samples=N_SAMPLES)
    use_article_style()

    gauss = gauss_scenes(BAND, run)
    disks = disk_scenes(BAND, run)

    train_configs = [
        (r"(a) Gauss $1.0~km$", [gauss[0]]),
        (r"(b) Gauss $50.0~km$", [gauss[2]]),
        (r"(c) Full Gauss", gauss),
        (r"(d) Full Gauss + Disk", gauss + disks),
    ]

    psf_modules = gg_parameter_grid(BAND, run, run.n_samples)
    loss_fn = Loss(Metric.RMSE_RAD)
    aspect = (EXTENT[1] - EXTENT[0]) / (EXTENT[3] - EXTENT[2])

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.4, 6))

    im = None
    for ax, (title, images) in zip(axes.flat, train_configs):
        losses = loss_landscape(
            train_images=TrainingImages(images=images, weights=[1.0] * len(images)),
            band=BAND,
            psf_modules=psf_modules,
            loss=loss_fn,
            device="cpu",
        ).reshape(run.n_samples, run.n_samples)

        im = ax.imshow(losses, extent=EXTENT, origin="upper")
        ax.set_aspect(aspect)
        ax.set_title(title, pad=10.0, fontsize=font())
        style_axes(ax)

    for ax in axes[:, 0]:
        ax.set_yticks([1e-6, 0.333, 0.666, 1.0])
        ax.set_yticklabels([r"$10^{0}$", r"$10^{-2}$", r"$10^{-4}$", r"$10^{-6}$"])
        ax.set_ylabel(r"$\sigma~[km]$", fontsize=font())

    for ax in axes[-1, :]:
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4"])
        ax.set_xlabel(r"$n$", fontsize=font())

    assert im is not None
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    style_colorbar(cbar, r"Radial $\mathrm{RMSE}$")

    save(fig, "figure2", run.figs_dir)
    if not run.smoke:
        plt.show()


if __name__ == "__main__":
    main()
