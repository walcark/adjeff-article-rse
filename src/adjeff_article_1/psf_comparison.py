"""Compare an optimised PSF against the fixed Gauss-330 m reference.

Figures 4 and 5 answer the same question for two kernel families: how
well does an optimised PSF recover ``rho_s`` from ``rho_unif`` on uniform
disks, next to the Gaussian of 330 m used as a baseline in the
literature.  Only the PSF class and its initial parameters differ, so the
whole comparison lives here and each figure script declares its kernel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from adjeff.api import make_model
from adjeff.core import GaussPSF, PSFGrid, SensorBand, psf_tree
from adjeff.core._psf import PSFModule
from adjeff.modules.models import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages, fit

from .runconfig import RunConfig
from .scenes import DISK_RADII, disk_scenes
from .shim import sym_profile
from .style import font, panel_title, save, style_axes, use_article_style

__all__ = ["GAUSS_REFERENCE_KM", "psf_comparison_figure"]

# Gaussian width used as the fixed reference kernel, in km.
GAUSS_REFERENCE_KM = 0.330


def psf_comparison_figure(
    name: str,
    band: SensorBand,
    run: RunConfig,
    psf_type: type[PSFModule],
    init_parameters: dict[str, float],
    label: str,
) -> None:
    """Draw and save the disk-recovery comparison for one kernel family.

    Parameters
    ----------
    name : str
        Output stem, e.g. ``"figure4"``.
    band : SensorBand
        Band to simulate.
    run : RunConfig
        Grid size, photon budget, device and output directory.
    psf_type : type[PSFModule]
        Kernel family to optimise, e.g. ``KingPSF``.
    init_parameters : dict[str, float]
        Starting point of the optimisation, one entry per free parameter.
    label : str
        Short LaTeX superscript for the legend, e.g. ``"King"``.
    """
    use_article_style()

    scenes = disk_scenes(band, run)
    train_images = TrainingImages(images=scenes, weights=[1.0] * len(scenes))

    model = make_model(
        Unif2Surface,
        psf_type,
        [band],
        res_km=run.res_km,
        n=run.n,
        init_parameters=init_parameters,
        device=run.device,
    )
    tree = fit(
        model, train_images, loss=Loss(Metric.RMSE_RAD), device=run.device
    )

    model_fitted = Unif2Surface(kernels=tree, device=run.device)
    reference_kernel = GaussPSF(
        PSFGrid(run.res_km, run.n), band, sigma=GAUSS_REFERENCE_KM
    ).to_dataarray()
    model_reference = Unif2Surface(
        kernels=psf_tree({band: reference_kernel}),
        device=run.device,
    )

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 3))

    for idx, (ax, scene, radius) in enumerate(zip(axes, scenes, DISK_RADII)):
        # SceneModule.forward() shallow-copies its input, so a prediction
        # only ever exists in the returned scene.  Reading it back from
        # `scene` would hand out the untouched ground truth instead.
        fitted = model_fitted(scene)[band]["rho_s"]
        reference = model_reference(scene)[band]["rho_s"]

        r, v_truth = sym_profile(scene[band]["rho_s"])
        _, v_unif = sym_profile(scene[band]["rho_unif"])
        _, v_reference = sym_profile(reference)
        _, v_fitted = sym_profile(fitted)

        opts = dict(linewidth=1.3)
        ax.plot(r, v_truth, label=r"$\rho_{s}$", **opts)
        ax.plot(r, v_unif, label=r"$\rho_{unif}$", **opts)
        ax.plot(r, v_reference, label=r"$\rho_{s}^{Gauss~(330m)}$", **opts)
        ax.plot(r, v_fitted, label=rf"$\rho_{{s}}^{{{label}}}$", **opts)

        ax.legend(loc="lower center", fontsize=font(10 / 12))
        panel_title(ax, idx, rf"Radius $= {radius:.1f}~km$")
        ax.set_xlim(-1.5 * radius, 1.5 * radius)
        ax.set_xlabel(r"Radial distance $[km]$", fontsize=font())
        style_axes(ax)

    axes[0].set_ylabel("Reflectance", fontsize=font())

    fig.tight_layout()
    save(fig, name, run.figs_dir)
    if not run.smoke:
        plt.show()
