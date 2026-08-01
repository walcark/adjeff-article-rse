"""Comparison of rho_s predictions: GeneralizedGaussianPSF vs Gauss-330.

Three disk training fields (radii 1, 5, 50 km) are run through the
forward pipeline.  A GeneralizedGaussianPSF is then optimised with
RMSE_RAD loss, and its predictions are compared to a fixed Gauss-330m
reference.
"""

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
from adjeff.core import (
    GeneralizedGaussianPSF,
    GaussPSF,
    PSFDict,
    PSFGrid,
    S2Band,
    disk_image_dict,
)
from adjeff.modules.models.unif2surface import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages

plt.style.use(["science", "nature"])

BAND = S2Band.B03
RES_KM = 0.05
N = 3999
N_PH = int(1e5)
FIGS_DIR = Path(__file__).parent.parent / "output"


def sym_profile(da):
    """Return a symmetric radial profile: (r_sym, values_sym)."""
    prof = da.squeeze().adjeff.radial()
    r = prof.coords["r"].values
    v = prof.values
    return np.concatenate([-r[::-1], r]), np.concatenate([v[::-1], v])


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

    disk_params = [
        (1.0, r"Radius $= 1.0~km$"),
        (5.0, r"Radius $= 5.0~km$"),
        (50.0, r"Radius $=50.0~km$"),
    ]

    # Build training scenes (each has rho_s + rho_unif + radiatives)
    scenes = []
    for radius, _ in disk_params:
        scene = disk_image_dict(radius=radius, res_km=RES_KM, bands=[BAND], n=N)
        scene = run_forward_pipeline(scene, **cfg, n_ph=N_PH)
        scenes.append(scene)

    train_images = TrainingImages(images=scenes, weights=[1.0, 1.0, 1.0])

    # Optimise GeneralizedGaussianPSF
    model = make_model(
        Unif2Surface,
        GeneralizedGaussianPSF,
        [BAND],
        res_km=RES_KM,
        n=N,
        init_parameters={"sigma": 1e-3, "n": 0.20},
    )
    psf_dict_opt = optimize_adam_lbfgs(
        model, train_images, Loss(Metric.RMSE_RAD)
    )

    # Prediction models
    model_gg = Unif2Surface(psf_dict=psf_dict_opt)
    model_gauss330 = Unif2Surface(
        psf_dict=PSFDict.from_kernels(
            {BAND: GaussPSF(PSFGrid(RES_KM, N), BAND, sigma=0.330).to_dataarray()}
        )
    )

    tick_factor = 1.2
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 3))

    for idx, (ax, scene, (radius, label)) in enumerate(
        zip(axes, scenes, disk_params)
    ):
        rho_s_gt = scene[BAND]["rho_s"]
        rho_unif_da = scene[BAND]["rho_unif"]

        # Run GG model first, then Gauss-330 (rho_unif is never overwritten)
        model_gg(scene)
        rho_s_gg_da = scene[BAND]["rho_s"]

        model_gauss330(scene)
        rho_s_g330_da = scene[BAND]["rho_s"]

        r, v_s = sym_profile(rho_s_gt)
        _, v_unif = sym_profile(rho_unif_da)
        _, v_gg = sym_profile(rho_s_gg_da)
        _, v_g330 = sym_profile(rho_s_g330_da)

        opts = dict(linewidth=1.3)
        ax.plot(r, v_s, label=r"$\rho_{s}$", **opts)
        ax.plot(r, v_unif, label=r"$\rho_{unif}$", **opts)
        ax.plot(r, v_g330, label=r"$\rho_{s}^{Gauss~(330m)}$", **opts)
        ax.plot(r, v_gg, label=r"$\rho_{s}^{GG}$", **opts)

        ax.legend(loc="lower center", fontsize=10 * tick_factor)
        ax.set_title(
            f"({'abc'[idx]}) " + label,
            pad=5,
            fontsize=12 * tick_factor,
        )
        ax.set_xlim(-1.5 * radius, 1.5 * radius)
        ax.set_xlabel(r"Radial distance $[km]$", fontsize=12 * tick_factor)
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

    axes[0].set_ylabel(r"Reflectance", fontsize=12 * tick_factor)
    fig.tight_layout()
    plt.savefig(FIGS_DIR / "figure4.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
