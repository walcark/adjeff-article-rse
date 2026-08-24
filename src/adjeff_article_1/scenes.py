"""Training scenes and parameter grids shared by several figures.

Figures 2 to 5 all start from the same reference atmosphere and the same
analytical landscapes.  Building them here keeps the scripts down to what
actually differs between figures.
"""

from __future__ import annotations

import numpy as np

from adjeff.api import FullConfig, make_full_config, run_forward_pipeline
from adjeff.core import (
    GeneralizedGaussianPSF,
    ImageDict,
    PSFGrid,
    SensorBand,
    disk_image_dict,
    gaussian_image_dict,
)
from adjeff.utils import CacheStore

from .runconfig import ARTICLE_ATMOSPHERE, RunConfig

__all__ = [
    "article_config",
    "disk_scenes",
    "gauss_scenes",
    "gg_parameter_grid",
    "make_cache",
]

# Landscape sizes used throughout the manuscript, in km.
GAUSS_SIGMAS: tuple[float, ...] = (1.0, 5.0, 50.0)
DISK_RADII: tuple[float, ...] = (1.0, 5.0, 50.0)


def article_config(bands: list[SensorBand]) -> FullConfig:
    """Return the manuscript's reference atmosphere for *bands*."""
    return make_full_config(bands=bands, **ARTICLE_ATMOSPHERE)  # type: ignore[arg-type]


def make_cache(run: RunConfig) -> CacheStore:
    """Return the on-disk cache for *run*."""
    return CacheStore(run.cache_dir)


def gauss_scenes(
    band: SensorBand,
    run: RunConfig,
    cfg: FullConfig | None = None,
    sigmas: tuple[float, ...] = GAUSS_SIGMAS,
) -> list[ImageDict]:
    """Run the forward pipeline on one Gaussian landscape per sigma.

    Parameters
    ----------
    band : SensorBand
        Band to simulate.
    run : RunConfig
        Grid size, photon budget and cache location.
    cfg : FullConfig or None
        Atmosphere to use.  Defaults to :func:`article_config`.
    sigmas : tuple[float, ...]
        Gaussian widths in km.

    Returns
    -------
    list[ImageDict]
        One scene per sigma, each carrying ``rho_s``, ``rho_unif`` and the
        radiative quantities.
    """
    cfg = cfg or article_config([band])
    cache = make_cache(run)
    return [
        run_forward_pipeline(
            gaussian_image_dict(
                sigma=s, res_km=run.res_km, bands=[band], n=run.n
            ),
            **cfg,
            n_ph=run.n_ph,
            cache=cache,
        )
        for s in sigmas
    ]


def disk_scenes(
    band: SensorBand,
    run: RunConfig,
    cfg: FullConfig | None = None,
    radii: tuple[float, ...] = DISK_RADII,
) -> list[ImageDict]:
    """Run the forward pipeline on one uniform disk per radius.

    Parameters
    ----------
    band : SensorBand
        Band to simulate.
    run : RunConfig
        Grid size, photon budget and cache location.
    cfg : FullConfig or None
        Atmosphere to use.  Defaults to :func:`article_config`.
    radii : tuple[float, ...]
        Disk radii in km.

    Returns
    -------
    list[ImageDict]
        One scene per radius.
    """
    cfg = cfg or article_config([band])
    cache = make_cache(run)
    return [
        run_forward_pipeline(
            disk_image_dict(
                radius=r, res_km=run.res_km, bands=[band], n=run.n
            ),
            **cfg,
            n_ph=run.n_ph,
            cache=cache,
        )
        for r in radii
    ]


def gg_parameter_grid(
    band: SensorBand,
    run: RunConfig,
    n_samples: int,
) -> list[GeneralizedGaussianPSF]:
    """Return the ``(sigma, n)`` grid of generalised Gaussian PSFs.

    ``sigma`` is log-spaced in ``[1e-6, 1]`` km and ``n`` linear in
    ``[0.1, 0.4]``, giving ``n_samples ** 2`` kernels in row-major order
    so that the result reshapes to ``(n_samples, n_samples)``.

    Parameters
    ----------
    band : SensorBand
        Band the kernels apply to.
    run : RunConfig
        Supplies the PSF grid resolution and size.
    n_samples : int
        Side of the parameter grid.  Pass ``run.n_samples or DEFAULT`` so
        that a smoke run shrinks it.
    """
    grid = PSFGrid(res=run.res_km, n=run.n)
    sigma_vals = np.logspace(-6, 0, n_samples).astype(np.float32)
    n_vals = np.linspace(0.1, 0.4, n_samples).astype(np.float32)
    return [
        GeneralizedGaussianPSF(grid, band, sigma=float(s), n=float(n))
        for s in sigma_vals
        for n in n_vals
    ]
