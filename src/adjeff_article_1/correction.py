"""Invert one scene with a given kernel.

This is the article's own experiment, not a gap in adjeff: it corrects a
scene with a PSF chosen independently of the atmosphere the scene was
simulated at, which is how the error is split between the scalar terms
of the 5S model and the kernel.
"""

from __future__ import annotations

import xarray as xr
from adjeff.core import ImageDict, SensorBand, psf_tree
from adjeff.modules.classic import Toa2Unif
from adjeff.modules.models import Unif2Surface

__all__ = ["correct"]


def correct(
    ds: xr.Dataset,
    band: SensorBand,
    kernel: xr.DataArray,
    device: str,
    rho_toa: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Invert *ds* with *kernel* and return ``(rho_s_est, rho_unif)``.

    Parameters
    ----------
    ds : xr.Dataset
        Scene carrying ``rho_toa`` and the six radiative quantities.
    band : SensorBand
        Band being corrected.
    kernel : xr.DataArray
        Frozen PSF kernel to deconvolve with.
    device : str
        Torch device for the convolution.
    rho_toa : xr.DataArray or None, optional
        Measured TOA reflectance, when it must come from a different
        atmospheric state than the scalar terms in *ds*.  Defaults to
        ``ds["rho_toa"]``.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        The retrieved surface reflectance and the uniform reflectance,
        with any length-one dimension dropped so that they compare
        directly against a 2-D truth.
    """
    scene = ImageDict({band: ds if rho_toa is None else ds.assign(rho_toa=rho_toa)})
    scene = Toa2Unif()(scene)

    model = Unif2Surface(
        kernels=psf_tree({band: kernel}),
        device=device,
        rename={"rho_s": "rho_s_est"},
    )
    model.eval()
    scene = model(scene)
    # Peel the length-one atmospheric dimensions a single-state scene
    # keeps: the error metrics will not broadcast them against a 2-D
    # truth.
    return (
        scene[band]["rho_s_est"].squeeze(drop=True),
        scene[band]["rho_unif"].squeeze(drop=True),
    )
