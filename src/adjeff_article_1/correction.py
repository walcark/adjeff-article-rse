"""Invert one scene with a given kernel.

This is the article's own experiment, not a gap in adjeff: it corrects a
scene with a PSF chosen independently of the atmosphere the scene was
simulated at, which is how the error is split between the scalar terms
of the 5S model and the kernel.

It used to live in ``shim.py`` for a different reason.  ``Unif2Surface``
writes ``rho_s``, which is also the name of the ground truth, so a scene
carrying both lost one of them and the helper had to strip the truth out
before running.  adjeff 0.10.0 binds roles to slots, so the estimate now
travels under its own name and the truth stays where it is.
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
        The retrieved surface reflectance and the uniform reflectance.
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
    return scene[band]["rho_s_est"], scene[band]["rho_unif"]
