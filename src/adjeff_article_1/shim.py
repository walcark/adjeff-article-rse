"""Helpers that stand in for primitives adjeff does not expose yet.

Everything in this module exists because the corresponding operation is
missing upstream, not because the article needs something special.  Each
function names the adjeff addition that would delete it.  This module is
meant to shrink to nothing: its size measures how much of the article's
plumbing adjeff still pushes onto its callers.
"""

from __future__ import annotations

import adjeff  # noqa: F401  (registers the .adjeff accessor)
import numpy as np
import xarray as xr
from adjeff.core import ImageDict, SensorBand, psf_tree
from adjeff.modules.classic import Toa2Unif
from adjeff.modules.models import Unif2Surface
from adjeff.modules.samplers import RADIATIVE_VARS

__all__ = [
    "correct",
]


def correct(
    ds: xr.Dataset,
    band: SensorBand,
    kernel: xr.DataArray,
    device: str,
    rho_toa: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Invert one scene with *kernel* and return ``(rho_s_est, rho_unif)``.

    The Dataset handed to the modules is rebuilt from the radiative
    quantities alone: :class:`Unif2Surface` writes its output under the
    name ``rho_s``, which is also the name of the ground truth, so a
    scene carrying both would lose the truth to the estimate.

    Would be deleted by: an ``output_var=`` argument on ``PSFConvModule``,
    or an output named ``rho_s_est``.

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
    rho_toa : xr.DataArray or None
        Measured TOA reflectance, when it must come from a different
        atmospheric state than the scalar terms in *ds*.  Defaults to
        ``ds["rho_toa"]``.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        The retrieved surface reflectance and the uniform reflectance.
    """
    measured = ds["rho_toa"] if rho_toa is None else rho_toa
    trimmed = xr.Dataset(
        {
            "rho_toa": measured.squeeze(drop=True),
            **{
                var: ds[var].squeeze(drop=True)
                for var in RADIATIVE_VARS
                if var in ds
            },
        }
    )

    scene = Toa2Unif()(ImageDict({band: trimmed}))
    model = Unif2Surface(
        kernels=psf_tree({band: kernel}), device=device
    )
    model.eval()
    scene = model(scene)
    return scene[band]["rho_s"], scene[band]["rho_unif"]


