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
from adjeff.optim import Metric

__all__ = [
    "correct",
    "radial_rmse",
    "select_scalar",
    "sym_profile",
]


def sym_profile(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return a symmetric radial profile ``(r, values)`` for plotting.

    Mirrors the azimuthal mean around ``r = 0`` so that a profile can be
    drawn across the full transect rather than on the positive half only.

    Would be deleted by: ``da.adjeff.radial(symmetric=True)``.

    Parameters
    ----------
    da : xr.DataArray
        Two-dimensional field, extra singleton dimensions allowed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Radii from ``-r_max`` to ``+r_max`` and the mirrored values.
    """
    prof = da.squeeze().adjeff.radial()
    r = prof.coords["r"].values
    v = prof.values
    return np.concatenate([-r[::-1], r]), np.concatenate([v[::-1], v])


def select_scalar(obj: xr.Dataset | xr.DataArray, **coords: float):
    """Select one point of a swept dimension and drop every singleton dim.

    adjeff coerces scalar configuration values to length-one arrays, so
    its outputs carry singleton ``aot``, ``rh``, ``h`` and ``href``
    dimensions that the caller has to peel off before any comparison.

    Would be deleted by: ``psf_kernel(tree, band).sel(aot=0.4)`` losing its singleton dims,
    configs that keep a scalar scalar.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Output of a sampler, a pipeline or a frozen PSF tree.
    **coords
        Coordinate values to select, matched to the nearest neighbour.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Same type as *obj*, without the selected or singleton dimensions.
    """
    if coords:
        obj = obj.sel(coords, method="nearest", drop=True)
    return obj.squeeze(drop=True)


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


def radial_rmse(
    pred: xr.DataArray,
    truth: xr.DataArray,
    mask_on: xr.DataArray,
    device: str,
) -> float:
    """Radial RMSE between *pred* and *truth*, masked on *mask_on*.

    ``Metric`` only accepts tensors, so every caller repeats the same
    four ``.adjeff.to_tensor().to(device)`` conversions.  The shape guard
    matters too: a leftover singleton dimension on one operand would be
    silently broadcast by the metric and yield a meaningless number
    rather than an error.

    Would be deleted by: ``Metric`` accepting DataArrays, or an accessor
    ``pred.adjeff.rmse(truth, mask=mask_on)``.

    Raises
    ------
    ValueError
        If the three arrays do not share the same shape.
    """
    if not pred.shape == truth.shape == mask_on.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape}, truth {truth.shape}, "
            f"mask {mask_on.shape}. Extra dimensions were not squeezed."
        )
    return float(
        Metric.RMSE_RAD(
            pred.adjeff.to_tensor().to(device),
            truth.adjeff.to_tensor().to(device),
            truth.adjeff.dists.to(device),
            mask_on.adjeff.to_tensor().to(device),
        )
    )
