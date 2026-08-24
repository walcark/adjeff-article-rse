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

__all__ = ["sym_profile"]


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
