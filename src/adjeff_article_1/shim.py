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
from adjeff.core import S2Band, SensorBand

__all__ = ["sym_profile", "wl_to_band"]

# SensorBand carries wl_nm but offers no reverse lookup, so every script
# that names a band by its wavelength has to build this table itself.
# Would be deleted by: ``S2Band.from_wl(665.0)``.
_WL_TO_BAND: dict[float, SensorBand] = {b.wl_nm: b for b in S2Band}


def wl_to_band(wl_nm: float) -> SensorBand:
    """Return the Sentinel-2 band centred on *wl_nm*.

    Would be deleted by: ``S2Band.from_wl(wl_nm)``.

    Parameters
    ----------
    wl_nm : float
        Central wavelength in nanometres, as written on the command line.

    Raises
    ------
    KeyError
        If no band has that exact central wavelength.
    """
    try:
        return _WL_TO_BAND[float(wl_nm)]
    except KeyError:
        known = ", ".join(f"{w:.0f}" for w in sorted(_WL_TO_BAND))
        raise KeyError(
            f"No Sentinel-2 band at {wl_nm} nm. Known: {known}."
        ) from None


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
