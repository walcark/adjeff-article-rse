"""The diffuse fraction of the downwelling flux, from the MODIS table.

A MODIS BRDF fit gives two hemispheric reflectances, and neither is the
one an adjacency computation needs on its own.  The black-sky albedo,
or directional-hemispherical reflectance, is what the surface returns
under a purely direct beam; the white-sky albedo, or bihemispherical
reflectance, is what it returns under a purely isotropic sky.  A real
surface sees both, so the quantity that feeds ``rho_env`` is the
blue-sky albedo, the two mixed by how much of the downwelling flux is
diffuse::

    alpha_blue = f_diffuse * BHR + (1 - f_diffuse) * DHR

That fraction is not a free choice: it depends on the sun zenith angle,
on the aerosol optical thickness and on the band.  The MODIS albedo
products carry it as a lookup table, and this reads that table rather
than guessing a round number.

The table is ``skyl_lut.dat``, taken from the USGS reference
implementation *Computation of Blue-Sky Albedo from MODIS MCD43A1 Data*
(https://code.usgs.gov/cfwsc/modis-mcd43a1-blue-sky-albedo).  It holds
two aerosol types, Continental and Maritime, ten bands each, on a grid
of 91 sun zenith angles (0 to 90 degrees, one degree apart) and 50
optical thicknesses (0.00 to 0.98, 0.02 apart).

Note the band numbering is MODIS's, not Sentinel-2's: band 1 is
620-670 nm, band 3 is 459-479 nm.  The BRDF coefficients and the
skylight fraction must be read on the same band, which is the caller's
responsibility.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from ._logging import get_logger

__all__ = ["MODIS_BANDS", "blue_sky", "skylight_fraction"]

logger = get_logger(__name__)

#: Shipped with the repository: it is 540 kB of ASCII and the figure
#: must not depend on a host being up.
LUT_PATH = Path(__file__).resolve().parents[2] / "data" / "modis" / "skyl_lut.dat"

#: Wavelength range of each band the table covers, in nanometres, so a
#: caller can check it is asking on the band its coefficients came from.
MODIS_BANDS: dict[int, tuple[float, float]] = {
    1: (620.0, 670.0),
    2: (841.0, 876.0),
    3: (459.0, 479.0),
    4: (545.0, 565.0),
    5: (1230.0, 1250.0),
    6: (1628.0, 1652.0),
    7: (2105.0, 2155.0),
    8: (400.0, 700.0),
    9: (700.0, 4000.0),
    10: (250.0, 4000.0),
}

#: The two atmospheres the table was computed for.
AEROSOLS = ("Continental", "Maritime")


@functools.cache
def _read(
    aerosol: str, band: int, path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(sza, aot, fraction)`` grid of one aerosol and band.

    Parameters
    ----------
    aerosol : str
        ``"Continental"`` or ``"Maritime"``.
    band : int
        MODIS band number, 1 to 10.
    path : str
        Table to read.  A string rather than a `Path` so that the cache
        can key on it.

    Returns
    -------
    tuple of ndarray
        Sun zenith angles [deg], optical thicknesses, and the diffuse
        fraction on their outer product, shaped ``(n_sza, n_aot)``.

    Raises
    ------
    ValueError
        When the aerosol or the band is not in the table, which would
        otherwise silently return the wrong atmosphere.
    """
    if aerosol not in AEROSOLS:
        raise ValueError(f"aerosol must be one of {AEROSOLS}, got {aerosol!r}")
    if band not in MODIS_BANDS:
        raise ValueError(f"band must be one of {sorted(MODIS_BANDS)}, got {band!r}")

    lines = Path(path).read_text().splitlines()
    starts = [i for i, s in enumerate(lines) if s.startswith("Aerosol_type:")]
    first = next(
        i for i in starts if lines[i].split(":", 1)[1].strip() == aerosol
    )
    last = next((i for i in starts if i > first), len(lines))

    # Bands 1 to 7 are labelled MODIS_Band_n, 8 to 10 Broad_Band_n.
    tag = ("MODIS_Band_" if band <= 7 else "Broad_Band_") + f"{band}:"
    head = next(i for i in range(first, last) if lines[i].startswith(tag))

    aot = np.array([float(x) for x in lines[head + 1].split()[1:]])
    rows = []
    for line in lines[head + 2 : last]:
        if not line or not line[0].isdigit():
            break
        rows.append([float(x) for x in line.split()])
    table = np.array(rows)
    return table[:, 0], aot, table[:, 1:]


def skylight_fraction(
    sza_deg: float,
    aot: float,
    *,
    band: int = 1,
    aerosol: str = "Continental",
    path: Path | None = None,
) -> float:
    """Return the diffuse share of the downwelling flux.

    Bilinear in sun zenith and optical thickness, clipped at the edges
    of the table rather than extrapolated: the table stops at an
    optical thickness of 0.98, and a linear continuation past it would
    cross one.

    Parameters
    ----------
    sza_deg : float
        Sun zenith angle [deg], 0 to 90.
    aot : float
        Aerosol optical thickness at the band's wavelength.
    band : int
        MODIS band number, 1 to 10.  See `MODIS_BANDS`.
    aerosol : str
        ``"Continental"`` or ``"Maritime"``.
    path : Path or None
        Table to read.  Defaults to the shipped one.

    Returns
    -------
    float
        A number in ``[0, 1]``: zero for a purely direct beam, one for
        a purely diffuse sky.

    Examples
    --------
    >>> round(skylight_fraction(40.0, 0.4), 3)
    0.304
    """
    zenith, thickness, grid = _read(aerosol, band, str(path or LUT_PATH))
    at_sza = np.array(
        [np.interp(sza_deg, zenith, grid[:, j]) for j in range(grid.shape[1])]
    )
    return float(np.interp(aot, thickness, at_sza))


def blue_sky(
    dhr: float,
    bhr: float,
    sza_deg: float,
    aot: float,
    **kwargs: object,
) -> float:
    """Return the blue-sky albedo, mixing the two by the diffuse share.

    Parameters
    ----------
    dhr : float
        Black-sky albedo, the directional-hemispherical reflectance.
    bhr : float
        White-sky albedo, the bihemispherical reflectance.
    sza_deg : float
        Sun zenith angle [deg].
    aot : float
        Aerosol optical thickness.
    **kwargs
        Passed to `skylight_fraction`: ``band``, ``aerosol``, ``path``.

    Returns
    -------
    float
        The hemispheric reflectance the surface actually presents.
    """
    fraction = skylight_fraction(sza_deg, aot, **kwargs)  # type: ignore[arg-type]
    logger.debug(
        "skylight.fraction",
        sza=sza_deg,
        aot=aot,
        fraction=round(fraction, 4),
    )
    return fraction * bhr + (1.0 - fraction) * dhr
