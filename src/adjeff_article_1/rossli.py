"""The Ross-Thick Li-Sparse BRDF, as Smart-G's CUDA kernel computes it.

MODIS distributes, per pixel and per band, three weights of the
decomposition ``rho = f_iso + f_geo F1 + f_vol F2``.  `device.cu`
consumes them relative to the isotropic one, ``k1p = f_geo / f_iso`` and
``k2p = f_vol / f_iso``, and this mirrors it so that the Python side and
the Monte-Carlo kernel share one convention.  The surface reflectance is
``k0 * rtls_shape(...)``.

The Li-Sparse crown shape is MODIS's, ``b/r = 1`` and ``h/b = 2``, which
is why no shape ratio appears and the zenith angles are used unmodified.
Changing it would break the meaning of the distributed coefficients.

The fit is not constrained positive and goes below zero at grazing view
for an anisotropic surface.  `shape_of` can hold it at zero, which
Smart-G can be made to do too, see ``scripts/clamp_brdf.py``: the two
must agree, or a correction is judged against a truth simulated for
another surface.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bihemispheric",
    "brf_onset",
    "f1_rtls",
    "f2_rtls",
    "hemispheric",
    "negative_flux",
    "rtls_shape",
    "shape_of",
]


def _rtls_angles(
    sza_deg: float | np.ndarray,
    vza_deg: float | np.ndarray,
    raa_deg: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ths, thv, phi, cos_xi)`` in the convention of device.cu.

    The relative azimuth is folded onto ``[0, pi]`` and forced to zero
    when either direction is exactly vertical, because ``BRDF()`` builds
    it from a horizontal projection that is then undefined.
    """
    ths = np.radians(np.asarray(sza_deg, dtype=float))
    thv = np.radians(np.asarray(vza_deg, dtype=float))
    phi = np.radians(np.asarray(raa_deg, dtype=float))

    phi = np.abs(phi) % (2.0 * np.pi)
    phi = np.where(phi > np.pi, 2.0 * np.pi - phi, phi)
    phi = np.where((ths == 0.0) | (thv == 0.0), 0.0, phi)

    cos_xi = np.cos(ths) * np.cos(thv) + np.sin(ths) * np.sin(thv) * np.cos(
        phi
    )
    return ths, thv, phi, np.clip(cos_xi, -1.0, 1.0)


def f1_rtls(
    sza_deg: float | np.ndarray,
    vza_deg: float | np.ndarray,
    raa_deg: float | np.ndarray,
) -> np.ndarray:
    """Return the Li-Sparse geometric kernel ``F1``.

    Mirrors ``F1_rtls`` of ``device.cu``, i.e. the reciprocal Li-Sparse
    kernel with the MODIS crown shape (``b/r = 1``, ``h/b = 2``), which
    is why no shape ratio appears and the zenith angles are used
    unmodified.

    Parameters
    ----------
    sza_deg, vza_deg, raa_deg : float or ndarray
        Sun zenith, view zenith and relative azimuth [deg].

    Returns
    -------
    ndarray
        The dimensionless geometric kernel.
    """
    ths, thv, phi, cos_xi = _rtls_angles(sza_deg, vza_deg, raa_deg)

    mm = 1.0 / np.cos(thv) + 1.0 / np.cos(ths)
    tthv, tths = np.tan(thv), np.tan(ths)

    cos_t = (2.0 / mm) * np.sqrt(
        tthv**2
        + tths**2
        - 2.0 * tthv * tths * np.cos(phi)
        + (tthv * tths * np.sin(phi)) ** 2
    )
    cos_t = np.clip(cos_t, -1.0, 1.0)
    t = np.arccos(cos_t)
    big_o = mm * (t - np.sin(t) * cos_t) / np.pi

    return np.asarray(
        big_o - mm + (1.0 + cos_xi) / (np.cos(thv) * np.cos(ths)) / 2.0,
        dtype=float,
    )


def f2_rtls(
    sza_deg: float | np.ndarray,
    vza_deg: float | np.ndarray,
    raa_deg: float | np.ndarray,
) -> np.ndarray:
    """Return the Ross-Thick volumetric kernel ``F2``.

    Mirrors ``F2_rtls`` of ``device.cu``.

    Parameters
    ----------
    sza_deg, vza_deg, raa_deg : float or ndarray
        Sun zenith, view zenith and relative azimuth [deg].

    Returns
    -------
    ndarray
        The dimensionless volumetric kernel.
    """
    ths, thv, _, cos_xi = _rtls_angles(sza_deg, vza_deg, raa_deg)
    xi = np.arccos(cos_xi)
    return np.asarray(
        ((np.pi / 2.0 - xi) * cos_xi + np.sin(xi))
        / (np.cos(thv) + np.cos(ths))
        - np.pi / 4.0,
        dtype=float,
    )


def rtls_shape(
    sza_deg: float | np.ndarray,
    vza_deg: float | np.ndarray,
    raa_deg: float | np.ndarray,
    k1p: float,
    k2p: float,
) -> np.ndarray:
    """Return the RTLS angular factor, without the ``k0`` normalisation.

    Mirrors ``1 + k1p*F1_rtls + k2p*F2_rtls`` of ``device.cu`` so that
    the Python side and the Monte-Carlo kernel share one convention.
    The surface reflectance is ``k0 * rtls_shape(...)``.

    Parameters
    ----------
    sza_deg, vza_deg, raa_deg : float or ndarray
        Sun zenith, view zenith and relative azimuth [deg].
    k1p : float
        Geometric weight relative to the isotropic one, ``f_geo/f_iso``.
    k2p : float
        Volumetric weight relative to the isotropic one, ``f_vol/f_iso``.

    Returns
    -------
    ndarray
        The dimensionless angular factor.
    """
    f1 = f1_rtls(sza_deg, vza_deg, raa_deg)
    f2 = f2_rtls(sza_deg, vza_deg, raa_deg)
    return np.asarray(1.0 + k1p * f1 + k2p * f2, dtype=float)


def brf_onset(
    sza_deg: float,
    k1p: float,
    k2p: float,
    n_theta: int = 90,
    n_phi: int = 181,
) -> float:
    """Return the view zenith at which the BRF first goes negative.

    The Li-Sparse kernel is unbounded below: for a large geometric
    weight it drives ``1 + k1p*F1 + k2p*F2`` negative at grazing view,
    which is not a reflectance and which ``device.cu`` does **not**
    clip, so Smart-G would weight those photons negatively.  It matters
    here more than elsewhere because the far-field adjacency signal
    leaves the ground at exactly those angles.

    This function is a diagnostic, not a fix: clipping on the Python
    side would silently disagree with the Monte-Carlo kernel.

    Parameters
    ----------
    sza_deg : float
        Sun zenith angle [deg].
    k1p, k2p : float
        Relative geometric and volumetric weights.
    n_theta, n_phi : int
        Search resolution in zenith and azimuth.

    Returns
    -------
    float
        The smallest offending view zenith [deg], or 90 when the BRF
        stays positive over the whole hemisphere.
    """
    theta = np.linspace(0.0, 89.0, n_theta)
    phi = np.linspace(0.0, 360.0, n_phi)
    grid_t, grid_p = np.meshgrid(theta, phi, indexing="ij")
    negative = rtls_shape(sza_deg, grid_t, grid_p, k1p, k2p).min(axis=1) < 0.0
    return float(theta[negative][0]) if negative.any() else 90.0


def shape_of(
    sza_deg: float | np.ndarray,
    vza_deg: float | np.ndarray,
    raa_deg: float | np.ndarray,
    k1p: float,
    k2p: float,
    clamp: bool,
) -> np.ndarray:
    """Return the angular factor, clamped at zero or not.

    Clamping has to happen on both sides or neither.  `device.cu` decides
    what the photons do; this decides what `a`, the hemispheric integral
    and the rescaling describe.  If the two disagree, the study compares
    a correction against a truth simulated for another surface.
    """
    shape = rtls_shape(sza_deg, vza_deg, raa_deg, k1p, k2p)
    return np.maximum(shape, 0.0) if clamp else shape


def hemispheric(sza_deg: float, k1p: float, k2p: float, clamp: bool) -> float:
    """Return the black-sky albedo of the shape, one for a Lambertian.

    Reimplemented here rather than taken from `hotspot.rtls_dhr` so that
    the clamp can be applied inside the integrand: clamping afterwards
    would not remove the negative contribution, which is the point.
    """
    theta = np.linspace(1e-4, np.pi / 2.0 - 1e-4, 400)
    phi = np.linspace(0.0, 2.0 * np.pi, 241)
    grid_t, grid_p = np.meshgrid(theta, phi, indexing="ij")
    shape = shape_of(
        sza_deg, np.degrees(grid_t), np.degrees(grid_p), k1p, k2p, clamp
    )
    weighted = shape * np.cos(grid_t) * np.sin(grid_t)
    return float(
        np.trapezoid(np.trapezoid(weighted, phi, axis=1), theta) / np.pi
    )


def bihemispheric(k1p: float, k2p: float, clamp: bool, n_theta: int = 200) -> float:
    """Return the white-sky albedo, one for a Lambertian.

    The black-sky albedo averaged over an isotropic sky, which is the
    same cosine-weighted hemispheric average applied a second time, now
    over the illumination rather than the view.  It does not depend on
    the sun, which is the point: it is what the surface returns of the
    part of the downwelling flux that has already been scattered.

    Parameters
    ----------
    k1p, k2p : float
        Relative geometric and volumetric weights.
    clamp : bool
        Whether the negative tail of the fit is held at zero, as in
        `shape_of`.  Both integrals must agree on this.
    n_theta : int
        Quadrature resolution in illumination zenith.

    Returns
    -------
    float
        The bihemispherical reflectance of the shape.
    """
    theta = np.linspace(1e-4, np.pi / 2.0 - 1e-4, n_theta)
    black = np.array(
        [hemispheric(np.degrees(t), k1p, k2p, clamp) for t in theta]
    )
    return float(
        2.0 * np.trapezoid(black * np.cos(theta) * np.sin(theta), theta)
    )


def negative_flux(sza_deg: float, k1p: float, k2p: float, n_theta: int = 200) -> float:
    """Return the share of upward flux leaving where the BRF is negative.

    Weighted as the hemispheric integral is, by ``cos(theta) sin(theta)``,
    because that is the flux that feeds the adjacency term.  The solid
    angle alone overstates it: the grazing directions where the fit
    misbehaves are the ones the cosine suppresses.

    Parameters
    ----------
    sza_deg : float
        Sun zenith angle [deg].
    k1p, k2p : float
        Relative geometric and volumetric weights.
    n_theta : int
        Quadrature resolution in zenith.

    Returns
    -------
    float
        Percentage of the positive flux carried by the negative region.
    """
    theta = np.linspace(1e-4, np.pi / 2.0 - 1e-4, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, 121)
    grid_t, grid_p = np.meshgrid(theta, phi, indexing="ij")
    shape = rtls_shape(sza_deg, np.degrees(grid_t), np.degrees(grid_p), k1p, k2p)
    weight = np.cos(grid_t) * np.sin(grid_t)

    def integrate(values: np.ndarray) -> float:
        return float(np.trapezoid(np.trapezoid(values, phi, axis=1), theta))

    negative = integrate(np.where(shape < 0.0, shape, 0.0) * weight)
    positive = integrate(np.where(shape < 0.0, 0.0, shape) * weight)
    return 100.0 * abs(negative) / positive
