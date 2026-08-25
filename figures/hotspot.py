r"""Accuracy of a Lambertian-trained PSF applied to real MODIS BRDFs.

Answers reviewer question T1: quantify the accuracy degradation of the
King kernel when the surface is not Lambertian, over the *realistic*
range of land surfaces rather than for one hand-picked BRDF.

Protocol
--------
1. ``N`` MCD43A1 pixels are sampled over contrasted land covers.  Each
   pixel gives a Ross-Thick Li-Sparse (RTLS) triplet ``(f_iso, f_geo,
   f_vol)``, i.e. an inverted surface, not a chosen one.
2. Every triplet is ranked by its anisotropy index ``a = DHR / v``, the
   ratio between what drives the hemispheric terms of the 5S inversion
   and what the algorithm actually retrieves.  A Lambertian surface has
   ``a == 1`` by definition, so ``|a - 1|`` measures how far a surface
   sits from the hypothesis under test.
3. The **min**, **median** and **max** surfaces of that ensemble are
   kept.  They bracket the realistic range.
4. A King PSF is optimised on the six **Lambertian** landscapes of
   Section 2.3, exactly as in the article.  This is the operational
   kernel, and it is the only kernel this study uses.
5. The three Gaussian landscapes (1, 5 and 50 km) are simulated again
   with each of the three RTLS surfaces, corrected with that unchanged
   kernel, and the radial RMSE is compared with the Lambertian case.

Any degradation is therefore attributable to the surface BRDF alone,
and the result is reported as a *range* over real surfaces rather than
as a single number for one parameter set.

Why RTLS and not RPV
--------------------
- RTLS is what operational products invert and distribute (MCD43A1,
  VIIRS VNP43, the Sentinel-2 BRDF correction), so every coefficient
  triplet used here is a measured surface.  RPV coefficients have to be
  chosen, and sweeping them freely generates surfaces that no land
  cover actually has.
- The RPV Minnaert term diverges as ``mu^(k-1)`` at grazing view, where
  a large share of the far-field adjacency signal originates.  The RTLS
  kernels grow only as ``sec(theta)``, which stays integrable and much
  better behaved over the angular range the environment term samples.

Why this works in Smart-G
-------------------------
The analytical Gaussian landscape of adjeff is built with ``ENV=2``,
which modulates the surface albedo by ``exp(-r^2 / ENV_SIZE)`` *after*
the BRDF factor has been applied (``device.cu``, ``surfaceBRDF_new``)::

    ph->weight *= BRDF(ilam, -v0, ph->v, spectrum);
    if (ENVd==2) ph->weight *= gauss_albedo(pos) * (alb_surface - alb_env)
                               + alb_env;

Swapping ``LambSurface`` for ``RTLSSurface`` therefore keeps the spatial
landscape strictly identical and only changes its angular signature.
Note that this is **not** true of the ``ENV=5`` albedo-map path used for
arbitrary fields, which multiplies by ``alb_envs[ispec]`` and never
calls ``BRDF`` at all: an arbitrary landscape cannot carry a BRDF in
this version of Smart-G.

Normalisation
-------------
An RTLS surface of isotropic parameter ``k0`` does not reflect ``k0`` in
the observation direction: it reflects ``k0 * (1 + k1p*F1 + k2p*F2)``.
``k0`` is rescaled so that the surface reflectance in the exact
sun-sensor geometry equals ``rho_max``, i.e. the *directly transmitted*
signal is identical in both runs and the truth field is the same array
for the Lambertian and the RTLS run.  What is left is the angular
redistribution of the environment contribution, which is precisely what
the PSF models.

Matching the hemispheric integral instead would be equally defensible
physically, but it moves the mismatch into the retrieval target itself:
the truth would then differ between runs, and the measured error would
mix a genuine degradation with a trivial offset in the quantity being
retrieved.  No normalisation can match both, since ``a != 1`` *is* the
definition of a non-Lambertian surface.

Caveats
-------
- MCD43A1 is a 500 m product and carries its own inversion uncertainty.
  Only full-inversion retrievals (mandatory QA flag 0) are kept.
- RTLS underestimates the hotspot by construction.  The degradation
  reported here is therefore a realistic-surface estimate, not a
  hotspot worst case.
- The Li-Sparse kernel is unbounded below.  For a large geometric
  weight the fitted BRF turns negative at grazing view, which is not a
  reflectance and which ``device.cu`` does not clip.  Sampled pixels
  that do so inside ``--min-brf-vza`` are dropped, and the onset angle
  of every kept surface is reported in the selection CSV.
- ``rho_toa_sym`` assumes the TOA field is radially symmetric.  This
  holds exactly for a Lambertian surface.  With an anisotropic BRDF the
  environment contribution is only symmetric when the sun is at zenith,
  so the simulated profile is the one sampled along the azimuth of
  ``_radial_sensors``.  The radial metric averages the azimuthal
  asymmetry away and can only *under*-estimate the degradation.

Usage
-----
python hotspot.py --wl 665 --aot 0.4 --sza 40 --vza 0 --rho-max 0.3

The MCD43A1 ensemble is fetched once and cached to
``--sites-csv`` (default ``output/hotspot_mcd43a1_sites.csv``); later
runs reuse it.  A CSV produced by any other means (AppEEARS, Earth
Engine export) is accepted as long as it carries the columns ``site``,
``f_iso``, ``f_geo`` and ``f_vol``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import xarray as xr

from adjeff.api import (
    make_full_config,
    make_model,
    run_forward_pipeline,
)
from adjeff.core import (
    ImageDict,
    KingPSF,
    S2Band,
    SensorBand,
    disk_image_dict,
    gaussian_image_dict,
    psf_kernel,
)
from adjeff.modules.models import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages, fit
from adjeff.utils import CacheStore
from adjeff_article_1.runconfig import RunConfig, parse_run
from adjeff_article_1.shim import (
    correct,
    radial_rmse,
    sym_profile,
)
from adjeff_article_1.style import save

plt.style.use(["science", "nature"])

# Sentinel-2 central wavelengths mapped to the MODIS land band whose
# BRDF shape is closest.  Only the *shape* is borrowed: the isotropic
# level is overwritten by the view normalisation, so a small spectral
# offset between the two sensors does not propagate into the study.
_WL_TO_MODIS_BAND = {
    645.0: 1, 665.0: 1, 858.0: 2, 865.0: 2, 469.0: 3, 490.0: 3,
    555.0: 4, 560.0: 4, 1240.0: 5, 1610.0: 6, 1640.0: 6, 2130.0: 7,
    2190.0: 7,
}

# Sampling sites, chosen to span the anisotropy range of vegetated and
# bare land covers.  These are coordinates only: every BRDF coefficient
# comes from the MCD43A1 product, none is set here.
SITES: tuple[tuple[str, float, float], ...] = (
    ("harvard-forest-decid", 42.5378, -72.1715),
    ("hyytiala-conifer", 61.8474, 24.2948),
    ("tapajos-tropical", -2.8567, -54.9589),
    ("konza-grassland", 39.0824, -96.5603),
    ("barrax-cropland", 39.0569, -2.1020),
    ("skukuza-savanna", -25.0197, 31.4969),
    ("railroad-valley-playa", 38.4970, -115.6900),
    ("libya4-desert", 28.5500, 23.3900),
)

# MCD43A1 stores the three RTLS weights in this order, and scales them
# by 1e-3.  device.cu expects the weights *relative* to the isotropic
# one, so the loader divides by f_iso.
MCD43A1_PARAM_ORDER = ("iso", "vol", "geo")
MCD43A1_SCALE = 1.0e-3
ORNL_ROOT = "https://modis.ornl.gov/rst/api/v1"


# ----------------------------------------------------------------------
# Ross-Thick Li-Sparse model, mirroring smartg/src/device.cu
# ----------------------------------------------------------------------


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


def rtls_dhr(
    sza_deg: float,
    k1p: float,
    k2p: float,
    n_theta: int = 180,
    n_phi: int = 121,
) -> float:
    """Return the directional-hemispherical reflectance factor.

    The black-sky albedo of the *shape*, normalised so that a Lambertian
    surface returns exactly one::

        DHR = (1/pi) * int_0^2pi int_0^pi/2 shape * cos(t) sin(t) dt dp

    The integrand carries a ``sec(theta)`` from the Li-Sparse kernel,
    which the ``cos(theta)`` weight cancels, so the quadrature converges.
    The upper bound stops just short of 90 deg for that reason.

    Parameters
    ----------
    sza_deg : float
        Sun zenith angle [deg].
    k1p, k2p : float
        Relative geometric and volumetric weights.
    n_theta, n_phi : int
        Quadrature resolution in zenith and azimuth.

    Returns
    -------
    float
        The dimensionless hemispheric factor.
    """
    theta = np.linspace(1e-4, np.pi / 2.0 - 1e-4, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    grid_t, grid_p = np.meshgrid(theta, phi, indexing="ij")

    shape = rtls_shape(
        sza_deg, np.degrees(grid_t), np.degrees(grid_p), k1p, k2p
    )
    weighted = shape * np.cos(grid_t) * np.sin(grid_t)
    return float(np.trapezoid(np.trapezoid(weighted, phi, axis=1), theta)
                 / np.pi)


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


def anisotropy(
    sza_deg: float, vza_deg: float, raa_deg: float, k1p: float, k2p: float
) -> tuple[float, float, float]:
    """Return ``(shape_view, dhr, a)`` for one surface and geometry.

    ``a = dhr / shape_view`` is the ratio between what drives the
    hemispheric terms of the 5S inversion (diffuse illumination, diffuse
    upward transmittance, spherical albedo) and what the algorithm
    retrieves, which is the bidirectional reflectance in the view
    direction.  ``a == 1`` is the Lambertian case, so ``|a - 1|`` ranks
    surfaces by how much they violate the hypothesis under test.
    """
    view = float(rtls_shape(sza_deg, vza_deg, raa_deg, k1p, k2p))
    hemi = rtls_dhr(sza_deg, k1p, k2p)
    return view, hemi, hemi / view


# ----------------------------------------------------------------------
# MCD43A1 ensemble
# ----------------------------------------------------------------------


def _ornl_get(path: str, **params: object) -> dict:
    """Return the decoded JSON of one ORNL MODIS web service call.

    Raises
    ------
    RuntimeError
        If the service is unreachable or answers with anything that is
        not JSON, which is how its outages present themselves.
    """
    url = f"{ORNL_ROOT}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"ORNL MODIS service unreachable: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        head = payload[:200].decode("utf-8", "replace")
        raise RuntimeError(
            f"ORNL MODIS service did not return JSON for {url}: {head}"
        ) from exc


def _mcd43a1_band_names(modis_band: int) -> tuple[str, str]:
    """Return the parameter and quality band names for *modis_band*.

    The names are discovered from the service rather than hard-coded,
    because the layer naming of the subset API has changed across
    releases.  Discovery also fails loudly when the product is not
    served, instead of silently returning an empty subset.
    """
    listing = _ornl_get("MCD43A1/bands")
    names = [entry["band"] for entry in listing.get("bands", [])]

    params = [
        name
        for name in names
        if f"Parameters_Band{modis_band}" in name
        or f"Parameters_band{modis_band}" in name
    ]
    quality = [
        name
        for name in names
        if "Mandatory_Quality" in name and str(modis_band) in name
    ]
    if not params:
        raise RuntimeError(
            f"No MCD43A1 parameter layer for MODIS band {modis_band}. "
            f"Served layers: {names}"
        )
    return params[0], quality[0] if quality else ""


def _subset_values(product: str, band: str, lat: float, lon: float,
                   start: str, end: str) -> list[list[float]]:
    """Return the per-date pixel values of one band at one point."""
    payload = _ornl_get(
        f"{product}/subset",
        latitude=lat,
        longitude=lon,
        band=band,
        startDate=start,
        endDate=end,
        kmAboveBelow=0,
        kmLeftRight=0,
    )
    return [entry["data"] for entry in payload.get("subset", [])]


def fetch_mcd43a1(
    sites: Sequence[tuple[str, float, float]],
    modis_band: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch the RTLS triplet of every site from the MCD43A1 product.

    Only full-BRDF-inversion retrievals are kept (mandatory quality flag
    equal to zero); magnitude inversions carry a climatological shape
    and would defeat the purpose of sampling real surfaces.  The dates
    available in the window are averaged, which suppresses part of the
    inversion noise without smearing across a phenological cycle if the
    window is kept short.

    Parameters
    ----------
    sites : sequence of (str, float, float)
        Site name, latitude and longitude.
    modis_band : int
        MODIS land band index, 1 to 7.
    start, end : str
        Bounding dates, ``AYYYYDDD`` in the MODIS convention.

    Returns
    -------
    pd.DataFrame
        Columns ``site``, ``lat``, ``lon``, ``f_iso``, ``f_geo``,
        ``f_vol``, ``k1p``, ``k2p`` and ``n_dates``.
    """
    param_band, quality_band = _mcd43a1_band_names(modis_band)
    i_iso = MCD43A1_PARAM_ORDER.index("iso")
    i_geo = MCD43A1_PARAM_ORDER.index("geo")
    i_vol = MCD43A1_PARAM_ORDER.index("vol")

    rows: list[dict[str, object]] = []
    for name, lat, lon in sites:
        print(f"    MCD43A1 {name} ({lat:.3f}, {lon:.3f})", flush=True)
        triplets = _subset_values(
            "MCD43A1", param_band, lat, lon, start, end
        )
        flags = (
            _subset_values("MCD43A1", quality_band, lat, lon, start, end)
            if quality_band
            else [[0]] * len(triplets)
        )

        kept = [
            np.asarray(values, dtype=float) * MCD43A1_SCALE
            for values, flag in zip(triplets, flags, strict=False)
            if len(values) >= 3 and float(flag[0]) == 0.0
        ]
        kept = [v for v in kept if np.all(np.isfinite(v)) and v[i_iso] > 0.0]
        if not kept:
            print("      no full-inversion retrieval, skipped", flush=True)
            continue

        mean = np.mean(np.stack(kept), axis=0)
        f_iso, f_geo, f_vol = mean[i_iso], mean[i_geo], mean[i_vol]
        rows.append(
            {
                "site": name,
                "lat": lat,
                "lon": lon,
                "f_iso": float(f_iso),
                "f_geo": float(f_geo),
                "f_vol": float(f_vol),
                "k1p": float(f_geo / f_iso),
                "k2p": float(f_vol / f_iso),
                "n_dates": len(kept),
            }
        )

    if not rows:
        raise RuntimeError(
            "MCD43A1 returned no usable pixel for any site."
        )
    return pd.DataFrame(rows)


def load_ensemble(args: argparse.Namespace) -> pd.DataFrame:
    """Return the MCD43A1 ensemble, from the cache or from the service.

    The cached CSV is the supported hand-off point: any tool able to
    export ``site``, ``f_iso``, ``f_geo`` and ``f_vol`` (AppEEARS, Earth
    Engine, a local granule read) can feed this study without going
    through the ORNL web service.
    """
    path = Path(args.sites_csv)
    if path.exists() and not args.refresh_sites:
        frame = pd.read_csv(path)
        missing = {"site", "f_iso", "f_geo", "f_vol"} - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path} lacks the columns {sorted(missing)}")
        if "k1p" not in frame:
            frame["k1p"] = frame["f_geo"] / frame["f_iso"]
        if "k2p" not in frame:
            frame["k2p"] = frame["f_vol"] / frame["f_iso"]
        print(f">>> MCD43A1 ensemble read from {path}", flush=True)
        return frame

    print(">>> MCD43A1 ensemble, fetching from the ORNL service",
          flush=True)
    frame = fetch_mcd43a1(
        SITES, args.modis_band, args.start_date, args.end_date
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"    cached to {path}", flush=True)
    return frame


def rank_surfaces(
    frame: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Annotate the ensemble with its anisotropy and admissibility.

    Adds ``shape_view``, ``dhr``, ``a``, ``brf_neg_vza`` and
    ``admissible``, and returns the frame sorted by ``a``.  Nothing is
    dropped here: :func:`plot_ensemble` shows the rejected pixels too,
    so that the figure says what the sampling found rather than only
    what survived it.
    """
    frame = frame.copy()
    indices = [
        anisotropy(args.sza, args.vza, args.saa - args.vaa, k1p, k2p)
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]
    frame["shape_view"] = [v for v, _, _ in indices]
    frame["dhr"] = [h for _, h, _ in indices]
    frame["a"] = [a for _, _, a in indices]
    frame["brf_neg_vza"] = [
        brf_onset(args.sza, k1p, k2p)
        for k1p, k2p in zip(frame["k1p"], frame["k2p"], strict=True)
    ]

    # A surface whose fit turns negative inside the angular range that
    # feeds the adjacency term would be simulated with negative photon
    # weights.  Such a pixel is rejected rather than clipped, so that
    # the Python side never disagrees with device.cu.
    frame["admissible"] = frame["brf_neg_vza"] >= args.min_brf_vza
    return frame.sort_values("a").reset_index(drop=True)


def select_surfaces(
    ranked: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Keep the min, median and max surfaces of the admissible ensemble.

    Ranking is on ``a`` itself rather than on ``|a - 1|``: the sign
    matters, because a surface brighter in the hemisphere than in the
    view direction biases the retrieval the other way round.  Taking the
    two extremes and the median therefore brackets both the amplitude
    and the sign of the effect.
    """
    rejected = list(ranked.loc[~ranked["admissible"], "site"])
    if rejected:
        print(
            f"    rejected {len(rejected)} site(s) with a negative BRF "
            f"below {args.min_brf_vza:g} deg: {rejected}",
            flush=True,
        )
    frame = ranked[ranked["admissible"]].reset_index(drop=True)
    if frame.empty:
        raise SystemExit(
            "No sampled surface keeps a positive BRF up to "
            f"{args.min_brf_vza:g} deg."
        )

    picks = {"min": 0, "median": len(frame) // 2, "max": len(frame) - 1}
    selected = frame.iloc[list(picks.values())].copy()
    selected.insert(0, "rank", list(picks.keys()))
    return selected.reset_index(drop=True)


# ----------------------------------------------------------------------
# Surface patch
# ----------------------------------------------------------------------


@contextlib.contextmanager
def rtls_surface(k1p: float, k2p: float) -> Iterator[None]:
    """Make ``SurfaceFactory.surface`` return an ``RTLSSurface``.

    ``k0`` is read from the landscape itself rather than fixed, because
    Smart-G applies the BRDF factor to the *whole* ``ENV=2``
    expression::

        weight *= BRDF * (gauss * (alb_surface - alb_env) + alb_env)

    Both ``alb_surface`` (this object) and ``alb_env`` (the Environment,
    which is left untouched) must therefore already carry the ``1 /
    rtls_shape`` normalisation, otherwise a landscape with a non-zero
    background stops being reproduced: a flat field would pick up a
    spurious Gaussian modulation from ``alb_surface != alb_env``.  This
    is handled upstream by :func:`scaled_landscape`.

    Notes
    -----
    The coefficients are passed through the deprecated ``kp`` argument
    on purpose: the ``k0=``/``k1p=``/``k2p=`` keywords of Smart-G 1.2.0
    assign into a tuple and raise ``TypeError``.
    """
    from smartg.smartg import RTLSSurface
    from smartg.water import Albedo_cst

    from adjeff.atmosphere import SurfaceFactory

    original = SurfaceFactory.surface

    def patched(self, arr: xr.Dataset):  # noqa: ANN001, ANN202
        params = arr["rho_s"].adjeff.params() or {}
        return RTLSSurface(
            kp=(
                Albedo_cst(float(params["rho_max"])),
                Albedo_cst(k1p),
                Albedo_cst(k2p),
            )
        )

    SurfaceFactory.surface = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        SurfaceFactory.surface = original  # type: ignore[method-assign]


def scaled_landscape(scale: float, **kwargs: float) -> ImageDict:
    """Return the Gaussian landscape rescaled for an RTLS run.

    Multiplying both ``rho_min`` and ``rho_max`` by the inverse angular
    factor in the sun-sensor geometry makes the *observed* field
    identical to the Lambertian one, background included.  The truth to
    compare against therefore stays the unscaled landscape.
    """
    kwargs = dict(kwargs)
    kwargs["rho_min"] = float(kwargs["rho_min"]) * scale
    kwargs["rho_max"] = float(kwargs["rho_max"]) * scale
    return gaussian_image_dict(**kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Scalar bias, measured without any adjacency contribution
# ----------------------------------------------------------------------


def uniform_response(
    levels: list[float],
    scale: float,
    k1p: float,
    k2p: float,
    tag: str,
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
) -> pd.DataFrame:
    """Measure the scalar BRDF bias of one surface on a uniform field.

    A uniform landscape produces no adjacency effect at all, and the
    correction of Eq. 5 reduces to the identity on it: the kernel is
    normalised, so ``rho_unif * P == rho_unif`` for a constant field,
    the coupling ratio collapses to one and the remaining prefactor is
    ``(T_up - T_up_dif) / T_up_dir == 1``.  Whatever separates the
    retrieved reflectance from the truth is therefore attributable to
    the scalar terms alone, with the PSF provably out of the picture.

    Sweeping several reflectance levels gives the full transfer function
    rather than a single gain, which matters because the bias is not
    exactly multiplicative: the coupling term ``1 / (1 - s * rho_env)``
    makes it depend on the reflectance itself.

    Returns
    -------
    pd.DataFrame
        Columns ``rho_true``, ``rho_hat_lamb`` and ``rho_hat_rtls``.
    """
    cfg = make_full_config(
        bands=[band],
        aot=args.aot,
        rh=args.rh,
        h=args.h,
        href=args.href,
        sza=args.sza,
        vza=args.vza,
        saa=args.saa,
        vaa=args.vaa,
        species={args.species: 1.0},
    )
    flat = dict(
        sigma=1.0, res_km=run.res_km, bands=[band], n=args.uniform_n
    )

    def centre_mean(scene: ImageDict) -> float:
        da = scene[band]["rho_unif"].squeeze(drop=True)
        n = da.sizes["x"]
        core = da.isel(
            x=slice(n // 4, 3 * n // 4), y=slice(n // 4, 3 * n // 4)
        )
        return float(core.mean())

    rows = []
    for rho in levels:
        lamb = run_forward_pipeline(
            gaussian_image_dict(rho_min=rho, rho_max=rho, **flat),
            **cfg,
            cache=CacheStore(run.cache_dir + "/uniform_lambertian"),
            nr=args.uniform_nr,
            n_ph=run.n_ph,
        )
        with rtls_surface(k1p, k2p):
            rtls = run_forward_pipeline(
                scaled_landscape(scale, rho_min=rho, rho_max=rho, **flat),
                **cfg,
                cache=CacheStore(run.cache_dir + f"/uniform_rtls_{tag}"),
                nr=args.uniform_nr,
                n_ph=run.n_ph,
            )
        rows.append(
            {
                "rho_true": rho,
                "rho_hat_lamb": centre_mean(lamb),
                "rho_hat_rtls": centre_mean(rtls),
            }
        )
        print(f"    {rows[-1]}", flush=True)

    return pd.DataFrame(rows)


def unbias(field: xr.DataArray, resp: pd.DataFrame) -> xr.DataArray:
    """Remove the measured scalar BRDF response from *field*.

    The transfer function of :func:`uniform_response` is inverted by
    monotone interpolation and applied pixel by pixel.  Unlike a fitted
    gain, nothing here is adjusted on the very data being evaluated.
    """
    x = resp["rho_hat_rtls"].to_numpy()
    y = resp["rho_true"].to_numpy()
    order = np.argsort(x)
    return xr.DataArray(
        np.interp(field.values, x[order], y[order]),
        dims=field.dims,
        coords=field.coords,
        attrs=field.attrs,
    )


# ----------------------------------------------------------------------
# Landscapes, forward runs and training
# ----------------------------------------------------------------------


def training_landscapes(
    band: SensorBand, args: argparse.Namespace, run: RunConfig
) -> list[tuple[str, ImageDict]]:
    """Return the 6 Lambertian training landscapes of Section 2.3.1.

    Three Gaussians and three disks, at 1, 5 and 50 km.  The operational
    kernel is fitted jointly on all six, exactly as in the article, and
    this set never changes: only the evaluation surface does.
    """
    common = dict(
        res_km=run.res_km,
        rho_min=0.0,
        rho_max=args.rho_max,
        bands=[band],
        n=run.n,
    )
    out: list[tuple[str, ImageDict]] = []
    for v in args.scales:
        out.append((f"gauss{v:g}", gaussian_image_dict(sigma=v, **common)))
    for v in args.scales:
        out.append((f"disk{v:g}", disk_image_dict(radius=v, **common)))
    return out


def gauss_landscapes(
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
    scale: float = 1.0,
) -> list[tuple[str, ImageDict]]:
    """Return the Gaussian evaluation landscapes, optionally rescaled."""
    return [
        (
            f"gauss{v:g}",
            scaled_landscape(
                scale,
                sigma=v,
                res_km=run.res_km,
                rho_min=0.0,
                rho_max=args.rho_max,
                bands=[band],
                n=run.n,
            ),
        )
        for v in args.scales
    ]


def simulate(
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
    selected: pd.DataFrame,
) -> tuple[
    list[tuple[str, ImageDict]],
    list[tuple[str, ImageDict]],
    dict[str, list[tuple[str, ImageDict]]],
]:
    """Run the forward pipeline, Lambertian then once per RTLS surface.

    Returns
    -------
    tuple
        The six Lambertian training scenes, the three Lambertian
        Gaussian evaluation scenes, and the RTLS evaluation scenes keyed
        by rank.
    """
    cfg = make_full_config(
        bands=[band],
        aot=args.aot,
        rh=args.rh,
        h=args.h,
        href=args.href,
        sza=args.sza,
        vza=args.vza,
        saa=args.saa,
        vaa=args.vaa,
        species={args.species: 1.0},
    )
    pipeline = dict(nr=args.nr, n_ph=run.n_ph)
    lamb_store = CacheStore(run.cache_dir + "/lambertian")

    print(">>> forward pipeline, Lambertian training set", flush=True)
    train_scenes = [
        (name, run_forward_pipeline(
            img, **cfg, cache=lamb_store, **pipeline))
        for name, img in training_landscapes(band, args, run)
    ]

    # The Gaussian evaluation scenes are a subset of the training set and
    # share its cache, so this loop costs nothing beyond the lookups.
    lamb_eval = [
        (name, run_forward_pipeline(
            img, **cfg, cache=lamb_store, **pipeline))
        for name, img in gauss_landscapes(band, args, run)
    ]

    # The cache is keyed on the scene configuration, which the surface
    # patch does not alter, so each RTLS surface needs its own store or
    # it would silently return another surface's result.
    rtls_eval: dict[str, list[tuple[str, ImageDict]]] = {}
    for row in selected.itertuples():
        print(f">>> forward pipeline, RTLS {row.rank} ({row.site})",
              flush=True)
        with rtls_surface(row.k1p, row.k2p):
            rtls_eval[row.rank] = [
                (name, run_forward_pipeline(
                    img, **cfg,
                    cache=CacheStore(run.cache_dir + f"/rtls_{row.rank}"),
                    **pipeline))
                for name, img in gauss_landscapes(
                    band, args, run, 1.0 / row.shape_view
                )
            ]

    return train_scenes, lamb_eval, rtls_eval


def train(
    scenes: list[tuple[str, ImageDict]],
    band: SensorBand,
    run: RunConfig,
) -> tuple[xr.DataArray, dict[str, float]]:
    """Optimise a single King PSF over the whole training set.

    This is the operational kernel of the article: **one** kernel fitted
    jointly on the six Lambertian landscapes, not one kernel per
    landscape.  It is the only kernel this study uses; the whole point
    is to apply it, unchanged, to surfaces it was not trained for.
    """
    images = [img for _, img in scenes]

    model = make_model(
        Unif2Surface,
        KingPSF,
        [band],
        res_km=run.res_km,
        n=run.n,
        init_parameters={"sigma": 0.1, "gamma": 1.0},
        device=run.device,
    )
    tree = fit(
        model,
        TrainingImages(images=images),
        loss=Loss(Metric.RMSE_RAD),
        device=run.device,
    )
    kernel = psf_kernel(tree, band).squeeze(drop=True)
    params = model.psf_params(band)

    # Joint training over six 3999x3999 landscapes is memory hungry.
    # The model is released before returning so that a second training
    # in the same process does not add to the peak.
    del model, tree
    if run.device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()

    return kernel, params


def evaluate(
    lamb_eval: list[tuple[str, ImageDict]],
    rtls_eval: dict[str, list[tuple[str, ImageDict]]],
    responses: dict[str, pd.DataFrame],
    kernel: xr.DataArray,
    band: SensorBand,
    run: RunConfig,
    selected: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, dict[str, xr.DataArray]]]:
    """Correct every landscape with the shared operational kernel.

    Returns
    -------
    tuple
        The metric rows, and the retrieved fields keyed by landscape
        then by series name, for the figures.
    """
    rows: list[dict[str, object]] = []
    fields: dict[str, dict[str, xr.DataArray]] = {}
    by_rank = {row.rank: row for row in selected.itertuples()}

    for index, (name, ls) in enumerate(lamb_eval):
        truth = ls[band]["rho_s"].squeeze(drop=True)
        lamb_est, lamb_unif = correct(ls[band], band, kernel, run.device)
        fields[name] = {
            "truth": truth, "unif": lamb_unif, "lamb": lamb_est
        }

        base = radial_rmse(lamb_est, truth, lamb_unif, run.device)
        rows.append(
            {
                "landscape": name,
                "rank": "lambertian",
                "site": "-",
                "a": 1.0,
                "no_corr": radial_rmse(
                    lamb_unif, truth, lamb_unif, run.device),
                "corr": base,
                "corr_unbiased": base,
                "eps_total": 0.0,
                "eps_psf": 0.0,
            }
        )
        print(f"    {rows[-1]}", flush=True)

        for rank, scenes in rtls_eval.items():
            hs = scenes[index][1]
            est, unif = correct(hs[band], band, kernel, run.device)
            flat = unbias(est, responses[rank])
            fields[name][rank] = est
            fields[name][f"{rank}_flat"] = flat

            row = {
                "landscape": name,
                "rank": rank,
                "site": by_rank[rank].site,
                "a": float(by_rank[rank].a),
                "no_corr": radial_rmse(unif, truth, unif, run.device),
                "corr": radial_rmse(est, truth, unif, run.device),
                "corr_unbiased": radial_rmse(
                    flat, truth, unif, run.device),
            }
            # eps_total is what an operational chain actually suffers;
            # eps_psf is what is left once the scalar 5S bias, measured
            # out of sample on a uniform field, has been removed.  The
            # gap between the two says whether a BRDF-aware kernel or
            # BRDF-aware scalar terms are the useful fix.
            row["eps_total"] = row["corr"] - base
            row["eps_psf"] = row["corr_unbiased"] - base
            rows.append(row)
            print(f"    {row}", flush=True)

    return rows, fields


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

_RANK_STYLE = {
    "min": dict(color="tab:green", lw=0.9, ls="--"),
    "median": dict(color="tab:orange", lw=0.9, ls="-."),
    "max": dict(color="tab:red", lw=0.9, ls=":"),
}


def plot_ensemble(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    args: argparse.Namespace,
    run: RunConfig,
) -> None:
    """Draw the sampled ensemble and the three surfaces kept from it."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    ax = axes[0]
    ordered = frame.reset_index(drop=True)
    keep = ordered["admissible"].to_numpy()
    ax.axhline(1.0, color="k", lw=0.6, ls="-")
    ax.annotate(
        "Lambertian", xy=(0, 1.0), xytext=(2, 3),
        textcoords="offset points", fontsize=6,
    )
    position = np.arange(len(ordered))
    ax.plot(position[keep], ordered["a"][keep], "o", ms=3, color="0.4")
    # Rejected pixels stay on the plot as crosses: the reader should see
    # how much of the sampled ensemble the admissibility test removed.
    ax.plot(
        position[~keep], ordered["a"][~keep], "x", ms=4, color="0.7",
        label="negative BRF, rejected",
    )
    for row in selected.itertuples():
        at = list(ordered["site"]).index(row.site)
        ax.plot([at], [row.a], "o", ms=6, mfc="none",
                color=_RANK_STYLE[row.rank]["color"])
        ax.annotate(
            row.rank, xy=(at, row.a), xytext=(4, 0),
            textcoords="offset points", fontsize=6,
            color=_RANK_STYLE[row.rank]["color"],
        )
    if not keep.all():
        ax.legend(fontsize=5, frameon=False, loc="lower right")
    ax.set_xticks(position)
    ax.set_xticklabels(ordered["site"], rotation=60, ha="right", fontsize=5)
    ax.set_ylabel(r"$a = \mathrm{DHR}\,/\,\rho(\theta_v)$")
    ax.set_title(
        f"(a) MCD43A1 ensemble, {int(keep.sum())}/{len(ordered)} kept"
    )

    ax = axes[1]
    vza = np.linspace(-80.0, 80.0, 601)
    raa = np.where(vza < 0.0, 180.0, 0.0)
    for row in selected.itertuples():
        shape = rtls_shape(args.sza, np.abs(vza), raa, row.k1p, row.k2p)
        ax.plot(
            vza,
            args.rho_max * shape / row.shape_view,
            label=f"{row.rank}: {row.site} ($a={row.a:.2f}$)",
            **_RANK_STYLE[row.rank],
        )
    ax.axhline(args.rho_max, color="k", lw=0.6)
    ax.axvline(args.vza, color="tab:blue", ls="--", lw=0.8)
    ax.annotate(
        "view", xy=(args.vza, args.rho_max), xytext=(4, 4),
        textcoords="offset points", color="tab:blue", fontsize=6,
    )
    ax.set_xlabel(r"view zenith angle [$^\circ$]")
    ax.set_ylabel(r"$\rho_\mathrm{s}$")
    ax.legend(fontsize=5, frameon=False)
    ax.set_title(
        rf"(b) RTLS in the principal plane, $\theta_s={args.sza:.0f}^\circ$"
    )

    fig.tight_layout()
    print(f"    wrote {save(fig, 'hotspot_ensemble', run.figs_dir, dpi=300)}",
          flush=True)


def plot_scale(
    name: str,
    series: dict[str, xr.DataArray],
    selected: pd.DataFrame,
    args: argparse.Namespace,
    run: RunConfig,
    scale_km: float,
) -> None:
    """Draw the three-panel figure for one Gaussian landscape."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    truth = series["truth"]
    lim = 4.0 * scale_km

    # (a) the BRDFs actually simulated, on the scale of the landscape
    ax = axes[0]
    vza = np.linspace(-80.0, 80.0, 601)
    raa = np.where(vza < 0.0, 180.0, 0.0)
    for row in selected.itertuples():
        shape = rtls_shape(args.sza, np.abs(vza), raa, row.k1p, row.k2p)
        ax.plot(
            vza,
            args.rho_max * shape / row.shape_view,
            label=f"{row.rank} ($a={row.a:.2f}$)",
            **_RANK_STYLE[row.rank],
        )
    # The normalisation is what makes the curves cross rho_max at the
    # view angle: plotting k0 * shape instead would show a surface that
    # was never simulated.
    ax.axhline(args.rho_max, color="k", lw=0.6)
    ax.axvline(args.vza, color="tab:blue", ls="--", lw=0.8)
    ax.set_xlabel(r"view zenith angle [$^\circ$]")
    ax.set_ylabel(r"$\rho_\mathrm{s}$ (RTLS)")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(rf"(a) RTLS, $\theta_s={args.sza:.0f}^\circ$")

    # (b) retrieved surface reflectance
    ax = axes[1]
    curves = [
        (series["truth"], r"$\rho_\mathrm{s}$ (truth)",
         dict(color="k", lw=1.0)),
        (series["unif"], r"$\rho_\mathrm{unif}$",
         dict(color="0.6", lw=0.8, ls=":")),
        (series["lamb"], "Lambertian", dict(color="tab:blue", lw=0.9)),
    ]
    curves += [
        (series[row.rank], row.rank, _RANK_STYLE[row.rank])
        for row in selected.itertuples()
    ]
    for da, label, style in curves:
        r, v = sym_profile(da)
        ax.plot(r, v, label=label, **style)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"$r$ [km]")
    ax.set_ylabel(r"$\rho_\mathrm{s}$")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(rf"(b) retrieval, $\sigma={scale_km:g}$ km")

    # (c) retrieval error
    ax = axes[2]
    errors = [(series["lamb"], "Lambertian", dict(color="tab:blue", lw=0.9))]
    for row in selected.itertuples():
        errors.append((series[row.rank], row.rank, _RANK_STYLE[row.rank]))
        errors.append(
            (
                series[f"{row.rank}_flat"],
                f"{row.rank}, scalar bias removed",
                dict(_RANK_STYLE[row.rank], lw=0.6, alpha=0.55),
            )
        )
    for da, label, style in errors:
        r, v = sym_profile(da - truth)
        ax.plot(r, v, label=label, **style)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"$r$ [km]")
    ax.set_ylabel(r"$\hat{\rho}_\mathrm{s} - \rho_\mathrm{s}$")
    ax.legend(fontsize=5, frameon=False)
    ax.set_title("(c) retrieval error")

    fig.tight_layout()
    out = save(fig, f"hotspot_{name}km", run.figs_dir, dpi=300)
    print(f"    wrote {out}", flush=True)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the script's own argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wl", type=float, default=665.0)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=[1.0, 5.0, 50.0],
        help="Gaussian widths and disk radii in km.",
    )
    parser.add_argument("--aot", type=float, default=0.4)
    parser.add_argument("--rh", type=float, default=50.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--href", type=float, default=2.0)
    parser.add_argument("--sza", type=float, default=40.0)
    parser.add_argument("--vza", type=float, default=0.0)
    parser.add_argument("--saa", type=float, default=0.0)
    parser.add_argument("--vaa", type=float, default=0.0)
    parser.add_argument("--species", type=str, default="sulphate")
    parser.add_argument(
        "--rho-max", type=float, default=0.3,
        help="Peak surface reflectance of the landscapes.  Kept below "
        "the value that would drive the hemispheric albedo above 1.",
    )
    parser.add_argument(
        "--sites-csv", type=Path,
        default=Path("output") / "hotspot_mcd43a1_sites.csv",
        help="Cache of the MCD43A1 ensemble.  Any CSV carrying site, "
        "f_iso, f_geo and f_vol is accepted.",
    )
    parser.add_argument(
        "--min-brf-vza", type=float, default=85.0,
        help="Drop a sampled surface whose RTLS fit turns negative below "
        "this view zenith [deg]; device.cu does not clip the BRDF.",
    )
    parser.add_argument(
        "--refresh-sites", action="store_true",
        help="Refetch the ensemble even if the cache exists.",
    )
    parser.add_argument(
        "--modis-band", type=int, default=0,
        help="MODIS land band, 1 to 7.  Defaults to the band closest to "
        "--wl.",
    )
    parser.add_argument("--start-date", type=str, default="A2020169")
    parser.add_argument("--end-date", type=str, default="A2020217")
    parser.add_argument(
        "--uniform-levels", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Reflectance levels probed on a uniform surface to measure "
        "the scalar BRDF bias without any adjacency contribution.",
    )
    parser.add_argument("--uniform-n", type=int, default=399)
    parser.add_argument("--uniform-nr", type=int, default=60)
    parser.add_argument("--nr", type=int, default=500)
    return parser


def main() -> None:
    """Quantify the PSF degradation caused by real MODIS BRDFs."""
    run, args = parse_run(__doc__.splitlines()[0], build_parser())
    run = run.resolve(n=3999, res_km=0.05, cache_dir="/tmp/adjeff-hotspot")

    # The study is a six-landscape joint training on top of one
    # Lambertian and three RTLS forward pipelines: a smoke run keeps one
    # landscape and three reflectance levels, which still walks the
    # whole chain.
    if run.smoke:
        args.scales = [1.0]
        args.uniform_levels = [0.0, 0.5, 1.0]
        args.uniform_n = 99
        args.uniform_nr = 20

    band = S2Band.from_wl(args.wl)
    if not args.modis_band:
        args.modis_band = _WL_TO_MODIS_BAND.get(args.wl, 1)
    run.figs_dir.mkdir(parents=True, exist_ok=True)

    ensemble = rank_surfaces(load_ensemble(args), args)
    selected = select_surfaces(ensemble, args)
    ensemble.to_csv(
        run.figs_dir / "hotspot_mcd43a1_ranked.csv", index=False
    )
    out = run.figs_dir / "hotspot_mcd43a1_selected.csv"
    selected.to_csv(out, index=False)
    print(selected.to_string(index=False), flush=True)
    print(f"    wrote {out}", flush=True)

    # A view-anchored RTLS surface reflects rho_max in the view
    # direction but a * rho_max over the hemisphere.  Above one that
    # surface stops being physical, and the run would be meaningless.
    worst = float(selected["a"].max()) * args.rho_max
    if worst > 1.0:
        raise SystemExit(
            f"hemispheric albedo would reach {worst:.3f} > 1 with "
            f"--rho-max {args.rho_max}. Lower it below "
            f"{1.0 / float(selected['a'].max()):.3f}."
        )
    print(f"    peak hemispheric albedo: {worst:.3f}", flush=True)

    print(">>> scalar BRDF response on a uniform surface", flush=True)
    responses = {}
    for row in selected.itertuples():
        print(f"  {row.rank} ({row.site})", flush=True)
        responses[row.rank] = uniform_response(
            args.uniform_levels, 1.0 / row.shape_view, row.k1p, row.k2p,
            row.rank, band, args, run,
        )
        responses[row.rank].to_csv(
            run.figs_dir / f"hotspot_uniform_response_{row.rank}.csv",
            index=False,
        )

    train_scenes, lamb_eval, rtls_eval = simulate(band, args, run, selected)

    print(">>> King PSF, trained jointly on the 6 Lambertian landscapes",
          flush=True)
    kernel, p_lamb = train(train_scenes, band, run)
    print(f"    Lambertian training : {p_lamb}", flush=True)
    pd.DataFrame([{"trained_on": "lambertian", **p_lamb}]).to_csv(
        run.figs_dir / "hotspot_king_params.csv", index=False
    )

    rows, fields = evaluate(
        lamb_eval, rtls_eval, responses, kernel, band, run, selected
    )

    print(">>> figures", flush=True)
    plot_ensemble(ensemble, selected, args, run)
    for name, series in fields.items():
        plot_scale(
            name, series, selected, args, run,
            scale_km=float(name.removeprefix("gauss")),
        )

    df = pd.DataFrame(rows)
    out = run.figs_dir / "hotspot_rmse.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
