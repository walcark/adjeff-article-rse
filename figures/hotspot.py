r"""Accuracy of a Lambertian-trained PSF applied to a hotspot BRDF.

Answers reviewer question T1: quantify the accuracy degradation of the
King kernel when the surface is not Lambertian.

Protocol
--------
1. A Gaussian training landscape is simulated with a **Lambertian**
   surface, and a King PSF is optimised on it, exactly as in Section
   2.3.  This is the operational kernel.
2. The **same** landscape is simulated again, with the same atmosphere
   and the same geometry, but the surface is now a Rahman-Pinty-
   Verstraete (RPV) reflector, i.e. a canopy BRDF with a hotspot.
3. The second scene is corrected with the kernel trained in step 1, and
   the radial RMSE is compared with the Lambertian case.

Any degradation is therefore attributable to the surface BRDF alone.

Why this works in Smart-G
-------------------------
The analytical Gaussian landscape of adjeff is built with ``ENV=2``,
which modulates the surface albedo by ``exp(-r^2 / ENV_SIZE)`` *after*
the BRDF factor has been applied (``device.cu``, ``surfaceBRDF_new``)::

    ph->weight *= BRDF(ilam, -v0, ph->v, spectrum);
    if (ENVd==2) ph->weight *= gauss_albedo(pos) * (alb_surface - alb_env)
                               + alb_env;

Swapping ``LambSurface`` for ``RPVSurface`` therefore keeps the spatial
landscape strictly identical and only changes its angular signature.
Note that this is **not** true of the ``ENV=5`` albedo-map path used for
arbitrary fields, which multiplies by ``alb_envs[ispec]`` and never
calls ``BRDF`` at all: an arbitrary landscape cannot carry a BRDF in
this version of Smart-G.

Normalisation
-------------
An RPV surface of parameter ``r0`` does not reflect ``r0`` in the
observation direction.  With ``--normalise view`` (the default), ``r0``
is rescaled so that the surface reflectance in the exact sun-sensor
geometry equals ``rho_max``, i.e. the *directly transmitted* signal is
identical in both runs.  What is left is the angular redistribution of
the environment contribution, which is precisely what the PSF models.
Pass ``--normalise none`` to compare at equal ``r0`` instead.

Caveats
-------
- The RPV coefficients below are representative values for a vegetated
  canopy in the red, where the hotspot is most pronounced.  They are
  **not** taken from a specific inversion: override them on the command
  line to match a documented case.
- ``rho_toa_sym`` assumes the TOA field is radially symmetric.  This
  holds exactly for a Lambertian surface.  With an anisotropic BRDF the
  environment contribution is only symmetric when the sun is at zenith,
  so the simulated profile is the one sampled along the azimuth of
  ``_radial_sensors``.  The effect is second order as long as the
  environment term is small compared with the direct term, but it is an
  approximation, not an identity.

Usage
-----
python hotspot.py --wl 665 --scales 1.0 --aot 0.4 \\
    --sza 40 --vza 0 --saa 0 --vaa 0 \\
    --r0 0.03 --k 0.65 --bt -0.15 --rc 0.10
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import xarray as xr

from adjeff.api import (
    make_full_config,
    make_model,
    optimize_adam_lbfgs,
    run_forward_pipeline,
)
from adjeff.core import (
    ImageDict,
    KingPSF,
    SensorBand,
    disk_image_dict,
    gaussian_image_dict,
)
from adjeff.modules.models import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages
from adjeff.utils import CacheStore
from adjeff_article_1.runconfig import RunConfig, parse_run
from adjeff_article_1.shim import (
    correct,
    fitted_params,
    radial_rmse,
    sym_profile,
    wl_to_band,
)
from adjeff_article_1.style import save

plt.style.use(["science", "nature"])

# Representative RPV coefficients for a vegetated canopy at 665 nm, the
# band where the hotspot is most visible on crops: the red is strongly
# absorbed, so the shadow-hiding contrast between the backscatter and
# the forward direction is at its largest.
#   r0  normalisation (close to the nadir red reflectance of a canopy)
#   k   Minnaert exponent, k < 1 means a bowl-shaped angular response
#   bt  Henyey-Greenstein asymmetry, bt < 0 means backscattering
#   rc  hotspot amplitude, the peak factor is (2 - rc) at exact backscatter
RPV_RED = {"r0": 0.03, "k": 0.65, "bt": -0.15, "rc": 0.10}


# ----------------------------------------------------------------------
# RPV model, mirroring smartg/src/device.cu
# ----------------------------------------------------------------------


def rpv_shape(
    sza_deg: float | np.ndarray,
    vza_deg: float | np.ndarray,
    raa_deg: float | np.ndarray,
    k: float,
    bt: float,
    rc: float,
) -> np.ndarray:
    """Return the RPV angular factor, without the ``r0`` normalisation.

    Mirrors ``Minnaert_rpv * HG_rpv * H_rpv`` of ``device.cu`` so that the
    Python side and the Monte-Carlo kernel share the same convention.
    The surface reflectance is ``r0 * rpv_shape(...)``.

    Parameters
    ----------
    sza_deg : float or ndarray
        Sun zenith angle [deg].
    vza_deg : float or ndarray
        View zenith angle [deg].
    raa_deg : float or ndarray
        Relative azimuth [deg] between the direction towards the sun and
        the direction towards the sensor.  Zero is the backscatter
        (hotspot) half-plane, following the ``dph`` of ``device.cu``.
    k : float
        Minnaert exponent.
    bt : float
        Henyey-Greenstein asymmetry parameter, negative for backscatter.
    rc : float
        Hotspot parameter.

    Returns
    -------
    ndarray
        The dimensionless angular factor.
    """
    th0 = np.radians(np.asarray(sza_deg, dtype=float))
    th1 = np.radians(np.asarray(vza_deg, dtype=float))
    dph = np.radians(np.asarray(raa_deg, dtype=float))

    mu0 = np.abs(np.cos(th0))
    mu1 = np.abs(np.cos(th1))

    # device.cu leaves dph at 0 when either direction is exactly vertical,
    # since the horizontal projection is then undefined.
    dph = np.where((th0 == 0.0) | (th1 == 0.0), 0.0, dph)

    minnaert = (mu0 * mu1) ** (k - 1.0) / (mu0 + mu1) ** (1.0 - k)

    cosg = mu0 * mu1 + np.sqrt(1.0 - mu0**2) * np.sqrt(1.0 - mu1**2) * np.cos(
        dph
    )
    hg = (1.0 - bt**2) / (1.0 + bt**2 + 2.0 * bt * cosg) ** 1.5

    t0, t1 = np.tan(th0), np.tan(th1)
    big_g = np.sqrt(t0**2 + t1**2 - 2.0 * t0 * t1 * np.cos(dph))
    hot = 1.0 + (1.0 - rc) / (1.0 + big_g)

    return np.asarray(minnaert * hg * hot, dtype=float)


@contextlib.contextmanager
def rpv_surface(k: float, bt: float, rc: float) -> Iterator[None]:
    """Make ``SurfaceFactory.surface`` return an ``RPVSurface``.

    ``r0`` is read from the landscape itself rather than fixed, because
    Smart-G applies the BRDF factor to the *whole* ``ENV=2`` expression::

        weight *= BRDF * (gauss * (alb_surface - alb_env) + alb_env)

    Both ``alb_surface`` (this object) and ``alb_env`` (the Environment,
    which is left untouched) must therefore already carry the ``1 /
    rpv_shape`` normalisation, otherwise a landscape with a non-zero
    background stops being reproduced: a flat field would pick up a
    spurious Gaussian modulation from ``alb_surface != alb_env``.  This
    is handled upstream by :func:`rpv_landscape`.

    Notes
    -----
    The coefficients are passed through the deprecated ``kp`` argument on
    purpose: the ``r0=``/``k=``/``bt=``/``rc=`` keywords of Smart-G 1.2.0
    assign into a tuple and raise ``TypeError``.
    """
    from smartg.smartg import RPVSurface
    from smartg.water import Albedo_cst

    from adjeff.atmosphere import SurfaceFactory

    original = SurfaceFactory.surface

    def patched(self, arr: xr.Dataset):  # noqa: ANN001, ANN202
        params = arr["rho_s"].adjeff.params() or {}
        return RPVSurface(
            kp=(
                Albedo_cst(float(params["rho_max"])),
                Albedo_cst(k),
                Albedo_cst(bt),
                Albedo_cst(rc),
            )
        )

    SurfaceFactory.surface = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        SurfaceFactory.surface = original  # type: ignore[method-assign]


# ----------------------------------------------------------------------
# Correction and metrics
# ----------------------------------------------------------------------





def rpv_landscape(scale: float, **kwargs: float) -> ImageDict:
    """Return the Gaussian landscape rescaled for the RPV run.

    Dividing both ``rho_min`` and ``rho_max`` by the RPV angular factor
    in the sun-sensor geometry makes the *observed* field identical to
    the Lambertian one, background included.  The truth to compare
    against therefore stays the unscaled landscape.
    """
    kwargs = dict(kwargs)
    kwargs["rho_min"] = float(kwargs["rho_min"]) * scale
    kwargs["rho_max"] = float(kwargs["rho_max"]) * scale
    return gaussian_image_dict(**kwargs)  # type: ignore[arg-type]


def uniform_response(
    levels: list[float],
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
) -> pd.DataFrame:
    """Measure the scalar BRDF bias on a strictly uniform surface.

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
        Columns ``rho_true``, ``rho_hat_lamb`` and ``rho_hat_rpv``.
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

    def flat(rho: float) -> ImageDict:
        return gaussian_image_dict(
            sigma=1.0,
            res_km=run.res_km,
            rho_min=rho,
            rho_max=rho,
            bands=[band],
            n=args.uniform_n,
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
            flat(rho),
            **cfg,
            cache=CacheStore(run.cache_dir + "/uniform_lambertian"),
            nr=args.uniform_nr,
            n_ph=run.n_ph,
        )
        with rpv_surface(args.k, args.bt, args.rc):
            rpv = run_forward_pipeline(
                rpv_landscape(
                    args.rpv_scale,
                    sigma=1.0,
                    res_km=run.res_km,
                    rho_min=rho,
                    rho_max=rho,
                    bands=[band],
                    n=args.uniform_n,
                ),
                **cfg,
                cache=CacheStore(run.cache_dir + "/uniform_hotspot"),
                nr=args.uniform_nr,
                n_ph=run.n_ph,
            )
        rows.append(
            {
                "rho_true": rho,
                "rho_hat_lamb": centre_mean(lamb),
                "rho_hat_rpv": centre_mean(rpv),
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
    x = resp["rho_hat_rpv"].to_numpy()
    y = resp["rho_true"].to_numpy()
    order = np.argsort(x)
    return xr.DataArray(
        np.interp(field.values, x[order], y[order]),
        dims=field.dims,
        coords=field.coords,
        attrs=field.attrs,
    )


def scalar_gain(pred: xr.DataArray, truth: xr.DataArray) -> float:
    """Return the least-squares scalar gain between *pred* and *truth*.

    A non-Lambertian surface biases the 5S scalar inversion of Eq. 3,
    because ``rho_unif`` is built from hemispheric quantities (diffuse
    illumination, diffuse upward transmittance, spherical albedo) that
    sample the BRDF over angles other than the observation direction.
    The resulting error is *multiplicative*, i.e. proportional to
    ``rho_s``, and has nothing to do with the shape of the PSF.

    Removing this gain before computing the radial RMSE is what
    separates the two contributions, exactly as ``eps_scalar`` and
    ``eps_psf`` are separated in the AOT sensitivity table.
    """
    a = np.asarray(pred.values, dtype=float).ravel()
    b = np.asarray(truth.values, dtype=float).ravel()
    denom = float(np.dot(a, a))
    return float(np.dot(a, b) / denom) if denom > 0.0 else 1.0



# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------


def plot(
    truth: xr.DataArray,
    lamb: xr.DataArray,
    hots: xr.DataArray,
    hots_flat: xr.DataArray,
    unif: xr.DataArray,
    args: argparse.Namespace,
    run: RunConfig,
    scale_km: float,
    name: str,
) -> None:
    """Draw the three-panel hotspot figure and save it to *out*."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    # (a) the BRDF itself, in the principal plane
    ax = axes[0]
    vza = np.linspace(-80.0, 80.0, 601)
    raa = np.where(vza < 0.0, 180.0, 0.0)
    shape = rpv_shape(args.sza, np.abs(vza), raa, args.k, args.bt, args.rc)
    ax.plot(vza, args.r0 * shape, color="k", lw=1.0)
    ax.axvline(args.sza, color="crimson", ls=":", lw=0.8)
    ax.annotate(
        "hotspot",
        xy=(args.sza, args.r0 * shape.max()),
        xytext=(4, -2),
        textcoords="offset points",
        color="crimson",
        fontsize=7,
    )
    ax.axvline(args.vza, color="tab:blue", ls="--", lw=0.8)
    ax.annotate(
        "view",
        xy=(args.vza, args.r0 * shape.min()),
        xytext=(4, 2),
        textcoords="offset points",
        color="tab:blue",
        fontsize=7,
    )
    ax.set_xlabel(r"view zenith angle [$^\circ$]")
    ax.set_ylabel(r"$\rho_\mathrm{s}$ (RPV)")
    ax.set_title(
        rf"(a) RPV, $\theta_s={args.sza:.0f}^\circ$ "
        rf"($k={args.k}$, $\Theta={args.bt}$, $\rho_c={args.rc}$)"
    )

    # (b) retrieved surface reflectance
    ax = axes[1]
    lim = 4.0 * scale_km
    for da, label, style in (
        (truth, r"$\rho_\mathrm{s}$ (truth)", dict(color="k", lw=1.0)),
        (unif, r"$\rho_\mathrm{unif}$", dict(color="0.6", lw=0.8, ls=":")),
        (
            lamb,
            r"$\hat{\rho}_\mathrm{s}$, Lambertian",
            dict(color="tab:blue", lw=0.9),
        ),
        (
            hots,
            r"$\hat{\rho}_\mathrm{s}$, hotspot",
            dict(color="tab:red", lw=0.9, ls="--"),
        ),
    ):
        r, v = sym_profile(da)
        ax.plot(r, v, label=label, **style)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"$r$ [km]")
    ax.set_ylabel(r"$\rho_\mathrm{s}$")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(rf"(b) retrieval, $\sigma={scale_km:g}$ km")

    # (c) retrieval error
    ax = axes[2]
    for da, label, style in (
        (lamb, "Lambertian", dict(color="tab:blue", lw=0.9)),
        (hots, "hotspot", dict(color="tab:red", lw=0.9, ls="--")),
        (
            hots_flat,
            "hotspot, scalar bias removed",
            dict(color="tab:orange", lw=0.9, ls="-."),
        ),
    ):
        r, v = sym_profile(da - truth)
        ax.plot(r, v, label=label, **style)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"$r$ [km]")
    ax.set_ylabel(r"$\hat{\rho}_\mathrm{s} - \rho_\mathrm{s}$")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("(c) retrieval error")

    fig.tight_layout()
    out = save(fig, name, run.figs_dir, dpi=300)
    print(f"    wrote {out}", flush=True)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def build_landscapes(
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
    scale: float = 1.0,
) -> list[tuple[str, ImageDict]]:
    """Return the 6 training landscapes of Section 2.3.1, as in the article.

    Three Gaussians and three disks, at 1, 5 and 50 km.  ``scale``
    rescales the reflectance for the RPV run so that the *observed*
    field matches the Lambertian one.

    Disks keep ``rho_min = 0``, which matters here: Smart-G applies the
    BRDF only inside the ``ENV=1`` disk and leaves the outside on the
    Lambertian ``alb_env``.  With a black background that asymmetry has
    no effect, but it would break for a non-zero ``rho_min``.
    """
    common = dict(
        res_km=run.res_km,
        rho_min=0.0,
        rho_max=args.rho_max * scale,
        bands=[band],
        n=run.n,
    )
    out: list[tuple[str, ImageDict]] = []
    for v in args.scales:
        out.append((f"gauss{v:g}", gaussian_image_dict(sigma=v, **common)))
    for v in args.scales:
        out.append((f"disk{v:g}", disk_image_dict(radius=v, **common)))
    return out


def simulate(
    band: SensorBand, args: argparse.Namespace, run: RunConfig
) -> tuple[list[tuple[str, ImageDict]], list[tuple[str, ImageDict]]]:
    """Run the forward pipeline on the 6 landscapes, Lambertian and RPV."""
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

    print(">>> forward pipeline, Lambertian", flush=True)
    lamb = [
        (name, run_forward_pipeline(
            img, **cfg,
            cache=CacheStore(run.cache_dir + "/lambertian"), **pipeline))
        for name, img in build_landscapes(band, args, run)
    ]

    # The cache is keyed on the scene configuration, which the surface
    # patch does not alter, so the RPV run needs its own store or it
    # would silently return the Lambertian result.
    print(">>> forward pipeline, RPV hotspot", flush=True)
    with rpv_surface(args.k, args.bt, args.rc):
        hots = [
            (name, run_forward_pipeline(
                img, **cfg,
                cache=CacheStore(run.cache_dir + "/hotspot"), **pipeline))
            for name, img in build_landscapes(band, args, run, args.rpv_scale)
        ]

    return lamb, hots


def train(
    scenes: list[tuple[str, ImageDict]],
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
) -> tuple[xr.DataArray, dict[str, float]]:
    """Optimise a single King PSF over the whole training set.

    This is the operational kernel of the article: **one** kernel fitted
    jointly on the six Lambertian landscapes, not one kernel per
    landscape.  It is the only kernel this study uses; the whole point
    is to apply it, unchanged, to a surface it was not trained for.
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
    psf_dict = optimize_adam_lbfgs(
        model,
        TrainingImages(images=images, weights=[1.0] * len(images)),
        Loss(Metric.RMSE_RAD),
        device=run.device,
    )
    kernel = psf_dict.kernel(band).squeeze(drop=True)
    params = fitted_params(model)

    # Joint training over six 3999x3999 landscapes is memory hungry.
    # The model is released before returning so that a second training
    # in the same process does not add to the peak.
    del model, psf_dict
    if run.device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()

    return kernel, params


def evaluate(
    lamb: list[tuple[str, ImageDict]],
    hots: list[tuple[str, ImageDict]],
    kernel: xr.DataArray,
    band: SensorBand,
    args: argparse.Namespace,
    run: RunConfig,
    resp: pd.DataFrame,
) -> list[dict[str, float]]:
    """Correct every landscape with the shared operational kernel."""
    rows: list[dict[str, float]] = []

    for (name, ls), (_, hs) in zip(lamb, hots, strict=True):
        truth = ls[band]["rho_s"].squeeze(drop=True)
        lamb_est, lamb_unif = correct(ls[band], band, kernel, run.device)
        hots_est, hots_unif = correct(hs[band], band, kernel, run.device)

        gain = scalar_gain(hots_est, truth)
        row = {
            "landscape": name,
            "wl_nm": band.wl_nm,
            "no_corr_lamb": radial_rmse(
                lamb_unif, truth, lamb_unif, run.device),
            "corr_lamb": radial_rmse(lamb_est, truth, lamb_unif, run.device),
            "no_corr_hots": radial_rmse(
                hots_unif, truth, hots_unif, run.device),
            "corr_hots": radial_rmse(hots_est, truth, hots_unif, run.device),
            "brdf_gain": gain,
            "corr_hots_degained": radial_rmse(
                hots_est * gain, truth, hots_unif, run.device),
            "corr_hots_unbiased": radial_rmse(
                unbias(hots_est, resp), truth, hots_unif, run.device),
        }
        row["eps_total"] = row["corr_hots"] - row["corr_lamb"]
        rows.append(row)
        print(f"    {row}", flush=True)

        if name.startswith("gauss"):
            plot(
                truth=truth,
                lamb=lamb_est,
                hots=hots_est,
                hots_flat=unbias(hots_est, resp),
                unif=lamb_unif,
                args=args,
                run=run,
                scale_km=float(name.removeprefix("gauss")),
                name=f"hotspot_{name}km",
            )

    return rows


def main() -> None:
    """Quantify the PSF degradation caused by a hotspot BRDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wl", type=float, default=665.0)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--aot", type=float, default=0.4)
    parser.add_argument("--rh", type=float, default=50.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--href", type=float, default=2.0)
    parser.add_argument("--sza", type=float, default=40.0)
    parser.add_argument("--vza", type=float, default=0.0)
    parser.add_argument("--saa", type=float, default=0.0)
    parser.add_argument("--vaa", type=float, default=0.0)
    parser.add_argument("--species", type=str, default="sulphate")
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=RPV_RED["r0"])
    parser.add_argument("--k", type=float, default=RPV_RED["k"])
    parser.add_argument("--bt", type=float, default=RPV_RED["bt"])
    parser.add_argument("--rc", type=float, default=RPV_RED["rc"])
    parser.add_argument(
        "--normalise",
        choices=["view", "none"],
        default="view",
        help=(
            "'view' rescales r0 so the RPV reflectance in the sun-sensor "
            "geometry equals --rho-max, isolating the angular effect. "
            "'none' keeps --r0 as given."
        ),
    )
    parser.add_argument(
        "--uniform-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Reflectance levels probed on a uniform surface to measure "
        "the scalar BRDF bias without any adjacency contribution.",
    )
    parser.add_argument("--uniform-n", type=int, default=399)
    parser.add_argument("--uniform-nr", type=int, default=60)
    parser.add_argument("--nr", type=int, default=500)
    run, args = parse_run(__doc__.splitlines()[0], parser)
    run = run.resolve(n=3999, res_km=0.05, cache_dir="/tmp/adjeff-hotspot")

    # The hotspot study is a six-landscape joint training on top of two
    # forward pipelines: a smoke run keeps one landscape and three
    # reflectance levels, which still walks the whole chain.
    if run.smoke:
        args.scales = [1.0]
        args.uniform_levels = [0.0, 0.5, 1.0]
        args.uniform_n = 99
        args.uniform_nr = 20

    band = wl_to_band(args.wl)

    shape_view = float(
        rpv_shape(
            args.sza, args.vza, args.saa - args.vaa, args.k, args.bt, args.rc
        )
    )
    if args.normalise == "view":
        args.rpv_scale = 1.0 / shape_view
    else:
        args.rpv_scale = args.r0 / args.rho_max

    shape_hot = float(
        rpv_shape(args.sza, args.sza, 0.0, args.k, args.bt, args.rc)
    )
    print(
        f"RPV: k={args.k}, Theta={args.bt}, rho_c={args.rc}\n"
        f"  angular factor in the view geometry : {shape_view:.4f}\n"
        f"  angular factor at exact backscatter : {shape_hot:.4f}"
        f"  ({shape_hot / shape_view:.2f}x)\n"
        f"  landscape rescaling for the RPV run : {args.rpv_scale:.4f}",
        flush=True,
    )

    run.figs_dir.mkdir(parents=True, exist_ok=True)

    print(">>> scalar BRDF response on a uniform surface", flush=True)
    resp = uniform_response(args.uniform_levels, band, args, run)
    resp.to_csv(run.figs_dir / "hotspot_uniform_response.csv", index=False)
    print(resp.to_string(index=False), flush=True)

    lamb, hots = simulate(band, args, run)

    print(">>> King PSF, trained jointly on the 6 landscapes", flush=True)
    kernel, p_lamb = train(lamb, band, args, run)
    print(f"    Lambertian training : {p_lamb}", flush=True)
    pd.DataFrame([{"trained_on": "lambertian", **p_lamb}]).to_csv(
        run.figs_dir / "hotspot_king_params.csv", index=False
    )

    rows = evaluate(lamb, hots, kernel, band, args, run, resp)

    df = pd.DataFrame(rows)
    out = run.figs_dir / "hotspot_rmse.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
