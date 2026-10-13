"""Build the lookup table of King PSF parameters deposited on SEDOO.

One netCDF per CAMS species, holding the ``sigma`` and ``gamma`` coefficients
of the King PSF model:

=========  ==  ==================================================
dimension   n  values
=========  ==  ==================================================
``band``   12  the Sentinel-2 bands except B10, the cirrus band
``aot``     9  0.0 to 0.8 by 0.1
``rh``      3  50, 90, 95 %
``href``    2  2, 4 km
``h``       2  0, 3 km
``sza``     1  40 degrees
``vza``     1  0 degrees
=========  ==  ==================================================

Usage
-----
    pixi run build-lut
    python generate/sedoo/build_psf_lut.py --species sulphate --dry-run
    python generate/sedoo/build_psf_lut.py --species dust --force
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import xarray as xr
from adjeff.api import make_full_config, make_model
from adjeff.core import KingPSF, S2Band, SensorBand, psf_params
from adjeff.modules.models import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages, fit

from adjeff_article_1.runconfig import RunConfig, parse_run
from adjeff_article_1.scenes import disk_scenes, gauss_scenes, mixed_scenes
from adjeff_article_1.utils import get_results_path

BANDS: tuple[SensorBand, ...] = tuple(b for b in S2Band if b is not S2Band.B10)
AOT = tuple(round(0.1 * i, 1) for i in range(9))
RH = (50.0, 90.0, 95.0)
HREF = (2.0, 4.0)
H = (0.0, 3.0)
SZA = (40.0,)
VZA = (0.0,)

#: Fixed across the cube, and recorded in the file's attributes.
SAA = 0.0
VAA = 0.0

#: Grid of the training landscapes, as the deposited tables declare it.
RES_KM = 0.12
N = 1999

#: Species, named as the CAMS tables of ``inputs/cams`` are.
SPECIES = (
    "ammonium",
    "blackcar",
    "dust",
    "nitrate",
    "organicm",
    "seasalt",
    "secondary_organic",
    "sulphate",
)

#: Landscapes to fit on.
TRAINING_SETS = {
    "disks": (disk_scenes, "Disk radius=[1.0, 5.0, 50.0] km"),
    "gauss": (gauss_scenes, "Gaussian sigma=[1.0, 5.0, 50.0] km"),
    "mixed": (mixed_scenes, "Gaussian and disk, [1.0, 5.0, 50.0] km"),
}

#: Parameters of the King model, in the order the files carry them.
PARAMETERS = ("sigma", "gamma")


def empty_cube(species: str, training: str) -> xr.Dataset:
    """Return the cube of the right shape, filled with NaN.

    Parameters
    ----------
    species : str
        CAMS species, one of `SPECIES`.
    training : str
        Key of `TRAINING_SETS`.

    Returns
    -------
    xr.Dataset
        ``sigma`` and ``gamma`` over the seven dimensions, all NaN.
    """
    coords = {
        "band": [float(b.wl_nm) for b in BANDS],
        "sza": list(SZA),
        "vza": list(VZA),
        "h": list(H),
        "href": list(HREF),
        "rh": list(RH),
        "aot": list(AOT),
    }
    shape = tuple(len(v) for v in coords.values())
    return xr.Dataset(
        {name: (tuple(coords), np.full(shape, np.nan)) for name in PARAMETERS},
        coords=coords,
        attrs={
            "psf_model": "king",
            "species": species,
            "sza_deg": SZA[0],
            "vza_deg": VZA[0],
            "saa_deg": SAA,
            "vaa_deg": VAA,
            "band_coordinate_unit": "nm",
            "training_scenes": TRAINING_SETS[training][1],
            "res_km": RES_KM,
        },
    )


def load_or_create(path: Path, species: str, training: str) -> xr.Dataset:
    """Return the cube to fill, read back from *path* when it exists.

    A cube whose axes do not match `empty_cube` is refused rather than
    silently extended: a restart has to resume the same run, not merge
    two different ones.

    Parameters
    ----------
    path : Path
        Where the cube is written.
    species : str
        CAMS species.
    training : str
        Key of `TRAINING_SETS`.

    Returns
    -------
    xr.Dataset
        The cube, with NaN wherever work remains.

    Raises
    ------
    ValueError
        If an existing file describes a different cube.
    """
    fresh = empty_cube(species, training)
    if not path.is_file():
        return fresh

    with xr.open_dataset(path) as stored:
        cube = stored.load()
    if dict(cube.sizes) != dict(fresh.sizes):
        raise ValueError(
            f"{path} has shape {dict(cube.sizes)}, this run builds "
            f"{dict(fresh.sizes)}; move it aside or pass --force"
        )
    for axis, values in fresh.coords.items():
        if not np.allclose(cube[axis].values, values.values):
            raise ValueError(
                f"{path} is tabulated on a different {axis}; move it aside "
                "or pass --force"
            )
    if cube.attrs.get("training_scenes") != fresh.attrs["training_scenes"]:
        raise ValueError(
            f"{path} was fitted on {cube.attrs.get('training_scenes')!r}, "
            f"this run fits on {fresh.attrs['training_scenes']!r}"
        )
    return cube


def save(cube: xr.Dataset, path: Path) -> None:
    """Write *cube* to *path*, atomically.

    Through a temporary file and a rename: the script is meant to be
    interrupted, and a half-written netCDF would lose every slab it
    already held.

    Parameters
    ----------
    cube : xr.Dataset
        The cube to write.
    path : Path
        Destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".nc.tmp")
    cube.to_netcdf(temporary)
    temporary.replace(path)


def slab_done(cube: xr.Dataset, where: dict[str, float]) -> bool:
    """Return whether the slab at *where* is already filled.

    Parameters
    ----------
    cube : xr.Dataset
        The cube being filled.
    where : dict
        One value per dimension except ``aot``.

    Returns
    -------
    bool
        True when no parameter is NaN anywhere in the slab.
    """
    return not any(bool(np.isnan(cube[name].sel(**where)).any()) for name in PARAMETERS)


def fit_slab(
    band: SensorBand,
    href: float,
    species: str,
    run: RunConfig,
    training: str,
) -> dict[str, np.ndarray]:
    """Optimise one King PSF per ``(h, rh, aot)``, at a fixed band and href.

    `RhoToaSymSampler` declares ``loop(sza, vza) vec(aot, rh, h, href)``, so
    every atmospheric state of the slab travels in one Smart-G call: 54
    states here against the 9 of an AOT-only slab, which is 24 launches for
    the whole cube rather than 144.

    ``href`` stays out of the sweep on purpose.  The sampler returns
    ``rho_toa(aot, rh, href, h, y, x)`` materialised in memory, about 32 MB
    per state at ``n = 1999``, and the pipeline holds ``rho_unif`` beside it
    for each of the three training scenes.  54 states peak near 10 GB; the
    108 of a full sweep would double that.

    Parameters
    ----------
    band : SensorBand
        Band to fit.
    href : float
        Aerosol scale height, in km.
    species : str
        CAMS species.
    run : RunConfig
        Grid size, photon budget and cache location.
    training : str
        Key of `TRAINING_SETS`.

    Returns
    -------
    dict
        ``{parameter: values over (h, rh, aot)}``, in the order the cube
        holds those dimensions.
    """
    config = make_full_config(
        bands=[band],
        aot=list(AOT),
        rh=list(RH),
        h=list(H),
        href=href,
        sza=SZA[0],
        vza=VZA[0],
        saa=SAA,
        vaa=VAA,
        species={species: 1.0},
    )
    builder = TRAINING_SETS[training][0]
    scenes = builder(band, run, cfg=config)

    model = make_model(
        Unif2Surface,
        KingPSF,
        [band],
        res_km=run.res_km,
        n=run.n,
        init_parameters={"sigma": 0.1, "gamma": 2.0},
        device=run.device,
    )
    tree = fit(
        model,
        TrainingImages(images=scenes),
        loss=Loss(Metric.RMSE_RAD),
        device=run.device,
    )
    fitted = psf_params(tree, band)
    return {name: _over_sweep(fitted[name], name) for name in PARAMETERS}


def _over_sweep(values: xr.DataArray, name: str) -> np.ndarray:
    """Return *values* over ``(h, rh, aot)``, as the cube orders them.

    Parameters
    ----------
    values : xr.DataArray
        A fitted parameter, carrying whatever dimensions the optimisation
        produced.
    name : str
        Parameter name, for the error message.

    Returns
    -------
    np.ndarray
        Shape ``(len(H), len(RH), len(AOT))``.

    Raises
    ------
    ValueError
        If the fit did not return one value per swept state, which would
        mean the configurations collapsed into fewer combos.
    """
    wanted = {"h": list(H), "rh": list(RH), "aot": list(AOT)}
    missing = [dim for dim in wanted if dim not in values.dims]
    if missing:
        raise ValueError(
            f"{name} came back without {missing}; dims were {values.dims}"
        )
    ordered = values.sel(wanted).transpose("h", "rh", "aot")
    return np.asarray(ordered.values, dtype=float)


def build(
    species: str,
    run: RunConfig,
    training: str,
    out: Path,
    dry_run: bool = False,
) -> Path:
    """Fill one species' cube, slab by slab, resuming where it stopped.

    Parameters
    ----------
    species : str
        CAMS species, one of `SPECIES`.
    run : RunConfig
        Grid size, photon budget and cache location.
    training : str
        Key of `TRAINING_SETS`.
    out : Path
        Directory to write into.
    dry_run : bool
        Report what remains and fit nothing.

    Returns
    -------
    Path
        The file written.
    """
    path = out / f"king_psf_{species}.nc"
    cube = load_or_create(path, species, training)

    slabs = [
        {
            "band": float(band.wl_nm),
            "href": href,
            "sza": SZA[0],
            "vza": VZA[0],
        }
        for (band, href) in itertools.product(BANDS, HREF)
    ]
    todo = [w for w in slabs if not slab_done(cube, w)]
    print(
        f"{species}: {len(todo)} of {len(slabs)} slabs left, "
        f"{len(todo) * len(H) * len(RH) * len(AOT)} fits",
        flush=True,
    )
    if dry_run:
        return path

    by_wavelength = {float(b.wl_nm): b for b in BANDS}
    for index, where in enumerate(todo, 1):
        band = by_wavelength[where["band"]]
        print(
            f"  [{index}/{len(todo)}] {band.id} href={where['href']:g}",
            flush=True,
        )
        fitted = fit_slab(band, where["href"], species, run, training)
        for name, values in fitted.items():
            cube[name].loc[where] = values
        # Written after every slab: this is the whole point of the
        # checkpoint, and a netCDF of 1296 floats costs nothing to rewrite.
        save(cube, path)

    print(f"{species}: {path}", flush=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for this script's own options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--species",
        nargs="+",
        choices=sorted(SPECIES),
        default=sorted(SPECIES),
        help="CAMS species to build (default: all eight)",
    )
    parser.add_argument(
        "--training",
        choices=sorted(TRAINING_SETS),
        default="disks",
        help=(
            "landscapes to fit on. The deposited tables used `gauss`; "
            "`disks` is what the article's figures use (default)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=get_results_path() / "sedoo",
        help="directory to write the cubes into",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many fits remain and do none of them",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="start from an empty cube, discarding what was already fitted",
    )
    return parser


def main() -> None:
    """Build every requested species' cube."""
    run, args = parse_run(__doc__.splitlines()[0], build_parser())
    run = run.resolve(n=N, res_km=RES_KM)

    for species in args.species:
        path = args.out / f"king_psf_{species}.nc"
        if args.force and path.is_file():
            print(f"{species}: discarding {path}", flush=True)
            path.unlink()
        build(species, run, args.training, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
