"""Build the lookup table of King PSF parameters deposited on SEDOO.

One netCDF per CAMS species, holding the ``sigma`` and ``gamma`` coefficients
of the King PSF model:

=========  ==  ==================================================
dimension   n  values
=========  ==  ==================================================
``band``   12  the sensor's bands, Sentinel-2 except B10 by default
``aot``     9  0.0 to 0.8 by 0.1
``rh``      4  0, 50, 90, 95 %
``href``    3  1, 2, 4 km
``h``       2  0, 3 km
``sza``     1  40 degrees
``vza``     1  0 degrees
=========  ==  ==================================================

Each ``(band, href)`` slab is fitted in one go, its 72 states travelling in
a single Smart-G call, and written to its own file under ``slabs/``.  A run
that stops is resumed by running it again: a slab whose file exists is
skipped.  The cube is assembled from those files once they are all there.

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

from adjeff_article_1.runconfig import parse_run
from adjeff_article_1.scenes import disk_scenes, gauss_scenes, mixed_scenes
from adjeff_article_1.utils import get_results_path

SENSORS: dict[str, tuple[SensorBand, ...]] = {
    "sentinel2": tuple(b for b in S2Band if b is not S2Band.B10),
}

AOT = tuple(round(0.1 * i, 1) for i in range(9))
RH = (0.0, 50.0, 90.0, 95.0)
HREF = (1.0, 2.0, 4.0)
H = (0.0, 3.0)
SZA = (40.0,)
VZA = (0.0,)

SAA = 0.0
VAA = 0.0

RES_KM = 0.060
N = 3999

BATCH_SIZE = 8

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

TRAINING_SETS = {
    "disks": (disk_scenes, "Disk radius=[1.0, 5.0, 50.0] km"),
    "gauss": (gauss_scenes, "Gaussian sigma=[1.0, 5.0, 50.0] km"),
    "mixed": (mixed_scenes, "Gaussian and disk, [1.0, 5.0, 50.0] km"),
}

PARAMETERS = ("sigma", "gamma")


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for this script's own options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sensor",
        choices=sorted(SENSORS),
        default="sentinel2",
        help="sensor whose bands the cube is tabulated over",
    )
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
        "--slabs",
        type=int,
        default=None,
        help="stop after that many slabs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many slabs remain and fit none of them",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard the slabs already fitted",
    )
    return parser


def main() -> None:
    """Fit every missing slab, then assemble one cube per species."""
    run, args = parse_run(__doc__.splitlines()[0], build_parser())
    run = run.resolve(n=N, res_km=RES_KM)
    builder, landscapes = TRAINING_SETS[args.training]
    bands = SENSORS[args.sensor]
    slab_dir = args.out / "slabs"
    slab_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"h": list(H), "rh": list(RH), "aot": list(AOT)}

    for species in args.species:
        stem = f"king_psf_{args.sensor}_{species}"
        slabs = {
            slab_dir / f"{stem}_{band.id}_href{href:g}.nc": (band, href)
            for band, href in itertools.product(bands, HREF)
        }
        if args.force:
            for path in slabs:
                path.unlink(missing_ok=True)

        todo = [p for p in slabs if not p.is_file()]
        if args.slabs is not None:
            todo = todo[: args.slabs]
        print(
            f"{species}: {len(todo)} of {len(slabs)} slabs left, "
            f"{len(todo) * len(H) * len(RH) * len(AOT)} fits",
            flush=True,
        )
        if args.dry_run:
            continue

        for index, path in enumerate(todo, 1):
            band, href = slabs[path]
            print(f"  [{index}/{len(todo)}] {band.id} href={href:g}", flush=True)
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
            scenes = builder(band, run, cfg=config, batch_size=BATCH_SIZE)
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
            xr.Dataset(
                {
                    name: fitted[name]
                    .sel(wanted)
                    .squeeze(drop=True)
                    .transpose("h", "rh", "aot")
                    .expand_dims(
                        {
                            "band": [float(band.wl_nm)],
                            "sza": list(SZA),
                            "vza": list(VZA),
                            "href": [href],
                        }
                    )
                    for name in PARAMETERS
                }
            ).to_netcdf(path)

        missing = [p for p in slabs if not p.is_file()]
        if missing:
            print(
                f"{species}: {len(missing)} slabs left, cube not assembled",
                flush=True,
            )
            continue

        cube = xr.combine_by_coords(
            [xr.load_dataset(p) for p in slabs], combine_attrs="drop"
        ).transpose("band", "sza", "vza", "h", "href", "rh", "aot")
        for name in PARAMETERS:
            assert not np.isnan(cube[name]).any(), f"{species}: {name} holds NaN"
        cube.attrs = {
            "psf_model": "king",
            "sensor": args.sensor,
            "species": species,
            "sza_deg": SZA[0],
            "vza_deg": VZA[0],
            "saa_deg": SAA,
            "vaa_deg": VAA,
            "band_coordinate_unit": "nm",
            "training_scenes": landscapes,
            "res_km": RES_KM,
        }
        written = args.out / f"{stem}.nc"
        cube.to_netcdf(written)
        print(f"{species}: {written}", flush=True)


if __name__ == "__main__":
    main()
