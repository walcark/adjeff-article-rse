"""Rebuild the eight CAMS aerosol tables from MOPSMAP.

``inputs/cams`` holds the optical properties of the eight CAMS
aerosol species, in the look-up table format Smart-G reads. They are
versioned because reproducing the figures must not require MOPSMAP, whose
optical dataset is a separate multi-gigabyte download. This script is how
they were produced, and what to run to extend the set to another species
or another grid.

The grid is the one the article uses: twelve relative humidities, nineteen
wavelengths from 250 to 2250 nm, two thousand scattering angles. The
wavelengths are the ones MOPSMAP's own dataset is tabulated on, so Smart-G
interpolates rather than extrapolates at every Sentinel-2 band.

Requirements
------------
`pymopsmap` >= 0.5.2, the MOPSMAP binary, and its optical dataset:

    export PYMOPSMAP_DATASET_SOURCE=/path/to/mopsmap/optical_dataset

Usage
-----
    python generate/sedoo/build_cams_auxdata.py
    python generate/sedoo/build_cams_auxdata.py --out /tmp/check --species sulphate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adjeff_article_1.utils import get_auxdata_aer_path

DEFAULT_OUT = get_auxdata_aer_path()

#: Relative humidities, in percent. Smart-G interpolates ``ext`` and
#: ``ssa`` linearly between them, so the grid is denser where the
#: hygroscopic species grow fastest.
HUMIDITIES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95]

#: Wavelengths, in nanometres.
WAVELENGTHS = [
    250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
    900, 1000, 1250, 1500, 1750, 2000, 2250,
]

#: Scattering angles. Two thousand is what resolves the forward peak of
#: the coarse modes, dust and sea salt, without aliasing it.
N_ANGLES = 2000

#: File name to CAMS species. Smart-G identifies a mixture by the stem of
#: its file, which is the name ``adjeff`` passes to ``AerOPAC``, so these
#: keys are part of the article's configuration and must not drift.
SPECIES: dict[str, str] = {
    "ammonium": "AMMONIUM",
    "blackcar": "BLACK_CARBON",
    "dust": "DUST",
    "nitrate": "NITRATE",
    "organicm": "ORGANIC_MATTER",
    "seasalt": "SEA_SALT",
    "secondary_organic": "SECONDARY_ORGANIC",
    "sulphate": "SULPHATE",
}


def build(names: list[str], out: Path) -> None:
    """Write one Smart-G look-up table per named species.

    Parameters
    ----------
    names : list of str
        File stems to build, keys of `SPECIES`.
    out : Path
        Directory to write ``<name>_sol.nc`` into.

    Notes
    -----
    ``OutputType.SCATTERING_MATRIX`` is what makes the table polarised.
    A scalar phase function would leave F21, F33, F34 and F44 at zero,
    and Smart-G would transport no polarisation at all.
    """
    import pymopsmap as pm

    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        result = pm.load(getattr(pm.CAMS, SPECIES[name])).compute(
            wl=[w / 1000.0 for w in WAVELENGTHS],
            rh=HUMIDITIES,
            outputs={pm.OutputType.SCATTERING_MATRIX},
            n_angles=N_ANGLES,
        )
        result.mopsmap.to_smartg(out / f"{name}_sol.nc", name=name)
        print(f"[INFO] wrote {name}_sol.nc")


def main() -> None:
    """Parse the command line and build the requested tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--species", nargs="+", choices=sorted(SPECIES), default=sorted(SPECIES)
    )
    args = parser.parse_args()
    build(args.species, args.out)


if __name__ == "__main__":
    main()
