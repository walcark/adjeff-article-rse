"""Build the Smart-G auxiliary dataset."""

import shutil
from pathlib import Path

from smartg.auxdata import download

from .utils import (
    get_auxdata_aer_path,
    get_auxdata_atmo_path,
    get_auxdata_path,
    get_root_path,
)

ROOT_PATH: Path = get_root_path()
AUXDATA_PATH: Path = get_auxdata_path()
AUXDATA_AER_PATH: Path = get_auxdata_aer_path()
AUXDATA_ATMO_PATH: Path = get_auxdata_atmo_path()


#: What a figure needs from the Smart-G dataset.  `acs` and `atmospheres`
#: are versioned here; the aerosol mixtures are the CAMS tables of
#: `inputs/cams`, copied in below.
REQUIRED = ("acs", "atmospheres", "aerosols/OPAC/mixtures")


def copy_cams_and_afgl() -> None:
    """Copy the versioned CAMS tables and AFGL profile into the dataset."""
    AUXDATA_AER_PATH.mkdir(parents=True, exist_ok=True)
    AUXDATA_ATMO_PATH.mkdir(parents=True, exist_ok=True)

    camsdata_path: Path = ROOT_PATH / "inputs/cams"
    if camsdata_path.is_dir():
        for file in sorted(camsdata_path.glob("*_sol.nc")):
            print(f"[INFO] Copying file {file} to {AUXDATA_AER_PATH} ...")
            shutil.copy(src=file, dst=AUXDATA_AER_PATH)
    else:
        print(f"[ERROR] Did not find CAMS optical properties at: {camsdata_path}")

    profile_file: Path = ROOT_PATH / "inputs/afgl/afgl_exp_h8km.nc"
    if profile_file.is_file():
        print(f"[INFO] Copying file {profile_file} to {AUXDATA_ATMO_PATH} ...")
        shutil.copy(src=profile_file, dst=AUXDATA_ATMO_PATH)
    else:
        print(f"[ERROR] Did not find AFGL exponential profile file: {profile_file}")


def missing() -> list[str]:
    """Return the required directories the dataset does not hold."""
    return [d for d in REQUIRED if not (AUXDATA_PATH / d).is_dir()]


def download_smartg_auxdata_and_cams() -> None:
    """Assemble the dataset, downloading only what the repository lacks."""
    copy_cams_and_afgl()
    absent = missing()
    if not absent:
        print("[INFO] Smart-G auxiliary data complete, nothing to download.")
        return

    print(f"[INFO] Missing {', '.join(absent)}. Downloading ...")
    download(AUXDATA_PATH, data_type="all")
    copy_cams_and_afgl()
    still = missing()
    if still:
        print(f"[ERROR] Still missing after the download: {', '.join(still)}")


if __name__ == "__main__":
    download_smartg_auxdata_and_cams()
