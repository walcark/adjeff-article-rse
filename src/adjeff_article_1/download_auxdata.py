"""Build the Smart-G auxiliary dataset."""

from pathlib import Path
import shutil

from smartg.auxdata import download

from .utils import (
    get_root_path,
    get_auxdata_path,
    get_auxdata_aer_path,
    get_auxdata_atmo_path,
)


ROOT_PATH: Path = get_root_path()
AUXDATA_PATH: Path = get_auxdata_path()
AUXDATA_AER_PATH: Path = get_auxdata_aer_path()
AUXDATA_ATMO_PATH: Path = get_auxdata_atmo_path()


def download_smartg_auxdata_and_cams() -> None:
    """Download Smart-G auxiliary data.

    Then, copies the following items:
    - CAMS aerosol optical properties,
    - The simplified exponential AFGL profile.
    """
    # Download Smart-G auxdata in data_path
    if not AUXDATA_PATH.is_dir():
        print("[INFO] Did not find Smart-G auxiliary data. Downloading ...")
        download(AUXDATA_PATH, data_type="all")
    else:
        print("[INFO] Smart-G auxiliary data already found. Checking content ...")
        if not (AUXDATA_PATH / "aerosols").is_dir():
            print("[INFO] Missing aerosols folder. Downloading ...")
            download(AUXDATA_PATH, data_type="all")

    # Merge CAMS aerosol optical properties with Smart-G auxdata
    camsdata_path: Path = ROOT_PATH / "data/cams_aer_auxdata"
    if camsdata_path.is_dir():
        for file in camsdata_path.glob("*"):
            if not str(file).endswith("_sol.nc"):
                continue
            print(f"[INFO] Copying file {file} to {AUXDATA_AER_PATH} ...")
            shutil.copy(src=file, dst=AUXDATA_AER_PATH)
    else:
        print(f"[ERROR] Did not find CAMS optical properties at: {camsdata_path}")

    # Merge the exponential AFGL profile with Smart-G auxdata
    profile_file: Path = ROOT_PATH / "data/afgl_auxdata/afgl_exp_h8km.nc"
    if profile_file.is_file():
        print(f"[INFO] Copying file {profile_file} to {AUXDATA_ATMO_PATH} ...")
        shutil.copy(src=profile_file, dst=AUXDATA_ATMO_PATH)
    else:
        print(f"[ERROR] Did not find AFGL exponential profile file: {profile_file}")
