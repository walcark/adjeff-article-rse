"""Define useful helper functions."""

from pathlib import Path
import os


def get_root_path() -> Path:
    path: Path = Path(__file__).resolve().parent.parent.parent
    assert path.glob("src"), f"[ERROR] Root path not resolved: {path}"
    return path


def get_auxdata_path() -> Path:
    return get_root_path() / "data/smartg_auxdata"


def get_auxdata_aer_path() -> Path:
    return get_auxdata_path() / "aerosols/OPAC/mixtures"


def get_auxdata_atmo_path() -> Path:
    return get_auxdata_path() / "atmospheres"
