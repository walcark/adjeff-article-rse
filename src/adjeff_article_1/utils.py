"""Where the repository keeps its inputs and its results.

Three roots, and the name of each says what is in it: `inputs/` holds
what the repository needs and does not produce, `results/` what it
produces, and the cache lives outside the tree entirely.
"""

from pathlib import Path


def get_root_path() -> Path:
    """Return the repository root."""
    path: Path = Path(__file__).resolve().parent.parent.parent
    assert path.glob("src"), f"[ERROR] Root path not resolved: {path}"
    return path


def get_inputs_path() -> Path:
    """Return the directory holding what the repository does not produce."""
    return get_root_path() / "inputs"


def get_results_path() -> Path:
    """Return the directory every generated file is written under."""
    return get_root_path() / "results"


def get_cache_path() -> Path:
    """Return the cache root, deliberately outside the repository.

    A Smart-G cache runs to gigabytes and is reproducible, so it has no
    business under version control or in a directory anyone might
    archive.  Not ``/tmp`` either: a lookup table takes long enough that
    it has to survive a reboot.
    """
    return Path.home() / ".cache" / "adjeff-article"


def get_auxdata_path() -> Path:
    """Return the Smart-G auxiliary data directory."""
    return get_inputs_path() / "smartg_auxdata"


def get_auxdata_aer_path() -> Path:
    """Return where Smart-G reads OPAC aerosol mixtures from."""
    return get_auxdata_path() / "aerosols/OPAC/mixtures"


def get_auxdata_atmo_path() -> Path:
    """Return where Smart-G reads AFGL profiles from."""
    return get_auxdata_path() / "atmospheres"
