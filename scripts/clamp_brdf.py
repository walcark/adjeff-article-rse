"""Turn the Ross-Li BRDF clamp in Smart-G's CUDA kernel on and off.

The Ross-Li fit MODIS distributes is not constrained positive.  For a
strongly anisotropic surface, ``1 + k1p F1 + k2p F2`` goes negative at
grazing view, where ``F1`` diverges as ``-sec(theta)``.  Smart-G does not
clip it, so those photons carry a negative weight, and they are the ones
that leave the ground almost horizontally, which is to say the ones that
feed the far-field adjacency signal this study measures.

Measured over the hemisphere at a sun zenith of 40 degrees, that region
carries 0.011 % of the upward flux for Libya 4, 0.36 % for Konza and
0.79 % for Skukuza, with weights reaching -2500 against +1 for a normal
photon.  Whether that is enough to explain Skukuza's degraded retrieval,
or whether the degradation is a genuine cost of the Lambertian
assumption, is what clamping is meant to separate.

Why a script rather than an edit
--------------------------------
``device.cu`` lives inside the pixi environment, which pixi owns: an
install or a lock change can rewrite it without warning.  So the patch is
applied and removed by one command that refuses to act on a file it does
not recognise, and the original is kept outside the tree.

``pycuda`` reads and compiles ``device.cu`` on every run, keyed on the
source hash, so a change takes effect at the next run and needs no build
step.

Usage
-----
    python scripts/clamp_brdf.py --status
    python scripts/clamp_brdf.py --on
    python scripts/clamp_brdf.py --off
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

__all__ = ["clamp_state", "device_source"]

#: The line Smart-G computes the Ross-Li weight on, verbatim.  Matching
#: it exactly is what makes the patch refuse a file it does not know.
ANCHOR = """                wbrdf = 1. + spectrum[ilam].k1p_surface*F1_rtls(th0,th1,dph) +
                    spectrum[ilam].k2p_surface*F2_rtls(th0,th1,dph);"""

#: What replaces it.  One added line, and a marker that makes the state
#: readable without comparing checksums.
PATCH = (
    ANCHOR
    + """
                // adjeff-clamp: a reflectance is not negative.  The
                // Ross-Li fit is unconstrained and goes below zero at
                // grazing view for an anisotropic surface, which would
                // otherwise send photons of negative weight into the
                // far field.  See scripts/clamp_brdf.py.
                wbrdf = fmaxf(wbrdf, 0.f);"""
)

#: Marker searched for to report the state.
MARKER = "// adjeff-clamp:"

#: Where the untouched original is kept, outside anything pixi manages.
BACKUP = Path.home() / ".cache" / "adjeff-article" / "device.cu.orig"


def device_source() -> Path:
    """Return the path of Smart-G's CUDA source, as the running env sees it.

    Returns
    -------
    Path
        Location of ``device.cu``.

    Raises
    ------
    RuntimeError
        When Smart-G is not importable or the file is not where it says.
    """
    try:
        from smartg.smartg import src_device
    except ImportError as exc:
        raise RuntimeError(
            "smartg is not importable from this environment; run this "
            "through pixi, e.g. `pixi run -e local python "
            "scripts/clamp_brdf.py --status`"
        ) from exc
    path = Path(src_device)
    if not path.is_file():
        raise RuntimeError(f"smartg points at {path}, which does not exist")
    return path


def clamp_state(path: Path | None = None) -> bool:
    """Return whether the clamp is currently applied.

    Parameters
    ----------
    path : Path or None
        Source to inspect.  Defaults to the running environment's.

    Returns
    -------
    bool
        ``True`` when the kernel clamps the Ross-Li weight at zero.
    """
    return MARKER in (path or device_source()).read_text(errors="ignore")


def _digest(path: Path) -> str:
    """Return the SHA-256 of a file, short form."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def turn_on(path: Path) -> None:
    """Apply the clamp, keeping the original outside the pixi tree."""
    text = path.read_text(errors="ignore")
    if MARKER in text:
        print("already on; nothing to do")
        return
    if text.count(ANCHOR) != 1:
        raise RuntimeError(
            f"{path} does not hold the expected Ross-Li line exactly once "
            f"(found {text.count(ANCHOR)}). Smart-G has changed; re-read the "
            "kernel before patching it."
        )

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP)
    path.write_text(text.replace(ANCHOR, PATCH, 1))
    print(f"clamp ON   {path}")
    print(f"  original kept at {BACKUP} (sha {_digest(BACKUP)})")
    print(f"  patched  sha {_digest(path)}")
    print("  pycuda recompiles on the next run; no build step needed")


def turn_off(path: Path) -> None:
    """Remove the clamp, from the backup when there is one."""
    if not clamp_state(path):
        print("already off; nothing to do")
        return
    if BACKUP.is_file():
        shutil.copy2(BACKUP, path)
        print(f"clamp OFF  {path}")
        print(f"  restored from {BACKUP} (sha {_digest(path)})")
        return

    # No backup: undo the substitution textually.  Kept as a fallback so
    # that a lost cache cannot strand the environment in a patched state.
    text = path.read_text(errors="ignore")
    if text.count(PATCH) != 1:
        raise RuntimeError(
            f"no backup at {BACKUP} and {path} does not hold the patch "
            "verbatim, so it cannot be undone safely. Reinstall smartg: "
            "`pixi clean` then `pixi install -e local`."
        )
    path.write_text(text.replace(PATCH, ANCHOR, 1))
    print(f"clamp OFF  {path}  (undone textually, no backup was present)")


def main() -> int:
    """Read the flags and act, reporting the state either way."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--on", action="store_true", help="apply the clamp")
    group.add_argument("--off", action="store_true", help="remove it")
    group.add_argument(
        "--status", action="store_true", help="report without changing anything"
    )
    args = parser.parse_args()

    path = device_source()
    if args.on:
        turn_on(path)
    elif args.off:
        turn_off(path)

    state = "ON" if clamp_state(path) else "OFF"
    print(f"\nBRDF clamp is {state}")
    print(f"  {path}")
    print(f"  sha {_digest(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
