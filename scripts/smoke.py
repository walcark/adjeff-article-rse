"""Run every smoke-capable figure script and report which ones survive.

This is the check to run after bumping the pinned adjeff release: it does
not verify the physics, only that each script still finds the adjeff API
it calls and still gets usable shapes back.  A figure script joins the
run automatically as soon as its ``--help`` advertises ``--smoke``.

Prefer ``--cold`` for a version check.  A warm cache short-circuits the
Smart-G calls, so a run that reuses it only proves that zarr still reads
back, not that the simulation chain still works.

Usage
-----
    pixi run python scripts/smoke.py --cold
    pixi run python scripts/smoke.py figure4 figure5
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from adjeff_article_1.runconfig import RunConfig
from adjeff_article_1.utils import get_auxdata_path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
TIMEOUT_S = 1800


def child_env() -> dict[str, str]:
    """Return the environment the figure scripts are run under.

    Importing adjeff pulls in Smart-G, which raises at import time when
    ``SMARTG_DIR_AUXDATA`` is unset.  Filling it in here means the runner
    behaves the same whether or not the caller sourced
    ``scripts/export_smartg_auxdata.py`` first.
    """
    env = dict(os.environ)
    env.setdefault("SMARTG_DIR_AUXDATA", str(get_auxdata_path()))
    return env


def purge_cache() -> None:
    """Delete the smoke cache so that every Smart-G call runs for real."""
    cache_dir = Path(RunConfig.smoke_run().cache_dir)
    if not cache_dir.is_dir():
        print(f"cold: {cache_dir} is already absent")
        return
    size_mb = sum(
        f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
    ) / 1e6
    print(f"cold: removing {cache_dir} ({size_mb:.1f} MB)")
    shutil.rmtree(cache_dir)


def supports_smoke(script: Path) -> tuple[bool, str]:
    """Return ``(supported, reason)`` for *script*.

    A script that cannot even print its own help is reported as broken
    rather than as not migrated: the whole point of this runner is to
    catch a script that a new adjeff release stopped working with, and
    an import error at ``--help`` time is exactly that.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            env=child_env(),
        )
    except subprocess.TimeoutExpired:
        return False, "--help timed out"
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:]
        return False, f"--help failed: {tail[0] if tail else 'no output'}"
    if "--smoke" not in proc.stdout:
        return False, "no --smoke option yet"
    return True, ""


def run(script: Path) -> tuple[bool, float, str]:
    """Run *script* in smoke mode; return ``(ok, seconds, last_error)``."""
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script), "--smoke"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=child_env(),
    )
    elapsed = time.monotonic() - start
    tail = proc.stderr.strip().splitlines()[-1:] if proc.returncode else []
    return proc.returncode == 0, elapsed, tail[0] if tail else ""


def main() -> int:
    """Run the selected scripts and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cold",
        action="store_true",
        help="purge the smoke cache first, so the runs are not memoised",
    )
    parser.add_argument(
        "figures",
        nargs="*",
        help="figure stems to run (default: every smoke-capable script)",
    )
    args = parser.parse_args()

    if args.cold:
        purge_cache()

    wanted = set(args.figures)
    scripts = sorted(
        s
        for s in FIGURES_DIR.glob("*.py")
        if not s.name.startswith("_")
        and (not wanted or s.stem in wanted)
    )

    probed = [(s, *supports_smoke(s)) for s in scripts]
    selected = [s for s, ok, _ in probed if ok]
    skipped = [(s, why) for s, ok, why in probed if not ok]
    broken = [(s, why) for s, why in skipped if why != "no --smoke option yet"]

    failures = 0
    for script in selected:
        ok, elapsed, error = run(script)
        status = "ok  " if ok else "FAIL"
        print(f"{status} {script.stem:26s} {elapsed:6.1f}s  {error}")
        failures += not ok

    for script, why in skipped:
        status = "skip" if why == "no --smoke option yet" else "BROKEN"
        print(f"{status:4s} {script.stem:26s}         {why}")

    print(
        f"\n{len(selected) - failures}/{len(selected)} smoke runs passed, "
        f"{len(skipped) - len(broken)} not migrated yet"
        + (f", {len(broken)} BROKEN" if broken else "")
    )
    return 1 if failures or broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
