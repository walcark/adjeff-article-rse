"""Run every smoke-capable figure script and report which ones survive.

This is the check to run after bumping the pinned adjeff release: it does
not verify the physics, only that each script still finds the adjeff API
it calls and still gets usable shapes back.  A figure script joins the
run automatically as soon as its ``--help`` advertises ``--smoke``.

Usage
-----
    pixi run python scripts/smoke.py
    pixi run python scripts/smoke.py figure4 figure5
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
TIMEOUT_S = 1800


def supports_smoke(script: Path) -> bool:
    """Return True when *script* advertises a ``--smoke`` option."""
    try:
        help_text = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except subprocess.TimeoutExpired:
        return False
    return "--smoke" in help_text


def run(script: Path) -> tuple[bool, float, str]:
    """Run *script* in smoke mode; return ``(ok, seconds, last_error)``."""
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script), "--smoke"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    elapsed = time.monotonic() - start
    tail = proc.stderr.strip().splitlines()[-1:] if proc.returncode else []
    return proc.returncode == 0, elapsed, tail[0] if tail else ""


def main() -> int:
    """Run the selected scripts and return a process exit code."""
    wanted = set(sys.argv[1:])
    scripts = sorted(
        s
        for s in FIGURES_DIR.glob("*.py")
        if not s.name.startswith("_")
        and (not wanted or s.stem in wanted)
    )

    selected = [s for s in scripts if supports_smoke(s)]
    skipped = [s for s in scripts if s not in selected]

    failures = 0
    for script in selected:
        ok, elapsed, error = run(script)
        status = "ok  " if ok else "FAIL"
        print(f"{status} {script.stem:26s} {elapsed:6.1f}s  {error}")
        failures += not ok

    for script in skipped:
        print(f"skip {script.stem:26s}         no --smoke option yet")

    print(
        f"\n{len(selected) - failures}/{len(selected)} smoke runs passed, "
        f"{len(skipped)} not migrated yet"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
