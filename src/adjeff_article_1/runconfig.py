"""Run parameters shared by every figure script, plus the smoke mode.

The article figures are expensive: a 3999 pixel grid and 1e5 photons per
Smart-G call take hours on a GPU.  :class:`RunConfig` carries those
numbers so that ``--smoke`` can shrink them to something that runs in
minutes.  Smoke output is scientifically meaningless on purpose: its only
job is to prove that the adjeff API a script calls still exists and still
returns the expected shapes, which is what must be re-checked at every
adjeff version bump.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import adjeff

from . import _logging as article_logging

__all__ = ["ARTICLE_ATMOSPHERE", "RunConfig", "add_run_arguments", "parse_run"]

# Reference atmosphere and geometry of the manuscript.  Every figure that
# does not sweep these values uses exactly this state.
ARTICLE_ATMOSPHERE: dict[str, float | dict[str, float]] = {
    "aot": 0.4,
    "h": 0.0,
    "rh": 50.0,
    "href": 2.0,
    "sza": 45.0,
    "vza": 8.0,
    "saa": 0.0,
    "vaa": 0.0,
    "species": {"sulphate": 1.0},
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    """Grid size, photon budget and output location for one figure run.

    Parameters
    ----------
    n : int
        Grid side in pixels.  Must be odd.
    n_ph : int
        Photons per Smart-G call.
    res_km : float
        Pixel size in km.
    n_samples : int or None
        Side of the parameter landscape grids (figures 2 and 3).  ``None``
        means "keep the value the script asks for"; smoke runs force it
        down so that a landscape stays a handful of kernels.
    device : str
        Torch device for the convolution and optimisation steps.
    figs_dir : Path
        Where the PNG files are written.
    cache_dir : str
        Root of the adjeff on-disk cache.
    smoke : bool
        ``True`` when the run is a shape check rather than a real figure.
    explicit : frozenset[str]
        Fields the caller set on the command line.  :meth:`resolve` never
        overwrites those.
    """

    n: int = 3999
    n_ph: int = int(1e5)
    res_km: float = 0.05
    n_samples: int | None = None
    device: str = "cuda"
    figs_dir: Path = _REPO_ROOT / "output"
    cache_dir: str = "/tmp/adjeff-figures"
    smoke: bool = False
    explicit: frozenset[str] = frozenset()

    def resolve(self, **native: object) -> "RunConfig":
        """Fill in a script's native run parameters.

        Figures do not all work on the same grid: the sweeps of figures 7
        to 17 run at 120 m over 1999 pixels where figures 2 to 5 run at
        50 m over 3999.  A script declares its own values here and they
        apply unless something more specific already decided: an explicit
        command-line option always wins, and so does ``--smoke``, whose
        whole purpose is to shrink the run.

        Parameters
        ----------
        **native
            Field values native to the calling script, e.g.
            ``run.resolve(n=1999, res_km=0.12)``.

        Returns
        -------
        RunConfig
            A copy carrying the native values that nothing overrode.
        """
        if self.smoke:
            return self
        updates = {
            k: v for k, v in native.items() if k not in self.explicit
        }
        return replace(self, **updates) if updates else self

    @property
    def extent_km(self) -> float:
        """Side of the simulated field of view, in km."""
        return self.n * self.res_km

    @classmethod
    def smoke_run(cls) -> "RunConfig":
        """Return the smallest configuration that still exercises the API.

        The pixel size is coarsened rather than the field of view shrunk:
        the article's landscapes go up to a 50 km radius, and a grid that
        cannot hold them would diverge for reasons that have nothing to do
        with the API being checked.
        """
        return cls(
            n=401,
            res_km=0.5,
            n_ph=int(1e3),
            n_samples=2,
            device="cpu",
            figs_dir=_REPO_ROOT / "output" / "smoke",
            cache_dir="/tmp/adjeff-figures-smoke",
            smoke=True,
        )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the run options shared by every figure script."""
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny grid and photon count: check the API, not the physics",
    )
    parser.add_argument("--n", type=int, help="grid side in pixels (odd)")
    parser.add_argument("--n-ph", type=int, help="photons per Smart-G call")
    parser.add_argument("--res-km", type=float, help="pixel size in km")
    parser.add_argument(
        "--n-samples", type=int, help="side of the parameter landscape grid"
    )
    parser.add_argument("--device", help="torch device (cuda or cpu)")
    parser.add_argument("--figs-dir", type=Path, help="output directory")
    parser.add_argument("--cache-dir", help="adjeff cache root")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="how much adjeff and xsweep report (default: info)",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="one JSON object per line, for a run whose output is collected",
    )


def parse_run(
    description: str,
    parser: argparse.ArgumentParser | None = None,
) -> tuple[RunConfig, argparse.Namespace]:
    """Parse the command line and return ``(RunConfig, namespace)``.

    Pass an existing *parser* when the script has arguments of its own;
    the shared options are added to it.  Explicit options always win over
    ``--smoke``, so ``--smoke --n 501`` runs a slightly larger smoke test.

    Parameters
    ----------
    description : str
        Used as the parser description when *parser* is ``None``.
    parser : argparse.ArgumentParser or None
        Script-specific parser to extend.

    Returns
    -------
    tuple[RunConfig, argparse.Namespace]
        The resolved run configuration and the full parsed namespace, so
        that scripts can read their own arguments from the latter.

    Notes
    -----
    Logging is configured here, from ``--log-level`` and ``--log-json``.
    A figure script is an application, so it is entitled to; adjeff is a
    library, so it no longer does it on the script's behalf.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description=description)
    add_run_arguments(parser)
    args = parser.parse_args()

    # adjeff 0.13.0 prints nothing until asked: it stopped configuring
    # logging for the process it is imported into, which is a library's
    # business to leave alone.  Turning it on is the application's, and
    # here that is the figure script.  Every one of them reaches this
    # function, so this is the one place it needs saying.
    adjeff.setup_logging(level=args.log_level, json=args.log_json)
    # This repository has a namespace of its own, which `setup_logging`
    # knows nothing about; it follows the same level.
    logging.getLogger(article_logging.ROOT).setLevel(args.log_level.upper())

    base = RunConfig.smoke_run() if args.smoke else RunConfig()
    overrides = {
        field: value
        for field, value in (
            ("n", args.n),
            ("n_ph", args.n_ph),
            ("res_km", args.res_km),
            ("n_samples", args.n_samples),
            ("device", args.device),
            ("figs_dir", args.figs_dir),
            ("cache_dir", args.cache_dir),
        )
        if value is not None
    }
    return replace(base, **overrides, explicit=frozenset(overrides)), args
