"""Companion package for Article 1 on adjacency effects in Sentinel-2.

The scripts in ``generate/`` hold nothing but what they produce.
Everything they share lives in the submodules below:

- :mod:`.style` matplotlib theme, tick and spine weights, panel titles.
- :mod:`.runconfig` grid size and photon budget, plus the ``--smoke``
  mode used to check that a new adjeff release still runs the scripts.
- :mod:`.scenes` reference atmosphere, training landscapes, PSF grids.
- :mod:`.correction` the article's own inversion experiment, where the
  PSF is chosen independently of the atmosphere the scene was simulated
  at.
- :mod:`.psf_comparison` the figure shared by two of the scripts.

- :mod:`.utils` the three roots: ``inputs/`` for what the repository
  needs, ``results/`` for what it produces, and a cache outside the tree.

Only the environment bootstrap is re-exported here.  ``tools/smartg_env.py``
imports this package to locate the Smart-G auxiliary data, which happens
*before* ``SMARTG_DIR_AUXDATA`` is set, so importing anything that reaches
``adjeff`` (and therefore ``smartg``) from this module would make the
bootstrap fail.  Scripts import the submodules they need explicitly.
"""

from .utils import (
    get_auxdata_path,
    get_cache_path,
    get_inputs_path,
    get_results_path,
)

__all__ = [
    "get_auxdata_path",
    "get_cache_path",
    "get_inputs_path",
    "get_results_path",
]
