"""Companion package for Article 1 on adjacency effects in Sentinel-2.

The figure scripts in ``figures/`` hold nothing but the figure they draw.
Everything they share lives in the submodules below:

- :mod:`.style` matplotlib theme, tick and spine weights, panel titles.
- :mod:`.runconfig` grid size and photon budget, plus the ``--smoke``
  mode used to check that a new adjeff release still runs the scripts.
- :mod:`.scenes` reference atmosphere, training landscapes, PSF grids.
- :mod:`.correction` the article's own inversion experiment, where the
  PSF is chosen independently of the atmosphere the scene was simulated
  at.
- :mod:`.psf_comparison` the figure shared by two of the scripts.

``shim``, which held stand-ins for primitives adjeff did not expose, is
gone: adjeff 0.11.0 provides all of them.

Only the environment bootstrap is re-exported here.  ``scripts/
export_smartg_auxdata.py`` imports this package to locate the Smart-G
auxiliary data, which happens *before* ``SMARTG_DIR_AUXDATA`` is set, so
importing anything that reaches ``adjeff`` (and therefore ``smartg``)
from this module would make the bootstrap fail.  Figure scripts import
the submodules they need explicitly.
"""

from .download_auxdata import download_smartg_auxdata_and_cams
from .utils import get_auxdata_path

__all__ = ["download_smartg_auxdata_and_cams", "get_auxdata_path"]
