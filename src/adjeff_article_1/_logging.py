"""This repository's own logger namespace, on adjeff's transport.

adjeff routes everything it emits through the standard library, under
the `adjeff` name.  Its `get_logger` re-homes any other name below that
one, which is right for adjeff's own modules and wrong for a consumer:
these lines come from the article's code, not from the library, and
saying so is the point of a logger name.

So the wrapping is repeated here, without the re-homing, and
`runconfig.parse_run` sets this namespace's level alongside adjeff's.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from adjeff._logging import _PROCESSORS

__all__ = ["ROOT", "get_logger"]

#: Root of this repository's logger namespace, matching the package name.
ROOT = "adjeff_article_1"

logging.getLogger(ROOT).addHandler(logging.NullHandler())


def get_logger(name: str) -> Any:
    """Return the logger a module of this repository writes through.

    Parameters
    ----------
    name : str
        Name below :data:`ROOT`, with or without the prefix.

    Returns
    -------
    structlog.stdlib.BoundLogger
        Bound logger delegating to :mod:`logging`, taking key-value
        pairs, and rendered by whatever `adjeff.setup_logging` installed.
    """
    if name != ROOT and not name.startswith(ROOT + "."):
        name = f"{ROOT}.{name}"
    return structlog.wrap_logger(
        logging.getLogger(name),
        processors=_PROCESSORS,
        wrapper_class=structlog.stdlib.BoundLogger,
    )
