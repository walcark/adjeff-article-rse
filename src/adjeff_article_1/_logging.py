"""This repository's own logger namespace, on adjeff's transport."""

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
    """Return the logger a module of this repository writes through."""
    if name != ROOT and not name.startswith(ROOT + "."):
        name = f"{ROOT}.{name}"
    return structlog.wrap_logger(
        logging.getLogger(name),
        processors=_PROCESSORS,
        wrapper_class=structlog.stdlib.BoundLogger,
    )
