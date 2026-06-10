"""Application logging configuration.

Provides a single :func:`configure_logging` entry point and a :func:`get_logger`
helper so every module logs through a consistent, structured format.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the whole process (idempotent)."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party libraries.
    for noisy in ("urllib3", "chromadb", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured."""

    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
