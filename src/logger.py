"""Centralized logging for the pipeline."""

from __future__ import annotations

import logging
import sys

_default_level = logging.INFO


def get_logger(name: str = "nlp_pipeline", level: int | None = None) -> logging.Logger:
    """Return a configured logger. Use same name to reuse."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level or _default_level)
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(log.level)
        log.addHandler(h)
    return log


def set_level(level: int | str) -> None:
    """Set default log level (e.g. logging.DEBUG or 'DEBUG')."""
    global _default_level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    _default_level = level
