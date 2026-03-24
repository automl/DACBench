"""Logging utilities for the project."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging() -> None:
    """Setup logging module."""
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )


def get_logger(logger_name: str) -> logging.Logger:
    """Get the logger by name.

    Parameters
    ----------
    logger_name : str
        Name of the logger.

    Returns:
    -------
    logging.Logger
        Logger object.
    """
    setup_logging()
    return logging.getLogger(logger_name)
