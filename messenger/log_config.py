"""
messenger.log_config
--------------------
Self-contained logging configuration for the messenger module.
Can be overridden by the parent project's logging configuration.

VERSION = "0.1.0"
LAST_UPDATED = "2025-09-23"
AUTHOR = "Prithvi Srivastava"
LICENSE = "MIT"
"""

# 1st party imports
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.

    If the parent project has already configured logging, this will use that config.
    Otherwise, it sets up a basic console handler.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.
    """

    logger = logging.getLogger(f"messenger.{name}")

    # only add handler if logger has no handlers (not configured by parent project)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger
