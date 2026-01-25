"""
This module contains decorators to handle exceptions for the messenger project.

VERSION = "0.1.0"
LAST_UPDATED = "2025-09-23"
AUTHOR = "Prithvi Srivastava"
LICENSE = "MIT"

Functions
---------

stop_all_exceptions:
    Decorator to stop all exceptions. Logs the exception and returns None.
"""

# 1st party imports
import logging
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

# create a logger
LOGGER = logging.getLogger("Exception Handler")

# create constants
P = ParamSpec("P")
R = TypeVar("R")


def stop_all_exceptions(func: Callable[P, R]) -> Callable[P, R | None]:
    """
    Decorator to stop all exceptions. Logs the exception and returns None.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            LOGGER.exception(
                f"Error while executing function {func.__name__}",
                extra={
                    "event": "function",
                    "function": f"{func.__module__}.{func.__name__}",
                    "args": [repr(a) for a in args],
                    "kwargs": {k: repr(v) for k, v in kwargs.items()},
                },
            )
            return None

    return wrapper
