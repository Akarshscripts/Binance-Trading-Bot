"""
This module contains the abstract indicator logic.
"""

# 1st party imports
from abc import ABC, abstractmethod
from typing import Any, List


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.

    All indicator implementations must inherit from this class
    and implement the get_value method.
    """

    NAME: str = "Indicator"

    def __init_subclass__(cls, **kwargs: Any) -> None:

        # sub class inherited
        super().__init_subclass__(**kwargs)

        # check if the name was over-riden
        if cls.NAME == "Indicator":
            raise TypeError(f"Subclass {cls.__name__} must override the NAME attribute")

    @abstractmethod
    def get_value(self, *args: List[float], **kwargs: Any) -> Any:
        """
        Calculate indicator values for a series of candle data.

        Args:
            *args: Variable candle data lists (e.g., high, low, close prices).
            **kwargs: Additional keyword arguments specific to the indicator.

        Returns:
            Indicator values (type varies by implementation).
        """
        pass
