"""
This module contains the abstract indicator logic.
"""

# 1st party imports
from enum import Enum
from typing import Any, List
from abc import ABC, abstractmethod


class AvailableIndicators(str, Enum):
    """
    Enum of available technical indicators.

    Each value corresponds to an indicator implementation that can be
    used for technical analysis of market data.
    """

    EMA = "ema"
    RSI = "rsi"
    ADX = "adx"
    FRACTALS = "fractals"
    INDICATOR = "indicator"


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.

    All indicator implementations must inherit from this class
    and implement the get_value method.
    """

    NAME: AvailableIndicators = AvailableIndicators.INDICATOR
    LENGTH: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:

        # sub class inherited
        super().__init_subclass__(**kwargs)

        # check if the name was over-riden
        if cls.NAME == AvailableIndicators.INDICATOR:
            raise TypeError(f"Subclass {cls.__name__} must override the NAME attribute")

        # check if the name is valid
        if cls.NAME.lower() not in [
            indicator.value for indicator in AvailableIndicators
        ]:
            raise TypeError(
                f"Subclass {cls.__name__} has invalid NAME '{cls.NAME}'. "
                f"Must be one of: {[i.value for i in AvailableIndicators]}"
            )

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
