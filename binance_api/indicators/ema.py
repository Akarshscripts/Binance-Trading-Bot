"""
This module contains the EMA indicator logic.
"""

# 1st party imports
from typing import List

# 3rd party imports
import numpy as np

# local imports
from binance_api.indicators.abstract import Indicator, AvailableIndicators


class EMA(Indicator):
    """
    Exponential Moving Average (EMA) indicator.

    EMA gives more weight to recent prices, making it more responsive to new
    information compared to a Simple Moving Average (SMA).

    Attributes:
        length (int): The period/lookback length for the EMA calculation.
    """

    NAME = AvailableIndicators.EMA

    def __init__(self, length: int) -> None:
        """
        Initialize the EMA indicator.

        Args:
            length: The period/lookback length for the EMA calculation.
        """
        self.LENGTH = length

    def get_value(self, candles_open: List[float] | np.ndarray) -> np.ndarray:
        """
        Calculate EMA values for a series of candle data (vectorized).

        Uses the standard EMA formula: EMA = α * price + (1 - α) * previous_EMA
        where α (alpha) = 2 / (length + 1).

        The first EMA value is seeded with an SMA (same approach as TradingView).
        Values before enough data points are available are set to 0.0.

        Args:
            candles_open: List of candle open prices (oldest to newest).

        Returns:
            np.ndarray of EMA values corresponding to each candle position.
        """

        # convert to numpy array
        prices = np.asarray(candles_open, dtype=np.float64)
        n = len(prices)

        # initialize output array with zeros
        ema_values = np.zeros(n, dtype=np.float64)

        # not enough data
        if n < self.LENGTH:
            return ema_values

        # smoothing factor
        alpha = 2.0 / (self.LENGTH + 1)

        # seed with SMA of first `length` prices
        ema_values[self.LENGTH - 1] = np.mean(prices[: self.LENGTH])

        # EMA calculation (recursive dependency requires loop)
        for i in range(self.LENGTH, n):
            ema_values[i] = alpha * prices[i] + (1 - alpha) * ema_values[i - 1]

        # return values
        return ema_values
