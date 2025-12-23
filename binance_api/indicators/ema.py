"""
This module contains the EMA indicator logic.
"""

# 1st party imports
from typing import List

# local imports
from binance_api.indicators.abstract import Indicator


class EMA(Indicator):
    """
    Exponential Moving Average (EMA) indicator.

    EMA gives more weight to recent prices, making it more responsive to new
    information compared to a Simple Moving Average (SMA).

    Attributes:
        length (int): The period/lookback length for the EMA calculation.
    """

    NAME = "EMA"

    def __init__(self, length: int) -> None:
        """
        Initialize the EMA indicator.

        Args:
            length: The period/lookback length for the EMA calculation.
        """
        self.length = length

    def get_value(self, candles_open: List[float]) -> List[float]:
        """
        Calculate EMA values for a series of candle data.

        Uses the standard EMA formula: EMA = α * price + (1 - α) * previous_EMA
        where α (alpha) = 2 / (length + 1).

        The first EMA value is seeded with an SMA (same approach as TradingView).
        Values before enough data points are available are set to 0.0.

        Args:
            candles_open: List of candle open prices (oldest to newest).

        Returns:
            List of EMA values corresponding to each candle position.
        """

        # the ema value at each idx
        ema_values = []

        # smoothing factor
        alpha = 2 / (self.length + 1)

        # iterate through all candles
        for idx in range(len(candles_open)):

            # not enough data points yet
            if idx < self.length - 1:
                ema_values.append(0.0)
                continue

            # first EMA = SMA of first `length` prices
            if idx == self.length - 1:
                sma = sum(candles_open[: self.length]) / self.length
                ema_values.append(sma)

            # EMA = alpha * current_price + (1 - alpha) * previous_EMA
            else:
                ema_value = alpha * candles_open[idx] + (1 - alpha) * ema_values[-1]
                ema_values.append(ema_value)

        # return the EMA values
        return ema_values
