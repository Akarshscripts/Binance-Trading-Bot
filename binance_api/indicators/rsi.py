"""
This module contains the RSI indicator logic.
"""

# 1st party imports
from typing import List

# 3rd party imports
import numpy as np

# local imports
from binance_api.indicators.abstract import Indicator, AvailableIndicators


class RSI(Indicator):
    """
    Relative Strength Index (RSI) indicator.

    RSI measures the speed and magnitude of price changes to evaluate
    overbought or oversold conditions. Values range from 0 to 100.
    - RSI > 70: Typically considered overbought
    - RSI < 30: Typically considered oversold

    Attributes:
        length (int): The period/lookback length for the RSI calculation.
    """

    NAME = AvailableIndicators.RSI

    def __init__(self, length: int = 14) -> None:
        """
        Initialize the RSI indicator.

        Args:
            length: The period/lookback length for the RSI calculation.
                    Default is 14 (standard RSI period).
        """
        self.length = length

    def get_value(self, candles_close: List[float] | np.ndarray) -> np.ndarray:
        """
        Calculate RSI values for a series of candle data.

        Uses the standard RSI formula with RMA (Wilder's smoothing):
        1. Calculate price changes
        2. Separate gains and losses
        3. Apply RMA smoothing to both
        4. RSI = 100 - (100 / (1 + avg_gain / avg_loss))

        The first RSI value is seeded with SMA of gains/losses (same as TradingView).
        Values before enough data points are available are set to 0.0.

        Args:
            candles_close: List of candle close prices (oldest to newest).

        Returns:
            List of RSI values corresponding to each candle position.
        """

        # convert if not already
        candles_close = np.asarray(candles_close, dtype=np.float64)
        n = len(candles_close)

        # the rsi value at each idx
        rsi_values = np.zeros(n, dtype=np.float64)

        # need at least length + 1 prices to calculate first RSI
        if n < self.length + 1:
            return rsi_values

        # calculate price changes
        changes = candles_close[1:] - candles_close[:-1]

        # separate gains and losses
        gains = np.maximum(changes, 0)
        losses = -np.minimum(changes, 0)

        # RMA smoothing factor (Wilder's smoothing: alpha = 1/length)
        alpha = 1 / self.length

        # first RSI = SMA of first `length` gains/losses
        avg_gain = np.mean(gains[: self.length])
        avg_loss = np.mean(losses[: self.length])

        # calculate first RSI at index length (offset by 1 for the first value)
        if avg_loss == 0:
            rsi_values[self.length] = 100.0
        elif avg_gain == 0:
            rsi_values[self.length] = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[self.length] = 100 - (100 / (1 + rs))

        # RMA calculation for remaining values
        for i in range(self.length, len(changes)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

            # calculate RSI
            if avg_loss == 0:
                rsi_values[i + 1] = 100.0
            elif avg_gain == 0:
                rsi_values[i + 1] = 0.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i + 1] = 100 - (100 / (1 + rs))

        # return the RSI values
        return rsi_values
