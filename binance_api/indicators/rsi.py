"""
This module contains the RSI indicator logic.
"""

# 1st party imports
from typing import List

# local imports
from binance_api.indicators.abstract import Indicator


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

    NAME = "RSI"

    def __init__(self, length: int = 14) -> None:
        """
        Initialize the RSI indicator.

        Args:
            length: The period/lookback length for the RSI calculation.
                    Default is 14 (standard RSI period).
        """
        self.length = length

    def get_value(self, candles_close: List[float]) -> List[float]:
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

        # the rsi value at each idx
        rsi_values = []

        # need at least length + 1 prices to calculate first RSI
        if len(candles_close) < self.length + 1:
            return [0.0] * len(candles_close)

        # calculate price changes
        changes = []
        for i in range(1, len(candles_close)):
            changes.append(candles_close[i] - candles_close[i - 1])

        # separate gains and losses
        gains = [max(c, 0) for c in changes]
        losses = [-min(c, 0) for c in changes]

        # RMA smoothing factor (Wilder's smoothing: alpha = 1/length)
        alpha = 1 / self.length

        # track RMA values for gains and losses
        avg_gain = 0.0
        avg_loss = 0.0

        # first value has no change, so RSI is 0
        rsi_values.append(0.0)

        # iterate through changes
        for idx in range(len(changes)):

            # not enough data points yet
            if idx < self.length - 1:
                rsi_values.append(0.0)
                continue

            # first RSI = SMA of first `length` gains/losses
            if idx == self.length - 1:
                avg_gain = sum(gains[: self.length]) / self.length
                avg_loss = sum(losses[: self.length]) / self.length

            # RMA = alpha * current + (1 - alpha) * previous_RMA
            else:
                avg_gain = alpha * gains[idx] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[idx] + (1 - alpha) * avg_loss

            # calculate RSI
            if avg_loss == 0:
                rsi = 100.0
            elif avg_gain == 0:
                rsi = 0.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

        # return the RSI values
        return rsi_values
