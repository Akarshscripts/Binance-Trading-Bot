"""
This module contains the Supertrend indicator logic.
"""

# 1st party imports
from typing import List, Tuple

# 3rd party imports
import numpy as np

# local imports
from .atr import ATR
from .abstract import Indicator, AvailableIndicators


class SuperTrendIndicator(Indicator):
    """
    SuperTrend indicator implementation based on TradingView Pine Script.

    SuperTrend is a trend-following indicator that uses ATR to calculate
    dynamic support and resistance levels.
    """

    NAME: AvailableIndicators = AvailableIndicators.SUPERTREND
    LENGTH: int = 10

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0):
        """
        Initialize SuperTrend indicator.

        Args:
            atr_period: Period for ATR calculation (default: 10)
            multiplier: Multiplier for ATR to calculate bands (default: 3.0)
        """

        # set values
        self.LENGTH = atr_period
        self.atr_period = atr_period
        self.multiplier = multiplier

        # linked instances
        self.atr_indicator = ATR(length=atr_period)

    def get_value(
        self,
        high: List[float] | np.ndarray,
        low: List[float] | np.ndarray,
        close: List[float] | np.ndarray,
    ) -> List[Tuple[float, int]]:
        """
        Calculate SuperTrend values for candle history.

        Args:
            high: List of high prices (oldest to newest).
            low: List of low prices (oldest to newest).
            close: List of close prices (oldest to newest).

        Returns:
            List of tuples containing (supertrend_value, direction)
            where direction is 1 for uptrend and -1 for downtrend
        """

        # check if the candle history is less than atr period
        if len(high) < self.atr_period:
            return [(0.0, 0)] * len(high)

        # get the atr values
        atr_values = self.atr_indicator.get_value(high, low, close)

        # values
        supertrend_values = []
        upper_band = []
        lower_band = []
        final_upper = []
        final_lower = []
        direction = []

        # calculate the supertrend values
        for idx in range(len(high)):

            # Calculate HL/2 (average of high and low)
            hl2 = (high[idx] + low[idx]) / 2

            # Calculate basic upper and lower bands
            up = hl2 + (self.multiplier * atr_values[idx])
            dn = hl2 - (self.multiplier * atr_values[idx])

            # append the values
            upper_band.append(up)
            lower_band.append(dn)

            # calculate the final upper and lower bands
            if idx == 0:
                final_upper.append(up)
                final_lower.append(dn)
                direction.append(1)
            else:
                # Calculate final upper band
                if up < final_upper[idx - 1] or high[idx - 1] > final_upper[idx - 1]:
                    final_up = up
                else:
                    final_up = final_upper[idx - 1]
                final_upper.append(final_up)

                # Calculate final lower band
                if dn > final_lower[idx - 1] or low[idx - 1] < final_lower[idx - 1]:
                    final_dn = dn
                else:
                    final_dn = final_lower[idx - 1]
                final_lower.append(final_dn)

                # Determine trend direction
                if direction[idx - 1] == 1:
                    if low[idx] <= final_lower[idx]:
                        trend = -1
                    else:
                        trend = 1
                else:
                    if high[idx] >= final_upper[idx]:
                        trend = 1
                    else:
                        trend = -1

                direction.append(trend)

            # SuperTrend value is the final upper band in downtrend
            # and final lower band in uptrend
            if direction[idx] == 1:
                supertrend = final_lower[idx]
            else:
                supertrend = final_upper[idx]

            supertrend_values.append((supertrend, direction[idx]))

        # return
        return supertrend_values
