"""
This module contains the ADX indicator logic.
"""

# 1st party imports
from typing import List, Tuple

# local imports
from binance_api.indicators.abstract import Indicator


class ADX(Indicator):
    """
    Average Directional Index (ADX) indicator.

    ADX measures trend strength regardless of direction. Values range from 0 to 100.
    - ADX > 25: Strong trend
    - ADX < 20: Weak trend or ranging market

    Also calculates +DI and -DI (Directional Indicators):
    - +DI > -DI: Bullish trend
    - -DI > +DI: Bearish trend

    Attributes:
        di_length (int): The period for +DI/-DI calculation.
        adx_smoothing (int): The period for ADX smoothing.
    """

    NAME = "ADX"

    def __init__(self, di_length: int = 14, adx_smoothing: int = 14) -> None:
        """
        Initialize the ADX indicator.

        Args:
            di_length: The period for DI calculation. Default is 14.
            adx_smoothing: The period for ADX smoothing. Default is 14.
        """
        self.di_length = di_length
        self.adx_smoothing = adx_smoothing

    def _calc_true_range(self, high: float, low: float, prev_close: float) -> float:
        """Calculate True Range for a single bar."""
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

    def _rma(self, values: List[float], length: int) -> List[float]:
        """
        Calculate RMA (Wilder's smoothing) for a list of values.

        Args:
            values: List of values to smooth.
            length: Smoothing period.

        Returns:
            List of RMA values.
        """
        rma_values = []
        alpha = 1 / length

        for idx in range(len(values)):

            # not enough data points yet
            if idx < length - 1:
                rma_values.append(0.0)
                continue

            # first RMA = SMA of first `length` values
            if idx == length - 1:
                sma = sum(values[:length]) / length
                rma_values.append(sma)

            # RMA = alpha * current + (1 - alpha) * previous_RMA
            else:
                rma = alpha * values[idx] + (1 - alpha) * rma_values[-1]
                rma_values.append(rma)

        return rma_values

    def get_value(
        self,
        candles_high: List[float],
        candles_low: List[float],
        candles_close: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate ADX, +DI, and -DI values for a series of candle data.

        Args:
            candles_high: List of candle high prices (oldest to newest).
            candles_low: List of candle low prices (oldest to newest).
            candles_close: List of candle close prices (oldest to newest).

        Returns:
            Tuple of (adx_values, plus_di_values, minus_di_values).
        """
        n = len(candles_high)

        if n < 2:
            return [0.0] * n, [0.0] * n, [0.0] * n

        # calculate True Range, +DM, -DM for each bar
        tr_list = [0.0]  # first bar has no TR
        plus_dm_list = [0.0]
        minus_dm_list = [0.0]

        for i in range(1, n):
            # true range
            tr = self._calc_true_range(
                candles_high[i], candles_low[i], candles_close[i - 1]
            )
            tr_list.append(tr)

            # directional movement
            up = candles_high[i] - candles_high[i - 1]
            down = candles_low[i - 1] - candles_low[i]

            plus_dm = up if (up > down and up > 0) else 0.0
            minus_dm = down if (down > up and down > 0) else 0.0

            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

        # apply RMA smoothing
        tr_rma = self._rma(tr_list, self.di_length)
        plus_dm_rma = self._rma(plus_dm_list, self.di_length)
        minus_dm_rma = self._rma(minus_dm_list, self.di_length)

        # calculate +DI and -DI
        plus_di = []
        minus_di = []

        for i in range(n):
            if tr_rma[i] == 0:
                plus_di.append(0.0)
                minus_di.append(0.0)
            else:
                plus_di.append(100 * plus_dm_rma[i] / tr_rma[i])
                minus_di.append(100 * minus_dm_rma[i] / tr_rma[i])

        # calculate DX
        dx_list = []
        for i in range(n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum == 0:
                dx_list.append(0.0)
            else:
                dx_list.append(100 * abs(plus_di[i] - minus_di[i]) / di_sum)

        # apply RMA to DX to get ADX
        adx_values = self._rma(dx_list, self.adx_smoothing)

        return adx_values, plus_di, minus_di
