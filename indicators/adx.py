"""
This module contains the ADX indicator logic.
"""

# 1st party imports
from typing import List, Tuple

# 3rd party imports
import numpy as np

# local imports
from .abstract import Indicator, AvailableIndicators


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

    NAME = AvailableIndicators.ADX

    def __init__(self, di_length: int = 14, adx_smoothing: int = 14) -> None:
        """
        Initialize the ADX indicator.

        Args:
            di_length: The period for DI calculation. Default is 14.
            adx_smoothing: The period for ADX smoothing. Default is 14.
        """
        self.di_length = di_length
        self.adx_smoothing = adx_smoothing
        self.LENGTH = di_length

    def _rma(self, values: np.ndarray, length: int) -> np.ndarray:
        """
        Calculate RMA (Wilder's smoothing) for an array of values (vectorized).

        Args:
            values: Array of values to smooth.
            length: Smoothing period.

        Returns:
            np.ndarray of RMA values.
        """
        n = len(values)
        rma_values = np.zeros(n, dtype=np.float64)

        if n < length:
            return rma_values

        alpha = 1.0 / length

        # first RMA = SMA of first `length` values
        rma_values[length - 1] = np.mean(values[:length])

        # RMA = alpha * current + (1 - alpha) * previous_RMA
        for i in range(length, n):
            rma_values[i] = alpha * values[i] + (1 - alpha) * rma_values[i - 1]

        return rma_values

    def get_value(
        self,
        high: List[float] | np.ndarray,
        low: List[float] | np.ndarray,
        close: List[float] | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ADX, +DI, and -DI values for a series of candle data (vectorized).

        Args:
            high: Array of candle high prices (oldest to newest).
            low: Array of candle low prices (oldest to newest).
            close: Array of candle close prices (oldest to newest).

        Returns:
            Tuple of (adx_values, plus_di_values, minus_di_values) as np.ndarray.
        """
        # convert to numpy arrays
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        n = len(high)

        if n < 2:
            zeros = np.zeros(n, dtype=np.float64)
            return zeros, zeros.copy(), zeros.copy()

        # calculate True Range (vectorized)
        tr = np.zeros(n, dtype=np.float64)
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )

        # calculate directional movement (vectorized)
        up = np.zeros(n, dtype=np.float64)
        down = np.zeros(n, dtype=np.float64)
        up[1:] = high[1:] - high[:-1]
        down[1:] = low[:-1] - low[1:]

        # +DM and -DM conditions
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        # apply RMA smoothing
        tr_rma = self._rma(tr, self.di_length)
        plus_dm_rma = self._rma(plus_dm, self.di_length)
        minus_dm_rma = self._rma(minus_dm, self.di_length)

        # calculate +DI and -DI (vectorized with safe division)
        plus_di = np.zeros(n, dtype=np.float64)
        minus_di = np.zeros(n, dtype=np.float64)
        np.divide(100 * plus_dm_rma, tr_rma, out=plus_di, where=tr_rma != 0)
        np.divide(100 * minus_dm_rma, tr_rma, out=minus_di, where=tr_rma != 0)

        # calculate DX (vectorized with safe division)
        di_sum = plus_di + minus_di
        dx = np.zeros(n, dtype=np.float64)
        np.divide(100 * np.abs(plus_di - minus_di), di_sum, out=dx, where=di_sum != 0)

        # apply RMA to DX to get ADX
        adx_values = self._rma(dx, self.adx_smoothing)

        return adx_values, plus_di, minus_di
