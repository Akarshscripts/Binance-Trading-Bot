"""
This module contains the ATR indicator logic.
"""

# 1st party imports
from enum import Enum
from typing import List

# 3rd party imports
import numpy as np

# local imports
from .abstract import Indicator, AvailableIndicators


class SmoothingType(str, Enum):
    """
    Smoothing types for ATR calculation.
    """

    RMA = "RMA"
    SMA = "SMA"
    EMA = "EMA"
    WMA = "WMA"


class ATR(Indicator):
    """
    Average True Range (ATR) indicator.

    ATR measures market volatility by decomposing the entire range of an asset
    price for that period. Specifically, it's the moving average of the true ranges.

    True Range is the greatest of:
    - Current high minus current low
    - Current high minus previous close (absolute)
    - Current low minus previous close (absolute)

    Attributes:
        length (int): The period for ATR calculation.
        smoothing (SmoothingType): The smoothing method to use.
    """

    NAME = AvailableIndicators.ATR

    def __init__(
        self, length: int = 14, smoothing: SmoothingType = SmoothingType.RMA
    ) -> None:
        """
        Initialize the ATR indicator.

        Args:
            length: The period for ATR calculation. Default is 14.
            smoothing: The smoothing method to use. Default is RMA.
        """
        self.smoothing = smoothing
        self.LENGTH = length

    def _rma(self, values: np.ndarray, length: int) -> np.ndarray:
        """Calculate RMA (Wilder's smoothing)."""
        n = len(values)
        result = np.zeros(n, dtype=np.float64)

        if n < length:
            return result

        alpha = 1.0 / length

        # First RMA = SMA of first `length` values
        result[length - 1] = np.mean(values[:length])

        # RMA = alpha * current + (1 - alpha) * previous_RMA
        for i in range(length, n):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]

        return result

    def _sma(self, values: np.ndarray, length: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        result = np.zeros(len(values), dtype=np.float64)
        for i in range(length - 1, len(values)):
            result[i] = np.mean(values[i - length + 1 : i + 1])
        return result

    def _ema(self, values: np.ndarray, length: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        result = np.zeros(len(values), dtype=np.float64)
        alpha = 2.0 / (length + 1)

        # Start with SMA for the first value
        if len(values) >= length:
            result[length - 1] = np.mean(values[:length])

            # Calculate EMA for remaining values
            for i in range(length, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]

        return result

    def _wma(self, values: np.ndarray, length: int) -> np.ndarray:
        """Calculate Weighted Moving Average."""
        result = np.zeros(len(values), dtype=np.float64)
        weights = np.arange(1, length + 1, dtype=np.float64)
        weight_sum = np.sum(weights)

        for i in range(length - 1, len(values)):
            result[i] = np.sum(values[i - length + 1 : i + 1] * weights) / weight_sum

        return result

    def _apply_smoothing(self, values: np.ndarray, length: int) -> np.ndarray:
        """Apply the selected smoothing method."""
        if self.smoothing == SmoothingType.RMA:
            return self._rma(values, length)
        elif self.smoothing == SmoothingType.SMA:
            return self._sma(values, length)
        elif self.smoothing == SmoothingType.EMA:
            return self._ema(values, length)
        elif self.smoothing == SmoothingType.WMA:
            return self._wma(values, length)
        else:
            raise ValueError(f"Unsupported smoothing type: {self.smoothing}")

    def get_value(
        self,
        high: np.ndarray | List[float],
        low: np.ndarray | List[float],
        close: np.ndarray | List[float],
    ) -> np.ndarray:
        """
        Calculate ATR values for a series of candle data.

        Args:
            high: Array of candle high prices (oldest to newest).
            low: Array of candle low prices (oldest to newest).
            close: Array of candle close prices (oldest to newest).

        Returns:
            Array of ATR values as np.ndarray.
        """
        # Convert to numpy arrays
        high = np.asarray([i for i in high], dtype=np.float64)
        low = np.asarray([i for i in low], dtype=np.float64)
        close = np.asarray([i for i in close], dtype=np.float64)

        # check if the arrays are empty
        n = len(high)
        if n < 2:
            return np.zeros(n, dtype=np.float64)

        # Calculate True Range (vectorized)
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )

        # Apply smoothing to get ATR
        atr_values = self._apply_smoothing(tr, self.LENGTH)
        return atr_values
