"""
This module contains the Bollinger Bands indicator logic.
"""

# 1st party imports
from enum import Enum
from typing import List, Tuple

# 3rd party imports
import numpy as np

# local imports
from .abstract import Indicator, AvailableIndicators


class MAType(str, Enum):
    """
    Moving average types for Bollinger Bands calculation.
    """

    SMA = "SMA"
    EMA = "EMA"
    SMMA = "SMMA (RMA)"
    WMA = "WMA"
    VWMA = "VWMA"


class BollingerBands(Indicator):
    """
    Bollinger Bands indicator.

    Bollinger Bands consist of a middle band (SMA/EMA/etc) and two outer bands
    that are typically 2 standard deviations away from the middle band.

    - Upper Band: Middle Band + (Multiplier × Standard Deviation)
    - Middle Band: Moving Average of closing prices
    - Lower Band: Middle Band - (Multiplier × Standard Deviation)

    Attributes:
        length (int): The period for moving average calculation.
        ma_type (MAType): Type of moving average to use.
        multiplier (float): Standard deviation multiplier (typically 2.0).
    """

    NAME = AvailableIndicators.BOLLINGER_BANDS

    def __init__(
        self, length: int = 20, ma_type: MAType = MAType.SMA, multiplier: float = 2.0
    ) -> None:
        """
        Initialize the Bollinger Bands indicator.

        Args:
            length: The period for moving average calculation. Default is 20.
            ma_type: Type of moving average to use. Default is SMA.
            multiplier: Standard deviation multiplier. Default is 2.0.
        """
        self.LENGTH = length
        self.ma_type = ma_type
        self.multiplier = multiplier

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

    def _wma(self, values: np.ndarray, length: int) -> np.ndarray:
        """Calculate Weighted Moving Average."""
        result = np.zeros(len(values), dtype=np.float64)
        weights = np.arange(1, length + 1, dtype=np.float64)
        weight_sum = np.sum(weights)

        for i in range(length - 1, len(values)):
            result[i] = np.sum(values[i - length + 1 : i + 1] * weights) / weight_sum

        return result

    def _vwma(self, values: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
        """Calculate Volume Weighted Moving Average."""
        result = np.zeros(len(values), dtype=np.float64)

        for i in range(length - 1, len(values)):
            values_slice = values[i - length + 1 : i + 1]
            volume_slice = volume[i - length + 1 : i + 1]
            result[i] = np.sum(values_slice * volume_slice) / np.sum(volume_slice)

        return result

    def _calculate_ma(
        self, values: np.ndarray, volume: np.ndarray = None
    ) -> np.ndarray:
        """Calculate moving average based on the specified type."""
        if self.ma_type == MAType.SMA:
            return self._sma(values, self.LENGTH)
        elif self.ma_type == MAType.EMA:
            return self._ema(values, self.LENGTH)
        elif self.ma_type == MAType.SMMA:
            return self._rma(values, self.LENGTH)
        elif self.ma_type == MAType.WMA:
            return self._wma(values, self.LENGTH)
        elif self.ma_type == MAType.VWMA:
            if volume is None:
                raise ValueError("Volume data is required for VWMA calculation")
            return self._vwma(values, volume, self.LENGTH)
        else:
            raise ValueError(f"Unsupported MA type: {self.ma_type}")

    def get_value(
        self,
        candles_close: List[float] | np.ndarray,
        candles_volume: List[float] | np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands values for a series of candle data.

        Args:
            candles_close: Array of candle close prices (oldest to newest).
            candles_volume: Array of candle volumes (required only for VWMA).

        Returns:
            Tuple of (upper_band, middle_band, lower_band) as np.ndarray.
        """
        # Convert to numpy arrays
        close = np.asarray(candles_close, dtype=np.float64)

        if self.ma_type == MAType.VWMA:
            if candles_volume is None:
                raise ValueError("Volume data is required for VWMA calculation")
            volume = np.asarray(candles_volume, dtype=np.float64)
        else:
            volume = None

        n = len(close)

        if n < self.LENGTH:
            zeros = np.zeros(n, dtype=np.float64)
            return zeros, zeros.copy(), zeros.copy()

        # Calculate middle band (moving average)
        middle_band = self._calculate_ma(close, volume)

        # Calculate standard deviation
        std_dev = np.zeros(n, dtype=np.float64)
        for i in range(self.LENGTH - 1, n):
            std_dev[i] = np.std(close[i - self.LENGTH + 1 : i + 1], ddof=0)

        # Calculate upper and lower bands
        deviation = self.multiplier * std_dev
        upper_band = middle_band + deviation
        lower_band = middle_band - deviation

        return upper_band, middle_band, lower_band
