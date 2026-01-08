"""
This module contains the VWAP indicator logic.
"""

# 1st party imports
from typing import List, Tuple, Optional
from enum import Enum

# 3rd party imports
import numpy as np

# local imports
from .abstract import Indicator, AvailableIndicators


class AnchorPeriod(str, Enum):
    """
    Anchor period types for VWAP calculation.
    """

    SESSION = "Session"
    WEEK = "Week"
    MONTH = "Month"
    QUARTER = "Quarter"
    YEAR = "Year"
    DECADE = "Decade"
    CENTURY = "Century"


class BandCalculationMode(str, Enum):
    """
    Band calculation modes for VWAP bands.
    """

    STANDARD_DEVIATION = "Standard Deviation"
    PERCENTAGE = "Percentage"


class VWAP(Indicator):
    """
    Volume Weighted Average Price (VWAP) indicator.

    VWAP is the trading volume-weighted average price for a given period.
    It's often used as a trading benchmark by institutional investors.

    The indicator also includes optional bands around the VWAP line:
    - Upper bands: VWAP + (multiplier × band calculation)
    - Lower bands: VWAP - (multiplier × band calculation)

    Attributes:
        anchor_period (AnchorPeriod): The reset period for VWAP calculation.
        band_calculation_mode (BandCalculationMode): How to calculate band distances.
        band_multipliers (List[float]): Multipliers for the bands.
    """

    NAME = AvailableIndicators.VWAP

    def __init__(
        self,
        anchor_period: AnchorPeriod = AnchorPeriod.SESSION,
        band_calculation_mode: BandCalculationMode = BandCalculationMode.STANDARD_DEVIATION,
        band_multipliers: List[float] = None,
    ) -> None:
        """
        Initialize the VWAP indicator.

        Args:
            anchor_period: The reset period for VWAP calculation. Default is Session.
            band_calculation_mode: How to calculate band distances. Default is Standard Deviation.
            band_multipliers: Multipliers for the bands. Default is [1.0, 2.0, 3.0].
        """
        self.anchor_period = anchor_period
        self.band_calculation_mode = band_calculation_mode
        self.band_multipliers = (
            band_multipliers if band_multipliers is not None else [1.0, 2.0, 3.0]
        )
        self.LENGTH = 0  # VWAP doesn't have a fixed length

    def _calculate_vwap(
        self,
        typical_price: np.ndarray,
        volume: np.ndarray,
        reset_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate VWAP and standard deviation bands.

        Args:
            typical_price: Array of typical prices (HLC/3).
            volume: Array of volumes.
            reset_points: Boolean array indicating where to reset VWAP calculation.

        Returns:
            Tuple of (vwap_values, stdev_values, cumulative_volumes).
        """
        n = len(typical_price)
        vwap_values = np.zeros(n, dtype=np.float64)
        stdev_values = np.zeros(n, dtype=np.float64)
        cumulative_volumes = np.zeros(n, dtype=np.float64)

        # Initialize tracking variables
        cumulative_price_volume = 0.0
        cumulative_volume = 0.0
        price_squared_volume_sum = 0.0

        for i in range(n):
            # Reset if this is a reset point
            if reset_points[i] or i == 0:
                cumulative_price_volume = 0.0
                cumulative_volume = 0.0
                price_squared_volume_sum = 0.0

            # Update cumulative values
            price_volume = typical_price[i] * volume[i]
            cumulative_price_volume += price_volume
            cumulative_volume += volume[i]
            price_squared_volume_sum += (typical_price[i] ** 2) * volume[i]

            # Calculate VWAP
            if cumulative_volume > 0:
                vwap_values[i] = cumulative_price_volume / cumulative_volume
                cumulative_volumes[i] = cumulative_volume

                # Calculate standard deviation
                if cumulative_volume > 1:
                    variance = (price_squared_volume_sum / cumulative_volume) - (
                        vwap_values[i] ** 2
                    )
                    stdev_values[i] = np.sqrt(max(variance, 0))

        return vwap_values, stdev_values, cumulative_volumes

    def _get_reset_points(self, n: int) -> np.ndarray:
        """
        Get reset points based on the anchor period.

        For this implementation, we'll use a simple session-based reset.
        More sophisticated implementations could use actual time data.

        Args:
            n: Number of data points.

        Returns:
            Boolean array indicating reset points.
        """
        reset_points = np.zeros(n, dtype=bool)

        # For now, implement simple session reset (every day)
        # In a real implementation, this would use actual timestamps
        if self.anchor_period == AnchorPeriod.SESSION:
            # Reset at the beginning (simplified)
            if n > 0:
                reset_points[0] = True
        elif self.anchor_period == AnchorPeriod.WEEK:
            # Reset weekly (simplified - every 5th trading day)
            for i in range(0, n, 5):
                if i < n:
                    reset_points[i] = True
        elif self.anchor_period == AnchorPeriod.MONTH:
            # Reset monthly (simplified - every 21 trading days)
            for i in range(0, n, 21):
                if i < n:
                    reset_points[i] = True
        # Add more sophisticated period detection as needed

        return reset_points

    def get_value(
        self,
        candles_high: List[float] | np.ndarray,
        candles_low: List[float] | np.ndarray,
        candles_close: List[float] | np.ndarray,
        candles_volume: List[float] | np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Calculate VWAP values and bands for a series of candle data.

        Args:
            candles_high: Array of candle high prices (oldest to newest).
            candles_low: Array of candle low prices (oldest to newest).
            candles_close: Array of candle close prices (oldest to newest).
            candles_volume: Array of candle volumes (oldest to newest).

        Returns:
            Tuple of (vwap_values, upper_bands, lower_bands).
            - vwap_values: VWAP line values
            - upper_bands: List of upper band arrays (one for each multiplier)
            - lower_bands: List of lower band arrays (one for each multiplier)
        """
        # Convert to numpy arrays
        high = np.asarray(candles_high, dtype=np.float64)
        low = np.asarray(candles_low, dtype=np.float64)
        close = np.asarray(candles_close, dtype=np.float64)
        volume = np.asarray(candles_volume, dtype=np.float64)

        n = len(close)

        if n == 0:
            return np.array([]), [], []

        # Calculate typical price (HLC/3)
        typical_price = (high + low + close) / 3.0

        # Check for zero volume
        if np.sum(volume) == 0:
            raise ValueError("No volume data provided or all volumes are zero")

        # Get reset points based on anchor period
        reset_points = self._get_reset_points(n)

        # Calculate VWAP and standard deviation
        vwap_values, stdev_values, _ = self._calculate_vwap(
            typical_price, volume, reset_points
        )

        # Calculate bands
        upper_bands = []
        lower_bands = []

        for multiplier in self.band_multipliers:
            upper_band = np.zeros(n, dtype=np.float64)
            lower_band = np.zeros(n, dtype=np.float64)

            if self.band_calculation_mode == BandCalculationMode.STANDARD_DEVIATION:
                # Use standard deviation
                upper_band = vwap_values + stdev_values * multiplier
                lower_band = vwap_values - stdev_values * multiplier
            else:  # PERCENTAGE
                # Use percentage of VWAP
                band_basis = vwap_values * 0.01  # 1% of VWAP
                upper_band = vwap_values + band_basis * multiplier
                lower_band = vwap_values - band_basis * multiplier

            upper_bands.append(upper_band)
            lower_bands.append(lower_band)

        return vwap_values, upper_bands, lower_bands
