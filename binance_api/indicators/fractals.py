"""
This module contains the ADX indicator logic.
"""

# 1st party imports
from typing import List, Tuple

# 3rd party imports
import numpy as np

# local imports
from binance_api.indicators.abstract import Indicator, AvailableIndicators


class FractalsIndicator(Indicator):
    """
    Fractal indicator returning price levels for stop-loss usage.
    """

    NAME = AvailableIndicators.FRACTALS

    def __init__(self, length: int = 5):
        """
        Initialize the fractals indicator.
        """

        # set the length
        self.LENGTH = length

    def get_value(
        self,
        high: List[float],
        low: List[float],
        *,
        ht_high: List[float] | None = None,
        ht_low: List[float] | None = None,
        filter_bw: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate fractal stop-loss levels.

        Args:
            high: High prices
            low: Low prices
            ht_high: Higher timeframe high (optional)
            ht_low: Higher timeframe low (optional)
            filter_bw: True -> Regular fractals
                       False -> Bill Williams fractals

        Returns:
            (top_sl, bottom_sl)
            - top_sl: high price at top fractals, else NaN
            - bottom_sl: low price at bottom fractals, else NaN
        """

        # convert to numpy arrays
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        # get the length
        n = len(high)
        top_sl = np.full(n, np.nan)
        bottom_sl = np.full(n, np.nan)

        # check if the length is less than the required length
        if n < self.LENGTH:
            return top_sl, bottom_sl

        # Shifted arrays (PineScript style)
        h4, h3, h2, h1, h0 = (high[:-4], high[1:-3], high[2:-2], high[3:-1], high[4:])
        l4, l3, l2, l1, l0 = (low[:-4], low[1:-3], low[2:-2], low[3:-1], low[4:])

        # Fractal rules
        if filter_bw:
            top_core = (h4 < h3) & (h3 < h2) & (h2 > h1) & (h1 > h0)
            bot_core = (l4 > l3) & (l3 > l2) & (l2 < l1) & (l1 < l0)
        else:
            top_core = (h4 < h2) & (h3 <= h2) & (h2 >= h1) & (h2 > h0)
            bot_core = (l4 > l2) & (l3 >= l2) & (l2 <= l1) & (l2 < l0)

        # get the indices
        idx = np.arange(2, n - 2)

        # update the top and bottom sl
        top_sl[idx[top_core]] = high[idx[top_core]]
        bottom_sl[idx[bot_core]] = low[idx[bot_core]]

        # Higher timeframe filter
        if ht_high is not None:
            ht_high = np.asarray(ht_high, dtype=float)
            top_sl = np.where(high >= ht_high, top_sl, np.nan)

        # update the bottom sl
        if ht_low is not None:
            ht_low = np.asarray(ht_low, dtype=float)
            bottom_sl = np.where(low <= ht_low, bottom_sl, np.nan)

        # return the top and bottom sl
        return top_sl, bottom_sl
