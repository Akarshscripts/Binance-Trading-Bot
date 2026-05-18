"""Abstract base class for all trading strategies."""

# 1st party imports
from abc import ABC, abstractmethod
from typing import List, Union

# 3rd party imports
import numpy as np

# local imports
from strategies.models import PredictionOutput


class Strategy(ABC):
    """
    Abstract base class that all trading strategies must implement.

    Concrete strategies subclass this and implement `process_candles`,
    allowing execution logic (backtest, predict) to depend on this
    interface rather than any specific strategy.
    """

    @abstractmethod
    def process_candles(
        self,
        open_prices: Union[List[float], np.ndarray],
        high_prices: Union[List[float], np.ndarray],
        low_prices: Union[List[float], np.ndarray],
        close_prices: Union[List[float], np.ndarray],
        volumes: Union[List[float], np.ndarray],
        round_off: int = 4,
    ) -> PredictionOutput:
        """
        Process a window of OHLCV candles and return a trade signal.

        Args:
            open_prices: Open prices for each candle.
            high_prices: High prices for each candle.
            low_prices: Low prices for each candle.
            close_prices: Close prices for each candle.
            volumes: Volume for each candle.
            round_off: Decimal places to round output prices to.

        Returns:
            PredictionOutput with action, entry, exit, stop-loss, and R:R ratio.
        """
