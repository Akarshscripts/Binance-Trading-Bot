"""
This module contains the core logic for the predict command.
"""

# 1st party imports
import logging
from typing import Any

# local imports
from .upstox import predict_upstox
from .binance import predict_binance
from upstox_api import UpstoxSymbols, UpstoxIntervals
from binance_api import BinanceSymbols, ChartIntervals

# get logger
logger = logging.getLogger("predict")


def start_algorithm(
    symbol: Any,
    interval: Any,
    discord_webhook: str,
    lag: int = 30,
):
    """
    Start tracking a crypto pair and make real-time predictions.

    Args:
        symbol: The trading symbol to track (e.g., BTCUSDT).
        interval: The chart interval for candlestick data.
        discord_webhook: The Discord webhook URL for sending predictions.
        lag: Seconds to wait after interval ends before fetching (default: 30).
    """

    # start prediction based on the symbol
    if isinstance(symbol, BinanceSymbols):

        # convert the interval to binance interval
        interval = ChartIntervals(interval)

        # start prediction for binance
        predict_binance(
            symbol=symbol,
            interval=interval,
            discord_webhook=discord_webhook,
            lag=lag,
        )

    elif isinstance(symbol, UpstoxSymbols):

        # convert the interval to upstox interval
        interval = UpstoxIntervals(interval)

        # start prediction for upstox
        predict_upstox(
            symbol=symbol,
            interval=interval,
            discord_webhook=discord_webhook,
            lag=lag,
        )

    # log the end message
    logger.info("Prediction completed.")
