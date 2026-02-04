"""
This module contains the core logic for the quant research command.
"""

# NEXT STEPS
# 1. Use walk forward validation and take average of the best 5 trials
# 2. Optimize for stability, not max profit use win ratio or max drawdown as the metric
# 3. Reduce parameter freedom

# 1st party imports
from typing import Any
from datetime import datetime

# local imports
from .upstox import quant_research_upstox
from .binance import quant_research_binance
from binance_api import ChartIntervals, BinanceSymbols
from upstox_api import UpstoxSymbols, UpstoxIntervals


def run_quant_research(
    symbol: Any,
    interval: Any,
    start_time: datetime,
    end_time: datetime,
):
    """
    Run quant research for a trading pair.
    """

    # run quant research based on the symbol
    if isinstance(symbol, BinanceSymbols):

        # parse the interval
        interval = ChartIntervals(interval)

        # backtest binance
        stats = quant_research_binance(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    elif isinstance(symbol, UpstoxSymbols):

        # parse the interval
        interval = UpstoxIntervals(interval)

        # backtest upstox
        stats = quant_research_upstox(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    else:
        raise ValueError("Unsupported symbol type")

    # print the best trial
    from pprint import pprint

    pprint(vars(stats))
