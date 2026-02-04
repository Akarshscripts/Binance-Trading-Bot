"""
This module contains the core logic for the quant research command for binance.
"""

# 1st party imports
import logging
from datetime import datetime

# 3rd party imports
from optuna import create_study

# local imports
from .utility import create_research
from binance_api import ChartIntervals, BinanceSymbols, BinanceExchange, TimeZones

# create the logger
logger = logging.getLogger(__name__)


def quant_research_binance(
    symbol: BinanceSymbols,
    interval: ChartIntervals,
    start_time: datetime,
    end_time: datetime,
):
    """
    Run quant research for a trading pair on Binance exchange.

    This function performs hyperparameter optimization using Optuna to find
    the best trading strategy configuration for the given symbol and interval.

    Args:
        symbol: The Binance trading symbol to research (e.g., BTCUSDT).
        interval: The chart interval for candlestick data.
        start_time: The start time for fetching historical data.
        end_time: The end time for fetching historical data.

    Returns:
        The statistics from the best performing strategy configuration.
    """

    # fetch the data from binance
    binance_api = BinanceExchange(timezone=TimeZones.INDIA)

    # fetch the data from start to end
    logger.info(
        f"Fetching data for {symbol.value} from {start_time} to {end_time} for {interval.value} interval."
    )
    binance_df = binance_api.get_symbol_info(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
    )
    logger.info(f"Fetched {len(binance_df)} candles for Quant Research.")

    # create a research
    research = create_research(
        symbol=symbol,
        interval=interval,
        dataframe=binance_df,
    )

    # create a study
    study = create_study(
        direction="maximize",
        study_name=f"Quant Research {symbol.value} {interval.value}",
    )

    # run the study
    study.optimize(research, n_trials=500, n_jobs=1, show_progress_bar=True)

    # return the best trial
    return study.best_trial
