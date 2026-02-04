"""
This module contains the core logic for the quant research command for upstox.
"""

# 1st party imports
import logging
from datetime import datetime

# 3rd party imports
from optuna import create_study

# local imports
from .utility import create_research
from upstox_api import UpstoxSymbols, UpstoxIntervals, UpstoxExchange

# create the logger
logger = logging.getLogger(__name__)


def quant_research_upstox(
    symbol: UpstoxSymbols,
    interval: UpstoxIntervals,
    start_time: datetime,
    end_time: datetime,
):
    """
    Run quant research for a trading pair on upstox.
    """

    # fetch the data from upstox
    upstox_api = UpstoxExchange()

    # fetch the data from start to end
    logger.info(
        f"Fetching data for {symbol.value} from {start_time} to {end_time} for {interval.value} interval."
    )
    upstox_df = upstox_api.get_symbol_info(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
    )
    logger.info(f"Fetched {len(upstox_df)} candles for Quant Research.")

    # create a research
    research = create_research(
        symbol=symbol,
        interval=interval,
        dataframe=upstox_df,
    )

    # create a study
    study = create_study(
        direction="maximize",
        study_name=f"Quant Research {symbol.value} {interval.value}",
    )

    # run the study
    study.optimize(research, n_trials=100, n_jobs=-1, show_progress_bar=True)

    # return the best trial
    return study.best_trial
