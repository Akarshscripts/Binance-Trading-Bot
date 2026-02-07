"""
This module contains the core logic for the quant research command.
"""

# 1st party imports
import os
import json
import logging
from typing import Any, List
from datetime import datetime

# 3rd party imports
from dotenv import load_dotenv
from optuna import create_study
from optuna.trial import FrozenTrial

# local imports
from .utility import (
    create_research,
    create_time_windows,
    fetch_symbol_interval_and_data,
    create_trial_scoring_callback,
)

# create the logger
logger = logging.getLogger(__name__)


def run_quant_research(
    symbol: Any,
    interval: Any,
    start_time: datetime,
    end_time: datetime,
    time_windows: int,
    n_trials: int,
    n_jobs: int,
):
    """
    Run quant research for a trading pair.

    Args:
        symbol: The trading symbol to backtest (e.g., BTCUSDT, TATA_STEEL).
        interval: The chart interval for candlestick data.
        start_time: Start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        end_time: End time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        time_windows: The number of time windows to create for walk forward validation.
        n_trials: The number of trials to run.
        n_jobs: The number of jobs to run in parallel.
    """

    logger.info(
        "Starting quant research for %s %s (windows=%s, trials=%s, jobs=%s)",
        symbol,
        interval,
        time_windows,
        n_trials,
        n_jobs,
    )

    # load the environment variables
    load_dotenv()
    storage_url = os.getenv("OPTUNA_STORAGE_URL")
    if not storage_url:
        raise ValueError(
            "OPTUNA_STORAGE_URL is not set. Please add it to your environment or .env file."
        )

    # fetch the data
    parsed_symbol, parsed_interval, dataframe = fetch_symbol_interval_and_data(
        symbol, interval, start_time, end_time
    )
    logger.info(
        "Fetched %s data from '%s' to '%s' for %s - %s.",
        len(dataframe),
        start_time,
        end_time,
        parsed_symbol,
        parsed_interval,
    )

    # create the time windows
    time_windows = create_time_windows(dataframe=dataframe, time_windows=time_windows)
    logger.info(
        "Generated %s time windows for %s - %s candles.",
        len(time_windows),
        len(dataframe),
        interval,
    )

    # search for all the available time windows
    final_best_trials: List[FrozenTrial] = []
    seed_trials: List[FrozenTrial] = []
    for idx, (train_df, val_df) in enumerate(time_windows):

        logger.info(
            "Window %s: train=%s, val=%s",
            idx,
            len(train_df),
            len(val_df),
        )

        # create a research
        research = create_research(
            symbol=parsed_symbol,
            interval=parsed_interval,
            dataframe=train_df,
            params_config=None,
        )

        # create a study
        study = create_study(
            direction="maximize",
            study_name=f"Quant Research {parsed_symbol.value} {parsed_interval.value} - {idx}",
            storage=storage_url,
            load_if_exists=True,
        )

        # enqueue the previous window's best trials as warm starts
        if seed_trials:
            logger.info("Window %s: enqueuing %s seed trials", idx, len(seed_trials))
            for trial in seed_trials:
                if trial.params:
                    study.enqueue_trial(trial.params)

        # calculate the remaining trials
        remaining_trials = n_trials - len(study.trials)

        # create a callback to validate and sort best trials
        callback, scored_trials = create_trial_scoring_callback(
            symbol=parsed_symbol,
            interval=parsed_interval,
            dataframe=val_df,
        )

        # run the study
        study.optimize(
            func=research,
            n_trials=remaining_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            callbacks=[callback],
        )
        logger.info("Window %s: study completed with %s trials", idx, len(study.trials))

        # update the run config with the params and new boundries
        final_best_trials = scored_trials.snapshot(sort=True)
        seed_trials = final_best_trials
        logger.info(
            "Window %s: selected %s trials for boundary update. Score of best trial: %s",
            idx,
            len(final_best_trials),
            final_best_trials[0].value,
        )

    # dump the top 5 trial params from the final window
    top_trials_payload = [
        {"params": trial.params, "trial_score": trial.value}
        for trial in final_best_trials[:5]
    ]
    with open("top_trials.json", "w", encoding="utf-8") as file:
        json.dump(top_trials_payload, file, indent=2)
    logger.info("Saved %s trial params to top_trials.json", len(top_trials_payload))
