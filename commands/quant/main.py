"""
This module contains the core logic for the quant research command.
"""

# 1st party imports
import os
import json
import logging
from typing import Any, List, Tuple
from datetime import datetime

# 3rd party imports
from dotenv import load_dotenv
from optuna import create_study
from optuna.trial import FrozenTrial
from optuna.storages import RDBStorage

# local imports
from .utility import (
    create_research,
    hydrate_scores_from_study,
    split_dataframe_into_windows,
    fetch_symbol_interval_and_data,
    create_trial_scoring_callback,
    score_trial_params_on_window,
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

    # split the dataframe into windows
    windows = split_dataframe_into_windows(
        dataframe=dataframe, time_windows=time_windows
    )
    logger.info(
        "Generated %s time windows for %s - %s candles.",
        len(windows),
        len(dataframe),
        interval,
    )

    # separate the last window and preceding windows
    last_window = windows[-1]
    preceding_windows = windows[:-1]
    if not preceding_windows:
        logger.warning("No preceding windows available for cross-validation")
        raise ValueError("No preceding windows available for cross-validation")

    # --- Phase 1: run Optuna study on the last window ---
    logger.info("Phase 1: optimising on last window (%s candles)", len(last_window))
    research = create_research(
        symbol=parsed_symbol,
        interval=parsed_interval,
        dataframe=last_window,
        params_config=None,
    )

    # create a study
    storage = RDBStorage(
        url=storage_url,
        engine_kwargs={
            "pool_pre_ping": True,
            "pool_recycle": 300,
        },
    )
    study = create_study(
        direction="maximize",
        study_name=f"Quant Research {parsed_symbol.value} {parsed_interval.value}",
        storage=storage,
        load_if_exists=True,
    )

    # create a callback to validate and sort best trials
    callback, scored_trials = create_trial_scoring_callback(
        symbol=parsed_symbol,
        interval=parsed_interval,
        dataframe=last_window,
    )

    # hydrate the scores from storage
    hydrate_scores_from_study(scored_trials, study)
    hydrated_scores = scored_trials.snapshot(sort=False)
    if hydrated_scores:
        logger.info("Hydrated %s scored trials from storage", len(hydrated_scores))

    # run the study
    completed_count = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    remaining_trials = max(0, n_trials - completed_count)
    if remaining_trials > 0:
        study.optimize(
            func=research,
            n_trials=remaining_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            callbacks=[callback],
        )
        logger.info("Study completed with %s total trials", len(study.trials))
    else:
        logger.info(
            "No remaining trials to run (%s already in storage)", len(study.trials)
        )

    # collect all completed trials from the study
    completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed_trials:
        logger.warning("No completed trials found; cannot cross-validate.")
        return

    logger.info(
        "Phase 1 done. %s completed trials to cross-validate.", len(completed_trials)
    )

    # --- Phase 2: cross-validate each trial on all preceding windows ---
    logger.info(
        "Phase 2: cross-validating %s trials across %s preceding windows",
        len(completed_trials),
        len(preceding_windows),
    )

    # rank each trial on all preceding windows
    trial_scores: List[Tuple[FrozenTrial, float]] = []
    for trial in completed_trials:

        # score of current trial on all preceding windows
        window_scores = []

        # score each window
        for win_idx, window in enumerate(preceding_windows):

            score = score_trial_params_on_window(
                trial=trial,
                symbol=parsed_symbol,
                interval=parsed_interval,
                dataframe=window,
            )
            window_scores.append(score)
            logger.debug(
                "Trial #%s | window %s/%s | score=%.4f",
                trial.number,
                win_idx + 1,
                len(preceding_windows),
                score,
            )

        # append the trial and its average score across all windows
        trial_scores.append((trial, sum(window_scores) / len(window_scores)))

    # sort the trials by their average score
    trial_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_trials = [t for t, _ in trial_scores]
    logger.info(
        "Cross-validation done. Best average score: %.4f (trial #%s)",
        trial_scores[0][1],
        trial_scores[0][0].number,
    )

    # --- Phase 3: select top 10 and dump ---
    top_trials = ranked_trials[:10]
    top_trials_payload = [
        {"params": trial.params, "last_window_score": trial.value}
        for trial in top_trials
    ]

    # save the top trials to a json file
    file_name = f"top_trials_{parsed_symbol.value}_{parsed_interval.value}.json"
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(top_trials_payload, file, indent=2)
    logger.info("Saved top %s trial params to %s", len(top_trials_payload), file_name)
