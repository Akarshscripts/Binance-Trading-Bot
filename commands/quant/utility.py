"""
This module contains the utility functions for the quant research command.
"""

# 1st party imports
import statistics
from enum import Enum
from threading import Lock
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Union

# 3rd party imports
import pandas as pd
from optuna import Trial
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

# local imports
from brokers import PaperTrader, TradeType
from strategies.models import TradeAction
from strategies import SupertrendStrategy, SupertrendStrategyConfig
from upstox_api import UpstoxSymbols, UpstoxIntervals, UpstoxExchange
from binance_api import ChartIntervals, BinanceSymbols, BinanceExchange, TimeZones


class TrialScoreStore:
    """Thread-safe container for scored trials."""

    def __init__(self) -> None:
        """Initialize an empty thread-safe trial score store."""
        self._lock = Lock()
        self._items: List[Tuple[FrozenTrial, float]] = []
        self._seen: set[int] = set()

    @staticmethod
    def _trial_key(trial: FrozenTrial) -> int:
        if getattr(trial, "number", None) is not None:
            return int(trial.number)
        if hasattr(trial, "_trial_id"):
            return int(trial._trial_id)
        return id(trial)

    def add(self, trial: FrozenTrial, score: float) -> None:
        """
        Add a trial and its score to the store.

        Args:
            trial: The frozen trial to store.
            score: The validation score for the trial.
        """
        with self._lock:
            key = self._trial_key(trial)
            if key in self._seen:
                return
            self._items.append((trial, score))
            self._seen.add(key)

    def snapshot(self, sort: bool = True) -> List[Tuple[FrozenTrial, float]]:
        """
        Return a thread-safe copy of all stored trials and scores sorted by score (highest first).

        Returns:
            A list of (trial, score) tuples.
        """
        with self._lock:
            if sort:
                return sorted(list(self._items), key=lambda item: item[1], reverse=True)
            return list(self._items)


def test_research(
    symbol: Enum,
    interval: Any,
    dataframe: pd.DataFrame,
    strategy: SupertrendStrategy,
    paper_trader: PaperTrader,
) -> float:
    """
    Test the research for a trading pair.

    Args:
        symbol: The trading symbol to backtest (e.g., BTCUSDT).
        interval: The chart interval for candlestick data.
        dataframe: DataFrame containing historical candlestick data.
        strategy: The trading strategy instance to use for generating signals.
        paper_trader: The paper trader instance for simulating trades.

    Returns:
        float: The success rate from the test.
    """

    # main loop
    n = len(dataframe)
    initial_investment = paper_trader.capital
    for idx in range(strategy.config.minimum_history, n):

        # get the data up to current candle (excluding incomplete current candle)
        current_df = dataframe.iloc[: idx + 1].copy()

        # make predictions using process_candles
        predictions = strategy.process_candles(
            open_prices=current_df["open"],
            high_prices=current_df["high"],
            low_prices=current_df["low"],
            close_prices=current_df["close"],
            volumes=current_df["volume"],
        )

        # pass the new candle info to paper trader
        paper_trader.update(
            low=dataframe.iloc[idx]["low"],
            high=dataframe.iloc[idx]["high"],
            current_candle_timestamp=dataframe.iloc[idx]["timestamp"],
        )

        # if no action, continue
        if predictions.action == TradeAction.NEUTRAL:
            continue

        # if buy action
        if predictions.action == TradeAction.ENTER_LONG:

            try:
                # open the position
                paper_trader.open_position(
                    symbol=symbol,
                    interval=interval,
                    entry_price=predictions.entry_price,
                    exit_price=predictions.exit_price,
                    stop_loss=predictions.stop_loss,
                    trade_type=TradeType.LONG,
                    risk_reward_ratio=predictions.risk_reward_ratio,
                    trade_start_timestamp=dataframe.iloc[idx]["timestamp"],
                )

            # if value error, stop the loop
            except ValueError:
                break

        # if sell action
        elif predictions.action == TradeAction.ENTER_SHORT:

            try:
                # open the position
                paper_trader.open_position(
                    symbol=symbol,
                    interval=interval,
                    entry_price=predictions.entry_price,
                    exit_price=predictions.exit_price,
                    stop_loss=predictions.stop_loss,
                    trade_type=TradeType.SHORT,
                    risk_reward_ratio=predictions.risk_reward_ratio,
                    trade_start_timestamp=dataframe.iloc[idx]["timestamp"],
                )

            # if value error, stop the loop
            except ValueError:
                break

    # calculate sharpe ratio using per-trade returns
    if initial_investment <= 0:
        return 0.0

    # calculate returns/trade
    returns = []
    for trade in paper_trader.closed:
        pnl = trade.profit - trade.loss - trade.total_brokerage
        returns.append(pnl / initial_investment)

    # if no returns, return 0
    if not returns:
        return 0.0

    # calculate average and standard deviation
    avg_return = statistics.mean(returns)
    std_return = statistics.pstdev(returns)
    if std_return == 0:
        return 0.0

    # calculate sharpe ratio
    sharpe_ratio = avg_return / std_return
    return sharpe_ratio


def create_trial_scoring_callback(
    *,
    symbol: Enum,
    interval: Any,
    dataframe: pd.DataFrame,
) -> Tuple[Callable[[Study, FrozenTrial], None], TrialScoreStore]:
    """
    Create a callback that evaluates each completed trial on the provided dataframe.

    Returns:
        Tuple containing the callback function and a list of (trial, score).
    """

    # create a thread-safe store
    scored_trials = TrialScoreStore()

    # params config
    required_params = {"factor", "atr_period", "fractal_period"}
    optional_params = {
        "adx_period",
        "adx_smoothing",
        "adx_threshold",
        "ema_1_period",
        "ema_2_period",
        "ema_3_period",
        "bbands_period",
        "rsi_period",
    }

    def callback(study: Study, trial: FrozenTrial) -> None:
        """
        Callback function to evaluate a completed trial on validation data.

        This callback is invoked by Optuna after each trial completes. It validates
        the trial's hyperparameters, creates a strategy instance, and computes
        a validation score using the test_research function.

        Args:
            study: The Optuna study object.
            trial: The frozen trial that just completed.
        """

        # if trial is not complete, skip it
        if trial.state != TrialState.COMPLETE:
            return

        # if params is missing, skip the trial
        params = trial.params or {}
        missing = sorted(required_params - params.keys())
        if missing:
            return

        # construct the config
        config_values = {
            "factor": params["factor"],
            "atr_period": params["atr_period"],
            "fractal_period": params["fractal_period"],
            "risk_reward_ratio": 2,
        }
        for key in optional_params:
            if key in params:
                config_values[key] = params[key]

        # create the strategy and paper trader
        strategy_config = SupertrendStrategyConfig(**config_values)
        strategy = SupertrendStrategy(config=strategy_config)
        paper_trader = PaperTrader(
            brokerage=0.001,
            capital=10_000,
            risk_investment=0.02,
        )

        # test the research
        score = test_research(
            symbol=symbol,
            interval=interval,
            dataframe=dataframe,
            strategy=strategy,
            paper_trader=paper_trader,
        )
        scored_trials.add(trial, score)

        if getattr(trial, "number", None) is not None:
            study.set_user_attr(f"testing_score_{trial.number}", score)

    return callback, scored_trials


def hydrate_scores_from_study(scored_trials: TrialScoreStore, study: Study) -> None:
    """Load stored validation scores from Optuna into the score store."""

    study_attrs = study.user_attrs
    for trial in study.trials:
        score = trial.user_attrs.get("testing_score")
        if score is None and getattr(trial, "number", None) is not None:
            score = study_attrs.get(f"testing_score_{trial.number}")
        if score is None:
            continue
        scored_trials.add(trial, float(score))


def get_best_trials_from_scores(
    scored_trials: Union[TrialScoreStore, List[Tuple[FrozenTrial, float]]],
    top_k_percent: float = 0.2,
) -> List[FrozenTrial]:
    """
    Sort scored trials by validation score and return the top-performing trials.
    """

    scored_list = (
        scored_trials.snapshot()
        if isinstance(scored_trials, TrialScoreStore)
        else scored_trials
    )

    if not scored_list:
        return []

    sorted_trials = [
        trial
        for trial, _ in sorted(scored_list, key=lambda item: item[1], reverse=True)
    ]
    limit = int(len(sorted_trials) * top_k_percent)

    if top_k_percent >= 1:
        return sorted_trials
    return sorted_trials[:limit]


def create_research(
    symbol: Enum,
    interval: Any,
    dataframe: pd.DataFrame,
    params_config: Dict[str, Any] = None,
) -> Callable[[Trial], float]:
    """
    Create a research function for optimizing trading strategy parameters.

    Args:
        symbol: The trading symbol to research.
        interval: The chart interval for candlestick data.
        paper_trader: The paper trader instance for simulating trades.
        dataframe: DataFrame containing historical price.
        params_config: The strategy configuration to use for backtesting.

    Returns:
        A function that takes a trial object and returns the profit from backtesting.
    """

    # if params_config is None, initialize it as an empty dictionary
    default_params_config = {
        "factor": {"min": 1, "max": 5},
        "atr_period": {"min": 10, "max": 30},
        "fractal_period": {"min": 2, "max": 15},
        "adx_period": {"min": 10, "max": 30},
        "adx_smoothing": {"min": 10, "max": 30},
        "adx_threshold": {"min": 10, "max": 30},
        "ema_1_period": {"min": 8, "max": 30},
        "ema_2_period": {"min": 10, "max": 50},
        "ema_3_period": {"min": 20, "max": 100},
        "bbands_period": {"min": 5, "max": 40},
        "rsi_period": {"min": 5, "max": 20},
    }

    # if params_config is None, use default_params_config
    if params_config is None:
        params_config = default_params_config
    else:
        for key, bounds in params_config.items():
            if key not in default_params_config:
                raise ValueError(f"Invalid parameter config key: {key}")
            if not isinstance(bounds, dict):
                raise ValueError(
                    f"Parameter config for {key} must be a dict with 'min' and 'max'."
                )
            default_params_config[key].update(bounds)

        params_config = default_params_config

    def run_research(trial: Trial):
        """
        Execute a single trial of the optimization study.

        This function is called by Optuna to evaluate different hyperparameter
        combinations for the trading strategy.

        Args:
            trial: Optuna trial object used to suggest hyperparameter values.

        Returns:
            float: The profit achieved with the suggested hyperparameters.
        """

        # create the paper trader
        paper_trader = PaperTrader(
            brokerage=0.001,
            capital=10_000,
            risk_investment=0.02,
        )

        # supertrend params
        risk_reward_ratio = 2
        factor = trial.suggest_int(
            "factor", params_config["factor"]["min"], params_config["factor"]["max"]
        )
        atr_period = trial.suggest_int(
            "atr_period",
            params_config["atr_period"]["min"],
            params_config["atr_period"]["max"],
        )
        fractal_period = trial.suggest_int(
            "fractal_period",
            params_config["fractal_period"]["min"],
            params_config["fractal_period"]["max"],
        )

        # adx params
        adx_period = trial.suggest_int(
            "adx_period",
            params_config["adx_period"]["min"],
            params_config["adx_period"]["max"],
        )
        adx_smoothing = trial.suggest_int(
            "adx_smoothing",
            params_config["adx_smoothing"]["min"],
            params_config["adx_smoothing"]["max"],
        )
        adx_threshold = trial.suggest_int(
            "adx_threshold",
            params_config["adx_threshold"]["min"],
            params_config["adx_threshold"]["max"],
        )

        # ema params
        ema_1_period = trial.suggest_int(
            "ema_1_period",
            params_config["ema_1_period"]["min"],
            params_config["ema_1_period"]["max"],
        )
        ema_2_period = trial.suggest_int(
            "ema_2_period",
            params_config["ema_2_period"]["min"],
            params_config["ema_2_period"]["max"],
        )
        ema_3_period = trial.suggest_int(
            "ema_3_period",
            params_config["ema_3_period"]["min"],
            params_config["ema_3_period"]["max"],
        )

        # bollinger bands params
        bbands_period = trial.suggest_int(
            "bbands_period",
            params_config["bbands_period"]["min"],
            params_config["bbands_period"]["max"],
        )

        # rsi params
        rsi_period = trial.suggest_int(
            "rsi_period",
            params_config["rsi_period"]["min"],
            params_config["rsi_period"]["max"],
        )

        # create the trading strategy
        strategy_config = SupertrendStrategyConfig(
            factor=factor,
            atr_period=atr_period,
            fractal_period=fractal_period,
            adx_period=adx_period,
            adx_smoothing=adx_smoothing,
            adx_threshold=adx_threshold,
            ema_1_period=ema_1_period,
            ema_2_period=ema_2_period,
            ema_3_period=ema_3_period,
            bbands_period=bbands_period,
            rsi_period=rsi_period,
            risk_reward_ratio=risk_reward_ratio,
        )
        strategy = SupertrendStrategy(config=strategy_config)

        # run backtesting
        success_rate = test_research(
            symbol=symbol,
            interval=interval,
            dataframe=dataframe,
            strategy=strategy,
            paper_trader=paper_trader,
        )

        # return the success_rate
        return success_rate

    return run_research


def fetch_symbol_interval_and_data(
    symbol: Any, interval: Any, start_time: datetime, end_time: datetime
) -> Tuple[
    Union[BinanceSymbols, UpstoxSymbols],
    Union[ChartIntervals, UpstoxIntervals],
    pd.DataFrame,
]:
    """
    Parse the symbol and interval and fetch the data from the exchange.

    Args:
        symbol: The trading symbol to backtest (e.g., BTCUSDT, TATA_STEEL).
        interval: The chart interval for candlestick data.
        start_time: Start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        end_time: End time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.

    Returns:
        Tuple[Union[BinanceSymbols, UpstoxSymbols], Union[ChartIntervals, UpstoxIntervals], pd.DataFrame]: Tuple containing (symbol, interval, dataframe).
    """

    # fetch the data from binance
    if isinstance(symbol, BinanceSymbols):

        # parse the interval
        interval: ChartIntervals = ChartIntervals(interval)

        # fetch the data from binance
        binance_api = BinanceExchange(timezone=TimeZones.INDIA)

        # fetch the data from start to end
        dataframe = binance_api.get_symbol_info(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    # fetch the data from upstox
    elif isinstance(symbol, UpstoxSymbols):

        # parse the interval
        interval: UpstoxIntervals = UpstoxIntervals(interval)

        # fetch the data from upstox
        upstox_api = UpstoxExchange()

        # fetch the data from start to end
        dataframe = upstox_api.get_symbol_info(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    # raise error if symbol type is not supported
    else:
        raise NotImplementedError("Unsupported symbol type")
    return symbol, interval, dataframe


def create_time_windows(
    dataframe: pd.DataFrame, time_windows: int, train_ratio: float = 0.8
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time windows for walk forward validation.

    Args:
        dataframe: DataFrame containing historical price data.
        time_windows: Number of time windows to create.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: List of tuples containing (train_df, test_df) for each time window.
    """

    # calculate the split size
    n = len(dataframe)
    window_size = n // time_windows
    train_split_size = int(window_size * train_ratio)
    test_split_size = window_size - train_split_size

    # validate the window size
    if train_split_size < 1000 and test_split_size < 600:
        raise ValueError("Too small dataset expand the time range and try again.")

    # create the test and train splits
    splited_df = []
    for idx in range(time_windows):

        # calculate the window start and end
        window_start = idx * window_size
        train_end = window_start + train_split_size
        test_end = train_end + test_split_size

        # create the train and test dataframes
        train_df = dataframe.iloc[window_start:train_end].reset_index(drop=True)
        test_df = dataframe.iloc[train_end:test_end].reset_index(drop=True)
        splited_df.append((train_df, test_df))

    # return the splits
    return splited_df
