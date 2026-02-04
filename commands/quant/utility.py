"""
This module contains the utility functions for the quant research command.
"""

# 1st party imports
from enum import Enum
from typing import Any, Callable

# 3rd party imports
import pandas as pd
from optuna import Trial

# local imports
from brokers import PaperTrader, TradeType
from strategies.models import TradeAction
from strategies import SupertrendStrategy, SupertrendStrategyConfig


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
                return paper_trader.stats().success_rate

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
                return paper_trader.stats().success_rate

    # return only the profit
    return paper_trader.stats().success_rate


def create_research(
    symbol: Enum,
    interval: Any,
    dataframe: pd.DataFrame,
) -> Callable[[Trial], float]:
    """
    Create a research function for optimizing trading strategy parameters.

    Args:
        symbol: The trading symbol to research.
        interval: The chart interval for candlestick data.
        paper_trader: The paper trader instance for simulating trades.
        dataframe: DataFrame containing historical price

    Returns:
        A function that takes a trial object and returns the profit from backtesting.
    """

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
        factor = trial.suggest_int("factor", 1, 5)
        atr_period = trial.suggest_int("atr_period", 10, 30)
        fractal_period = trial.suggest_int("fractal_period", 2, 15)

        # adx params
        adx_period = trial.suggest_int("adx_period", 10, 30)
        adx_smoothing = trial.suggest_int("adx_smoothing", 10, 30)
        adx_threshold = trial.suggest_int("adx_threshold", 10, 30)

        # ema params
        ema_1_period = trial.suggest_int("ema_1_period", 8, 30)
        ema_2_period = trial.suggest_int("ema_2_period", 10, 50)
        ema_3_period = trial.suggest_int("ema_3_period", 20, 100)

        # bollinger bands params
        bbands_period = trial.suggest_int("bbands_period", 5, 40)

        # rsi params
        rsi_period = trial.suggest_int("rsi_period", 5, 20)

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
