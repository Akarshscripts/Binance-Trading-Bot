"""
This module contains the core logic for the predict command.
"""

# 1st party imports
import time
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Literal

# 3rd party imports
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# local imports
from .nn_model import NnModel
from settings import Constants
from messenger import Messenger
from dataframe_handler import DataManager
from brokers import PaperTrader, TradeType
from binance_api import BinanceSymbols, ChartIntervals, BinanceApi, TimeZones
from binance_api.indicators import Indicator, ADX, RSI, EMA, FractalsIndicator


def __create_model_inputs(
    dataframe: pd.DataFrame, indicators: Optional[List[Indicator]] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create model inputs from a dataframe by applying indicators and preprocessing.

    Args:
        dataframe: The raw dataframe containing OHLCV data.
        indicators: Optional list of technical indicators to apply.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (X, Y) tensors ready for model inference.
    """

    # create the data manager
    dm = DataManager(
        device=Constants.DEVICE,
        sequence_length=Constants.SEQUENCE_LENGTH,
        df=dataframe.copy(),
    )

    # process the data
    dm.preprocess(
        indicators=indicators,
        threshold=Constants.LABEL_THRESH,
        look_ahead=Constants.LOOK_AHEAD_CANDLES,
    )

    # scale the whole data
    dm.scale(
        cols=Constants.SCALABLE_COLS,
        log_cols=Constants.COLS_TO_SCALE_LOG,
        scale_whole=True,
    )

    # get the tensor inputs
    tensorX, tensorY = dm.get_tensors(Constants.FEATURE_COLS, Constants.TARGET_COL)

    # check if the tensorX is empty
    if tensorX.shape[0] == 0:
        return None

    # return the tensors for the last batch only
    return tensorX[-1].unsqueeze(0), tensorY[-1].unsqueeze(0)


def trading_strategy(df: pd.DataFrame, n: int = 5) -> str:
    """
    Calculate 20 and 50 EMA and determine trend direction based on crossover.

    Args:
        df: DataFrame containing price data with 'close' column.
        n: Number of past candles to check for crossover.

    Returns:
        str: 'up' if 20 EMA crossed above 50 EMA, 'down' if crossed below, 'neutral' otherwise.
    """

    # create emas
    ema_20 = EMA(20)
    ema_50 = EMA(50)

    # Calculate EMAs
    ema_20_vals = ema_20.get_value(df["close"])
    ema_50_vals = ema_50.get_value(df["close"])

    # Get the difference between EMAs for the last n+1 candles
    diff = ema_20_vals - ema_50_vals
    recent_diff = diff[-(n + 1) :]

    # Check for crossover in the past n candles
    for i in range(1, len(recent_diff)):
        prev_diff = recent_diff[i - 1]
        curr_diff = recent_diff[i]

        # Bullish crossover: 20 EMA crosses above 50 EMA
        if prev_diff <= 0 and curr_diff > 0:
            return "up"

        # Bearish crossover: 20 EMA crosses below 50 EMA
        if prev_diff >= 0 and curr_diff < 0:
            return "down"

    # return neutral if no crossover
    return "neutral"


def apply_strategies(
    df: pd.DataFrame, model_output: Literal["neutral", "up", "down"]
) -> Tuple[bool, str]:
    """
    Apply trading strategies to validate model output.

    Args:
        df: DataFrame containing price data for strategy calculations.
        model_output: The model's prediction output ('neutral', 'up', or 'down').

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if the strategy agrees with the model output, False otherwise.
            - str: A message describing the strategy and model output comparison.
    """

    # apply the 20 and 50 EMA strategy
    ema_strategy = trading_strategy(df, Constants.LOOK_AHEAD_CANDLES * 2)

    # check if the ema strategy is neutral
    if ema_strategy == "neutral" or model_output == "neutral":
        return False, ""

    # create msg
    msg = f"EMA Strategy: **{ema_strategy}**.\nModel Output: **{model_output}**.\nTime: **{df.iloc[-1]["timestamp"].strftime('%Y-%m-%d %H:%M:%S')}**.\nCandle Close Price: **{df.iloc[-1]["close"]:.4f}**.\n"

    # return the final strategy
    return ema_strategy != model_output, msg


def start_predictions(
    symbol: BinanceSymbols,
    interval: ChartIntervals,
    time_zone: TimeZones,
    model_path: str,
    discord_webhook: str,
    lag: int = 30,
):
    """
    Start tracking the crypto pair and make predictions.
    """

    # instances
    binance = BinanceApi(timezone=time_zone)
    nn_model = NnModel(model_path, Constants.DEVICE)
    discord = Messenger(webhook_url=discord_webhook)

    # send the startup message
    discord.send_text_message(
        f"Starting predictions for {symbol.value} on {interval.value} interval."
    )

    # indicators
    indicators = [EMA(20), EMA(50), RSI(14), ADX(14)]

    # time variables
    curr_time = None
    last_success_fetched_at = None

    # main loop
    while True:
        # check if it is time to fetch
        curr_time = time.time()
        if last_success_fetched_at is not None:

            # calculate the next fetch time
            next_fetch_at = last_success_fetched_at + interval.value + lag

            # sleep for the remaining time
            sleep_for = next_fetch_at - curr_time

            # sleep if needed
            if sleep_for > 0:
                time.sleep(sleep_for)

        # fetch the data
        binance_df = binance.get_symbol_info(
            symbol=symbol,
            interval=interval,
        )

        # create model inputs
        x_input, _ = __create_model_inputs(dataframe=binance_df, indicators=indicators)
        if not x_input:

            # update the last success fetched at
            last_success_fetched_at = curr_time

            # send the error message
            discord.send_text_message(
                f"Error fetching latest data for {symbol.value} on {interval.value} interval. Df shape: {binance_df.shape}"
            )
            continue

        # make predictions
        predictions = nn_model.get_outputs(x_input)

        # convert the predictions to labels
        output = nn_model.convert_preds(predictions)

        # merge with a custom startegy and create output message
        send_msg, msg = apply_strategies(binance_df, output)

        # if send_msg is true, send the message
        if send_msg:
            discord.send_text_message(msg)
            print(msg)
        else:
            print("Strategy did not agree with model output.")

        # update the last success fetched at
        last_success_fetched_at = curr_time


def backtest_predictions(
    symbol: BinanceSymbols,
    interval: ChartIntervals,
    time_zone: TimeZones,
    model_path: str,
    discord_webhook: str,
    start_time: str,
    end_time: str,
    risk_to_reward: float,
):
    """
    Backtest predictions for a given symbol and interval.

    Args:
        symbol: The trading symbol to backtest (e.g., BTCUSDT).
        interval: The chart interval for candlestick data.
        time_zone: The timezone for time conversions.
        model_path: Path to the trained model checkpoint.
        discord_webhook: Discord webhook URL for sending predictions.
        start_time: Start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        end_time: End time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        risk_to_reward: Risk to reward ratio for backtesting.
    """

    # instances
    paper_trader = PaperTrader()
    binance = BinanceApi(timezone=time_zone)
    nn_model = NnModel(model_path, Constants.DEVICE)
    discord = Messenger(webhook_url=discord_webhook)

    # convert the start and end time to datetime
    start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")
    assert end_time - start_time >= timedelta(
        days=5
    ), "The time gap b/w start and end date should be greater than 5 days."

    # send the startup message
    msg = f"Starting Backtesting for {symbol.value} on {interval.value} interval.\nStart Time: {start_time}\nEnd Time: {end_time}\n"
    discord.send_text_message(msg)

    # indicators
    indicators = [EMA(20), EMA(50), RSI(14), ADX(14)]
    fractals = FractalsIndicator()

    # fetch the data from start to end
    binance_df = binance.get_symbol_info(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
    )

    # main loop
    for idx in tqdm(
        range(Constants.SEQUENCE_LENGTH, len(binance_df)),
        total=len(binance_df) - Constants.SEQUENCE_LENGTH,
    ):

        # pass the new candle info to paper trader
        paper_trader.update(
            low=binance_df.iloc[idx]["low"],
            high=binance_df.iloc[idx]["high"],
            current_candle_timestamp=binance_df.iloc[idx]["timestamp"],
        )

        # create model inputs
        curr_df = binance_df.iloc[:idx]
        inputs = __create_model_inputs(dataframe=curr_df, indicators=indicators)
        if not inputs:
            continue

        # make predictions
        predictions = nn_model.get_outputs(inputs[0])

        # convert the predictions to labels
        output = nn_model.convert_preds(predictions)

        # merge with a custom startegy and create output message
        action_allowed, msg = apply_strategies(curr_df, output)

        # if action_allowed, take a action
        if action_allowed:

            # get the data
            entry_price = binance_df.iloc[idx]["close"]
            trade_start_timestamp = binance_df.iloc[idx]["timestamp"]
            candle_highs = binance_df.iloc[:idx]["high"].values
            candle_lows = binance_df.iloc[:idx]["low"].values

            # get the fractal stop loss
            top_sl, bottom_sl = fractals.get_value(high=candle_highs, low=candle_lows)

            # check trade side
            if output == "up":

                # set trade type enum
                trade_type = TradeType.LONG

                # get the valid SL
                valid_sl = bottom_sl[~np.isnan(bottom_sl)]
                if len(valid_sl) == 0 or valid_sl[-1] >= entry_price:
                    continue
                stop_loss = valid_sl[-1]

                # calculate the exit price
                exit_price = entry_price + (
                    abs(entry_price - stop_loss) * risk_to_reward
                )
            else:

                # set trade type enum
                trade_type = TradeType.SHORT

                # get the valid SL
                valid_sl = top_sl[~np.isnan(top_sl)]
                if len(valid_sl) == 0 or valid_sl[-1] <= entry_price:
                    continue
                stop_loss = valid_sl[-1]

                # calculate the exit price
                exit_price = entry_price - (
                    abs(entry_price - stop_loss) * risk_to_reward
                )

            # open the position
            paper_trader.open_position(
                symbol=symbol,
                interval=interval,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=round(stop_loss, 4),
                trade_type=trade_type,
                trade_start_timestamp=trade_start_timestamp,
            )

    # get the stats
    stats = paper_trader.stats()

    # print the stats
    from pprint import pprint

    pprint(stats.model_dump(mode="json", indent=4))
