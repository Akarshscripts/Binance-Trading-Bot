"""
This module contains the core logic for the finding signals of upstox symbols.
"""

# 1st party imports
import pytz
import json
import time
import logging
from pathlib import Path

# 3rd party imports
from pandas import DataFrame
from datetime import datetime

# local imports
from messenger import Messenger
from upstox_api import UpstoxSymbols, UpstoxExchange, UpstoxIntervals
from strategies import SupertrendStrategy, SupertrendStrategyConfig, TradeAction

# get the logger
logger = logging.getLogger("predict.upstox")


def predict_upstox(
    symbol: UpstoxSymbols,
    interval: UpstoxIntervals,
    discord_webhook: str,
    time_zone: str = "Asia/Kolkata",
    lag: int = 30,
):
    """
    Find signals for a trading strategy on historical Upstox data.

    Args:
        symbol: The Upstox trading symbol to backtest (e.g., TATA_STEEL).
        interval: The chart interval for candlestick data.
        discord_webhook: The Discord webhook URL for sending predictions.
        time_zone: The timezone for time conversions.
        lag: Seconds to wait after interval ends before fetching (default: 30).
    """

    # create config
    strategy_config = SupertrendStrategyConfig(
        factor=3,
        atr_period=10,
        fractal_period=7,
        risk_reward_ratio=2,
    )

    # instances
    upstox_api = UpstoxExchange()
    discord = Messenger(webhook_url=discord_webhook)
    strategy = SupertrendStrategy(config=strategy_config)

    # time variables
    curr_time = None
    next_fetch_at = interval.get_next_fetch_time(lag=lag)
    local_next_fetch_at = datetime.fromtimestamp(
        next_fetch_at, tz=pytz.timezone(time_zone)
    )

    # create a json file to dump trades
    trades_jsonl = Path("trades.jsonl")
    trades_jsonl.touch()

    # send the startup message
    startup_msg = f"Starting predictions for {symbol.name} on {interval.value} interval. Next fetch at: {local_next_fetch_at.strftime('%Y-%m-%d %H:%M:%S')}."
    logger.info(startup_msg)
    discord.send_text_message(startup_msg)

    # main loop
    while True:

        # check if it is time to fetch
        curr_time = time.time()  # UTC timestamp
        sleep_for = next_fetch_at - curr_time
        if sleep_for > 0:
            time.sleep(sleep_for)

        # fetch the data
        logger.info(
            f"Fetching data for {symbol.value} on {interval.value} interval. time: {datetime.fromtimestamp(curr_time, tz=pytz.timezone(time_zone)).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        upstox_df = upstox_api.get_symbol_info(
            symbol=symbol,
            interval=interval,
            start_time=datetime.now(),
        )

        # calculate the next fetch time
        next_fetch_at = interval.get_next_fetch_time(lag=lag)
        local_next_fetch_at = datetime.fromtimestamp(
            next_fetch_at, tz=pytz.timezone(time_zone)
        )

        # remove the last row as it is not completed yet
        upstox_df: DataFrame = upstox_df.drop(upstox_df.index[-1])

        # make predictions
        predictions = strategy.process_candles(
            open_prices=upstox_df["open"],
            high_prices=upstox_df["high"],
            low_prices=upstox_df["low"],
            close_prices=upstox_df["close"],
            volumes=upstox_df["volume"],
        )
        logger.info(
            f"Latest candle: {upstox_df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. Latest Candle Close: {upstox_df.iloc[-1]['close']:.4f}"
        )

        # if the strategy is not neutral, send the message
        if predictions.action != TradeAction.NEUTRAL:

            # prepare msg
            msg = f"Strategy: **{predictions.action}**.\nPair: **{symbol.name}**.\nTime: **{upstox_df.iloc[-1]["timestamp"].strftime('%Y-%m-%d %H:%M:%S')}**.\nEntry Price: **{predictions.entry_price:.4f}**.\n Stop Loss: **{predictions.stop_loss:.4f}**.\n Take Profit: **{predictions.exit_price:.4f}**."
            discord.send_text_message(msg)
            logger.info(msg)

            # dump the trade to jsonl
            with trades_jsonl.open("a") as f:
                f.write(json.dumps(predictions.model_dump(mode="json")) + "\n")

        # print cycle complete message
        cycle_complete_msg = "-" * 20 + " Cycle Complete " + "-" * 20
        logger.info(cycle_complete_msg)
