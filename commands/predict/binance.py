"""
This module contains the core logic for finding signals on Binance symbols via WebSocket.
"""

# 1st party imports
import json
import logging
import time
import pytz
from datetime import datetime
from pathlib import Path

# 3rd party imports
import pandas as pd
import websocket

# local imports
from messenger import Messenger
from binance_api import BinanceSymbols, BinanceExchange, ChartIntervals
from strategies import Strategy, SupertrendStrategy, SupertrendStrategyConfig, TradeAction

# get the logger
logger = logging.getLogger("predict.binance")


def predict_binance(
    symbol: BinanceSymbols,
    interval: ChartIntervals,
    discord_webhook: str,
    time_zone: str = "Asia/Kolkata",
):
    """
    Stream live Binance candles via WebSocket and post trade signals to Discord.

    Args:
        symbol: The Binance trading symbol (e.g., BTCUSDT).
        interval: The chart interval for candlestick data.
        discord_webhook: Discord webhook URL for sending trade signals.
        time_zone: IANA timezone name for display timestamps.
    """

    # create config
    strategy_config = SupertrendStrategyConfig(
        factor=3,
        atr_period=10,
        fractal_period=7,
        risk_reward_ratio=2,
    )

    # instances
    binance_api = BinanceExchange()
    discord = Messenger(webhook_url=discord_webhook)
    strategy: Strategy = SupertrendStrategy(config=strategy_config)
    local_tz = pytz.timezone(time_zone)

    # create a jsonl file to log trades
    trades_jsonl = Path("trades.jsonl")
    trades_jsonl.touch()

    # seed historical data via HTTP before opening WebSocket
    logger.info("Seeding historical data for %s %s...", symbol.value, interval.value)
    df = binance_api.get_symbol_info(symbol=symbol, interval=interval)
    df = df.drop(df.index[-1])  # drop the live (incomplete) candle
    logger.info("Seeded %d candles.", len(df))

    # send startup message
    startup_msg = f"Starting WebSocket predictions for {symbol.value} on {interval.value} interval."
    logger.info(startup_msg)
    discord.send_text_message(startup_msg)

    # build WebSocket URL: wss://stream.binance.com:9443/ws/<symbol>@kline_<interval>
    interval_str = f"{interval.value.time_value}{interval.value.time_unit}"
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol.value.lower()}@kline_{interval_str}"

    def on_open(ws):
        logger.info("Connected to Binance WebSocket stream: %s", ws_url)

    def on_message(ws, message):
        nonlocal df

        data = json.loads(message)
        kline = data["k"]

        # only process closed candles
        if not kline["x"]:
            return

        # build new row from WebSocket kline payload
        ts = datetime.fromtimestamp(kline["t"] / 1000, tz=pytz.utc).astimezone(local_tz)
        new_row = pd.DataFrame([{
            "timestamp": ts,
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }])

        # append to rolling window and cap at 600 candles
        df = pd.concat([df, new_row], ignore_index=True).tail(600)

        logger.info(
            "Closed candle: %s | close=%.4f",
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            float(kline["c"]),
        )

        # run strategy
        predictions = strategy.process_candles(
            open_prices=df["open"],
            high_prices=df["high"],
            low_prices=df["low"],
            close_prices=df["close"],
            volumes=df["volume"],
        )

        # send Discord signal on non-neutral action
        if predictions.action != TradeAction.NEUTRAL:
            msg = (
                f"Strategy: **{predictions.action.value}**. Pair: **{symbol.value}**.\n"
                f"Time: **{ts.strftime('%Y-%m-%d %H:%M:%S')}**.\n"
                f"Entry Price: **{predictions.entry_price:.4f}**.\n"
                f"Stop Loss: **{predictions.stop_loss:.4f}**.\n"
                f"Take Profit: **{predictions.exit_price:.4f}**."
            )
            discord.send_text_message(msg)
            logger.info(msg)

            with trades_jsonl.open("a") as f:
                f.write(json.dumps(predictions.model_dump(mode="json")) + "\n")

        logger.info("-" * 20 + " Cycle Complete " + "-" * 20)

    def on_error(ws, error):
        logger.error("WebSocket error: %s", error)

    def on_close(ws, close_status_code, close_msg):
        logger.warning(
            "WebSocket closed (code=%s): %s", close_status_code, close_msg
        )

    # connect with automatic reconnect on disconnect
    while True:
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever()
        logger.warning("WebSocket disconnected. Reconnecting in 5 seconds...")
        time.sleep(5)
