"""
This module contains the core logic for the predict command.
"""

# 1st party imports
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# 3rd party imports
from tqdm import tqdm

# local imports
from messenger import Messenger
from brokers import PaperTrader, TradeType
from strategies import SupertrendStrategy, TradeAction
from binance_api import BinanceSymbols, ChartIntervals, BinanceApi, TimeZones


def start_algorithm(
    symbol: BinanceSymbols,
    interval: ChartIntervals,
    time_zone: TimeZones,
    discord_webhook: str,
    lag: int = 30,
):
    """
    Start tracking a crypto pair and make real-time predictions.

    Args:
        symbol: The trading symbol to track (e.g., BTCUSDT).
        interval: The chart interval for candlestick data.
        time_zone: The timezone for time conversions.
        discord_webhook: The Discord webhook URL for sending predictions.
        lag: Seconds to wait after interval ends before fetching (default: 30).
    """

    # instances
    binance = BinanceApi(timezone=time_zone)
    discord = Messenger(webhook_url=discord_webhook)
    trade_strategy = SupertrendStrategy(3, 10, 7, 2)

    # send the startup message
    discord.send_text_message(
        f"Starting predictions for {symbol.value} on {interval.value} interval."
    )

    # time variables
    curr_time = None
    last_success_fetched_at = None

    # create a json file to dump trades
    trades_jsonl = Path("trades.jsonl")
    trades_jsonl.touch()

    # initialize strategy with past month's data
    discord.send_text_message("Initializing strategy with past month's data...")
    one_month_ago = datetime.now() - timedelta(days=30)
    historical_data = binance.get_symbol_info(
        symbol=symbol,
        interval=interval,
        start_time=one_month_ago,
        end_time=datetime.now(),
    )

    # feed historical data to strategy for initialization
    for idx in tqdm(range(len(historical_data)), desc="Initializing indicators"):
        trade_strategy.new_candle(
            open_price=historical_data.iloc[idx]["open"],
            high_price=historical_data.iloc[idx]["high"],
            low_price=historical_data.iloc[idx]["low"],
            close_price=historical_data.iloc[idx]["close"],
            volume=historical_data.iloc[idx]["volume"],
        )

    discord.send_text_message(
        "Strategy initialization complete. Starting real-time predictions..."
    )

    # main loop
    while True:

        # check if it is time to fetch
        curr_time = time.time()
        if last_success_fetched_at is not None:

            # calculate the next fetch time
            next_fetch_at = last_success_fetched_at + float(interval) + lag

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

        # make predictions
        predictions = trade_strategy.new_candle(
            open_price=binance_df.iloc[-1]["open"],
            high_price=binance_df.iloc[-1]["high"],
            low_price=binance_df.iloc[-1]["low"],
            close_price=binance_df.iloc[-1]["close"],
            volume=binance_df.iloc[-1]["volume"],
        )
        print("Latest candle:", binance_df.iloc[-1]["timestamp"])

        # if the strategy is not neutral, send the message
        if predictions.action != TradeAction.NEUTRAL:

            # prepare msg
            msg = f"Strategy: **{predictions.action}**.\nTime: **{binance_df.iloc[-1]["timestamp"].strftime('%Y-%m-%d %H:%M:%S')}**.\nEntry Price: **{predictions.entry_price:.4f}**.\n Stop Loss: **{predictions.stop_loss:.4f}**.\n Take Profit: **{predictions.exit_price:.4f}**."
            discord.send_text_message(msg)

            # dump the trade to jsonl
            with trades_jsonl.open("a") as f:
                f.write(json.dumps(predictions.model_dump(mode="json")) + "\n")

        # update the last success fetched at
        last_success_fetched_at = curr_time


def backtest_algorithm(
    symbol: BinanceSymbols,
    interval: ChartIntervals,
    time_zone: TimeZones,
    start_time: str,
    end_time: str,
    brokerage: float,
    initial_capital: float,
    risk_investment: float,
):
    """
    Backtest predictions for a given symbol and interval.

    Args:
        symbol: The trading symbol to backtest (e.g., BTCUSDT).
        interval: The chart interval for candlestick data.
        time_zone: The timezone for time conversions.
        start_time: Start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        end_time: End time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        brokerage: The brokerage to use for backtesting.
        initial_capital: The initial capital to use for backtesting.
        risk_investment: The risk investment to use for backtesting.
    """

    # instances
    paper_trader = PaperTrader(
        brokerage=brokerage,
        capital=initial_capital,
        risk_investment=risk_investment,
    )
    binance = BinanceApi(timezone=time_zone)
    trade_strategy = SupertrendStrategy(3, 10, 7, 2)

    # convert the start and end time to datetime
    start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")

    # fetch the data from start to end
    binance_df = binance.get_symbol_info(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
    )

    # main loop
    for idx in tqdm(range(len(binance_df))):

        # pass the new candle info to paper trader
        paper_trader.update(
            low=binance_df.iloc[idx]["low"],
            high=binance_df.iloc[idx]["high"],
            current_candle_timestamp=binance_df.iloc[idx]["timestamp"],
        )

        # make predictions
        predictions = trade_strategy.new_candle(
            open_price=binance_df.iloc[idx]["open"],
            high_price=binance_df.iloc[idx]["high"],
            low_price=binance_df.iloc[idx]["low"],
            close_price=binance_df.iloc[idx]["close"],
            volume=binance_df.iloc[idx]["volume"],
        )

        # if no action, continue
        if predictions.action == TradeAction.NEUTRAL:
            continue

        # if buy action
        if predictions.action == TradeAction.ENTER_LONG:

            # open the position
            paper_trader.open_position(
                symbol=symbol,
                interval=interval,
                entry_price=predictions.entry_price,
                exit_price=predictions.exit_price,
                stop_loss=predictions.stop_loss,
                trade_type=TradeType.LONG,
                risk_reward_ratio=predictions.risk_reward_ratio,
                trade_start_timestamp=binance_df.iloc[idx]["timestamp"],
            )

        # if sell action
        elif predictions.action == TradeAction.ENTER_SHORT:

            # open the position
            paper_trader.open_position(
                symbol=symbol,
                interval=interval,
                entry_price=predictions.entry_price,
                exit_price=predictions.exit_price,
                stop_loss=predictions.stop_loss,
                trade_type=TradeType.SHORT,
                risk_reward_ratio=predictions.risk_reward_ratio,
                trade_start_timestamp=binance_df.iloc[idx]["timestamp"],
            )

    # get the stats
    stats = paper_trader.stats()

    # print the stats
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Trades: {stats.total_trades}")
    print(f"  Long: {stats.long_trades} | Short: {stats.short_trades}")
    print(f"  Won: {stats.trades_won} | Lost: {stats.trades_lost}")
    print(f"Success Rate: {stats.success_rate:.2%}")
    print("-" * 50)
    print(f"Total Brokerage: ${stats.total_brokerage:.2f}")
    print(
        f"Buy Brokerage: ${stats.buy_brokerage:.2f} | Sell Brokerage: ${stats.sell_brokerage:.2f}"
    )
    print("-" * 50)
    print(f"Profit: ${stats.profit:.2f} | Loss: ${stats.loss:.2f}")
    print(f"Net P/L: ${stats.net:.2f}")
    print(f"Profit Factor: {stats.profit_factor:.2f}")
    print("-" * 50)
    print(f"Avg Profit: ${stats.avg_profit:.2f} | Avg Loss: ${stats.avg_loss:.2f}")
    print(f"Max Profit: ${stats.max_profit:.2f} | Max Loss: ${stats.max_loss:.2f}")
    print("-" * 50)
    print(
        f"Trade Duration (candles): Avg: {stats.avg_trade_age:.1f} | Min: {stats.min_trade_age} | Max: {stats.max_trade_age}"
    )
    print("=" * 50)
    print("\nRisk-Reward Distribution:")
    for rr_ratio, values in stats.grouped_risk_reawards.items():
        print(
            f"  R:R {rr_ratio}: {values.total_trades} trades | Win Rate: {values.win_percent:.2f}%"
        )
