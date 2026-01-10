"""
This module contains the core logic for the predict command.
"""

# 1st party imports
import time
import json
import pytz
import logging
from pathlib import Path
from datetime import datetime

# 3rd party imports
from tqdm import tqdm
from pandas import DataFrame

# local imports
from messenger import Messenger
from brokers import PaperTrader, TradeType
from binance_api import BinanceSymbols, ChartIntervals, BinanceApi, TimeZones
from strategies import SupertrendStrategy, TradeAction, SupertrendStrategyConfig

# get logger
logger = logging.getLogger("predict")


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

    # create config
    strategy_config = SupertrendStrategyConfig(
        factor=3,
        atr_period=10,
        fractal_period=7,
        risk_reward_ratio=2,
    )

    # instances
    binance = BinanceApi(timezone=time_zone)
    discord = Messenger(webhook_url=discord_webhook)
    trade_strategy = SupertrendStrategy(config=strategy_config)

    # time variables
    curr_time = None
    next_fetch_at = interval.get_next_fetch_time(lag=lag)
    local_next_fetch_at = datetime.fromtimestamp(next_fetch_at, tz=pytz.timezone(time_zone.value.iana_name))

    # create a json file to dump trades
    trades_jsonl = Path("trades.jsonl")
    trades_jsonl.touch()

    # send the startup message
    startup_msg = f"Starting predictions for {symbol.value} on {interval.value} interval. Next fetch at: {local_next_fetch_at.strftime('%Y-%m-%d %H:%M:%S')}."
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
            f"Fetching data for {symbol.value} on {interval.value} interval. Time: {datetime.fromtimestamp(curr_time, tz=pytz.timezone(time_zone.value.iana_name)).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        binance_df = binance.get_symbol_info(
            symbol=symbol,
            interval=interval,
        )

        # calculate the next fetch time
        next_fetch_at = interval.get_next_fetch_time(lag=lag)
        local_next_fetch_at = datetime.fromtimestamp(next_fetch_at, tz=pytz.timezone(time_zone.value.iana_name))

        # remove the last row as it is not completed yet
        binance_df: DataFrame = binance_df.drop(binance_df.index[-1])

        # make predictions
        predictions = trade_strategy.process_candles(
            open_prices=binance_df["open"],
            high_prices=binance_df["high"],
            low_prices=binance_df["low"],
            close_prices=binance_df["close"],
            volumes=binance_df["volume"],
        )
        logger.info(
            f"Latest candle: {binance_df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. Latest Candle Close: {binance_df.iloc[-1]['close']:.4f}"
        )

        # if the strategy is not neutral, send the message
        if predictions.action != TradeAction.NEUTRAL:

            # prepare msg
            msg = f"Strategy: **{predictions.action}**.\nTime: **{binance_df.iloc[-1]["timestamp"].strftime('%Y-%m-%d %H:%M:%S')}**.\nEntry Price: **{predictions.entry_price:.4f}**.\n Stop Loss: **{predictions.stop_loss:.4f}**.\n Take Profit: **{predictions.exit_price:.4f}**."
            discord.send_text_message(msg)
            logger.info(msg)

            # dump the trade to jsonl
            with trades_jsonl.open("a") as f:
                f.write(json.dumps(predictions.model_dump(mode="json")) + "\n")

        # print cycle complete message
        cycle_complete_msg = "-" * 20 + " Cycle Complete " + "-" * 20
        logger.info(cycle_complete_msg)


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

    # create config
    strategy_config = SupertrendStrategyConfig(
        factor=3,
        atr_period=10,
        fractal_period=7,
        risk_reward_ratio=2,
    )

    # instances
    paper_trader = PaperTrader(
        brokerage=brokerage,
        capital=initial_capital,
        risk_investment=risk_investment,
    )
    binance = BinanceApi(timezone=time_zone)
    trade_strategy = SupertrendStrategy(config=strategy_config)

    # convert the start and end time to datetime
    start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")

    # fetch the data from start to end
    logger.info(
        f"Fetching backtest data for {symbol.value} from {start_time} to {end_time}"
    )
    binance_df = binance.get_symbol_info(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
    )
    logger.info(f"Fetched {len(binance_df)} candles for backtesting")

    # check if enough data is available
    if len(binance_df) < strategy_config.minimum_history:
        raise ValueError(
            f"Not enough data available for backtesting. Need at least {strategy_config.minimum_history} candles, got {len(binance_df)}"
        )

    # main loop
    n = len(binance_df)
    for idx in tqdm(range(strategy_config.minimum_history, n), desc="Backtesting progress"):

        # get the data up to current candle (excluding incomplete current candle)
        current_df = binance_df.iloc[: idx + 1].copy()

        # make predictions using process_candles
        predictions = trade_strategy.process_candles(
            open_prices=current_df["open"],
            high_prices=current_df["high"],
            low_prices=current_df["low"],
            close_prices=current_df["close"],
            volumes=current_df["volume"],
        )

        # pass the new candle info to paper trader
        paper_trader.update(
            low=binance_df.iloc[idx]["low"],
            high=binance_df.iloc[idx]["high"],
            current_candle_timestamp=binance_df.iloc[idx]["timestamp"],
        )

        # if no action, continue
        if predictions.action == TradeAction.NEUTRAL:
            continue

        # log the trade signal
        logger.info(
            f"Backtest {predictions.action} signal at {binance_df.iloc[idx]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - "
            f"Entry: {predictions.entry_price:.4f}, SL: {predictions.stop_loss:.4f}, TP: {predictions.exit_price:.4f}"
        )

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
