"""
This module contains the core logic for the backtest of binance symbols.
"""

# 1st party imports
import logging

# 3rd party imports
from tqdm import tqdm
from datetime import datetime

# local imports
from brokers import PaperTrader, TradeType, PaperTradeStats
from strategies import SupertrendStrategy, TradeAction, SupertrendStrategyConfig
from binance_api import ChartIntervals, BinanceSymbols, BinanceExchange, TimeZones

# get the logger
logger = logging.getLogger("backtest")


def backtest_binance(
    symbol: BinanceSymbols,
    interval: ChartIntervals,
    paper_trader: PaperTrader,
    start_time: str,
    end_time: str,
    get_approval: bool,
) -> PaperTradeStats:
    """
    Backtest a trading strategy on historical Binance data.

    Args:
        symbol: The Binance trading symbol to backtest (e.g., BTCUSDT).
        interval: The chart interval for candlestick data.
        paper_trader: The paper trader instance for simulating trades.
        start_time: Start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        end_time: End time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        get_approval: If True, ask user for approval before each trade.

    Returns:
        PaperTradeStats: Statistics from the backtest including profit/loss metrics.
    """

    # create the trading strategy
    strategy_config = SupertrendStrategyConfig(
        factor=3,
        atr_period=10,
        fractal_period=7,
        risk_reward_ratio=2,
    )
    strategy = SupertrendStrategy(config=strategy_config)

    # create binance instance
    start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")

    # make sure end time is after start time
    if end_time < start_time:
        raise ValueError("End time must be after start time")

    # create binance instance
    binance_api = BinanceExchange(timezone=TimeZones.INDIA)

    # fetch the data from start to end
    logger.info(
        f"Fetching backtest data for {symbol.value} from {start_time} to {end_time}"
    )
    binance_df = binance_api.get_symbol_info(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
    )
    logger.info(f"Fetched {len(binance_df)} candles for backtesting")

    # main loop
    n = len(binance_df)
    for idx in tqdm(
        range(strategy.config.minimum_history, n),
        desc="Backtesting progress",
        disable=get_approval,
    ):

        # get the data up to current candle (excluding incomplete current candle)
        current_df = binance_df.iloc[: idx + 1].copy()

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
            low=binance_df.iloc[idx]["low"],
            high=binance_df.iloc[idx]["high"],
            current_candle_timestamp=binance_df.iloc[idx]["timestamp"],
        )

        # if no action, continue
        if predictions.action == TradeAction.NEUTRAL:
            continue

        # prompt for approval
        if get_approval:

            # log the trade signal
            logger.info(
                f"Backtest {predictions.action} signal at {binance_df.iloc[idx]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - "
                f"Entry: {predictions.entry_price:.4f}, SL: {predictions.stop_loss:.4f}, TP: {predictions.exit_price:.4f}"
            )

            # ask for approval
            approval_msg = input("Approve this trade? (y/n): ")
            while approval_msg.lower() not in ["y", "n"]:
                approval_msg = input("Approve this trade? (y/n): ")
            if approval_msg.lower() == "n":
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
    return paper_trader.stats()
