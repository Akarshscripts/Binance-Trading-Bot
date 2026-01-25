"""
This module contains the core logic for the backtest command.
"""

# 1st party imports
from typing import Any
from pathlib import Path

# local imports
from brokers import PaperTrader, PaperTradeStats
from upstox_api import UpstoxSymbols, UpstoxIntervals
from binance_api import BinanceSymbols, ChartIntervals
from strategies import SupertrendStrategy, SupertrendStrategyConfig

from .binance import backtest_binance
from .upstox import backtest_upstox


def _print_stats(stats: PaperTradeStats):
    """
    Print formatted backtest statistics to console.

    Args:
        stats: PaperTradeStats object containing backtest results including
            trade counts, profit/loss metrics, brokerage fees, and performance ratios.
    """

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

    # dump to json
    file_path = "backtest_results.json"
    with open(file_path, "w") as f:
        f.write(stats.model_dump_json())


def execute_backtest(
    *,
    symbol: Any,
    interval: Any,
    start_time: str,
    end_time: str,
    brokerage: float,
    initial_capital: float,
    risk_investment: float,
    get_approval: bool,
):
    """
    Backtest predictions for a given symbol and interval.

    Args:
        symbol: The trading symbol to backtest (e.g., BTCUSDT, TATA_STEEL).
        interval: The chart interval for candlestick data.
        start_time: Start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        end_time: End time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
        brokerage: The brokerage fee percentage to use for backtesting.
        initial_capital: The initial capital to use for backtesting.
        risk_investment: The risk investment percentage to use for backtesting.
        get_approval: Ask user for approval before each trade in backtesting.
    """

    # create the paper trader
    paper_trader = PaperTrader(
        brokerage=brokerage,
        capital=initial_capital,
        risk_investment=risk_investment,
    )

    # run backtest based on the symbol
    if isinstance(symbol, BinanceSymbols):

        # parse the interval
        interval = ChartIntervals("15m")

        # backtest binance
        stats = backtest_binance(
            symbol=symbol,
            interval=interval,
            paper_trader=paper_trader,
            start_time=start_time,
            end_time=end_time,
            get_approval=get_approval,
        )

    elif isinstance(symbol, UpstoxSymbols):

        # parse the interval
        interval = UpstoxIntervals(interval)

        # backtest upstox
        stats = backtest_upstox(
            symbol=symbol,
            interval=interval,
            paper_trader=paper_trader,
            start_time=start_time,
            end_time=end_time,
            get_approval=get_approval,
        )

    # print the stats
    _print_stats(stats)
