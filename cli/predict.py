"""
This module contains the argument parser for the predict command.
"""

# 1st party imports
from enum import Enum
from datetime import datetime

# 3rd party imports
import typer

# local imports
from binance_api import BinanceSymbols, ChartIntervals, TimeZones


# create the predict app cli
predict_app = typer.Typer(help="Run price predictions")


class CommandGroups(str, Enum):
    """Enum for command groups."""

    REAL_TIME = "Real-time predictions"
    BACKTEST = "Backtesting"
    NOTIFICATION = "Notification"


@predict_app.command("predict")
def predict(
    pair: BinanceSymbols = typer.Argument(
        ..., help="Crypto pair (BTCUSDT)", metavar="Pair"
    ),
    interval: ChartIntervals = typer.Argument(
        ..., help="Chart interval (15m, 1h)", metavar="Interval"
    ),
    discord: str = typer.Option(
        None,
        "-d",
        "--discord",
        help="Discord webhook URL",
        metavar="Discord URL",
        rich_help_panel=CommandGroups.NOTIFICATION,
    ),
    start_time: str = typer.Option(
        None,
        "-st",
        "--start-time",
        help="Start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format",
        metavar="Start Time",
        rich_help_panel=CommandGroups.BACKTEST,
    ),
    end_time: str = typer.Option(
        None,
        "-et",
        "--end-time",
        help="End time for backtesting in 'MM/dd/yyyy HH:mm:ss' format",
        metavar="End Time",
        rich_help_panel=CommandGroups.BACKTEST,
    ),
    brokerage: float = typer.Option(
        0.001,
        "-b",
        "--brokerage",
        help="Brokerage to use for backtesting",
        metavar="Brokerage",
        rich_help_panel=CommandGroups.BACKTEST,
    ),
    capital: float = typer.Option(
        10000,
        "-c",
        "--capital",
        help="Capital to use for backtesting",
        metavar="Capital",
        rich_help_panel=CommandGroups.BACKTEST,
    ),
    risk_investment: float = typer.Option(
        0.02,
        "-ri",
        "--risk-investment",
        help="Risk investment to use for backtesting",
        metavar="Risk Investment",
        rich_help_panel=CommandGroups.BACKTEST,
    ),
    approval: bool = typer.Option(
        False,
        "-a",
        "--approval",
        help="User approves each trades while backtesting",
        metavar="Approval",
        rich_help_panel=CommandGroups.BACKTEST,
    ),
):
    """
    Run price predictions for a crypto pair.

    This command supports two modes:
    1. Real-time predictions: When no start/end time is provided, continuously
       monitors the pair and sends predictions to Discord.
    2. Backtesting: When start/end time is provided, runs historical backtesting
       with paper trading simulation.
    """

    # import and run the function
    from commands import start_algorithm, backtest_algorithm

    # check for start time availability
    if start_time:

        # set the end time as today if not provided
        if not end_time:
            end_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

        # start backtesting
        backtest_algorithm(
            symbol=pair,
            interval=interval,
            time_zone=TimeZones.INDIA,
            start_time=start_time,
            end_time=end_time,
            brokerage=brokerage,
            initial_capital=capital,
            risk_investment=risk_investment,
            approval=approval,
        )

    else:
        # check if the discord webhook is provided
        if not discord:
            raise ValueError("Discord webhook is required for realtime predictions.")

        # start realtime predictions
        start_algorithm(
            symbol=pair,
            interval=interval,
            time_zone=TimeZones.INDIA,
            discord_webhook=discord,
        )
