"""
This module contains the argument parser for the backtest command.
"""

# 3rd party imports
import typer

# local imports
from cli.models import CommandGroups, PairRegistry

# create the registry
registry = PairRegistry()

# help text
available_pairs = []
for pair in registry._registry.keys():
    available_pairs.extend(list(pair.__members__.keys()))

help_text = f"""
[bold cyan]Run backtesting for a trading pair.[/bold cyan]
This command runs historical backtesting with paper trading simulation.

[bold]Available Pairs:[/bold]
{"|".join(f" [bright_green]• {p}[/bright_green] |" for p in available_pairs)}
"""

# create the backtest app cli
backtest_app = typer.Typer()


@backtest_app.command(
    "backtest",
    help=help_text,
    short_help="Run backtesting on a given pair.",
    no_args_is_help=True,
)
def backtest(
    pair: str = typer.Argument(
        ..., help="Trading pair (BTCUSDT, TATA_STEEL)", metavar="Pair"
    ),
    interval: str = typer.Argument(
        ..., help="Chart interval (15m, 1h)", metavar="Interval"
    ),
    start_time: str = typer.Option(
        ...,
        "-st",
        "--start-time",
        help="Start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format",
        metavar="Start Time",
        rich_help_panel=CommandGroups.BACKTEST,
    ),
    end_time: str = typer.Option(
        ...,
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
        help="User approves each trades while backtesting.",
        metavar="Approval",
        rich_help_panel=CommandGroups.BACKTEST,
    ),
):
    """
    Run backtesting for a trading pair.
    This command runs historical backtesting with paper trading simulation.
    """

    # get the trading pair and the trading exchange for that pair
    trading_pair = registry.generate_pair_enum(pair)
    if not trading_pair:
        raise ValueError(
            f"Unsupported pair: {pair}. Try listing all the pairs and choosing one of them."
        )

    # import and run the function
    from commands import execute_backtest

    # start backtesting
    execute_backtest(
        symbol=trading_pair,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        brokerage=brokerage,
        initial_capital=capital,
        risk_investment=risk_investment,
        get_approval=approval,
    )
