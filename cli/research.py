"""
This module contains the argument parser for the quant-research command.
"""

# 3rd party imports
import typer
from datetime import datetime

# local imports
from cli.models import PairRegistry, CommandGroups

# create the registry and register the pairs
registry = PairRegistry()

# help text
available_pairs = []
for pair in registry._registry.keys():
    available_pairs.extend(list(pair.__members__.keys()))

help_text = f"""
[bold cyan]Find the best strategy configuration for a trading pair.[/bold cyan]
This command runs historical backtesting with paper trading simulation and tweaks the strategy configuration to find the best configuration.

[bold]Available Pairs:[/bold]
{"|".join(f" [bright_green]• {p}[/bright_green] |" for p in available_pairs)}
"""

# create the Quant research app cli
quant_research_app = typer.Typer()


@quant_research_app.command(
    "quant-research",
    help=help_text,
    no_args_is_help=True,
    short_help="Find the best strategy configuration for a pair.",
)
def quant_research(
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
        rich_help_panel=CommandGroups.QUANT_RESEARCH,
    ),
    end_time: str = typer.Option(
        ...,
        "-et",
        "--end-time",
        help="End time for backtesting in 'MM/dd/yyyy HH:mm:ss' format",
        metavar="End Time",
        rich_help_panel=CommandGroups.QUANT_RESEARCH,
    ),
):
    """
    Find the best strategy configuration for a trading pair.
    This command runs historical backtesting with paper trading
    simulation and tweaks the strategy configuration to
    find the best configuration.
    """

    # get the trading pair and the trading exchange for that pair
    trading_pair = registry.generate_pair_enum(pair)
    if not trading_pair:
        raise ValueError(
            f"Unsupported pair: {pair}. Try listing all the pairs and choosing one of them."
        )

    # parse the start and end time
    start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")

    # make sure end time is after start time
    if end_time < start_time:
        raise ValueError("End time must be after start time")

    # import and run the function
    from commands import run_quant_research

    # start realtime predictions
    run_quant_research(
        symbol=trading_pair,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
    )
