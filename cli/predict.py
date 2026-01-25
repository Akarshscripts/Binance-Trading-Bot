"""
This module contains the argument parser for the predict command.
"""

# 3rd party imports
import typer

# local imports
from cli.models import PairRegistry

# create the registry and register the pairs
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

# create the predict app cli
predict_app = typer.Typer()


@predict_app.command(
    "predict",
    help=help_text,
    no_args_is_help=True,
    short_help="Run price predictions for a crypto pair.",
)
def predict(
    pair: str = typer.Argument(
        ..., help="Crypto pair (BTCUSDT, TATA_STEEL)", metavar="Pair"
    ),
    interval: str = typer.Argument(
        ..., help="Chart interval (15m, 1h)", metavar="Interval"
    ),
    discord: str = typer.Option(
        ...,
        "-d",
        "--discord",
        help="Discord webhook URL",
        metavar="Discord URL",
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

    # get the trading pair and the trading exchange for that pair
    trading_pair = registry.generate_pair_enum(pair)
    if not trading_pair:
        raise ValueError(
            f"Unsupported pair: {pair}. Try listing all the pairs and choosing one of them."
        )

    # import and run the function
    from commands import start_algorithm

    # start realtime predictions
    start_algorithm(
        symbol=trading_pair,
        interval=interval,
        discord_webhook=discord,
    )
