"""
This module contains the argument parser for the stock predictor.
"""

# local imports
from cli.models import PairRegistry
from upstox_api import UpstoxSymbols, UpstoxExchange
from binance_api import BinanceSymbols, BinanceExchange

# 3rd party imports
import typer

app = typer.Typer(
    help="AI-powered crypto trading & prediction tool", add_completion=False
)

# create the registry and register the pairs
registry = PairRegistry()
registry.register(BinanceSymbols, BinanceExchange)
registry.register(UpstoxSymbols, UpstoxExchange)
