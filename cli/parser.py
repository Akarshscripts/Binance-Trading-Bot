"""
This module contains the argument parser for the stock predictor.
"""

# 1st party imports
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def create_parser() -> ArgumentParser:
    """
    Create and return the argument parser for the stock predictor CLI.

    Returns:
        ArgumentParser: The configured argument parser with all subcommands.
    """

    # create a parser
    parser = ArgumentParser(
        prog="stock-predictor",
        description="Start the AlgoTrader and let it trade!",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # create subparsers
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register commands
    from .predict import add_predict_command

    # add commands
    add_predict_command(subparsers)

    # return parser
    return parser
