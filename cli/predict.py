"""
This module contains the argument parser for the predict command.
"""

# 1st party imports
from datetime import datetime
from argparse import _SubParsersAction

# local imports
from binance_api.models import ChartIntervalInternal
from binance_api import BinanceSymbols, ChartIntervals, TimeZones


def add_predict_command(subparsers: _SubParsersAction):
    """
    Add the predict command to the argument parser.

    Args:
        subparsers (_SubParsersAction): The subparsers action from the main parser.
    """

    # create a parser for the predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Start tracking the crypto pair and make predictions.",
        usage="stock-predictor predict <- pos_args -> <- opt_args ->",
    )

    # add the pair argument
    predict_parser.add_argument(
        "pair",
        choices=[s.value for s in BinanceSymbols],
        metavar="PAIR",
        help="The crypto pair to track (e.g., XRPUSDT, BTCUSDT).",
    )

    # add the interval argument
    predict_parser.add_argument(
        "-i",
        "--interval",
        required=True,
        choices=[str(i.value) for i in ChartIntervals],
        metavar="INTERVAL",
        help="The chart interval to use.",
    )

    # add the discord webhook url argument
    predict_parser.add_argument(
        "-d",
        "--discord",
        metavar="URL",
        help="The discord webhook url to send the predictions to.",
    )

    # add the start time argument
    predict_parser.add_argument(
        "-st",
        "--start-time",
        metavar="START_TIME",
        help="The start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.",
        required=False,
    )

    # add the end time argument
    predict_parser.add_argument(
        "-et",
        "--end-time",
        metavar="END_TIME",
        help="The end time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.",
        required=False,
    )

    # add the brokerage argument
    predict_parser.add_argument(
        "-b",
        "--brokerage",
        metavar="BROKERAGE",
        help="The brokerage to use for backtesting.",
        required=False,
        default=0.001,
        type=float,
    )

    # add the capital argument
    predict_parser.add_argument(
        "-c",
        "--capital",
        metavar="CAPITAL",
        help="The capital to use for backtesting.",
        required=False,
        default=10000,
        type=float,
    )

    # add the risk_investment argument
    predict_parser.add_argument(
        "-ri",
        "--risk-investment",
        metavar="RISK_INVESTMENT",
        help="The risk investment to use for backtesting.",
        required=False,
        default=0.02,
        type=float,
    )

    # add the handler
    predict_parser.set_defaults(handler=handle_predict)


def handle_predict(args):
    """
    Handle the predict command by parsing arguments and starting predictions.

    Args:
        args: The parsed command line arguments containing:
            - pair: The crypto trading pair symbol (e.g., XRPUSDT).
            - interval: The chart interval string (e.g., 15m, 1h).
            - discord: Discord webhook URL for sending predictions.
            - start_time: The start time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
            - end_time: The end time for backtesting in 'MM/dd/yyyy HH:mm:ss' format.
    """

    # convert the pair from 'str' to 'BinanceSymbols'
    symbol = BinanceSymbols(args.pair)

    # convert the interval from 'str' to 'ChartIntervals'
    value = [i for i in args.interval if i.isdigit()]
    unit = [i for i in args.interval if not i.isdigit()]
    char_internal_val = ChartIntervalInternal(
        time_value=int("".join(value)), time_unit="".join(unit)
    )
    interval = ChartIntervals(char_internal_val)

    # import and run the function
    from commands import start_algorithm, backtest_algorithm

    # check for start time availability
    if args.start_time:

        # set the end time as today if not provided
        if not args.end_time:
            args.end_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

        # start backtesting
        backtest_algorithm(
            symbol=symbol,
            interval=interval,
            time_zone=TimeZones.INDIA,
            start_time=args.start_time,
            end_time=args.end_time,
            brokerage=args.brokerage,
            initial_capital=args.capital,
            risk_investment=args.risk_investment,
        )

    else:
        # check if the discord webhook is provided
        if not args.discord:
            raise ValueError("Discord webhook is required for realtime predictions.")

        # start realtime predictions
        start_algorithm(
            symbol=symbol,
            interval=interval,
            time_zone=TimeZones.INDIA,
            discord_webhook=args.discord,
        )
