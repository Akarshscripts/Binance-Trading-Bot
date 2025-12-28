"""
This module contains the argument parser for the predict command.
"""

# 1st party imports
import os
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

    # add the model path argument
    predict_parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        metavar="MODEL_PATH",
        help="The path to the trained model.",
    )

    # add the discord webhook url argument
    predict_parser.add_argument(
        "-d",
        "--discord",
        metavar="URL",
        help="The discord webhook url to send the predictions to.",
        required=True,
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

    # add the risk to reward argument
    predict_parser.add_argument(
        "-rr",
        "--risk-to-reward",
        default=1.5,
        type=float,
        metavar="RISK_TO_REWARD",
        help="The risk to reward ratio for backtesting.",
        required=False,
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
            - model_path: Path to the trained model file.
            - discord: Discord webhook URL for sending predictions.
    """

    # check if the model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")

    # convert the pair from str to BinanceSymbols
    symbol = BinanceSymbols(args.pair)

    # convert the interval from str to ChartIntervals
    value = [i for i in args.interval if i.isdigit()]
    unit = [i for i in args.interval if not i.isdigit()]
    char_internal_val = ChartIntervalInternal(
        time_value=int("".join(value)), time_unit="".join(unit)
    )
    interval = ChartIntervals(char_internal_val)

    # import and run the function
    from commands import start_predictions, backtest_predictions

    # check for start time and end time availability
    if args.start_time and args.end_time:
        # start backtesting
        backtest_predictions(
            symbol=symbol,
            interval=interval,
            time_zone=TimeZones.INDIA,
            model_path=args.model_path,
            discord_webhook=args.discord,
            start_time=args.start_time,
            end_time=args.end_time,
            risk_to_reward=args.risk_to_reward,
        )
    else:
        # start realtime predictions
        start_predictions(
            symbol=symbol,
            interval=interval,
            time_zone=TimeZones.INDIA,
            model_path=args.model_path,
            discord_webhook=args.discord,
        )
