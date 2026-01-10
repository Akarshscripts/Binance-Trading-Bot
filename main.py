"""
This is the main module for the stock prediction project.

It fetches the stock data from the binance API and processes it, yielding a dataset
which is then passed to the model for inference. The results are posted to the linked discord webhook.
"""

# 1st party imports
import os
import logging

# import locals
from cli import create_parser


def setup_logging(level=logging.INFO):
    """
    Setup logging for the application.
    """

    # create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="logs/app.log",
        filemode="w",
    )


def main():
    """
    Main function for the application.
    """

    # setup logging
    setup_logging()

    # parse args
    parser = create_parser()
    args = parser.parse_args()

    # call the handler attached to the command
    args.handler(args)


if __name__ == "__main__":
    main()
