"""
This is the main module for the stock prediction project.

It fetches the stock data from the binance API and processes it, yielding a dataset
which is then passed to the model for inference. The results are posted to the linked discord webhook.
"""

# 1st party imports
import logging
from pathlib import Path

# import locals
from cli import app


def setup_logging(level=logging.INFO):
    """
    Setup logging for the application.
    """

    # create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # log format
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    # create a file handler
    file_handler = logging.FileHandler(log_dir / "app.log", mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # setup root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # add handlers
    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def main():
    """
    Main function for the application.
    """

    # setup logging
    setup_logging()

    # run the app
    app()


if __name__ == "__main__":
    main()
