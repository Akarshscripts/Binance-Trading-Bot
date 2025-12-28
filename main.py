"""
This is the main module for the stock prediction project.

It fetches the stock data from the binance API and processes it, yielding a dataset
which is then passed to the model for inference. The result are posted to the linked discord webhook.
"""

# import locals
from cli import create_parser

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    # call the handler attached to the command
    args.handler(args)
