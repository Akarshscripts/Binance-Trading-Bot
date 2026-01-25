"""
This module contains the Binance class, which is used to interact with the Binance API.
It provides methods for retrieving market data, placing orders, and managing the trading bot's state.
"""

# 1st party imports
import pytz
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

# 3rd party imports
import requests
import pandas as pd

# local imports
from .models import UpstoxSymbols, UpstoxIntervals


class UpstoxExchange:
    """
    A class to interact with the Upstox API.
    This class provides methods to fetch historical candlestick data for various trading symbols and intervals.
    """

    def __init__(self):
        """
        Initializes the UpstoxExchange instance.
        """

        # set variables
        self.base_url = "https://api.upstox.com"

        # get logger
        self.logger = logging.getLogger(__name__)

        # configure logger if not already configured
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.info("Logger configured at %s", __name__)

    def _generate_url(
        self,
        exchange: str,
        symbol: UpstoxSymbols,
        interval: UpstoxIntervals,
        end_time: str,
        start_time: Optional[str] = None,
    ):
        """
        Generate the URL for fetching historical candlestick data from Upstox API.

        Args:
            exchange: The exchange code (e.g., "NSE", "BSE").
            symbol: The trading symbol to fetch data for.
            interval: The time interval for candlestick data.
            end_time: End time in 'yyyy-MM-dd' format.
            start_time: Optional start time in 'yyyy-MM-dd' format.

        Returns:
            str: The constructed URL for the Upstox API request.
        """

        # base url with exchange and symbol
        url = f"{self.base_url}/v3/historical-candle/{exchange}|{symbol.value}"

        # add the time unit and interval
        time_unit = interval.value.get_time_unit()
        url += f"/{time_unit}/{interval.value.time_value}"

        # add the end time
        url += f"/{end_time}"

        # add the start time if provided
        if start_time:
            url += f"/{start_time}"

        # return the url
        return url

    def get_symbol_info(
        self,
        symbol: UpstoxSymbols,
        interval: UpstoxIntervals,
        start_time: str | datetime,
        end_time: Optional[str | datetime] = None,
        exchnage: Literal["NSE_EQ"] = "NSE_EQ",
    ) -> pd.DataFrame:
        """
        Fetches historical candlestick (kline) data for a specific trading symbol and interval from the Upstox API.

        Args:
            symbol (UpstoxSymbols): The trading symbol for which to fetch data (e.g., BTCUSDT).
            interval (UpstoxIntervals): The interval for candlestick data (e.g., 1m, 1h, 1d).
            start_time (str | datetime): The start time for the data in `MM/dd/yyyy HH:mm:ss` format (24 hour time).
            end_time (Optional[str | datetime]): The end time for the data in `MM/dd/yyyy HH:mm:ss` format (24 hour time).

        Returns:
            pd.DataFrame: A DataFrame containing the candlestick data with columns such as open, high, low, close, volume, etc.
                          Returns an empty DataFrame if the API request fails.
        """

        # convert start_time and end_time to datetime if not already
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, "%Y-%m-%d")
        if end_time is None:
            end_time = datetime.now()
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, "%Y-%m-%d")


        # add utc awareness
        start_time = start_time.replace(tzinfo=pytz.utc)
        end_time = end_time.replace(tzinfo=pytz.utc)

        # fetch the url and keep iterating until the data is fetched
        last_candle_timestamp = end_time
        candles_history = []
        fetched_urls = set()
        while last_candle_timestamp >= start_time:

            # generate the url
            url = self._generate_url(
                exchange=exchnage,
                symbol=symbol,
                interval=interval,
                end_time=last_candle_timestamp.strftime("%Y-%m-%d"),
            )

            # check if url is already fetched
            if url in fetched_urls:
                break
            else:
                fetched_urls.add(url)

            # get the response from the API
            response = self.__make_get_request(url)

            # check if response is None
            if response is None:
                self.logger.error(
                    "Failed to fetch data from Binance API. Total fetched: %s candles",
                    len(candles_history),
                )
                raise Exception("Failed to fetch data from Binance API")

            # update the last candle timestamp
            candles_data = response.get("data", {}).get("candles", [])
            last_candle_timestamp = datetime.fromisoformat(candles_data[-1][0])
            last_candle_timestamp = last_candle_timestamp.replace(tzinfo=pytz.utc)
            self.logger.info(
                "Fetched %s candles for %s %s. Last candle timestamp: %s",
                len(candles_data),
                symbol.value,
                interval.value,
                last_candle_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            )

            # append the candles to the history
            candles_history.extend(candles_data)

        return self.__convert_to_dataframe(candles_history[::-1])

    def __convert_to_dataframe(self, data: List[List[Any]]) -> pd.DataFrame:
        """
        Converts the raw data from the API into a pandas DataFrame.

        Args:
            data (List[List[Any]]): The raw data from the API.

        Returns:
            pd.DataFrame: A DataFrame containing the converted data.
        """

        # convert to DataFrame
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume", "ignore"],
        )

        # convert time columns to datetime and localize to specified timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # numeric columns
        NUMERIC_COLS = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # convert the columns to numeric
        df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
        df[NUMERIC_COLS] = df[NUMERIC_COLS].astype("float32")

        # drop the ignore column
        df.drop(columns=["ignore"], inplace=True)

        # return
        return df

    def __make_get_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = {},
        headers: Optional[Dict[str, str]] = {},
    ) -> Optional[Dict[str, Any]]:
        """
        Makes a GET request to the specified endpoint with optional parameters and headers.

        Args:
            endpoint (str): The URL endpoint to send the GET request to.
            params (Optional[Dict[str, Any]]): A dictionary of query parameters to include in the request. Defaults to an empty dictionary.
            headers (Optional[Dict[str, str]]): A dictionary of HTTP headers to include in the request. Defaults to an empty dictionary.

        Returns:
            Optional[Dict[str, Any]]: The JSON response from the server as a dictionary if the request is successful,
            or None if an HTTP error occurs.

        Logs:
            Logs an error message if an HTTP error occurs, including the endpoint and the error details.
        """

        # make the request
        response = requests.get(endpoint, params=params, headers=headers, timeout=60)

        # check for errors
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            self.logger.error(
                "Endpoint: %s\nHTTP error: %s,\nResponse: %s",
                endpoint,
                str(err),
                response.text,
            )
            return None

        # return the response as JSON
        return response.json()
