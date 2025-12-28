"""
This module contains the Binance class, which is used to interact with the Binance API.
It provides methods for retrieving market data, placing orders, and managing the trading bot's state.
"""

# 1st party imports
import time
import pytz
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# 3rd party imports
import requests
import pandas as pd

# local imports
from .models import BinanceSymbols, ChartIntervals, TimeZones


class BinanceApi:
    """
    A class to interact with the Binance API.
    This class provides methods to fetch historical candlestick data for various trading symbols and intervals.
    """

    def __init__(self, timezone: TimeZones = TimeZones.INDIA):
        """
        Initializes the BinanceApi instance.

        Args:
            timezone (TimeZones): The timezone to use for time conversions. Defaults to TimeZones.INDIA.
        """

        # set variables
        self.base_url = "https://api4.binance.com"
        self.timezone = timezone

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
            self.logger.info("Logger configured for %s", __name__)

    def get_symbol_info(
        self,
        symbol: BinanceSymbols,
        interval: ChartIntervals,
        start_time: Optional[str | datetime] = None,
        end_time: Optional[str | datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetches historical candlestick (kline) data for a specific trading symbol and interval from the Binance API.

        Args:
            symbol (BinanceSymbols): The trading symbol for which to fetch data (e.g., BTCUSDT).
            interval (ChartIntervals): The interval for candlestick data (e.g., 1m, 1h, 1d).
            start_time (Optional[str], optional): The start time for the data in `MM/dd/yyyy HH:mm:ss` format (24 hour time). Defaults to None.
            end_time (Optional[str], optional): The end time for the data in `MM/dd/yyyy HH:mm:ss` format (24 hour time). Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the candlestick data with columns such as open, high, low, close, volume, etc.
                          Returns an empty DataFrame if the API request fails.
        """

        # construct variables
        url = f"{self.base_url}/api/v3/klines"
        headers = {"X-MBX-TIME-UNIT": "MILLISECOND"}
        params = {
            "symbol": symbol.value,
            "interval": str(interval.value),
            "timeZone": self.timezone.value.offset,
            "limit": 1000,
        }

        # if start_time and end_time are not provided, return the latest data
        if not start_time:

            # get the response from the API
            self.logger.info("Fetching latest data for symbol: %s", symbol.value)
            response = self.__make_get_request(url, params=params, headers=headers)

            # check if response is None
            if response is None:
                self.logger.error("Failed to fetch data from Binance API")
                return pd.DataFrame()

            # convert the response to a DataFrame
            return self.__convert_to_dataframe(response)

        # check if start_time and end_time are both provided
        if bool(start_time) ^ bool(end_time):
            self.logger.error(
                "Both start_time and end_time must be provided or neither."
            )
            raise ValueError(
                "Both start_time and end_time must be provided or neither."
            )

        # parse the start time, if not already
        if not isinstance(start_time, datetime):
            start_time: datetime = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
        start_time_millis = int(start_time.timestamp() * 1000)

        # parse the end time
        if not isinstance(end_time, datetime):
            end_time: datetime = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")
        end_time_millis = int(end_time.timestamp() * 1000)

        # adjust the params for start and end time if provided
        params["startTime"] = start_time_millis
        params["endTime"] = end_time_millis

        # Calculate the time covered per request based on the interval
        if interval.value.time_unit == "m":
            time_covered_per_req = timedelta(minutes=interval.value.time_value * 1000)
        elif interval.value.time_unit == "h":
            time_covered_per_req = timedelta(hours=interval.value.time_value * 1000)
        elif interval.value.time_unit == "d":
            time_covered_per_req = timedelta(days=interval.value.time_value * 1000)
        elif interval.value.time_unit == "w":
            time_covered_per_req = timedelta(weeks=interval.value.time_value * 1000)
        elif interval.value.time_unit == "M":
            time_covered_per_req = timedelta(days=30 * interval.value.time_value * 1000)

        # check if the whole time range can be covered in one request
        if start_time + time_covered_per_req > end_time:

            # get the response from the API
            response = self.__make_get_request(url, params=params, headers=headers)

            # check if response is None
            if response is None:
                self.logger.error("Failed to fetch data from Binance API")
                return pd.DataFrame()
            return self.__convert_to_dataframe(response)

        # store the result
        result = []
        req_idx = 0

        # iterate over the time range in chunks
        while (start_time + time_covered_per_req) <= end_time:

            # adjust the params for start and end time
            params["startTime"] = int(start_time.timestamp() * 1000)
            params["endTime"] = int(
                (start_time + time_covered_per_req).timestamp() * 1000
            )

            # get the response from the API
            self.logger.info(
                "Fetching data from Binance API. Start time: %s, End time: %s",
                str(start_time),
                str(start_time + time_covered_per_req),
            )
            response = self.__make_get_request(url, params=params, headers=headers)
            req_idx += 1

            # add to the result if not None
            if response is not None:
                result.extend(response)
            else:
                self.logger.error(
                    "Failed to fetch data from Binance API. Start time: %s, End time: %s",
                    str(start_time),
                    str(end_time),
                )
                break

            # increment the start time by the time covered per request
            start_time += time_covered_per_req

            # after every 3000 requests, wait for 2 minute to avoid hitting the API limit
            if req_idx % 3000 == 0:
                self.logger.info(
                    "Waiting for 2 minutes to avoid hitting the API limit..."
                )
                time.sleep(120)

        # return the result as a DataFrame
        return self.__convert_to_dataframe(result)

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
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )

        # convert time columns to datetime and localize to specified timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # create a local timezone object
        local_tz = pytz.timezone(self.timezone.value.iana_name)

        # localize the time columns to the specified timezone
        df["timestamp"] = df["timestamp"].apply(lambda x: x.astimezone(local_tz))
        df["close_time"] = df["close_time"].apply(lambda x: x.astimezone(local_tz))

        # numeric columns
        NUMERIC_COLS = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
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
