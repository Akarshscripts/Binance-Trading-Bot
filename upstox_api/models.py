"""
This module contains the models for the BinanceExchange Class.
It contains models for supported symbols, pydantic models for API responses, and other related data structures.
"""

# 1st party imports
import pytz
from enum import Enum
from datetime import datetime, timedelta
from typing import Literal, Dict, List, Optional, Tuple, Any

# 3rd party imports
import requests
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception_type,
    stop_after_attempt,
)
from pydantic import BaseModel


class ChartIntervalInternal(BaseModel):
    """
    Model for internal representation of chart intervals.
    """

    time_value: int
    time_unit: Literal["m", "h", "d"]

    model_config = {"frozen": True}

    def __str__(self) -> str:
        """
        Returns the string representation of the interval.
        """

        return f"{self.time_value}{self.time_unit}"

    def to_seconds(self) -> int:
        """
        Convert the interval to seconds.
        """

        multipliers = {"m": 60, "h": 3600, "d": 86400}
        return self.time_value * multipliers.get(self.time_unit, 1)

    def get_time_unit(self):
        """
        Get the time unit of the interval.

        Returns:
            str: The time unit of the interval (e.g. "minutes", "hours", "days").
        """

        match self.time_unit:
            case "m":
                return "minutes"
            case "h":
                return "hours"
            case "d":
                return "days"
            case _:
                raise ValueError(f"Invalid time unit: {self.time_unit}")

    def __int__(self) -> int:
        return self.to_seconds()

    def __lt__(self, other: int) -> bool:
        return self.to_seconds() < other

    def __le__(self, other: int) -> bool:
        return self.to_seconds() <= other

    def __gt__(self, other: int) -> bool:
        return self.to_seconds() > other

    def __ge__(self, other: int) -> bool:
        return self.to_seconds() >= other

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.to_seconds() == other
        if isinstance(other, str):
            return str(self) == other
        return super().__eq__(other)


class UpstoxIntervals(Enum):
    """
    Model for supported chart intervals in the Upstox API.
    """

    ONE_MINUTE = ChartIntervalInternal(time_value=1, time_unit="m")
    THREE_MINUTES = ChartIntervalInternal(time_value=3, time_unit="m")
    FIVE_MINUTES = ChartIntervalInternal(time_value=5, time_unit="m")
    FIFTEEN_MINUTES = ChartIntervalInternal(time_value=15, time_unit="m")
    THIRTY_MINUTES = ChartIntervalInternal(time_value=30, time_unit="m")
    ONE_HOUR = ChartIntervalInternal(time_value=1, time_unit="h")
    ONE_DAY = ChartIntervalInternal(time_value=1, time_unit="d")

    @retry(
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    )
    def _get_trading_session_info(
        self, date: datetime, exchange: str = "NSE"
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get trading session information for a given date and exchange.

        Args:
            date: The date for which to fetch trading session info.
            exchange: The exchange code (default: "NSE").

        Returns:
            Optional[Tuple[datetime, datetime]]: A tuple of (start_time, end_time) for the trading session,
                                                  or None if no session info is found for the exchange.

        Raises:
            requests.HTTPError: If the API request fails.
            KeyError: If the response JSON is missing expected data.
        """

        # NOTE: This url returns a epoch list of valid trading time for the given date
        formated_date = date.strftime(r"%Y-%m-%d")
        url = f"https://api.upstox.com/v2/market/timings/{formated_date}"

        # fetch and return
        response = requests.get(url)
        response.raise_for_status()

        # parse data
        resp_json: Dict[str, Any] = response.json()
        exchange_details: List[Dict[str, str | float | int]] = resp_json.get("data")
        if exchange_details is None:
            raise KeyError("Unable to parse response json. Key 'data' not found.")

        # iterate and find the start and end time for the exchange
        for exchange_detail in exchange_details:

            # get exchange name
            fetched_exchange = exchange_detail.get("exchange")
            if not isinstance(fetched_exchange, str):
                continue

            # match and return if found
            if fetched_exchange == exchange:
                start_time = datetime.fromtimestamp(
                    exchange_detail["start_time"] / 1000, tz=pytz.UTC
                )
                end_time = datetime.fromtimestamp(
                    exchange_detail["end_time"] / 1000, tz=pytz.UTC
                )
                return start_time, end_time

        return None

    def _get_next_trading_session(
        self, date: datetime, exchange: str = "NSE", days_range: int = 10
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the next trading session for a given date and exchange.

        Args:
            date: The date to check for trading session.
            exchange: The exchange code (default: "NSE").


        Returns:
            Optional[Tuple[datetime, datetime]]: A tuple of (start_time, end_time) for the trading session,
                                                  or None if no session info is found for the exchange.
        """
        # get the trading session
        start_time, end_time = None, None

        for idx in range(days_range + 1):
            target_date = date + timedelta(days=idx)
            session_info = self._get_trading_session_info(
                date=target_date, exchange=exchange
            )
            if isinstance(session_info, tuple):
                start_time, end_time = session_info
                return start_time, end_time

        return None

    def get_next_fetch_time(self, lag: int = 30, days_range: int = 10) -> int:
        """
        Get the next fetch time based on the last completely closed candle.

        Args:
            lag: Seconds to wait after the candle closes before fetching (max: 30)
            range: How far to see for the next fetch time in days (inclusive).

        Returns:
            int: Unix timestamp of when to fetch the next candle.
        """
        now = datetime.now()

        # get the trading session
        start_end_time = self._get_next_trading_session(
            date=now, exchange="NSE", days_range=days_range
        )

        # check if start_time and end_time is populated
        if start_end_time is None:
            raise ValueError(
                "Unable to find trading session info for the given date range."
            )
        start_time, end_time = start_end_time

        # variables
        interval = self.value
        lag = min(lag, 30)
        interval_sec = interval.to_seconds()
        now_ts = now.timestamp()
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()

        # If before market open → wait for first candle
        if now_ts < start_ts:
            last_closed_ts = start_ts - interval_sec

        # If after market close → last candle of session
        elif now_ts > end_ts:
            last_closed_ts = end_ts - interval_sec

        else:
            elapsed = now_ts - start_ts
            completed_candles = int(elapsed // interval_sec) - 1
            last_closed_ts = start_ts + completed_candles * interval_sec

        next_fetch_time = last_closed_ts + interval_sec + lag
        return int(next_fetch_time)

    def __float__(self) -> float:
        """Convert the interval to float (seconds)."""
        return float(self.value.to_seconds())

    def __lt__(self, other: float) -> bool:
        """Compare less than with float."""
        return float(self) < other

    def __le__(self, other: float) -> bool:
        """Compare less than or equal with float."""
        return float(self) <= other

    def __gt__(self, other: float) -> bool:
        """Compare greater than with float."""
        return float(self) > other

    def __ge__(self, other: float) -> bool:
        """Compare greater than or equal with float."""
        return float(self) >= other

    def __eq__(self, other: object) -> bool:
        """Compare equality with float or other objects."""
        if isinstance(other, (int, float)):
            return float(self) == float(other)
        return super().__eq__(other)

    def __str__(self) -> str:
        """Return the string representation of the interval."""
        return f"{self.value.time_value}{self.value.time_unit}"

    @classmethod
    def _missing_(cls, value):
        """Handle missing values for the enum."""
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


class UpstoxSymbols(Enum):
    """
    Model for supported symbols in the Upstox API.

    To add more visit: https://www.nseindia.com/get-quote/equity
    and search for a stock then add it's INE number in the value below
    """

    TATA_STEEL = "INE081A01020"
    TCS = "INE467B01029"
    OBEROI_REALITY_LTD = "INE093I01010"
    VODAFONE_IDEA_LTD = "INE669E01016"
    TATA_MOTORS = "INE1TAE01010"
    IFCI_LTD = "INE039A01010"
    RELIANCE_POWER = "INE614G01033"
    NESTLE_LTD = "INE239A01024"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name} ({self.value})"
