"""
This module contains the models for the BinanceApi Class.
It contains models for supported symbols, pydantic models for API responses, and other related data structures.
"""

# 1st party imports
from enum import Enum

# 3rd party imports
from pydantic import BaseModel


class ChartIntervalInternal(BaseModel):
    """
    Model for internal representation of chart intervals.
    """

    time_value: int
    time_unit: str

    def __str__(self) -> str:
        """
        Returns the string representation of the interval.
        """

        return f"{self.time_value}{self.time_unit}"


class ChartIntervals(Enum):
    """
    Model for supported chart intervals in the Binance API.
    """

    ONE_MINUTE = ChartIntervalInternal(time_value=1, time_unit="m")
    THREE_MINUTES = ChartIntervalInternal(time_value=3, time_unit="m")
    FIVE_MINUTES = ChartIntervalInternal(time_value=5, time_unit="m")
    FIFTEEN_MINUTES = ChartIntervalInternal(time_value=15, time_unit="m")
    THIRTY_MINUTES = ChartIntervalInternal(time_value=30, time_unit="m")
    ONE_HOUR = ChartIntervalInternal(time_value=1, time_unit="h")
    TWO_HOURS = ChartIntervalInternal(time_value=2, time_unit="h")
    FOUR_HOURS = ChartIntervalInternal(time_value=4, time_unit="h")
    SIX_HOURS = ChartIntervalInternal(time_value=6, time_unit="h")
    EIGHT_HOURS = ChartIntervalInternal(time_value=8, time_unit="h")
    TWELVE_HOURS = ChartIntervalInternal(time_value=12, time_unit="h")
    ONE_DAY = ChartIntervalInternal(time_value=1, time_unit="d")
    THREE_DAYS = ChartIntervalInternal(time_value=3, time_unit="d")
    ONE_WEEK = ChartIntervalInternal(time_value=1, time_unit="w")
    ONE_MONTH = ChartIntervalInternal(time_value=1, time_unit="M")


class BinanceSymbols(Enum):
    """
    Model for supported symbols in the Binance API.
    """

    XRPUSDT = "XRPUSDT"
    BTCUSDT = "BTCUSDT"
    ETHUSDT = "ETHUSDT"
    LTCUSDT = "LTCUSDT"
    DOGEUSDT = "DOGEUSDT"
    DOTUSDT = "DOTUSDT"
    ADAUSDT = "ADAUSDT"
    MATICUSDT = "MATICUSDT"
    LINKUSDT = "LINKUSDT"
    SOLUSDT = "SOLUSDT"
    TRXUSDT = "TRXUSDT"
    BNBUSDT = "BNBUSDT"
    UNIUSDT = "UNIUSDT"
    AVAXUSDT = "AVAXUSDT"
    BCHUSDT = "BCHUSDT"
    FILUSDT = "FILUSDT"
    AAVEUSDT = "AAVEUSDT"
    XLMUSDT = "XLMUSDT"
    XMRUSDT = "XMRUSDT"
    ALGOUSDT = "ALGOUSDT"
    XTZUSDT = "XTZUSDT"
    ATOMUSDT = "ATOMUSDT"


class TimeZoneInfo(BaseModel):
    """
    Model for timezone information containing both API offset and IANA name.
    """

    offset: str
    iana_name: str


class TimeZones(Enum):
    """
    Model for supported time zones in the Binance API.
    """

    INDIA = TimeZoneInfo(offset="05:30", iana_name="Asia/Kolkata")
    UTC = TimeZoneInfo(offset="00:00", iana_name="UTC")
    EST = TimeZoneInfo(offset="-05:00", iana_name="America/New_York")
