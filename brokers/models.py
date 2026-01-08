"""
This module contains the models used in the brokers
"""

# 1st party import
from enum import Enum
from typing import Optional
from datetime import datetime

# 3rd party import
from pydantic import BaseModel

# local imports
from binance_api import BinanceSymbols, ChartIntervals


class TradeType(str, Enum):
    """
    Trade type dataclass to store trade type details.
    """

    LONG = "long"
    SHORT = "short"


class Position(BaseModel):
    """
    Position dataclass to store position details.
    """

    # symbol details
    symbol: BinanceSymbols
    interval: ChartIntervals

    # trade details
    entry_price: float
    exit_price: float
    stop_loss: float
    trade_type: TradeType
    position_size: int
    risk_reward_ratio: float

    # P/L details
    profit: float = 0
    loss: float = 0
    total_brokerage: float = 0
    open_brokerage: float = 0
    close_brokerage: float = 0

    # trade age details
    age: int = 0
    trade_end_time: Optional[datetime] = None
    trade_start_time: Optional[datetime] = None
