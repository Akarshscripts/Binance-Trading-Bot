"""
This module contains the models used in the brokers
"""

# 1st party import
from enum import Enum
from typing import Optional
from datetime import datetime

# 3rd party import
from pydantic import BaseModel, field_serializer


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
    symbol: Enum
    interval: Enum

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

    @field_serializer("symbol")
    def serialize_symbol(self, symbol: Enum):
        return symbol.name
