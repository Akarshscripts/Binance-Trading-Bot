"""
This module contains the models for the strategies.
"""

# 1st party imports
from enum import Enum
from typing import Dict, Any, Optional

# 3rd party imports
from pydantic import BaseModel


class TradeAction(str, Enum):
    """
    This class contains the trade action for the strategy.
    """

    ENTER_LONG = "BUY"
    ENTER_SHORT = "SELL"
    NEUTRAL = "NEUTRAL"


class PredictionOutput(BaseModel):
    """
    This class contains the prediction output for the strategy.
    """

    # price variables
    entry_price: float
    exit_price: float
    stop_loss: float

    # action variables
    action: TradeAction = TradeAction.NEUTRAL

    # risk reward ratio
    risk_reward_ratio: float

    # indicator details
    indicator_details: Optional[Dict[str, Any]] = None


class CandleData(BaseModel):
    """
    This class stores a single candle data.
    """

    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
