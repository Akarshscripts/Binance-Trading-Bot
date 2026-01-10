"""
This module contains the models for the strategies.
"""

# 1st party imports
from enum import Enum
from typing import Dict, Any, Optional

# 3rd party imports
from pydantic import BaseModel, Field, field_validator


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


class SupertrendStrategyConfig(BaseModel):
    """This class is used to configure the Supertrend strategy."""

    # supertrend params
    factor: int = Field(
        ..., ge=1, le=500, description="The multiplier for ATR to calculate supertrend."
    )
    atr_period: int = Field(
        ..., ge=1, le=500, description="The period for ATR calculation in supertrend."
    )

    # fractal params
    fractal_period: int = Field(
        ..., ge=1, le=500, description="The period for fractal calculation."
    )

    # risk reward params
    risk_reward_ratio: float = Field(
        ..., ge=1, le=10, description="The risk reward ratio for the strategy."
    )

    # adx params
    adx_period: int = Field(
        14, ge=1, le=500, description="The period for ADX calculation."
    )
    adx_smoothing: int = Field(
        14, ge=1, le=500, description="The smoothing for ADX calculation."
    )
    adx_threshold: int = Field(
        25, ge=1, le=500, description="The threshold for ADX calculation."
    )

    # ema params
    ema_1_period: int = Field(
        14, ge=1, le=500, description="The period for EMA 1 calculation."
    )
    ema_2_period: int = Field(
        30, ge=1, le=500, description="The period for EMA 2 calculation."
    )
    ema_3_period: int = Field(
        50, ge=1, le=500, description="The period for EMA 3 calculation."
    )

    # bollinger bands params
    bbands_period: int = Field(
        20, ge=1, le=500, description="The period for Bollinger Bands calculation."
    )

    # rsi params
    rsi_period: int = Field(
        14, ge=1, le=500, description="The period for RSI calculation."
    )

    # internal config
    minimum_history: int = Field(
        500,
        ge=1,
        description="The minimum candle history length for the strategy.",
    )

    @field_validator("minimum_history", mode="after")
    @classmethod
    def minimize_history(cls, value: int):
        """minimize the history required."""
        return min(value, 500)
