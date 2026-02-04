"""
This module contains the models for the strategies.
"""

# 1st party imports
import json
from enum import Enum
from pathlib import Path
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
        20, ge=1, le=500, description="The threshold for ADX calculation."
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
    allow_dynamic_risk_reward: bool = Field(
        True,
        description="Allow dynamic risk reward ratio based on market conditions.",
    )

    @field_validator("minimum_history", mode="after")
    @classmethod
    def minimize_history(cls, value: int):
        """minimize the history required."""
        return min(value, 500)

    @classmethod
    def from_file(cls, file_path: str) -> "SupertrendStrategyConfig":
        """
        Load a strategy configuration from a JSON file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file content is not valid JSON or contains invalid keys.
        """

        # load the file
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # load the json
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file: {path}") from exc

        # validate the json
        if not isinstance(data, dict):
            raise ValueError("Strategy config must be a JSON object.")

        # validate the keys
        invalid_keys = sorted(set(data) - set(cls.model_fields))
        if invalid_keys:
            invalid_key_list = ", ".join(invalid_keys)
            raise ValueError(f"Invalid config keys: {invalid_key_list}")

        # return the config
        return cls(**data)

    def dump_to_file(self, file_path: str) -> None:
        """
        Dump the strategy configuration to a JSON file.
        """

        # dump the file
        path = Path(file_path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        # dump the json
        payload = self.model_dump(exclude_none=False, exclude_unset=False)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
