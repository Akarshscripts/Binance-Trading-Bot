"""
This module contains the settings for the neural network
"""

# 3rd party imports
from torch import cuda
from pydantic import Field
from pydantic_settings import BaseSettings

# TODO: Update with new values


class Settings(BaseSettings):

    # hyer_parameters for the model
    LSTM_INPUT_SIZE: int = Field(
        default=5, description="Input size for the LSTM layer."
    )
    LSTM_HIDDEN_SIZE: int = Field(
        default=128, description="Hidden size for the LSTM layer."
    )
    LSTM_NUM_LAYERS: int = Field(
        default=3, description="Number of layers for the LSTM layer."
    )
    LSTM_DROPOUT: float = Field(default=0.2, description="Dropout for the LSTM layer.")

    # 1st FC layer params
    FC1_OUT_FEATURES: int = Field(
        default=256, description="Output features for the 1st FC layer."
    )

    # dropout params
    DROPOUT: float = Field(default=0.2, description="Dropout for the model.")

    # 2nd FC layer params
    FC2_OUT_FEATURES: int = Field(
        default=128, description="Output features for the 2nd FC layer."
    )

    # final FC layer params
    LINEAR_OUT_FEATURES: int = Field(
        default=5,
        description="Output features for the final linear layer. (direction, max value, min value)",
    )

    # device params
    DEVICE: str = Field(default="cpu", description="Device to use for training.")

    # Length of indicators
    INDICATOR_EMA1_LENGTH: int = Field(
        default=20, description="Length for the first EMA indicator."
    )
    INDICATOR_EMA2_LENGTH: int = Field(
        default=50, description="Length for the second EMA indicator."
    )
    INDICATOR_RSI_LENGTH: int = Field(
        default=20, description="Length for the RSI indicator."
    )
    INDICATOR_ADX_LENGTH: int = Field(
        default=14, description="Length for the ADX indicator."
    )

    # the range of candles to predict
    PREDICT_CANDLES: int = Field(default=5, description="Number of candles to predict.")


# create instance
project_settings = Settings()

# update device param
project_settings.DEVICE = "cuda" if cuda.is_available() else "cpu"
