"""
This module contains the settings which will be used to train the NN and then to make preds
"""

# 1st party imports
import os
from typing import List

# 3rd party imports
import torch

# construct csv file path
csv_file_name = "XRPUSDT_5m_2024_2025.csv"
predicted_file_name = "xrp_usdt_processed.csv"
csv_file_path = os.path.join(os.path.dirname(__file__), csv_file_name)
processed_file_path = os.path.join(os.path.dirname(__file__), predicted_file_name)

# construct model paths
model_save_path = os.path.join(os.path.dirname(__file__), "trained_model.pth")


class Constants:
    """
    Constants for training the NN and then to make preds.
    """

    # PATHS
    CSV_FILE_PATH: str = csv_file_path
    PROCESSED_FILE_PATH: str = processed_file_path
    MODEL_SAVE_PATH: str = model_save_path

    # DATASET PARAMS
    COL_LAGS: List[int] = [1, 3, 5, 10, 15, 20]
    TRAIN_RATIO: float = 0.8
    ATR_COL_NAME: str = "atr_2"
    COLS_TO_DROP = [
        "timestamp",
        "close_time",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "number_of_trades",
        "quote_asset_volume",
    ]
    SCALABLE_COLS: List[str] = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bollinger_bands_0_diff",
        "vwap_1",
        "atr_2",
        "rsi_3_centered_diff",
    ]
    COLS_TO_STACK: List[str] = [
        "volume",
        "bollinger_bands_0_diff",
        "vwap_1",
        "atr_2",
        "rsi_3_centered_diff",
    ]
    R_MULTIPLE_REWARD: float = 1.75
    MAX_ALLOWED_TRADE_LENGTH: int = 36
    TARGET_COL: List[str] = ["r_multiple"]

    # MODEL ARCHITECTURE
    DROPOUT: float = 0.2
    FC1_INPUT_SIZE: int = 64
    FC2_INPUT_SIZE: int = 32

    # TRAINING PARAMS
    EPOCHS: int = 150
    BATCH_SIZE: int = 256
    VAL_PATIENCE: int = 10
    GRADIENT_CLIP: float = 1.0
    LEARNING_RATE: float = 0.001
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
