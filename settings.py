"""
This module contains the settings which will be used to train the NN and then to make preds
"""

# 1st party imports
import os
from typing import List

# 3rd party imports
import torch

# construct csv file path
csv_file_name = "XRPUSDT_15m_Jan_to_Dec_2025.csv"
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
    TARGET_COL: List[str] = ["label"]
    TRAIN_RATIO: float = 0.8
    LABEL_THRESH: float = 0.003
    LOOK_AHEAD_CANDLES: int = 8
    COLS_TO_SCALE_LOG: List[str] = ["open", "volume", "future_min", "future_max"]
    SCALABLE_COLS: List[str] = [
        "open",
        "volume",
        "future_min",
        "future_max",
        "ema_0_diff",
        "ema_1_diff",
    ]
    FEATURE_COLS: List[str] = [
        "open",
        "volume",
        "ema_0_diff",
        "ema_1_diff",
        "rsi_2",
        "adx_3",
        "return",
        "label",
    ]
    COLS_TO_EXTRACT: List[str] = [
        "open",
        "volume",
        "ema_0_diff",
        "ema_1_diff",
        "rsi_2",
        "adx_3",
        "return",
        "future_min",
        "future_max",
        "label",
    ]

    # MODEL ARCHITECTURE
    INPUT_SIZE: int = 8
    NUM_CLASSES: int = 3
    FC_DROPOUT: float = 0.4
    LSTM_DROPOUT: float = 0.4
    LSTM_NUM_LAYERS: int = 2
    LSTM_HIDDEN_SIZE: int = 32
    FC1_OUT_FEATURES: int = 128
    OUT_CLASS: List[str] = ["neutral", "up", "down"]

    # TRAINING PARAMS
    EPOCHS: int = 150
    BATCH_SIZE: int = 256
    VAL_PATIENCE: int = 10
    GRADIENT_CLIP: float = 1.0
    LEARNING_RATE: float = 0.001
    SEQUENCE_LENGTH: int = 192
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss weights for [Neutral, Up, Down]
    CROSS_ENTROPY_LOSS_WEIGHTS: List[float] = [1, 1, 1]
