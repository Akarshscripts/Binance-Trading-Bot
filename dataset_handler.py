"""
This module handles the dataset creation and conversion for stock price prediction.
"""

# 1st party imports
from typing import Tuple, List, Optional

# 3rd party imports
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# local imports
from settings import project_settings
from binance_api.indicators import Indicator


class DataManager:
    """
    A class to manage stock data loading, preprocessing, and tensor conversion.

    This class handles:
        - Loading CSV data with stock price information
        - Preprocessing (column filtering, timestamp parsing, return calculation)
        - Train/test splitting
        - Creating input sequences for LSTM models
        - Converting data to PyTorch tensors

    Attributes:
        csv_file (str): Path to the CSV file containing stock data.
        train_ratio (float): Ratio of data to use for training.
        sequence_length (int): Length of input sequences for the model.
        device (str): Device to run the model on.
        df (pd.DataFrame): The loaded and preprocessed dataframe.
        train_df (pd.DataFrame): Training data subset.
        test_df (pd.DataFrame): Testing data subset.
        indicators (List[Indicator]): List of technical indicators to compute.
        split_idx (int): Index at which to split training and testing data.
    """

    def __init__(
        self,
        csv_file: str,
        train_ratio: float = 0.8,
        sequence_length: int = 192,
        device: str = "cpu",
        indicators: Optional[List[Indicator]] = None,
    ):
        """
        Initialize the DataManager with a CSV file.

        Args:
            csv_file (str): Path to the CSV file containing stock data.
            train_ratio (float): Ratio of data to use for training (0-1).
                Defaults to 0.8.
            sequence_length (int): Length of input sequences for the model.
                Defaults to 192.
            device (str): Device to place tensors on (e.g., "cpu" or "cuda").
                Defaults to "cpu".
            indicators (List[Indicator]): List of technical indicators to compute
                during preprocessing. Defaults to empty list.
        """

        # initialize params
        self.csv_file = csv_file
        self.train_ratio = train_ratio
        self.sequence_length = sequence_length
        self.device = device

        # initialize dataframes
        self.df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        # initialize indicators and scalers
        self.indicators = indicators or []
        self.std_scaler = StandardScaler()

        # populate the base df and filter out cols
        self.df = pd.read_csv(self.csv_file)

        # split
        self.split_idx = int(len(self.df) * self.train_ratio)
        self.train_df = self.df[: self.split_idx].reset_index(drop=True)
        self.test_df = self.df[self.split_idx :].reset_index(drop=True)

    def preprocess(self, threshold: float, look_ahead: int = 5) -> None:
        """
        Preprocess the data: compute indicators, calculate future prices and returns.

        This method:
            1. Computes all configured technical indicators
            2. Calculates future maximum and minimum prices within a prediction window
            3. Computes returns based on the prediction window
            4. Assigns labels based on return threshold (0=neutral, 1=up, 2=down)
            5. Re-splits the data into train/test sets

        Args:
            threshold (float): Threshold for label assignment. Returns above threshold
                are labeled as 1 (up), below -threshold as 2 (down), otherwise 0 (neutral).
            look_ahead (int): Number of future candles to look ahead for calculating
                future max/min prices. Defaults to 5.
        """

        # regression cols to be scaled (use list for deterministic ordering)
        scalable_cols: List[str] = []

        # get the fields used by indicators
        candles_open = self.df["open"].tolist()
        candles_close = self.df["close"].tolist()
        candles_high = self.df["high"].tolist()
        candles_low = self.df["low"].tolist()

        # populate indicators into their column
        for idx, indicator in enumerate(self.indicators):

            # create col name
            col_name = f"{indicator.NAME}_{idx}"

            # handle each indicator type with its specific signature
            if indicator.NAME.lower() == "ema":

                # EMA takes candles_open
                self.df[col_name] = indicator.get_value(candles_open)

                # create the ema_diff col (percentage difference for better scaling)
                diff_col_name = f"{col_name}_diff"
                self.df[diff_col_name] = (
                    self.df[col_name] - self.df["open"]
                ) / self.df["open"]

                # add to be scaled later
                scalable_cols.append(diff_col_name)

            elif indicator.NAME.lower() == "rsi":

                # RSI takes candles_close
                self.df[col_name] = indicator.get_value(candles_close)

                # scale by dividing by 100
                self.df[col_name] = self.df[col_name] / 100

            elif indicator.NAME.lower() == "adx":

                # ADX takes (high, low, close) and returns tuple of 3 lists
                adx_vals, plus_di, minus_di = indicator.get_value(
                    candles_high, candles_low, candles_close
                )

                # populate all three columns, scaled by 100
                self.df[col_name] = [v / 100 for v in adx_vals]
                self.df[f"{col_name}_plus_di"] = [v / 100 for v in plus_di]
                self.df[f"{col_name}_minus_di"] = [v / 100 for v in minus_di]

        # The maximum high, low and returns in the NEXT N candles
        future_maxes, future_mins, returns = [], [], []

        # iterate
        for idx in range(len(self.df)):

            # current window
            curr_highs = candles_high[idx : idx + look_ahead]
            curr_lows = candles_low[idx : idx + look_ahead]

            # update
            future_maxes.append(max(curr_highs))
            future_mins.append(min(curr_lows))

            # update returns
            if idx + look_ahead >= len(self.df):
                # current candle and the last candle
                curr_close, nth_close = candles_close[idx], candles_close[-1]

            else:
                # current candle and the nth candle
                curr_close, nth_close = (
                    candles_close[idx],
                    candles_close[idx + look_ahead],
                )

            # append
            returns.append((nth_close - curr_close) / curr_close)

        # populate df
        self.df["return"] = returns
        self.df["future_min"] = future_mins
        self.df["future_max"] = future_maxes

        # add cols for scaling
        scalable_cols.append("open")
        scalable_cols.append("future_min")
        scalable_cols.append("future_max")

        # calculate labels
        self.df["label"] = 0
        self.df.loc[self.df["return"] > threshold, "label"] = 1
        self.df.loc[self.df["return"] < -threshold, "label"] = 2

        # split the test data before scaling
        self.test_df = self.df[self.split_idx :].reset_index(drop=True)
        self.train_df = self.df[: self.split_idx].reset_index(drop=True)

        # separate price cols (need log transform) from percentage cols (already normalized)
        price_cols = ["open", "future_min", "future_max"]

        # transform price columns with log (prices are always positive)
        self.train_df[price_cols] = np.log1p(self.train_df[price_cols])
        self.test_df[price_cols] = np.log1p(self.test_df[price_cols])

        # scale the train data using standard scaler (fit only on train)
        self.std_scaler.fit(self.train_df[scalable_cols])
        self.train_df[scalable_cols] = self.std_scaler.transform(
            self.train_df[scalable_cols]
        )

        # transform test data using the fitted scaler
        self.test_df[scalable_cols] = self.std_scaler.transform(
            self.test_df[scalable_cols]
        )

        # after preprocessing, join the data
        self.df = pd.concat([self.train_df, self.test_df], ignore_index=True)

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_col: List[str],
        target_col: List[str],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create input sequences and target values for the model.

        Generates sliding window sequences of length `sequence_length` from the
        input dataframe for use with sequential models like LSTM.

        Args:
            df (pd.DataFrame): Dataframe to create sequences from.
            feature_col (List[str]): Column names to use as input features.
            target_col (List[str]): Column names to use as target values.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: A tuple of (X, Y) where:
                - X: List of input sequences, each of shape (sequence_length, num_features)
                - Y: List of corresponding target values, each of shape (num_targets,)
        """

        # outputs
        X, Y = [], []

        # extract values once to avoid repeated df access
        feature_values = df[feature_col].values
        target_values = df[target_col].values

        # loop in seq length
        for i in range(self.sequence_length, len(df)):

            # append
            X.append(feature_values[i - self.sequence_length : i])
            Y.append(target_values[i])

        # return
        return X, Y

    def get_train_data(
        self,
        feature_col: List[str],
        target_col: List[str],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get training sequences and targets.

        Args:
            feature_col (List[str]): Column names to use as input features.
            target_col (List[str]): Column names to use as target values.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Training (X, Y) sequences.
        """
        return self.create_sequences(self.train_df, feature_col, target_col)

    def get_test_data(
        self,
        feature_col: List[str],
        target_col: List[str],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get testing sequences and targets.

        Args:
            feature_col (List[str]): Column names to use as input features.
            target_col (List[str]): Column names to use as target values.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Testing (X, Y) sequences.
        """
        return self.create_sequences(self.test_df, feature_col, target_col)

    def to_tensors(
        self,
        X: List[np.ndarray],
        Y: List[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert lists of numpy arrays to PyTorch tensors.

        Args:
            X (List[np.ndarray]): Input sequences as list of numpy arrays.
            Y (List[np.ndarray]): Target values as list of numpy arrays.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (X_tensor, Y_tensor) where:
                - X_tensor: Shape (num_samples, sequence_length, num_features)
                - Y_tensor: Shape (num_samples, num_targets, 1)
        """
        X_tensor = torch.tensor(np.array(X, np.float32)).to(self.device)
        Y_tensor = torch.tensor(np.array(Y, np.float32)).unsqueeze(-1).to(self.device)
        return X_tensor, Y_tensor

    def get_train_tensors(
        self,
        feature_col: List[str],
        target_col: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training data as PyTorch tensors.

        Args:
            feature_col (List[str]): Column names to use as input features.
            target_col (List[str]): Column names to use as target values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Training (X, Y) tensors where:
                - X: Shape (num_samples, sequence_length, num_features)
                - Y: Shape (num_samples, num_targets, 1)
        """
        X, Y = self.get_train_data(feature_col, target_col)
        return self.to_tensors(X, Y)

    def get_test_tensors(
        self,
        feature_col: List[str],
        target_col: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get testing data as PyTorch tensors.

        Args:
            feature_col (List[str]): Column names to use as input features.
            target_col (List[str]): Column names to use as target values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Testing (X, Y) tensors where:
                - X: Shape (num_samples, sequence_length, num_features)
                - Y: Shape (num_samples, num_targets, 1)
        """
        X, Y = self.get_test_data(feature_col, target_col)
        return self.to_tensors(X, Y)
