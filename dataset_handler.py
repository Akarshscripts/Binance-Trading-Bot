"""
This module handles the dataset creation and conversion for stock direction classification.
"""

# 1st party imports
from dataclasses import dataclass
from typing import Tuple, List, Optional

# 3rd party imports
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# local imports
from binance_api.indicators import Indicator, AvailableIndicators


@dataclass
class IndicatorColumn:
    """
    Represents a column of indicator values to be added to the dataset.

    Attributes:
        name: The name of the indicator column.
        values: Array of indicator values.
        is_main_column: Whether this is a primary indicator column. Defaults to True.
    """

    name: str
    values: np.ndarray
    is_main_column: bool = True


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

    def __calculate_indicators(
        self,
        candles_open: np.ndarray,
        candles_close: np.ndarray,
        candles_high: np.ndarray,
        candles_low: np.ndarray,
    ) -> List[IndicatorColumn]:
        """
        Calculate technical indicators and add them to the dataframe.

        Args:
            candles_open (np.ndarray): List of opening prices.
            candles_close (np.ndarray): List of closing prices.
            candles_high (np.ndarray): List of high prices.
            candles_low (np.ndarray): List of low prices.

        Returns:
            List[IndicatorColumn]: List of column names created for the indicators.
        """

        # created columns
        result = []

        # populate indicators into their column
        for idx, indicator in enumerate(self.indicators):

            # create col name
            col_name = f"{indicator.NAME.value}_{idx}"

            # handle each indicator type with its specific signature
            if indicator.NAME == AvailableIndicators.EMA:

                # EMA takes candles_open
                ema_values: np.ndarray = indicator.get_value(candles_open)
                ema_indicator_res = IndicatorColumn(name=col_name, values=ema_values)

                # create the ema_diff col (percentage difference for better scaling)
                diff_col_values = (ema_values - candles_open) / candles_open
                ema_diff_open_res = IndicatorColumn(
                    name=f"{col_name}_diff", values=diff_col_values
                )

                # append the indicator result
                result.append(ema_indicator_res)
                result.append(ema_diff_open_res)

            elif indicator.NAME == AvailableIndicators.RSI:

                # RSI takes candles_close and scale it b/w 0 - 1
                rsi_values = indicator.get_value(candles_close) / 100

                # append the indicator result
                result.append(IndicatorColumn(name=col_name, values=rsi_values))

            elif indicator.NAME == AvailableIndicators.ADX:

                # ADX takes (high, low, close) and returns tuple of 3 arrays
                adx_vals, plus_di, minus_di = indicator.get_value(
                    candles_high, candles_low, candles_close
                )

                # scale to 0-1 (already numpy arrays from ADX)
                adx_vals = adx_vals / 100
                plus_di = plus_di / 100
                minus_di = minus_di / 100

                # append the indicator results
                result.append(IndicatorColumn(name=col_name, values=adx_vals))
                result.append(
                    IndicatorColumn(
                        name=f"{col_name}_plus_di", values=plus_di, is_main_column=False
                    )
                )
                result.append(
                    IndicatorColumn(
                        name=f"{col_name}_minus_di",
                        values=minus_di,
                        is_main_column=False,
                    )
                )

        # return the indicator results
        return result

    def __calculate_max_min_return(
        self,
        look_ahead: int,
        candles_high: np.ndarray,
        candles_low: np.ndarray,
        candles_close: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate future maximum prices, minimum prices, and returns.

        This method looks ahead a specified number of candles to find the maximum
        high price, minimum low price, and calculates the return based on the
        current close price.

        Args:
            look_ahead: Number of future candles to look ahead.
            candles_high: Array of high prices for each candle.
            candles_low: Array of low prices for each candle.
            candles_close: Array of close prices for each candle.

        Returns:
            Tuple containing:
                - future_maxes: Maximum high prices within the look-ahead window.
                - future_mins: Minimum low prices within the look-ahead window.
                - returns: Calculated returns based on the prediction window.
        """

        # length
        n = len(candles_high)

        # pre-allocate arrays
        future_maxes = np.zeros(n, dtype=np.float64)
        future_mins = np.zeros(n, dtype=np.float64)
        returns = np.zeros(n, dtype=np.float64)

        # calculate rolling max/min using sliding window
        for i in range(n):
            end_idx = min(i + look_ahead, n)
            future_maxes[i] = np.max(candles_high[i:end_idx])
            future_mins[i] = np.min(candles_low[i:end_idx])

        # calculate returns (vectorized)
        # for indices where we can look ahead fully
        valid_end = n - look_ahead
        if valid_end > 0:
            returns[:valid_end] = (
                candles_close[look_ahead:] - candles_close[:valid_end]
            ) / candles_close[:valid_end]

        # for remaining indices, use last close
        if valid_end < n:
            returns[valid_end:] = (
                candles_close[-1] - candles_close[valid_end:]
            ) / candles_close[valid_end:]

        # return
        return future_maxes, future_mins, returns

    def __scale_columns(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cols_to_scale: List[str],
        cols_to_log_scale: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale specified columns using StandardScaler.

        Fits the scaler on training data only to prevent data leakage,
        then transforms both training and test data.

        Args:
            train_df: Training dataframe.
            test_df: Test dataframe.
            cols_to_scale: List of column names to scale using StandardScaler.
            cols_to_log_scale: List of column names to apply log1p transformation
                before scaling. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test dataframes.
        """

        # create copies to avoid modifying original
        train_df = train_df.copy()
        test_df = test_df.copy()

        # transform columns with log if any
        if cols_to_log_scale:
            train_df[cols_to_log_scale] = np.log1p(train_df[cols_to_log_scale])
            test_df[cols_to_log_scale] = np.log1p(test_df[cols_to_log_scale])

        # scale using standard scaler (fit only on train)
        self.std_scaler.fit(train_df[cols_to_scale])
        train_df[cols_to_scale] = self.std_scaler.transform(train_df[cols_to_scale])

        # transform test data using the fitted scaler
        test_df[cols_to_scale] = self.std_scaler.transform(test_df[cols_to_scale])

        # return the train and test dataframe
        return train_df, test_df

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

        # get the price fields
        candles_open = self.df["open"].to_numpy(dtype=np.float64)
        candles_close = self.df["close"].to_numpy(dtype=np.float64)
        candles_high = self.df["high"].to_numpy(dtype=np.float64)
        candles_low = self.df["low"].to_numpy(dtype=np.float64)

        # calculate indicators
        indicator_columns = self.__calculate_indicators(
            candles_open=candles_open,
            candles_close=candles_close,
            candles_high=candles_high,
            candles_low=candles_low,
        )

        # populate df with indicator values
        for col in indicator_columns:

            # only add the main cols
            if col.is_main_column:
                self.df[col.name] = col.values

        # The maximum high, low and returns in the NEXT N candles
        future_maxes, future_mins, returns = self.__calculate_max_min_return(
            look_ahead=look_ahead,
            candles_high=candles_high,
            candles_low=candles_low,
            candles_close=candles_close,
        )

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

    def perform_scaling(
        self, scalable_cols: List[str], cols_to_log_scale: List[str] | None
    ):
        """
        Perform scaling on the specified columns.

        Args:
            scalable_cols (List[str]): List of column names to scale using StandardScaler.
            cols_to_log_scale (List[str] | None): List of column names to apply log1p
                transformation before scaling. Can be None if no log scaling is needed.
        """

        # split the test data before scaling
        self.test_df = self.df[self.split_idx :].reset_index(drop=True)
        self.train_df = self.df[: self.split_idx].reset_index(drop=True)

        # scale the data
        scaled_train, scaled_test = self.__scale_columns(
            self.train_df, self.test_df, scalable_cols, cols_to_log_scale
        )

        # after preprocessing, join the data
        self.df = pd.concat([scaled_train, scaled_test], ignore_index=True)

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
