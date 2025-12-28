"""
This module handles the dataset creation and conversion for stock direction classification.

1. I can provide data to dm either as csv or a df
2. The dm should apply indicators, calculate min, max and return cols
3. It can scale the columns i want and apply log to customm cols, The scaling can be done by
    spliting or onto the whole df.
4. It should return tensors either for split or for the whole df
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
    Manages stock data preprocessing, scaling, and tensor generation for ML models.

    Handles the complete data pipeline: loading raw data, computing technical indicators,
    labeling based on future returns, scaling features, and creating sequences for training.
    """

    def __init__(
        self,
        device: str = "cpu",
        split_ratio: float = 0.8,
        sequence_length: int = 192,
        csv_file: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the DataManager.

        Args:
            sequence_length: Number of time steps in each input sequence.
            device: Target device for tensors ('cpu' or 'cuda').
        """

        # setup class vars
        self.device = device
        self.split_ratio = split_ratio
        self.sequence_length = sequence_length

        # variables used later
        self.df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        # scaler instance
        self.scaler = StandardScaler()

        # load the data into the class
        if csv_file:
            self.df = pd.read_csv(csv_file)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError(
                "Either 'csv_file' or 'df' should be provided to load the data."
            )

    def __to_tensor(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors on the configured device.

        Args:
            X: Input feature array.
            Y: Target label array.

        Returns:
            Tuple of (X_tensor, Y_tensor) on the configured device.
        """
        X_t = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        Y_t = (
            torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        return X_t, Y_t

    def __calculate_indicators(
        self,
        candles_open: np.ndarray,
        candles_close: np.ndarray,
        candles_high: np.ndarray,
        candles_low: np.ndarray,
        indicators: List[Indicator],
    ) -> List[IndicatorColumn]:
        """
        Calculate technical indicators and add them to the dataframe.

        Args:
            candles_open (np.ndarray): List of opening prices.
            candles_close (np.ndarray): List of closing prices.
            candles_high (np.ndarray): List of high prices.
            candles_low (np.ndarray): List of low prices.
            indicators (List[Indicator]): List of technical indicators to compute.

        Returns:
            List[IndicatorColumn]: List of column names created for the indicators.
        """

        # created columns
        result = []

        # populate indicators into their column
        for idx, indicator in enumerate(indicators):

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

    def preprocess(
        self,
        indicators: list[Indicator],
        threshold: float,
        look_ahead: int = 5,
    ) -> None:
        """
        Compute technical indicators and generate labels based on future returns.

        Args:
            indicators: List of technical indicators to compute.
            threshold: Return threshold for labeling (label=1 if return > threshold,
                       label=2 if return < -threshold, else label=0).
            look_ahead: Number of future candles to consider for max/min calculation.
        """

        # dataframe cols
        req_cols = {"open", "close", "high", "low"}
        if not req_cols.issubset(self.df.columns):
            missing_cols = req_cols - set(self.df.columns)
            raise ValueError(f"Missing required columns: {missing_cols}")

        # get values
        candles_open = self.df["open"].to_numpy(np.float64)
        candles_close = self.df["close"].to_numpy(np.float64)
        candles_high = self.df["high"].to_numpy(np.float64)
        candles_low = self.df["low"].to_numpy(np.float64)

        # calculate indicators
        indicator_cols = self.__calculate_indicators(
            candles_open,
            candles_close,
            candles_high,
            candles_low,
            indicators,
        )

        # populate indicator cols
        for col in indicator_cols:
            if col.is_main_column:
                self.df[col.name] = col.values

        # calculate future max, min and return
        future_max, future_min, returns = self.__calculate_max_min_return(
            look_ahead,
            candles_high,
            candles_low,
            candles_close,
        )

        # populate df
        self.df["return"] = returns
        self.df["future_max"] = future_max
        self.df["future_min"] = future_min

        # Assign labels: 0=hold, 1=buy, 2=sell
        self.df["label"] = 0
        self.df.loc[self.df["return"] > threshold, "label"] = 1
        self.df.loc[self.df["return"] < -threshold, "label"] = 2

        # trim the df with the max length of the indicator so all indicators have value
        max_ind_val = max([i.LENGTH for i in indicators])
        self.df = self.df.iloc[max_ind_val:].reset_index(drop=True)

    def scale(
        self,
        cols: list[str],
        log_cols: list[str] | None = None,
        scale_whole: bool = False,
    ) -> None:
        """
        Scale specified columns using StandardScaler.

        Args:
            cols: List of column names to scale.
            log_cols: Optional list of columns to apply log1p transformation before scaling.
            scale_whole: If True, scale the entire dataframe. If False, fit scaler on
                train data only and transform both train and test sets.

        Returns:
            None: Modifies dataframes in place.
        """

        # check if the split has already been done
        if not scale_whole and (self.train_df is None or self.test_df is None):

            # create the split
            self.split()

        # if the whole df is to be scaled
        elif scale_whole:

            # copt the df
            self.df = self.df.copy()

            # scale log cols
            if log_cols:
                self.df[log_cols] = np.log1p(self.df[log_cols])

            # Fit scaler on train data only to prevent data leakage
            self.scaler.fit(self.df[cols])

            # scale all the data
            self.df[cols] = self.scaler.transform(self.df[cols])
            return

        # create copies
        self.train_df = self.train_df.copy()
        self.test_df = self.test_df.copy()

        # scale log cols
        if log_cols:
            self.train_df[log_cols] = np.log1p(self.train_df[log_cols])
            self.test_df[log_cols] = np.log1p(self.test_df[log_cols])

        # Fit scaler on train data only to prevent data leakage
        self.scaler.fit(self.train_df[cols])

        # scale all the data
        self.train_df[cols] = self.scaler.transform(self.train_df[cols])
        self.test_df[cols] = self.scaler.transform(self.test_df[cols])

    def split(self, split_ratio: Optional[float] = None) -> None:
        """
        Split data chronologically into train and test sets.

        Args:
            split_ratio: Fraction of data to use for training.
        """

        if not split_ratio:
            split_ratio = self.split_ratio

        idx = int(len(self.df) * split_ratio)
        self.train_df = self.df[:idx].reset_index(drop=True)
        self.test_df = self.df[idx:].reset_index(drop=True)

    def _create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create sliding window sequences from the DataFrame.

        Args:
            df: Source DataFrame containing the data.
            feature_cols: List of column names to use as input features.
            target_cols: List of column names to use as targets.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: (X, Y) lists containing
                sequences of shape (sequence_length, num_features) and targets.
        """
        X, Y = [], []
        fv = df[feature_cols].values
        tv = df[target_cols].values

        for i in range(self.sequence_length, len(df)):
            X.append(fv[i - self.sequence_length : i])
            Y.append(tv[i])

        return X, Y

    def get_tensors(
        self,
        feature_cols: List[str],
        target_cols: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tensors for model from the whole dataframe.

        Args:
            feature_cols: List of column names to use as input features.
            target_cols: List of column names to use as targets.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y) tensors on the configured device.
        """
        X, Y = self._create_sequences(self.df, feature_cols, target_cols)
        return self.__to_tensor(X, Y)

    def get_test_tensors(
        self,
        feature_cols: List[str],
        target_cols: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate test tensors ready for model testing.

        Args:
            feature_cols: List of column names to use as input features.
            target_cols: List of column names to use as targets.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y) tensors on the configured device.
        """
        X_test, Y_test = self._create_sequences(self.test_df, feature_cols, target_cols)
        return self.__to_tensor(X_test, Y_test)

    def get_train_tensors(
        self,
        feature_cols: List[str],
        target_cols: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate train tensors ready for model training.

        Args:
            feature_cols: List of column names to use as input features.
            target_cols: List of column names to use as targets.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y) tensors on the configured device.
        """
        X_train, Y_train = self._create_sequences(
            self.train_df, feature_cols, target_cols
        )
        return self.__to_tensor(X_train, Y_train)
