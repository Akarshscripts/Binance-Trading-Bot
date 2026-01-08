"""
This module handles dataset creation and conversion for stock direction classification.

The DataManager class provides functionality to:
    1. Load data from CSV files or pandas DataFrames
    2. Compute technical indicators and generate labels based on future price conditions
    3. Scale columns using RobustScaler with optional log transformation
    4. Split data chronologically into train/test sets
    5. Convert processed data to PyTorch tensors for model training/inference
"""

# 1st party imports
from dataclasses import dataclass
from typing import Tuple, List, Optional

# 3rd party imports
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# local imports
from indicators import Indicator, AvailableIndicators


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
    Manages financial data preprocessing, scaling, and tensor generation for ML models.

    Handles the complete data pipeline including loading raw data, computing technical
    indicators, scaling features, splitting into train/test sets, and converting to
    PyTorch tensors for model training and inference.
    """

    def __init__(
        self,
        device: str = "cpu",
        split_ratio: float = 0.8,
        csv_file: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the DataManager.

        Args:
            device: Target device for tensors ('cpu' or 'cuda').
        """

        # setup class vars
        self.device = device
        self.split_ratio = split_ratio

        # variables used later
        self.df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        # load the data into the class
        if csv_file:
            self.df = pd.read_csv(csv_file)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError(
                "Either 'csv_file' or 'df' should be provided to load the data."
            )

    def __calculate_indicators(
        self,
        candles_open: np.ndarray,
        candles_close: np.ndarray,
        candles_high: np.ndarray,
        candles_low: np.ndarray,
        candles_volume: np.ndarray,
        indicators: List[Indicator],
    ) -> List[IndicatorColumn]:
        """
        Calculate technical indicators and add them to the dataframe.

        Args:
            candles_open (np.ndarray): List of opening prices.
            candles_close (np.ndarray): List of closing prices.
            candles_high (np.ndarray): List of high prices.
            candles_low (np.ndarray): List of low prices.
            candles_volume (np.ndarray): List of volumes.
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

                # RSI takes candles_close
                rsi_values = indicator.get_value(candles_close)

                # get the rsi diff from the centre (50)
                rsi_centered_diff = rsi_values - 50

                # append the indicator result
                result.append(
                    IndicatorColumn(
                        name=f"{col_name}_centered_diff", values=rsi_centered_diff
                    )
                )

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

            elif indicator.NAME == AvailableIndicators.BOLLINGER_BANDS:

                # Bollinger Bands takes candles_close and volume
                upper_band, _, lower_band = indicator.get_value(
                    candles_close=candles_close, candles_volume=candles_volume
                )

                # calculate the diff b/w upper and lower band
                diff_band = upper_band - lower_band

                # append the indicator results
                result.append(
                    IndicatorColumn(name=f"{col_name}_diff", values=diff_band)
                )

            elif indicator.NAME == AvailableIndicators.VWAP:

                # VWAP takes candles_high, candles_low, candles_close and candles_volume
                vwap_values, _, _ = indicator.get_value(
                    candles_high=candles_high,
                    candles_low=candles_low,
                    candles_close=candles_close,
                    candles_volume=candles_volume,
                )

                result.append(IndicatorColumn(name=col_name, values=vwap_values))

            elif indicator.NAME == AvailableIndicators.ATR:

                # ATR takes candles_high, candles_low and candles_close
                atr_values = indicator.get_value(
                    candles_high=candles_high,
                    candles_low=candles_low,
                    candles_close=candles_close,
                )

                # append the indicator result
                result.append(IndicatorColumn(name=col_name, values=atr_values))

            else:
                raise ValueError(f"Unsupported indicator: {indicator.NAME}")

        # return the indicator results
        return result

    def compute_indicators(self, indicators: List[Indicator]) -> None:
        """
        Compute technical indicators for the dataframe.

        Args:
            indicators: List of technical indicators to compute.
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
        candles_volume = self.df["volume"].to_numpy(np.float64)

        # calculate indicators
        indicator_cols = self.__calculate_indicators(
            candles_open,
            candles_close,
            candles_high,
            candles_low,
            candles_volume,
            indicators,
        )

        # populate indicator cols
        for col in indicator_cols:
            if col.is_main_column:
                self.df[col.name] = col.values

        # trim the df with the max length of the indicator so all indicators have value
        max_ind_val = max([i.LENGTH for i in indicators])
        self.df = self.df.iloc[max_ind_val:].reset_index(drop=True)

    def add_r_multiple(
        self, trade_length: int, atr_col_name: str, reward_r: float = 2
    ) -> None:
        """
        Calculate R-multiple values for each candle based on ATR-based stop loss and take profit levels.

        This method calculates the risk-reward outcome for each position by:
        1. Setting stop loss at current close minus ATR
        2. Setting take profit at current close plus (ATR * reward_r)
        3. Checking which level is hit first within the trade_length window

        Args:
            trade_length: Maximum number of candles to hold the trade.
            atr_col_name: Name of the ATR column in the DataFrame.
            reward_r: Risk-reward ratio for take profit calculation. Default is 2.

        Returns:
            None. Modifies self.df in place by adding an 'r_multiple' column.
        """

        # get the close and the atr values from the df
        close = self.df["close"].to_numpy(np.float64)
        lows = self.df["low"].to_numpy(np.float64)
        highs = self.df["high"].to_numpy(np.float64)
        atr = self.df[atr_col_name].to_numpy(np.float64)

        # length
        n = len(self.df)

        # calculate the r multiple
        r_multiples = np.zeros(n, dtype=np.float64)

        # iterate
        for idx in range(n - trade_length):

            # calculate the risk, sl and tp
            curr_risk = atr[idx]
            curr_sl = close[idx] - curr_risk
            curr_tp = close[idx] + (curr_risk * reward_r)

            # calculate the index where tp and sl are hit
            tp_hit_idx = np.where(highs[idx : idx + trade_length] >= curr_tp)[0]
            sl_hit_idx = np.where(lows[idx : idx + trade_length] <= curr_sl)[0]

            # calculate the r multiple when none are hit
            if tp_hit_idx.size == 0 and sl_hit_idx.size == 0:
                curr_exit = close[idx + trade_length - 1]
                r_multiples[idx] = (curr_exit - close[idx]) / curr_risk
                continue

            # extract the first indexes
            tp_hit_idx = tp_hit_idx[0] if tp_hit_idx.size > 0 else trade_length + 1
            sl_hit_idx = sl_hit_idx[0] if sl_hit_idx.size > 0 else trade_length + 1

            # when tp is hit
            if tp_hit_idx < sl_hit_idx:
                r_multiples[idx] = reward_r

            # when sl is hit
            else:
                r_multiples[idx] = -1

        # add the r multiple to the df
        self.df["r_multiple"] = r_multiples

    def scale_cols(
        self,
        cols: List[str],
        chunk_size: int = 100,
    ) -> None:
        """
        Scale specified columns using StandardScaler.

        Args:
            cols: List of column names to scale.
            chunk_size: Number of rows to process at a time.

        Returns:
            None: Modifies dataframes in place.
        """

        # hold scaled values
        n = len(self.df)
        scaled = np.full((n, len(cols)), np.nan)

        # create the scaler
        scaler = RobustScaler()

        # fit
        for start in range(1, n, chunk_size):
            end = min(start + chunk_size, n)
            scaler.fit(self.df[cols].iloc[:start])
            scaled[start:end] = scaler.transform(self.df[cols].iloc[start:end])

        # scale all the data
        self.df[cols] = scaled

    def stack_features(
        self,
        feature_cols: List[str],
        lags: List[int],
        dropna: bool = True,
    ) -> None:
        """
        Create stacked (lagged) features for time-series ML.

        Args:
            feature_cols (list[str]): Feature column names to stack.
            lags (list[int]): Lags to use (e.g. [1, 3, 5, 10]).
            dropna (bool): Whether to drop rows with NaNs after stacking.

        Returns:
            None: Modifies dataframes in place.
        """

        df = self.df.copy()

        # Create lagged features
        for col in feature_cols:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # Drop rows with incomplete history
        if dropna:
            df = df.dropna().reset_index(drop=True)

        # update the df
        self.df = df

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

    def get_tensors(
        self,
        target_cols: List[str],
        feature_cols: List[str] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tensors for model from the whole dataframe.

        Args:
            target_cols: List of column names to use as targets.
            feature_cols (None | List[str]): List of column names to use as input features or None if all cols except target cols are to be used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y) tensors on the configured device.
        """

        # get the feature cols
        if feature_cols is None:
            feature_cols = [col for col in self.df.columns if col not in target_cols]

        # get the values
        X = self.df[feature_cols].values
        Y = self.df[target_cols].values

        # convert to tensors
        X_t = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(np.array(Y), dtype=torch.float32).to(self.device)
        return X_t, Y_t

    def get_test_tensors(
        self,
        target_cols: List[str],
        feature_cols: List[str] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate test tensors ready for model testing.

        Args:
            target_cols: List of column names to use as targets.
            feature_cols (None | List[str]): List of column names to use as input features or None if all cols except target cols are to be used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y) tensors on the configured device.
        """

        # check if test df is available
        if self.test_df is None:
            self.split()

        # get the feature cols
        if feature_cols is None:
            feature_cols = [
                col for col in self.test_df.columns if col not in target_cols
            ]

        # get the values
        X = self.test_df[feature_cols].values
        Y = self.test_df[target_cols].values

        # convert to tensors
        X_t = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(np.array(Y), dtype=torch.float32).to(self.device)
        return X_t, Y_t

    def get_train_tensors(
        self,
        target_cols: List[str],
        feature_cols: List[str] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate train tensors ready for model training.

        Args:
            target_cols: List of column names to use as targets.
            feature_cols (None | List[str]): List of column names to use as input features or None if all cols except target cols are to be used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y) tensors on the configured device.
        """

        # check if train df is available
        if self.train_df is None:
            self.split()

        # get the feature cols
        if feature_cols is None:
            feature_cols = [
                col for col in self.train_df.columns if col not in target_cols
            ]

        # get the values
        X = self.train_df[feature_cols].values
        Y = self.train_df[target_cols].values

        # convert to tensors
        X_t = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(np.array(Y), dtype=torch.float32).to(self.device)
        return X_t, Y_t
