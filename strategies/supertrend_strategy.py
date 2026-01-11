"""
This module contains the supertrend strategy for backtesting.
"""

# 1st party imports
import logging
from typing import List
from collections import deque

# 3rd party imports
import numpy as np

# local imports
from strategies.models import PredictionOutput, TradeAction, SupertrendStrategyConfig
from indicators import (
    FractalsIndicator,
    SuperTrendIndicator,
    ADX,
    EMA,
    BollingerBands,
    RSI,
)


class SupertrendStrategy:
    """
    Supertrend strategy that combines Supertrend indicator with fractal-based stop losses.
    """

    def __init__(
        self,
        *,
        config: SupertrendStrategyConfig,
    ):
        """
        Initialize the Supertrend strategy.

        Args:
            config: The configuration for the Supertrend strategy.
        """

        # properties
        self.config = config
        self.logger = logging.getLogger("supertrend_strategy")

        # create indicators
        self.fractals_instance = FractalsIndicator(length=config.fractal_period)
        self.adx_instance = ADX(
            di_length=config.adx_period, adx_smoothing=config.adx_smoothing
        )
        self.supertrend_instance = SuperTrendIndicator(
            atr_period=config.atr_period, multiplier=config.factor
        )
        self.rsi = RSI(length=config.rsi_period)
        self.ema_1 = EMA(length=config.ema_1_period)
        self.ema_2 = EMA(length=config.ema_2_period)
        self.ema_3 = EMA(length=config.ema_3_period)
        self.bbands = BollingerBands(length=config.bbands_period)

        # minimum candle history
        self.open_price_history: deque[float] = deque(maxlen=config.minimum_history)
        self.high_price_history: deque[float] = deque(maxlen=config.minimum_history)
        self.low_price_history: deque[float] = deque(maxlen=config.minimum_history)
        self.close_price_history: deque[float] = deque(maxlen=config.minimum_history)
        self.volume_history: deque[float] = deque(maxlen=config.minimum_history)

    def get_dynamic_risk_reward_ratio(self, action: TradeAction) -> float:
        """
        Get the dynamic risk reward ratio based on the current market conditions.

        Args:
            action: The trade action (ENTER_LONG or ENTER_SHORT).

        Returns:
            The dynamic risk reward ratio.
        """

        # get values for all the indicators
        ema_1_vals = self.ema_1.get_value(self.open_price_history)
        ema_2_vals = self.ema_2.get_value(self.open_price_history)
        ema_3_vals = self.ema_3.get_value(self.open_price_history)
        rsi_vals = self.rsi.get_value(self.close_price_history)
        _, middle_band, _ = self.bbands.get_value(
            self.close_price_history, self.volume_history
        )

        # validate we have data
        if (
            len(ema_1_vals) == 0
            or len(ema_2_vals) == 0
            or len(ema_3_vals) == 0
            or len(rsi_vals) == 0
            or len(middle_band) == 0
        ):
            return self.config.risk_reward_ratio

        # calculate points
        points = 0

        # EMA alignment points (2 points for strong trend)
        if (
            action == TradeAction.ENTER_LONG
            and ema_1_vals[-1] > ema_2_vals[-1] > ema_3_vals[-1]
        ) or (
            action == TradeAction.ENTER_SHORT
            and ema_1_vals[-1] < ema_2_vals[-1] < ema_3_vals[-1]
        ):
            points += 2

        # RSI momentum points (2 points for strong momentum)
        if abs(rsi_vals[-1] - 50) > 10:
            points += 2

        # Bollinger Bands position points (2 points for trend continuation)
        if self.close_price_history[-1] > middle_band[-1]:
            points += 2

        # return risk reward ratio based on points
        if points >= 5:
            return self.config.risk_reward_ratio * 1.25  # 1.5 times the base R:R
        elif points >= 3:
            return self.config.risk_reward_ratio * 1.125  # 1.25 times the base R:R
        else:
            return self.config.risk_reward_ratio  # for normal setup

    def populate_candle_history(
        self,
        open_prices: List[float],
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        volumes: List[float],
        round_off: int = 4,
    ):
        """
        Populate the candle history with the provided data.

        Args:
            open_prices: List of open prices for each candle.
            high_prices: List of high prices for each candle.
            low_prices: List of low prices for each candle.
            close_prices: List of close prices for each candle.
            volumes: List of volumes for each candle.
            round_off: Number of decimal places to round values to. Set to 0 to disable rounding.
        """

        # get the total length
        n = len(open_prices)
        for idx in range(n - self.config.minimum_history, n):

            # round of the values
            if round_off > 0:
                curr_open_price = round(float(open_prices[idx]), round_off)
                curr_high_price = round(float(high_prices[idx]), round_off)
                curr_low_price = round(float(low_prices[idx]), round_off)
                curr_close_price = round(float(close_prices[idx]), round_off)
                curr_volume = round(float(volumes[idx]), round_off)
            else:
                curr_open_price = float(open_prices[idx])
                curr_high_price = float(high_prices[idx])
                curr_low_price = float(low_prices[idx])
                curr_close_price = float(close_prices[idx])
                curr_volume = float(volumes[idx])

            # add to history
            self.open_price_history.append(curr_open_price)
            self.high_price_history.append(curr_high_price)
            self.low_price_history.append(curr_low_price)
            self.close_price_history.append(curr_close_price)
            self.volume_history.append(curr_volume)

    def clear_candle_history(self):
        """Clear the candle history."""
        self.open_price_history.clear()
        self.high_price_history.clear()
        self.low_price_history.clear()
        self.close_price_history.clear()
        self.volume_history.clear()

    def process_candles(
        self,
        open_prices: List[float | int] | np.ndarray,
        high_prices: List[float | int] | np.ndarray,
        low_prices: List[float | int] | np.ndarray,
        close_prices: List[float | int] | np.ndarray,
        volumes: List[float | int] | np.ndarray,
        round_off: int = 4,
    ) -> PredictionOutput:
        """
        Process candle data and generate a trading prediction.

        Args:
            open_prices: List or array of open prices for each candle.
            high_prices: List or array of high prices for each candle.
            low_prices: List or array of low prices for each candle.
            close_prices: List or array of close prices for each candle.
            volumes: List or array of volumes for each candle.
            round_off: Number of decimal places to round values to. Set to 0 to disable rounding.

        Returns:
            PredictionOutput containing entry price, exit price, stop loss, trade action, and risk reward ratio.

        Raises:
            ValueError: If the data lists have different lengths or fewer than minimum_history candles.
        """

        # validate that the data is equal
        valid_data = (
            len(open_prices)
            == len(high_prices)
            == len(low_prices)
            == len(close_prices)
            == len(volumes)
        )
        if not valid_data or len(open_prices) < self.config.minimum_history:
            err_msg = (
                "Invalid data provided. All the price and volume lists must have the same length and must have at least "
                + str(self.config.minimum_history)
                + " candles."
            )
            raise ValueError(err_msg)

        # populate the candle history with the data
        self.populate_candle_history(
            open_prices, high_prices, low_prices, close_prices, volumes, round_off
        )

        # default prediction
        default_prediction = PredictionOutput(
            entry_price=self.close_price_history[-1],
            exit_price=self.close_price_history[-1],
            stop_loss=self.close_price_history[-1],
            action=TradeAction.NEUTRAL,
            risk_reward_ratio=self.config.risk_reward_ratio,
        )

        try:

            # Get indicator values
            supertrend_data = self.supertrend_instance.get_value(
                high=self.high_price_history,
                low=self.low_price_history,
                close=self.close_price_history,
            )
            fractal_data = self.fractals_instance.get_value(
                high=self.high_price_history,
                low=self.low_price_history,
            )
            adx_data, _, _ = self.adx_instance.get_value(
                high=self.high_price_history,
                low=self.low_price_history,
                close=self.close_price_history,
            )

            # validate if data is valid
            if not supertrend_data or not fractal_data or not adx_data.any():
                return default_prediction

            # Convert signal to action
            if (
                supertrend_data[-1][1] == 1
                and supertrend_data[-2][1] == -1
                and adx_data[-1] > self.config.adx_threshold
            ):
                action = TradeAction.ENTER_LONG
            elif (
                supertrend_data[-1][1] == -1
                and supertrend_data[-2][1] == 1
                and adx_data[-1] > self.config.adx_threshold
            ):
                action = TradeAction.ENTER_SHORT
            else:
                action = TradeAction.NEUTRAL

            # Calculate stop loss and exit price based on fractals
            last_close = self.close_price_history[-1]
            if action == TradeAction.ENTER_LONG:

                # get the risk to reward ratio
                risk_reward_ratio = self.get_dynamic_risk_reward_ratio(action)

                # get the last fractal bottom where the price was above it
                last_fractal_bottom = [i for i in fractal_data[0] if i < last_close]
                if not last_fractal_bottom:
                    return default_prediction

                # clip the SL so that it is maximum 2% away
                curr_stop_loss = last_fractal_bottom[-1]
                curr_stop_loss = max(last_close - 0.02 * last_close, curr_stop_loss)

                # calculate the exit price
                exit_price = last_close + (
                    abs(last_close - curr_stop_loss) * risk_reward_ratio
                )

            elif action == TradeAction.ENTER_SHORT:

                # get the risk to reward ratio
                risk_reward_ratio = self.get_dynamic_risk_reward_ratio(action)

                # check if the sl is valid
                last_fractal_top = [i for i in fractal_data[0] if i > last_close]
                if not last_fractal_top:
                    return default_prediction

                # clip the SL so that it is maximum 2% away
                curr_stop_loss = last_fractal_top[-1]
                curr_stop_loss = min(last_close + 0.02 * last_close, curr_stop_loss)

                # calculate the exit price
                exit_price = last_close - (
                    abs(last_close - curr_stop_loss) * risk_reward_ratio
                )

            # default values
            else:
                exit_price = self.close_price_history[-1]
                curr_stop_loss = self.close_price_history[-1]
                risk_reward_ratio = self.config.risk_reward_ratio

            # round the values
            if round_off > 0:
                exit_price = round(float(exit_price), round_off)
                curr_stop_loss = round(float(curr_stop_loss), round_off)

            # return the prediction output
            return PredictionOutput(
                action=action,
                entry_price=last_close,
                exit_price=exit_price,
                stop_loss=curr_stop_loss,
                risk_reward_ratio=risk_reward_ratio,
                indicator_details={
                    "supertrend_data[-1][1]": supertrend_data[-1][1],
                    "supertrend_data[-2][1]": supertrend_data[-2][1],
                    "adx_data[-1]": adx_data[-1],
                },
            )

        # handle exceptions
        except Exception as e:
            log_msg = f"Exception occurred in process_candles: {str(e)}"
            self.logger.exception(log_msg)

            return PredictionOutput(
                action=TradeAction.NEUTRAL,
                entry_price=0,
                exit_price=0,
                stop_loss=0,
                risk_reward_ratio=self.config.risk_reward_ratio,
            )

        # clear the candle history
        finally:
            self.clear_candle_history()
