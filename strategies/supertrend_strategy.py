"""
This module contains the supertrend strategy for backtesting.
"""

# 1st party imports
from collections import deque

# local imports
from strategies.models import PredictionOutput, CandleData, TradeAction
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
        factor: int,
        atr_period: int,
        fractal_period: int,
        risk_reward_ratio: float,
        adx_period: int = 14,
        adx_smoothing: int = 14,
        adx_threshold: int = 25,
        ema_1_period: int = 14,
        ema_2_period: int = 30,
        ema_3_period: int = 50,
        bbands_period: int = 20,
        rsi_period: int = 14,
    ):
        """
        Initialize the Supertrend strategy.

        Args:
            factor: The multiplier for ATR to calculate bands.
            atr_period: The period for ATR calculation.
            fractal_period: The period for fractal calculation.
            risk_reward_ratio: The risk reward ratio for the strategy.
        """

        # properties
        self.factor = factor
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.adx_smoothing = adx_smoothing
        self.adx_threshold = adx_threshold
        self.fractal_period = fractal_period
        self.risk_reward_ratio = risk_reward_ratio

        # linked instances
        self.fractals_instance = FractalsIndicator(length=fractal_period)
        self.adx_instance = ADX(di_length=adx_period, adx_smoothing=adx_smoothing)
        self.supertrend_instance = SuperTrendIndicator(
            atr_period=atr_period, multiplier=factor
        )
        self.rsi = RSI(length=rsi_period)
        self.ema_1 = EMA(length=ema_1_period)
        self.ema_2 = EMA(length=ema_2_period)
        self.ema_3 = EMA(length=ema_3_period)
        self.bbands = BollingerBands(length=bbands_period)

        # candle history
        self.candle_history: deque[CandleData] = deque(
            maxlen=int(
                max(
                    atr_period,
                    adx_period,
                    ema_1_period,
                    ema_2_period,
                    ema_3_period,
                    rsi_period,
                    bbands_period,
                )
                * 1.5
            )
        )

    def get_dynamic_risk_reward_ratio(self, action: TradeAction) -> float:
        """
        Get the dynamic risk reward ratio based on the current market conditions.

        Args:
            action: The trade action (ENTER_LONG or ENTER_SHORT).

        Returns:
            The dynamic risk reward ratio.
        """

        # ensure we have enough data
        if len(self.candle_history) < 50:
            return 1.5  # default for insufficient data

        # get the data for indicators
        open_price_hist = [i.open_price for i in self.candle_history]
        close_price_hist = [i.close_price for i in self.candle_history]
        volume_hist = [i.volume for i in self.candle_history]

        # get values for all the indicators
        ema_1_vals = self.ema_1.get_value(open_price_hist)
        ema_2_vals = self.ema_2.get_value(open_price_hist)
        ema_3_vals = self.ema_3.get_value(open_price_hist)
        rsi_vals = self.rsi.get_value(close_price_hist)
        _, middle_band, _ = self.bbands.get_value(open_price_hist, volume_hist)

        # validate we have data
        if (
            len(ema_1_vals) == 0
            or len(ema_2_vals) == 0
            or len(ema_3_vals) == 0
            or len(rsi_vals) == 0
            or len(middle_band) == 0
        ):
            return self.risk_reward_ratio

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
        if close_price_hist[-1] > middle_band[-1]:
            points += 2

        # return risk reward ratio based on points
        if points >= 5:
            return self.risk_reward_ratio * 1.25  # 1.5 times the base R:R
        elif points >= 3:
            return self.risk_reward_ratio * 1.125  # 1.25 times the base R:R
        else:
            return self.risk_reward_ratio  # for normal setup

    def new_candle(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        round_off: int = 4,
    ) -> PredictionOutput:
        """
        Process a new candle and generate a prediction.

        Args:
            open_price: The opening price of the candle.
            high_price: The highest price of the candle.
            low_price: The lowest price of the candle.
            close_price: The closing price of the candle.
            round_off: Number of decimal places to round values to.

        Returns:
            PredictionOutput containing entry, exit, stop loss prices and trade action.
        """

        # round of the values
        if round_off > 0:
            open_price = round(float(open_price), round_off)
            high_price = round(float(high_price), round_off)
            low_price = round(float(low_price), round_off)
            close_price = round(float(close_price), round_off)
            volume = round(float(volume), round_off)

        # create the candle data
        curr_candle = CandleData(
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
        )
        self.candle_history.append(curr_candle)

        # default prediction
        default_prediction = PredictionOutput(
            entry_price=close_price,
            exit_price=close_price,
            stop_loss=close_price,
            action=TradeAction.NEUTRAL,
            risk_reward_ratio=self.risk_reward_ratio,
        )

        # prepare the history
        high_price_hist = [i.high_price for i in self.candle_history]
        low_price_hist = [i.low_price for i in self.candle_history]
        close_price_hist = [i.close_price for i in self.candle_history]

        # Get indicator values
        supertrend_data = self.supertrend_instance.get_value(
            high_price_hist, low_price_hist, close_price_hist
        )
        fractal_data = self.fractals_instance.get_value(high_price_hist, low_price_hist)
        adx_data, _, _ = self.adx_instance.get_value(
            high_price_hist, low_price_hist, close_price_hist
        )

        # validate if data is valid
        if not supertrend_data or not fractal_data or not adx_data.any():
            return default_prediction

        # Convert signal to action
        if (
            supertrend_data[-1][1] == 1
            and supertrend_data[-2][1] == -1
            and adx_data[-1] > self.adx_threshold
        ):
            action = TradeAction.ENTER_LONG
        elif (
            supertrend_data[-1][1] == -1
            and supertrend_data[-2][1] == 1
            and adx_data[-1] > self.adx_threshold
        ):
            action = TradeAction.ENTER_SHORT
        else:
            action = TradeAction.NEUTRAL

        # Calculate stop loss and exit price based on fractals
        if action == TradeAction.ENTER_LONG:

            # get the risk to reward ratio
            risk_reward_ratio = self.get_dynamic_risk_reward_ratio(action)

            # get the last fractal bottom where the price was above it
            last_fractal_bottom = [i for i in fractal_data[0] if i < close_price]
            if not last_fractal_bottom:
                return default_prediction

            # clip the SL so that it is maximum 2% away
            curr_stop_loss = last_fractal_bottom[-1]
            curr_stop_loss = max(close_price - 0.02 * close_price, curr_stop_loss)

            # calculate the exit price
            exit_price = close_price + (
                abs(close_price - curr_stop_loss) * risk_reward_ratio
            )

        elif action == TradeAction.ENTER_SHORT:

            # get the risk to reward ratio
            risk_reward_ratio = self.get_dynamic_risk_reward_ratio(action)

            # check if the sl is valid
            last_fractal_top = [i for i in fractal_data[0] if i > close_price]
            if not last_fractal_top:
                return default_prediction

            # clip the SL so that it is maximum 2% away
            curr_stop_loss = last_fractal_top[-1]
            curr_stop_loss = min(close_price + 0.02 * close_price, curr_stop_loss)

            # calculate the exit price
            exit_price = close_price - (
                abs(close_price - curr_stop_loss) * risk_reward_ratio
            )

        # default values
        else:
            exit_price = close_price
            curr_stop_loss = close_price
            risk_reward_ratio = self.risk_reward_ratio

        # round the values
        if round_off > 0:
            exit_price = round(float(exit_price), round_off)
            curr_stop_loss = round(float(curr_stop_loss), round_off)

        # return the prediction output
        return PredictionOutput(
            action=action,
            entry_price=close_price,
            exit_price=exit_price,
            stop_loss=curr_stop_loss,
            risk_reward_ratio=risk_reward_ratio,
        )
