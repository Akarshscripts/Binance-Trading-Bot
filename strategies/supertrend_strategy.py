"""
This module contains the supertrend strategy for backtesting.
"""

# 1st party imports
from collections import deque

# local imports
from indicators import FractalsIndicator, SuperTrendIndicator
from strategies.models import PredictionOutput, CandleData, TradeAction


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
        self.fractal_period = fractal_period
        self.risk_reward_ratio = risk_reward_ratio

        # linked instances
        self.fractals_instance = FractalsIndicator(length=fractal_period)
        self.supertrend_instance = SuperTrendIndicator(
            atr_period=atr_period, multiplier=factor
        )

        # candle history
        self.candle_history: deque[CandleData] = deque(
            maxlen=max(atr_period, fractal_period) * 2
        )

    def new_candle(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
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

        # create the candle data
        curr_candle = CandleData(
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
        )
        self.candle_history.append(curr_candle)

        # default prediction
        default_prediction = PredictionOutput(
            entry_price=close_price, exit_price=close_price, stop_loss=close_price
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

        # validate if data is valid
        if not supertrend_data or not fractal_data:
            return default_prediction

        # Convert signal to action
        if supertrend_data[-1][1] == 1 and supertrend_data[-2][1] == -1:
            action = TradeAction.ENTER_LONG
        elif supertrend_data[-1][1] == -1 and supertrend_data[-2][1] == 1:
            action = TradeAction.ENTER_SHORT
        else:
            action = TradeAction.NEUTRAL

        # Calculate stop loss and exit price based on fractals
        if action == TradeAction.ENTER_LONG:

            # get the last fractal bottom where the price was above it
            last_fractal_bottom = [i for i in fractal_data[0] if i < close_price]
            if not last_fractal_bottom:
                return default_prediction

            # clip the SL so that it is maximum 2% away
            curr_stop_loss = last_fractal_bottom[-1]
            curr_stop_loss = max(close_price - 0.02 * close_price, curr_stop_loss)

            # calculate the exit price
            exit_price = close_price + (
                abs(close_price - curr_stop_loss) * self.risk_reward_ratio
            )

        elif action == TradeAction.ENTER_SHORT:
            # check if the sl is valid
            last_fractal_top = [i for i in fractal_data[0] if i > close_price]
            if not last_fractal_top:
                return default_prediction

            # clip the SL so that it is maximum 2% away
            curr_stop_loss = last_fractal_top[-1]
            curr_stop_loss = min(close_price + 0.02 * close_price, curr_stop_loss)

            # calculate the exit price
            exit_price = close_price - (
                abs(close_price - curr_stop_loss) * self.risk_reward_ratio
            )

        # default values
        else:
            exit_price = close_price
            curr_stop_loss = close_price

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
        )
