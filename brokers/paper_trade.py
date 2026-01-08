"""
This module contains the paper trader class which handles the paper trading logic.
"""

# 1st party imports
from datetime import datetime
from typing import List, Optional, Dict

# 3rd party imports
from pydantic import BaseModel

# local imports
from .models import Position, TradeType
from binance_api import BinanceSymbols, ChartIntervals


class PaperTradeStats(BaseModel):
    """
    This model is used to return the stats of all the trades taken by the PaperTrader
    """

    # overview
    total_trades: int
    long_trades: int
    short_trades: int
    trades_won: int
    trades_lost: int

    # P/L
    profit: float
    loss: float
    net: float
    buy_brokerage: float
    sell_brokerage: float
    total_brokerage: float
    grouped_risk_reawards: Dict[float, int]

    # age
    avg_trade_age: float
    max_trade_age: int
    min_trade_age: int

    # Success
    success_rate: float
    avg_profit: float
    avg_loss: float
    max_profit: float
    max_loss: float

    # profit factor
    profit_factor: float


class PaperTrader:
    """
    This class handles the paper trading logic.
    """

    def __init__(self, brokerage: float, capital: float, risk_investment: float):
        """
        Initialize the paper trader.
        """

        # variables
        self.capital = capital
        self.brokerage = brokerage
        self.current: Optional[Position] = None
        self.risk_investment = risk_investment

        # stats
        self.closed: List[Position] = []

    def open_position(
        self,
        symbol: BinanceSymbols,
        interval: ChartIntervals,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        trade_type: TradeType,
        risk_reward_ratio: float,
        trade_start_timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Open a new position.

        Args:
            symbol: The trading symbol (e.g., BTCUSDT).
            interval: The chart interval for candlestick data.
            entry_price: The price at which to enter the trade.
            exit_price: The target price to exit the trade.
            stop_loss: The stop loss price.
            trade_type: The type of trade (LONG or SHORT).
            risk_reward_ratio: The risk reward ratio for the trade.
            trade_start_timestamp: The timestamp when the trade starts.

        Returns:
            bool: True if the position was opened successfully.

        Raises:
            ValueError: If there is already an open position or if stop loss is invalid.
        """

        # check if there is an open position
        if self.current is not None:
            return False

        # if capital is less than 0
        if self.capital <= 0:
            raise ValueError("Capital exhausted.")

        # validate SL direction
        if trade_type == TradeType.LONG and stop_loss >= entry_price:
            raise ValueError(
                f"Stop loss must be less than entry price for long positions. Values: entry_price: {entry_price}, stop_loss: {stop_loss}"
            )
        if trade_type == TradeType.SHORT and stop_loss <= entry_price:
            raise ValueError(
                f"Stop loss must be greater than entry price for short positions. Values: entry_price: {entry_price}, stop_loss: {stop_loss}"
            )

        # calculate the position size based on risk
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = self.capital * self.risk_investment

        # calculate position size based on risk (how many shares for this risk amount)
        risk_based_size = max_risk_amount // risk_per_share

        # calculate position size based on total capital (how many shares can we afford)
        max_shares_by_capital = self.capital // (entry_price * (1 + self.brokerage))

        # use the smaller of the two to respect both risk and capital constraints
        position_size = int(min(risk_based_size, max_shares_by_capital))

        # ensure minimum position size of 1
        position_size = max(1, position_size)

        # calculate the brokerage
        open_brokerage = (entry_price * position_size) * self.brokerage

        # validate sufficient capital
        required_capital = (entry_price * position_size) + open_brokerage
        if required_capital > self.capital:
            raise ValueError(
                f"Insufficient capital. Required: ${required_capital:.2f}, Available: ${self.capital:.2f}"
            )

        # create the position
        self.current = Position(
            symbol=symbol,
            interval=interval,
            trade_type=trade_type,
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            stop_loss=round(stop_loss, 4),
            trade_start_timestamp=trade_start_timestamp,
            open_brokerage=open_brokerage,
            position_size=position_size,
            risk_reward_ratio=risk_reward_ratio
        )

        # return true
        return True

    def update(
        self,
        low: float,
        high: float,
        current_candle_timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Updates the current position with new candle data.

        Args:
            low: The lowest price of the candle.
            high: The highest price of the candle.
            current_candle_timestamp: The timestamp of the current candle.
        """

        # if no open position, return
        if self.current is None:
            return

        # get the position
        pos = self.current

        # increment the age
        pos.age += 1

        # LONG
        if pos.trade_type == TradeType.LONG:

            # SL
            if low <= pos.stop_loss:
                pos.loss = (pos.entry_price - pos.stop_loss) * pos.position_size
                close_brokerage = (pos.stop_loss * pos.position_size) * self.brokerage
                self._close(current_candle_timestamp, close_brokerage)

            # Target
            elif high >= pos.exit_price:
                pos.profit = (pos.exit_price - pos.entry_price) * pos.position_size
                close_brokerage = (pos.exit_price * pos.position_size) * self.brokerage
                self._close(current_candle_timestamp, close_brokerage)

        # SHORT
        else:

            # SL
            if high >= pos.stop_loss:
                pos.loss = (pos.stop_loss - pos.entry_price) * pos.position_size
                close_brokerage = (pos.stop_loss * pos.position_size) * self.brokerage
                self._close(current_candle_timestamp, close_brokerage)

            # Target
            elif low <= pos.exit_price:
                pos.profit = (pos.entry_price - pos.exit_price) * pos.position_size
                close_brokerage = (pos.exit_price * pos.position_size) * self.brokerage
                self._close(current_candle_timestamp, close_brokerage)

    def _close(
        self,
        current_candle_timestamp: Optional[datetime] = None,
        close_brokerage: float = 0,
    ):
        """
        Close the current position and move it to closed positions.

        Args:
            current_candle_timestamp: The timestamp when the position was closed.
            close_brokerage: The brokerage to use for closing the position.
        """

        # update the position
        self.current.trade_end_time = current_candle_timestamp
        self.current.close_brokerage = close_brokerage
        self.current.total_brokerage = (
            self.current.open_brokerage + self.current.close_brokerage
        )

        # update the capital
        self.capital += (
            self.current.profit - self.current.loss - self.current.total_brokerage
        )

        # append to closed
        self.closed.append(self.current)

        # reset the current
        self.current = None

    def stats(self) -> PaperTradeStats:
        """
        Calculate and return statistics for all closed trades.

        Returns:
            PaperTradeStats: Statistics including total trades, win/loss counts,
                profit/loss metrics, trade duration stats, and performance ratios.
        """

        # variables
        total_trades = len(self.closed)
        long_trades, short_trades, trades_won = 0, 0, 0
        profit, loss = 0, 0
        max_profit, max_loss = 0, 0
        total_trade_age = 0
        max_trade_age = 0
        min_trade_age = 0
        total_brokerage = 0
        total_buy_brokerage = 0
        total_sell_brokerage = 0
        grouped_risk_reawards = {}

        # iterate over the closed positions
        for pos in self.closed:

            # for long trades
            if pos.trade_type == TradeType.LONG:
                long_trades += 1
            # for short trades
            else:
                short_trades += 1

            # update trades won
            if pos.profit > 0:
                trades_won += 1

            # update P/L variables
            profit += pos.profit
            loss += pos.loss
            max_profit = max(max_profit, pos.profit)
            max_loss = max(max_loss, pos.loss)
            total_brokerage += pos.total_brokerage
            total_buy_brokerage += pos.open_brokerage
            total_sell_brokerage += pos.close_brokerage

            # update age variables
            total_trade_age += pos.age
            max_trade_age = max(max_trade_age, pos.age)
            min_trade_age = min(min_trade_age, pos.age)

            # update grouped_risk_reawards
            if pos.risk_reward_ratio not in grouped_risk_reawards:
                grouped_risk_reawards[pos.risk_reward_ratio] = 0
            grouped_risk_reawards[pos.risk_reward_ratio] += 1

        # calculate the total brokerage
        net_profit = profit - loss - total_brokerage

        # return the stats
        return PaperTradeStats(
            total_trades=total_trades,
            long_trades=long_trades,
            short_trades=short_trades,
            trades_won=trades_won,
            trades_lost=total_trades - trades_won,
            profit=round(profit, 4),
            loss=round(loss, 4),
            net=round(net_profit, 4),
            avg_trade_age=round(total_trade_age / max(1, total_trades), 4),
            max_trade_age=max_trade_age,
            min_trade_age=min_trade_age,
            success_rate=round(trades_won / max(1, total_trades), 4),
            avg_profit=round(profit / max(1, trades_won), 4),
            avg_loss=round(loss / max(1, total_trades - trades_won), 4),
            max_profit=max_profit,
            max_loss=max_loss,
            profit_factor=round(profit / max(1, loss), 4),
            buy_brokerage=round(total_buy_brokerage, 4),
            sell_brokerage=round(total_sell_brokerage, 4),
            total_brokerage=round(total_brokerage, 4),
            grouped_risk_reawards=grouped_risk_reawards,
        )
