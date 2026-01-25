"""
This module contains the models for the cli.
"""

# 1st party imports
import inspect
from enum import Enum
from typing import Self, Dict, Any, Optional


class CommandGroups(str, Enum):
    """Enum for command groups."""

    REAL_TIME = "Real-time predictions"
    BACKTEST = "Backtesting"
    NOTIFICATION = "Notification"


class AvailablePairs(str, Enum):
    """Enum for available pairs."""

    BTCUSDT = "BTCUSDT"
    TATA_STEEL = "TATA_STEEL"


class PairRegistry:
    """
    A registry for mapping pair enum classes to their associated exchange classes.
    """

    _instance: Optional["PairRegistry"] = None

    def __new__(cls) -> Self:
        """
        Create a new instance of PairRegistry or return the existing singleton instance.

        Returns:
            Self: The singleton instance of PairRegistry.
        """

        if cls._instance is None:
            cls._instance = super(PairRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the PairRegistry.

        This registry maintains a mapping between pair enum classes and their
        associated exchange classes, allowing for lookup of exchanges by pair.
        """

        if not hasattr(self, "_registry"):
            self._registry: Dict[Enum, Any] = {}

    def register(self, pair_cls: Enum, exchange_cls: Enum) -> None:
        """
        Register a pair enum class with its associated exchange class.

        Args:
            pair_cls (Enum): The pair enum class to register (e.g., BinanceSymbols).
            exchange_cls (Enum): The exchange class associated with the pair (e.g., BinanceExchange).
        """
        self._registry[pair_cls] = exchange_cls

    def get_exchange(self, pair_enum: Enum) -> Any:
        """
        Get the exchange class associated with a pair enum.

        Args:
            pair_enum (Enum): The pair enum class or instance to look up.

        Returns:
            Any: The exchange class associated with the pair, or None if not found.
        """

        if inspect.isclass(pair_enum):
            return self._registry.get(pair_enum, None)
        else:
            return self._registry.get(pair_enum.__class__, None)

    def generate_pair_enum(self, pair_str: str) -> Optional[Enum]:
        """
        Generate a pair enum from a string representation.

        Args:
            pair_str (str): The string representation of the pair (e.g., "BTCUSDT").

        Returns:
            Optional[Enum]: The pair enum if found, or None if not found in any registered pair class.
        """

        for pair_cls in self._registry.keys():
            try:
                return pair_cls[pair_str]
            except KeyError:
                continue
        return None
