"""Selection strategy interface and base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from finquant.selection import SelectionResult


class SelectionStrategy(ABC):
    """Abstract base class for stock selection strategies.

    All selection strategies must implement the select() method which takes
    market data and returns a SelectionResult containing selected stocks.

    Example:
        class MyStrategy(SelectionStrategy):
            def select(self, market_df, index_df, as_of_date):
                # Custom selection logic
                return SelectionResult(...)
    """

    @abstractmethod
    def select(
        self,
        market_df: pd.DataFrame,
        index_df: pd.DataFrame,
        as_of_date: str,
    ) -> SelectionResult:
        """Select stocks based on strategy logic.

        Args:
            market_df: Market data with columns [date, tic, open, high, low, close, volume]
            index_df: Index data for market state classification
            as_of_date: Selection date in YYYY-MM-DD format

        Returns:
            SelectionResult with selected tickers, scores, and metadata
        """
        pass
