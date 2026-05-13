"""Factor-based selection strategy wrapper."""

import pandas as pd

from finquant.selection import SelectionResult
from finquant.selection.pipeline import SelectionPipeline
from finquant.selection.strategy import SelectionStrategy


class FactorBasedStrategy(SelectionStrategy):
    """Wrapper for existing factor-based selection pipeline.

    This strategy uses market state classification, IC-weighted multi-factor
    scoring, and cross-sectional normalization to select stocks.

    Args:
        pipeline: Configured SelectionPipeline instance

    Example:
        pipeline = SelectionPipeline.from_config(config)
        strategy = FactorBasedStrategy(pipeline)
        result = strategy.select(market_df, index_df, "2023-01-01")
    """

    def __init__(self, pipeline: SelectionPipeline):
        self.pipeline = pipeline

    def select(
        self,
        market_df: pd.DataFrame,
        index_df: pd.DataFrame,
        as_of_date: str,
    ) -> SelectionResult:
        """Select stocks using factor-based pipeline.

        Args:
            market_df: Market data with OHLCV columns
            index_df: Index data for market state classification
            as_of_date: Selection date in YYYY-MM-DD format

        Returns:
            SelectionResult with selected tickers and factor-based scores
        """
        return self.pipeline.run(market_df, index_df, as_of_date)
