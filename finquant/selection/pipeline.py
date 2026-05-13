"""Selection pipeline orchestrating market state classification and factor selection.

Coordinates the full selection workflow: classify market state → select factors →
compute IC weights → normalize factors → score and screen stocks.
"""

from __future__ import annotations

import pandas as pd

from finquant.config.settings import AppConfig
from finquant.selection import MarketState, SelectionResult
from finquant.selection.factor_registry import FactorRegistry
from finquant.selection.factor_selector import FactorSelector
from finquant.selection.ic_weight import ICWeightCalculator
from finquant.selection.market_state import MarketStateClassifier
from finquant.selection.normalizer import FactorNormalizer
from finquant.selection.screener import ScreenConfig, StockScreener


class SelectionPipeline:
    """Orchestrate stock selection workflow."""

    def __init__(
        self,
        classifier: MarketStateClassifier,
        registry: FactorRegistry,
        selector: FactorSelector,
        ic_calculator: ICWeightCalculator,
        normalizer: FactorNormalizer,
        screener: StockScreener,
    ):
        """Initialize selection pipeline.

        Args:
            classifier: Market state classifier
            registry: Factor registry
            selector: Factor selector
            ic_calculator: IC weight calculator
            normalizer: Factor normalizer
            screener: Stock screener
        """
        self.classifier = classifier
        self.registry = registry
        self.selector = selector
        self.ic_calculator = ic_calculator
        self.normalizer = normalizer
        self.screener = screener

    def run(
        self,
        market_df: pd.DataFrame,
        index_df: pd.DataFrame,
        as_of_date: str,
    ) -> SelectionResult:
        """Execute full selection workflow for a single trading day.

        Args:
            market_df: Market DataFrame (all stocks, daily frequency)
            index_df: Index DataFrame (daily frequency)
            as_of_date: Date to select in "YYYY-MM-DD" format

        Returns:
            SelectionResult with selected tickers and metadata

        Workflow:
            1. Classify market state from index data
            2. Select active factors based on state
            3. Compute IC-weighted factor weights
            4. Compute and normalize factor values
            5. Screen stocks and select top-k
            6. Assemble SelectionResult
        """
        # Step 1: Classify market state
        market_state = self.classifier.classify(index_df, as_of_date)

        # Step 2: Select factors
        factor_selection = self.selector.select(market_state)

        # Step 3: Compute IC weights
        factor_weights = self.ic_calculator.compute_weights(
            factor_ids=factor_selection.active_factors,
            as_of_date=as_of_date,
            preset_weights=factor_selection.preset_weights,
        )

        # Step 4 & 5: Screen stocks (includes factor computation and normalization)
        selected_tickers, all_scores, exclusion_reasons = self.screener.screen(
            df=market_df,
            factor_weights=factor_weights,
            registry=self.registry,
            normalizer=self.normalizer,
            as_of_date=as_of_date,
        )

        # Step 6: Get index metrics for debugging
        index_metrics = self.classifier.get_index_metrics(index_df, as_of_date)

        # Extract scores for selected tickers only
        selected_scores = {tic: all_scores[tic] for tic in selected_tickers if tic in all_scores}

        # Assemble result
        return SelectionResult(
            date=as_of_date,
            selected_tickers=selected_tickers,
            scores=selected_scores,
            market_state=market_state,
            active_factors=factor_selection.active_factors,
            factor_weights=factor_weights,
            index_metrics=index_metrics or {},
            exclusion_reasons=exclusion_reasons,
        )

    @classmethod
    def from_config(cls, config: AppConfig) -> SelectionPipeline:
        """Create pipeline from configuration.

        Args:
            config: Application configuration

        Returns:
            Configured SelectionPipeline

        Raises:
            ValueError: If selection config not present
        """
        if config.selection is None:
            raise ValueError("config.selection is required")

        # Initialize components
        classifier = MarketStateClassifier(
            index_ticker=config.selection.index_ticker,
            auto_optimize=config.selection.market_state.auto_optimize,
        )

        registry = FactorRegistry.from_defaults()

        selector = FactorSelector(registry)

        ic_calculator = ICWeightCalculator(
            registry=registry,
            window=config.selection.ic_window,
            min_periods=config.selection.ic_min_periods,
        )

        normalizer = FactorNormalizer(method=config.selection.normalizer)

        screen_config = ScreenConfig(
            top_k=config.selection.top_k,
            exclude_st=config.selection.exclude_st,
            exclude_halt=config.selection.exclude_halt,
        )
        screener = StockScreener(screen_config)

        return cls(
            classifier=classifier,
            registry=registry,
            selector=selector,
            ic_calculator=ic_calculator,
            normalizer=normalizer,
            screener=screener,
        )
