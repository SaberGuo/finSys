"""Stock screener for multi-factor scoring and candidate selection.

Computes composite scores from normalized factor values and selects top-k stocks
after applying exclusion rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finquant.selection.factor_registry import FactorRegistry
from finquant.selection.normalizer import FactorNormalizer


@dataclass
class ScreenConfig:
    """Configuration for stock screening.

    Attributes:
        top_k: Number of stocks to select
        exclude_st: Exclude ST stocks
        exclude_halt: Exclude halted stocks
        exclude_limit_up: Exclude limit-up stocks
        score_range: Valid score range
    """

    top_k: int = 10
    exclude_st: bool = True
    exclude_halt: bool = True
    exclude_limit_up: bool = True
    score_range: tuple[float, float] = (-1.0, 1.0)


class StockScreener:
    """Screen stocks based on multi-factor composite scoring."""

    def __init__(self, config: ScreenConfig):
        """Initialize screener.

        Args:
            config: Screening configuration
        """
        self.config = config

    def screen(
        self,
        df: pd.DataFrame,
        factor_weights: dict[str, float],
        registry: FactorRegistry,
        normalizer: FactorNormalizer,
        as_of_date: str,
    ) -> tuple[list[str], dict[str, float], dict[str, str]]:
        """Screen stocks and return top-k candidates.

        Args:
            df: Market DataFrame with columns: date, tic, close, and factor columns
            factor_weights: IC-weighted factor weights
            registry: Factor registry
            normalizer: Factor normalizer
            as_of_date: Date to screen in "YYYY-MM-DD" format

        Returns:
            Tuple of (selected_tickers, scores, exclusion_reasons)
            - selected_tickers: List of top-k stock tickers
            - scores: Dict mapping all tickers to composite scores
            - exclusion_reasons: Dict mapping excluded tickers to reasons

        Note:
            Composite score formula:
            score_s = sum(w_i * direction_i * normalized_value_{s,i}) / sum(|w_i|)
        """
        # Compute factor values on full historical data first
        factor_dfs = {}
        for factor_id, weight in factor_weights.items():
            factor_def = registry.get(factor_id)

            # Compute factor values on full data (allows historical lookback)
            try:
                factor_values = factor_def.compute_fn(df)
                # Create a DataFrame with date, tic, and factor value
                factor_df = pd.DataFrame({
                    'date': df['date'],
                    'tic': df['tic'],
                    factor_id: factor_values
                })
                factor_dfs[factor_id] = factor_df
            except (KeyError, AttributeError) as e:
                # Factor computation failed (missing columns), skip
                continue

        if not factor_dfs:
            # No valid factors computed
            return [], {}, {}

        # Filter to as_of_date for cross-sectional analysis
        date_df = df[df["date"] == as_of_date].copy()
        if date_df.empty:
            return [], {}, {}

        # Merge factor values for as_of_date
        factor_scores = {}
        for factor_id, weight in factor_weights.items():
            if factor_id not in factor_dfs:
                continue

            factor_df = factor_dfs[factor_id]
            factor_date_df = factor_df[factor_df["date"] == as_of_date]

            if factor_date_df.empty:
                continue

            factor_def = registry.get(factor_id)

            # Get factor values for this date
            factor_values = factor_date_df.set_index('tic')[factor_id]

            # Normalize cross-sectionally
            normalized = normalizer.normalize(factor_values)

            # Apply direction
            factor_scores[factor_id] = normalized * factor_def.direction * weight

        if not factor_scores:
            # No valid factors computed
            return [], {}, {}

        # Compute composite scores
        composite_df = pd.DataFrame(factor_scores)
        composite_df["tic"] = composite_df.index
        composite_df = composite_df.set_index("tic")

        # Sum weighted normalized scores
        total_weight = sum(abs(w) for w in factor_weights.values())
        if total_weight == 0:
            total_weight = 1.0

        composite_scores = composite_df.sum(axis=1) / total_weight

        # Apply exclusion rules
        exclusion_reasons = {}
        valid_tickers = set(composite_scores.index)

        if self.config.exclude_st:
            # Exclude ST stocks (ticker contains "ST")
            st_tickers = {tic for tic in valid_tickers if "ST" in tic.upper()}
            for tic in st_tickers:
                exclusion_reasons[tic] = "ST"
            valid_tickers -= st_tickers

        if self.config.exclude_halt:
            # Exclude halted stocks (volume = 0)
            date_df_indexed = date_df.set_index("tic")
            halted_tickers = set(
                date_df_indexed[date_df_indexed["volume"] == 0].index
            )
            for tic in halted_tickers:
                if tic in valid_tickers:
                    exclusion_reasons[tic] = "halted"
            valid_tickers -= halted_tickers

        if self.config.exclude_limit_up:
            # Exclude limit-up stocks (close >= high * 1.099)
            # Simplified: assume limit-up if close == high and pct_change > 9.5%
            date_df_indexed = date_df.set_index("tic")
            if "high" in date_df_indexed.columns:
                limit_up_mask = (
                    (date_df_indexed["close"] >= date_df_indexed["high"] * 0.999)
                    & (date_df_indexed["close"].pct_change() > 0.095)
                )
                limit_up_tickers = set(date_df_indexed[limit_up_mask].index)
                for tic in limit_up_tickers:
                    if tic in valid_tickers:
                        exclusion_reasons[tic] = "limit_up"
                valid_tickers -= limit_up_tickers

        # Filter to valid tickers
        valid_scores = composite_scores[list(valid_tickers)]

        # Sort by score descending and select top-k
        sorted_scores = valid_scores.sort_values(ascending=False)
        selected_tickers = sorted_scores.head(self.config.top_k).index.tolist()

        # Return all scores (for debugging), selected tickers, and exclusion reasons
        # Clip scores to [-1.0, 1.0] range to satisfy SelectionResult invariant
        all_scores = {tic: max(-1.0, min(1.0, score)) for tic, score in composite_scores.to_dict().items()}

        return selected_tickers, all_scores, exclusion_reasons
