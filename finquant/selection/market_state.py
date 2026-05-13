"""Market state classification for dynamic factor selection.

Classifies market conditions based on index technical indicators to guide
factor selection strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from finquant.selection import MarketState


@dataclass
class MarketStateRule:
    """Rule for classifying market state.

    Attributes:
        state: Target market state
        condition: Function that takes index DataFrame and returns boolean Series
        priority: Rule priority (higher = checked first)
    """

    state: MarketState
    condition: Callable[[pd.DataFrame], pd.Series]
    priority: int


class MarketStateClassifier:
    """Classify market state based on index technical indicators.

    Uses hardcoded rules by default, with optional auto-optimization mode.
    """

    def __init__(
        self,
        index_ticker: str = "000905.SH",
        auto_optimize: bool = False,
    ):
        """Initialize classifier.

        Args:
            index_ticker: Index ticker for classification (default: CSI 500)
            auto_optimize: Enable auto-optimization mode (default: False, use hardcoded rules)
        """
        self.index_ticker = index_ticker
        self.auto_optimize = auto_optimize
        self.rules = self._build_default_rules()

    def _build_default_rules(self) -> list[MarketStateRule]:
        """Build default hardcoded classification rules.

        Rules are checked in priority order (highest first).
        """
        rules = [
            # Priority 1: Strong uptrend
            MarketStateRule(
                state=MarketState.UPTREND,
                condition=lambda df: (df["adx_14"] > 25) & (df["close"] > df["ma50"] * 1.02),
                priority=10,
            ),
            # Priority 2: Strong downtrend
            MarketStateRule(
                state=MarketState.DOWNTREND,
                condition=lambda df: (df["close"] < df["ma50"]) & (df["adx_14"] > 25),
                priority=9,
            ),
            # Priority 3: Volume contraction
            MarketStateRule(
                state=MarketState.VOLUME_CONTRACTION,
                condition=lambda df: df["volume_ma20_ratio"] < 0.8,
                priority=8,
            ),
            # Priority 4: Structural market (placeholder - needs refinement)
            MarketStateRule(
                state=MarketState.STRUCTURAL,
                condition=lambda df: pd.Series([False] * len(df), index=df.index),  # Placeholder
                priority=7,
            ),
            # Priority 5: Sentiment cautious (placeholder - needs sentiment data)
            MarketStateRule(
                state=MarketState.SENTIMENT_CAUTIOUS,
                condition=lambda df: pd.Series([False] * len(df), index=df.index),  # Placeholder
                priority=6,
            ),
            # Priority 6: Oscillation (default fallback)
            MarketStateRule(
                state=MarketState.OSCILLATION,
                condition=lambda df: df["adx_14"] < 20,
                priority=5,
            ),
        ]
        return sorted(rules, key=lambda r: r.priority, reverse=True)

    def classify(self, index_df: pd.DataFrame, as_of_date: str) -> MarketState:
        """Classify market state for a specific date.

        Args:
            index_df: Index DataFrame with columns: date, close, adx_14, ma50, volume_ma20_ratio
            as_of_date: Date to classify in "YYYY-MM-DD" format

        Returns:
            Classified market state

        Raises:
            ValueError: If as_of_date not found in index_df or required columns missing
        """
        # Validate required columns
        required = ["date", "close", "adx_14"]
        missing = [c for c in required if c not in index_df.columns]
        if missing:
            raise ValueError(f"Missing required columns in index_df: {missing}")

        # Compute derived indicators if not present
        df = index_df.copy()
        if "ma50" not in df.columns:
            df = df.sort_values("date")
            df["ma50"] = df["close"].rolling(50, min_periods=1).mean()

        if "volume_ma20_ratio" not in df.columns:
            if "volume" in df.columns:
                df = df.sort_values("date")
                volume_ma20 = df["volume"].rolling(20, min_periods=1).mean().replace(0, 1e-9)
                df["volume_ma20_ratio"] = df["volume"] / volume_ma20
            else:
                df["volume_ma20_ratio"] = 1.0

        # Filter to as_of_date
        date_data = df[df["date"] == as_of_date]
        if date_data.empty:
            raise ValueError(f"Date {as_of_date} not found in index_df")

        # Apply rules in priority order
        for rule in self.rules:
            try:
                matches = rule.condition(date_data)
                if matches.any():
                    return rule.state
            except (KeyError, AttributeError):
                # Rule condition failed (missing columns), skip
                continue

        # Default fallback: oscillation
        return MarketState.OSCILLATION

    def classify_series(
        self,
        index_df: pd.DataFrame,
        dates: list[str] | None = None,
    ) -> dict[str, MarketState]:
        """Classify market state for multiple dates.

        Args:
            index_df: Index DataFrame
            dates: List of dates to classify (default: all dates in index_df)

        Returns:
            Dictionary mapping date to market state
        """
        if dates is None:
            dates = index_df["date"].unique().tolist()

        results = {}
        for date in dates:
            try:
                results[date] = self.classify(index_df, date)
            except ValueError:
                # Date not found, skip
                continue

        return results

    def get_index_metrics(self, index_df: pd.DataFrame, as_of_date: str) -> dict[str, float]:
        """Get index metrics snapshot for debugging.

        Args:
            index_df: Index DataFrame
            as_of_date: Date in "YYYY-MM-DD" format

        Returns:
            Dictionary of index metrics
        """
        df = index_df.copy()

        # Compute derived indicators if not present
        if "ma50" not in df.columns:
            df = df.sort_values("date")
            df["ma50"] = df["close"].rolling(50, min_periods=1).mean()

        if "volume_ma20_ratio" not in df.columns:
            if "volume" in df.columns:
                df = df.sort_values("date")
                volume_ma20 = df["volume"].rolling(20, min_periods=1).mean().replace(0, 1e-9)
                df["volume_ma20_ratio"] = df["volume"] / volume_ma20
            else:
                df["volume_ma20_ratio"] = 1.0

        date_data = df[df["date"] == as_of_date]
        if date_data.empty:
            return

        row = date_data.iloc[0]
        return {
            "close": float(row.get("close", 0)),
            "adx": float(row.get("adx_14", 0)),
            "ma50": float(row.get("ma50", 0)),
            "ma50_ratio": float(row.get("close", 0) / row.get("ma50", 1)) if row.get("ma50", 0) > 0 else 1.0,
            "volume_ma20_ratio": float(row.get("volume_ma20_ratio", 1.0)),
        }
