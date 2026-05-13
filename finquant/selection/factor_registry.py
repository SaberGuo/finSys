"""Factor registry for managing factor definitions and IC history.

Provides centralized registration and retrieval of factor definitions,
along with IC (Information Coefficient) tracking for dynamic weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd


class FactorCategory(Enum):
    """Factor category classification."""

    MOMENTUM = "momentum"
    BETA = "beta"
    GROWTH = "growth"
    LOW_VOL = "low_volatility"
    LOW_TURNOVER = "low_turnover"
    HIGH_DIVIDEND = "high_dividend"
    SMALL_CAP = "small_cap"
    REVERSAL = "reversal"
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"


class MissingStrategy(Enum):
    """Strategy for handling missing factor data."""

    FILL_NEUTRAL = "fill_neutral"  # Fill with 0 (after normalization)
    SKIP = "skip"  # Skip this factor (weight = 0)
    REDUCE_WEIGHT = "reduce_weight"  # Reduce factor weight proportionally


@dataclass
class FactorDefinition:
    """Definition of a single factor.

    Attributes:
        id: Unique factor identifier
        name: Display name
        category: Factor category
        compute_fn: Function to compute factor values from DataFrame
        required_columns: Required columns in input DataFrame
        missing_strategy: Strategy for handling missing data
        direction: Expected direction (1=positive, -1=negative)
    """

    id: str
    name: str
    category: FactorCategory
    compute_fn: Callable[[pd.DataFrame], pd.Series]
    required_columns: list[str]
    missing_strategy: MissingStrategy = MissingStrategy.FILL_NEUTRAL
    direction: int = 1

    def __post_init__(self):
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be 1 or -1, got {self.direction}")


class FactorRegistry:
    """Registry for factor definitions and IC history."""

    def __init__(self):
        self._factors: dict[str, FactorDefinition] = {}
        self._ic_history: dict[str, list[tuple[str, float]]] = {}  # factor_id -> [(date, ic_value)]

    def register(self, factor: FactorDefinition) -> None:
        """Register a factor definition.

        Args:
            factor: Factor definition to register

        Raises:
            ValueError: If factor ID already registered
        """
        if factor.id in self._factors:
            raise ValueError(f"Factor {factor.id} already registered")
        self._factors[factor.id] = factor
        self._ic_history[factor.id] = []

    def get(self, factor_id: str) -> FactorDefinition:
        """Get factor definition by ID.

        Args:
            factor_id: Factor identifier

        Returns:
            Factor definition

        Raises:
            KeyError: If factor not found
        """
        if factor_id not in self._factors:
            raise KeyError(f"Factor {factor_id} not found in registry")
        return self._factors[factor_id]

    def list_by_category(self, category: FactorCategory) -> list[FactorDefinition]:
        """List all factors in a category.

        Args:
            category: Factor category

        Returns:
            List of factor definitions
        """
        return [f for f in self._factors.values() if f.category == category]

    def all_ids(self) -> list[str]:
        """Get all registered factor IDs.

        Returns:
            List of factor IDs
        """
        return list(self._factors.keys())

    def record_ic(self, factor_id: str, date: str, ic_value: float) -> None:
        """Record IC value for a factor on a specific date.

        Args:
            factor_id: Factor identifier
            date: Date in "YYYY-MM-DD" format
            ic_value: IC value (Spearman rank correlation)

        Raises:
            KeyError: If factor not found
        """
        if factor_id not in self._factors:
            raise KeyError(f"Factor {factor_id} not found in registry")
        self._ic_history[factor_id].append((date, ic_value))

    def get_ic_series(self, factor_id: str, window: int) -> pd.Series:
        """Get recent IC series for a factor.

        Args:
            factor_id: Factor identifier
            window: Number of recent periods to retrieve

        Returns:
            Series with date index and IC values

        Raises:
            KeyError: If factor not found
        """
        if factor_id not in self._factors:
            raise KeyError(f"Factor {factor_id} not found in registry")

        history = self._ic_history[factor_id]
        if not history:
            return pd.Series(dtype=float)

        # Get last `window` entries
        recent = history[-window:] if len(history) > window else history
        dates, values = zip(*recent) if recent else ([], [])
        return pd.Series(values, index=pd.Index(dates, name="date"))

    @classmethod
    def from_defaults(cls) -> FactorRegistry:
        """Create registry with 11 built-in factors.

        Returns:
            Registry with default factors registered
        """
        registry = cls()

        # Register 11 built-in factors
        registry.register(
            FactorDefinition(
                id="momentum_20d",
                name="20日动量",
                category=FactorCategory.MOMENTUM,
                compute_fn=lambda df: (df["close"] / df.groupby("tic")["close"].shift(20) - 1).fillna(0),
                required_columns=["close"],
                direction=1,
            )
        )

        registry.register(
            FactorDefinition(
                id="high_beta",
                name="高Beta",
                category=FactorCategory.BETA,
                compute_fn=lambda df: df.groupby("tic")["close"].transform(
                    lambda x: x.pct_change().rolling(60, min_periods=20).std()
                ).fillna(0),  # Use 60-day volatility as proxy for beta
                required_columns=["close"],
                direction=1,
            )
        )

        registry.register(
            FactorDefinition(
                id="growth_yoy",
                name="成长YoY",
                category=FactorCategory.GROWTH,
                compute_fn=lambda df: (df["close"] / df.groupby("tic")["close"].shift(250) - 1).fillna(0),  # Use 250-day (1 year) price growth as proxy
                required_columns=["close"],
                direction=1,
            )
        )

        registry.register(
            FactorDefinition(
                id="low_volatility",
                name="低波动",
                category=FactorCategory.LOW_VOL,
                compute_fn=lambda df: -df.groupby("tic")["close"].transform(
                    lambda x: x.pct_change().rolling(20, min_periods=1).std()
                ).fillna(0),
                required_columns=["close"],
                direction=1,  # Negative volatility, so positive direction
            )
        )

        registry.register(
            FactorDefinition(
                id="low_turnover",
                name="低换手",
                category=FactorCategory.LOW_TURNOVER,
                compute_fn=lambda df: -df.get("turnover_rate", pd.Series(0.0, index=df.index)),
                required_columns=["turnover_rate"],
                direction=1,  # Negative turnover, so positive direction
            )
        )

        registry.register(
            FactorDefinition(
                id="high_dividend",
                name="高股息",
                category=FactorCategory.HIGH_DIVIDEND,
                compute_fn=lambda df: df.get("dividend_yield", pd.Series(0.0, index=df.index)),
                required_columns=["dividend_yield"],
                direction=1,
            )
        )

        registry.register(
            FactorDefinition(
                id="small_cap",
                name="小市值",
                category=FactorCategory.SMALL_CAP,
                compute_fn=lambda df: -df.get("market_cap", pd.Series(0.0, index=df.index)),
                required_columns=["market_cap"],
                direction=1,  # Negative market cap, so positive direction
            )
        )

        registry.register(
            FactorDefinition(
                id="reversal_5d",
                name="5日反转",
                category=FactorCategory.REVERSAL,
                compute_fn=lambda df: -(df["close"] / df.groupby("tic")["close"].shift(5) - 1).fillna(0),
                required_columns=["close"],
                direction=1,  # Negative return, so positive direction
            )
        )

        registry.register(
            FactorDefinition(
                id="roe",
                name="ROE",
                category=FactorCategory.FUNDAMENTAL,
                compute_fn=lambda df: df.get("roe", pd.Series(0.0, index=df.index)),
                required_columns=["roe"],
                direction=1,
            )
        )

        registry.register(
            FactorDefinition(
                id="sue",
                name="SUE",
                category=FactorCategory.FUNDAMENTAL,
                compute_fn=lambda df: df.get("sue", pd.Series(0.0, index=df.index)),
                required_columns=["sue"],
                direction=1,
            )
        )

        registry.register(
            FactorDefinition(
                id="macro_trend",
                name="宏观趋势",
                category=FactorCategory.MACRO,
                compute_fn=lambda df: df.get("index_return_20d", pd.Series(0.0, index=df.index)),
                required_columns=[],  # Computed from index, not stock data
                direction=1,
            )
        )

        return registry
