"""Stock selection module for factor-based candidate pool generation.

This module implements daily stock selection based on market state classification
and dynamic factor selection with IC-weighted scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MarketState(Enum):
    """Market state classification for factor selection."""

    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    OSCILLATION = "oscillation"
    STRUCTURAL = "structural"
    VOLUME_CONTRACTION = "volume_contraction"
    SENTIMENT_CAUTIOUS = "sentiment_cautious"


@dataclass(frozen=True)
class SelectionResult:
    """Standard interface from selection module to RL trading module.

    Attributes:
        date: Trading date in "YYYY-MM-DD" format
        selected_tickers: List of selected stock tickers (length <= top_k)
        scores: Composite scores for selected stocks, range [-1.0, 1.0]
        market_state: Identified market state
        active_factors: List of factor IDs used in selection
        factor_weights: IC-weighted factor weights (sum to 1.0)
        index_metrics: Index metrics snapshot for debugging
        exclusion_reasons: Reasons for excluded stocks
    """

    date: str
    selected_tickers: list[str]
    scores: dict[str, float]
    market_state: MarketState
    active_factors: list[str]
    factor_weights: dict[str, float]
    index_metrics: dict[str, float]
    exclusion_reasons: dict[str, str]

    def __post_init__(self) -> None:
        """Validate invariants."""
        # INV-1: Tickers match scores
        if set(self.selected_tickers) != set(self.scores.keys()):
            raise ValueError("selected_tickers must match scores keys")

        # INV-2: Scores in valid range
        if not all(-1.0 <= s <= 1.0 for s in self.scores.values()):
            raise ValueError("All scores must be in range [-1.0, 1.0]")

        # INV-3: Weights sum to ~1.0
        if self.factor_weights and abs(sum(self.factor_weights.values()) - 1.0) > 1e-6:
            raise ValueError("factor_weights must sum to approximately 1.0")

        # INV-4: Active factors match weights
        if set(self.active_factors) != set(self.factor_weights.keys()):
            raise ValueError("active_factors must match factor_weights keys")

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "version": "1.0.0",
            "date": self.date,
            "selected_tickers": self.selected_tickers,
            "scores": self.scores,
            "market_state": self.market_state.value,
            "active_factors": self.active_factors,
            "factor_weights": self.factor_weights,
            "index_metrics": self.index_metrics,
            "exclusion_reasons": self.exclusion_reasons,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SelectionResult:
        """Create from dict with version handling."""
        version = data.get("version", "1.0.0")
        if version != "1.0.0":
            raise ValueError(f"Unsupported SelectionResult version: {version}")

        return cls(
            date=data["date"],
            selected_tickers=data["selected_tickers"],
            scores=data["scores"],
            market_state=MarketState(data["market_state"]),
            active_factors=data["active_factors"],
            factor_weights=data["factor_weights"],
            index_metrics=data.get("index_metrics", {}),
            exclusion_reasons=data.get("exclusion_reasons", {}),
        )

    def save(self, path: str) -> None:
        """Save to JSON file."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> SelectionResult:
        """Load from JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Import SelectionPipeline for convenience
from finquant.selection.pipeline import SelectionPipeline  # noqa: E402
from finquant.selection.strategy import SelectionStrategy  # noqa: E402
from finquant.selection.factory import create_strategy  # noqa: E402


__all__ = [
    "MarketState",
    "SelectionResult",
    "SelectionPipeline",
    "SelectionStrategy",
    "create_strategy",
]
