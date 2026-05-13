"""Factor selector for mapping market states to active factor combinations.

Implements state-to-factors mapping with preset weights for each market condition.
"""

from __future__ import annotations

from dataclasses import dataclass

from finquant.selection import MarketState
from finquant.selection.factor_registry import FactorRegistry


@dataclass
class FactorSelection:
    """Result of factor selection for a market state.

    Attributes:
        active_factors: List of selected factor IDs
        preset_weights: Initial weights before IC adjustment (sum to 1.0)
        market_state: Market state that triggered this selection
    """

    active_factors: list[str]
    preset_weights: dict[str, float]
    market_state: MarketState

    def __post_init__(self):
        """Validate invariants."""
        if set(self.active_factors) != set(self.preset_weights.keys()):
            raise ValueError("active_factors must match preset_weights keys")
        if abs(sum(self.preset_weights.values()) - 1.0) > 1e-6:
            raise ValueError("preset_weights must sum to 1.0")


class FactorSelector:
    """Select active factors based on market state."""

    def __init__(
        self,
        registry: FactorRegistry,
        state_map: dict[MarketState, list[str]] | None = None,
    ):
        """Initialize factor selector.

        Args:
            registry: Factor registry
            state_map: Custom state-to-factors mapping (default: use built-in mapping)
        """
        self.registry = registry
        self.state_map = state_map or self._default_state_map()

    def _default_state_map(self) -> dict[MarketState, list[str]]:
        """Build default state-to-factors mapping per spec FR-003.

        Returns:
            Dictionary mapping market state to list of factor IDs
        """
        return {
            MarketState.UPTREND: ["momentum_20d", "high_beta", "growth_yoy"],
            MarketState.DOWNTREND: ["low_volatility", "low_turnover", "high_dividend"],
            MarketState.OSCILLATION: ["low_volatility", "low_turnover", "high_dividend"],
            MarketState.STRUCTURAL: ["small_cap", "reversal_5d"],
            MarketState.VOLUME_CONTRACTION: ["low_volatility", "low_turnover", "high_dividend"],
            MarketState.SENTIMENT_CAUTIOUS: ["roe", "sue", "macro_trend"],
        }

    def select(self, state: MarketState) -> FactorSelection:
        """Select factors for a given market state.

        Args:
            state: Market state

        Returns:
            Factor selection with active factors and preset weights

        Raises:
            ValueError: If state not in mapping or factors not registered
        """
        if state not in self.state_map:
            raise ValueError(f"Market state {state} not in state_map")

        factor_ids = self.state_map[state]

        # Validate all factors are registered
        for fid in factor_ids:
            if fid not in self.registry.all_ids():
                raise ValueError(f"Factor {fid} not registered in registry")

        # Compute preset weights
        preset_weights = self._compute_preset_weights(state, factor_ids)

        return FactorSelection(
            active_factors=factor_ids,
            preset_weights=preset_weights,
            market_state=state,
        )

    def _compute_preset_weights(
        self,
        state: MarketState,
        factor_ids: list[str],
    ) -> dict[str, float]:
        """Compute preset weights for factors.

        Default: equal weights for all factors.
        Special case: VOLUME_CONTRACTION reduces low_vol/low_turnover to 0.5x.

        Args:
            state: Market state
            factor_ids: List of factor IDs

        Returns:
            Dictionary of preset weights (sum to 1.0)
        """
        if state == MarketState.VOLUME_CONTRACTION:
            # Reduce low_volatility and low_turnover weights
            weights = {}
            for fid in factor_ids:
                if fid in ("low_volatility", "low_turnover"):
                    weights[fid] = 0.5
                else:
                    weights[fid] = 1.0

            # Normalize to sum to 1.0
            total = sum(weights.values())
            return {k: v / total for k, v in weights.items()}

        # Default: equal weights
        weight = 1.0 / len(factor_ids)
        return {fid: weight for fid in factor_ids}
