"""Unit tests for factor selector."""

from __future__ import annotations

import pytest

from finquant.selection import MarketState
from finquant.selection.factor_registry import FactorRegistry
from finquant.selection.factor_selector import FactorSelection, FactorSelector


class TestFactorSelection:
    """Test FactorSelection dataclass."""

    def test_factor_selection_creation(self):
        """Test creating a factor selection."""
        selection = FactorSelection(
            active_factors=["momentum_20d", "high_beta"],
            preset_weights={"momentum_20d": 0.5, "high_beta": 0.5},
            market_state=MarketState.UPTREND,
        )
        assert len(selection.active_factors) == 2
        assert sum(selection.preset_weights.values()) == pytest.approx(1.0)

    def test_factor_selection_invalid_weights_mismatch(self):
        """Test that mismatched factors and weights raises error."""
        with pytest.raises(ValueError, match="must match preset_weights keys"):
            FactorSelection(
                active_factors=["momentum_20d", "high_beta"],
                preset_weights={"momentum_20d": 0.5},  # Missing high_beta
                market_state=MarketState.UPTREND,
            )

    def test_factor_selection_invalid_weights_sum(self):
        """Test that weights not summing to 1.0 raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            FactorSelection(
                active_factors=["momentum_20d", "high_beta"],
                preset_weights={"momentum_20d": 0.3, "high_beta": 0.3},  # Sum = 0.6
                market_state=MarketState.UPTREND,
            )


class TestFactorSelector:
    """Test FactorSelector."""

    def test_selector_initialization_with_defaults(self):
        """Test selector initialization with default mapping."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        assert selector.registry == registry
        assert len(selector.state_map) == 6  # 6 market states

    def test_selector_initialization_with_custom_map(self):
        """Test selector initialization with custom mapping."""
        registry = FactorRegistry.from_defaults()
        custom_map = {
            MarketState.UPTREND: ["momentum_20d"],
            MarketState.DOWNTREND: ["low_volatility"],
        }
        selector = FactorSelector(registry, state_map=custom_map)

        assert len(selector.state_map) == 2

    def test_select_uptrend(self):
        """Test selecting factors for uptrend."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        selection = selector.select(MarketState.UPTREND)

        assert selection.market_state == MarketState.UPTREND
        assert set(selection.active_factors) == {"momentum_20d", "high_beta", "growth_yoy"}
        assert sum(selection.preset_weights.values()) == pytest.approx(1.0)
        # Equal weights for 3 factors
        assert all(w == pytest.approx(1/3) for w in selection.preset_weights.values())

    def test_select_downtrend(self):
        """Test selecting factors for downtrend."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        selection = selector.select(MarketState.DOWNTREND)

        assert selection.market_state == MarketState.DOWNTREND
        assert set(selection.active_factors) == {"low_volatility", "low_turnover", "high_dividend"}
        assert sum(selection.preset_weights.values()) == pytest.approx(1.0)

    def test_select_oscillation(self):
        """Test selecting factors for oscillation."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        selection = selector.select(MarketState.OSCILLATION)

        assert selection.market_state == MarketState.OSCILLATION
        # Same as downtrend per spec
        assert set(selection.active_factors) == {"low_volatility", "low_turnover", "high_dividend"}

    def test_select_structural(self):
        """Test selecting factors for structural market."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        selection = selector.select(MarketState.STRUCTURAL)

        assert selection.market_state == MarketState.STRUCTURAL
        assert set(selection.active_factors) == {"small_cap", "reversal_5d"}
        assert sum(selection.preset_weights.values()) == pytest.approx(1.0)
        # Equal weights for 2 factors
        assert all(w == pytest.approx(0.5) for w in selection.preset_weights.values())

    def test_select_volume_contraction(self):
        """Test selecting factors for volume contraction with reduced weights."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        selection = selector.select(MarketState.VOLUME_CONTRACTION)

        assert selection.market_state == MarketState.VOLUME_CONTRACTION
        assert set(selection.active_factors) == {"low_volatility", "low_turnover", "high_dividend"}
        assert sum(selection.preset_weights.values()) == pytest.approx(1.0)

        # low_volatility and low_turnover should have reduced weights (0.5x)
        # high_dividend should have normal weight (1.0x)
        # Normalized: 0.5 + 0.5 + 1.0 = 2.0, so weights are 0.25, 0.25, 0.5
        assert selection.preset_weights["low_volatility"] == pytest.approx(0.25)
        assert selection.preset_weights["low_turnover"] == pytest.approx(0.25)
        assert selection.preset_weights["high_dividend"] == pytest.approx(0.5)

    def test_select_sentiment_cautious(self):
        """Test selecting factors for sentiment cautious."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        selection = selector.select(MarketState.SENTIMENT_CAUTIOUS)

        assert selection.market_state == MarketState.SENTIMENT_CAUTIOUS
        assert set(selection.active_factors) == {"roe", "sue", "macro_trend"}
        assert sum(selection.preset_weights.values()) == pytest.approx(1.0)

    def test_select_invalid_state(self):
        """Test selecting with state not in mapping raises error."""
        registry = FactorRegistry.from_defaults()
        custom_map = {
            MarketState.UPTREND: ["momentum_20d"],
        }
        selector = FactorSelector(registry, state_map=custom_map)

        with pytest.raises(ValueError, match="not in state_map"):
            selector.select(MarketState.DOWNTREND)

    def test_select_unregistered_factor(self):
        """Test selecting with unregistered factor raises error."""
        registry = FactorRegistry()  # Empty registry
        custom_map = {
            MarketState.UPTREND: ["nonexistent_factor"],
        }
        selector = FactorSelector(registry, state_map=custom_map)

        with pytest.raises(ValueError, match="not registered in registry"):
            selector.select(MarketState.UPTREND)


class TestFactorSelectorStateMapping:
    """Test state-to-factors mapping correctness."""

    def test_default_state_map_completeness(self):
        """Test that default mapping covers all market states."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        # All 6 market states should be mapped
        assert len(selector.state_map) == 6
        assert MarketState.UPTREND in selector.state_map
        assert MarketState.DOWNTREND in selector.state_map
        assert MarketState.OSCILLATION in selector.state_map
        assert MarketState.STRUCTURAL in selector.state_map
        assert MarketState.VOLUME_CONTRACTION in selector.state_map
        assert MarketState.SENTIMENT_CAUTIOUS in selector.state_map

    def test_default_state_map_factor_counts(self):
        """Test that default mapping has expected factor counts per state."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        # Per spec FR-003
        assert len(selector.state_map[MarketState.UPTREND]) == 3
        assert len(selector.state_map[MarketState.DOWNTREND]) == 3
        assert len(selector.state_map[MarketState.OSCILLATION]) == 3
        assert len(selector.state_map[MarketState.STRUCTURAL]) == 2
        assert len(selector.state_map[MarketState.VOLUME_CONTRACTION]) == 3
        assert len(selector.state_map[MarketState.SENTIMENT_CAUTIOUS]) == 3

    def test_preset_weights_normalization(self):
        """Test that preset weights are correctly normalized."""
        registry = FactorRegistry.from_defaults()
        selector = FactorSelector(registry)

        for state in MarketState:
            selection = selector.select(state)
            # Weights must sum to 1.0
            assert sum(selection.preset_weights.values()) == pytest.approx(1.0)
            # All weights must be positive
            assert all(w > 0 for w in selection.preset_weights.values())
