"""Integration test for factor selection across different market states."""

from __future__ import annotations

import pytest

from finquant.selection import MarketState
from finquant.selection.factor_registry import FactorRegistry
from finquant.selection.factor_selector import FactorSelector


@pytest.fixture
def registry():
    """Create factor registry with default factors."""
    return FactorRegistry.from_defaults()


@pytest.fixture
def selector(registry):
    """Create factor selector."""
    return FactorSelector(registry)


def test_select_uptrend_factors(selector):
    """Test factor selection for uptrend market state."""
    selection = selector.select(MarketState.UPTREND)

    # Should select momentum, high_beta, growth_yoy
    assert "momentum_20d" in selection.active_factors
    assert "high_beta" in selection.active_factors
    assert "growth_yoy" in selection.active_factors

    # Should have preset weights
    assert len(selection.preset_weights) == len(selection.active_factors)
    assert sum(selection.preset_weights.values()) == pytest.approx(1.0)


def test_select_downtrend_factors(selector):
    """Test factor selection for downtrend market state."""
    selection = selector.select(MarketState.DOWNTREND)

    # Should select defensive factors
    assert "low_volatility" in selection.active_factors
    assert "low_turnover" in selection.active_factors
    assert "high_dividend" in selection.active_factors

    # Weights should sum to 1.0
    assert sum(selection.preset_weights.values()) == pytest.approx(1.0)


def test_select_oscillation_factors(selector):
    """Test factor selection for oscillation market state."""
    selection = selector.select(MarketState.OSCILLATION)

    # Should select defensive factors (same as downtrend per implementation)
    assert "low_volatility" in selection.active_factors
    assert "low_turnover" in selection.active_factors
    assert "high_dividend" in selection.active_factors

    # Weights should sum to 1.0
    assert sum(selection.preset_weights.values()) == pytest.approx(1.0)


def test_select_structural_factors(selector):
    """Test factor selection for structural change market state."""
    selection = selector.select(MarketState.STRUCTURAL)

    # Should select small_cap and reversal_5d per implementation
    assert "small_cap" in selection.active_factors
    assert "reversal_5d" in selection.active_factors

    # Weights should sum to 1.0
    assert sum(selection.preset_weights.values()) == pytest.approx(1.0)


def test_select_volume_contraction_factors(selector):
    """Test factor selection for volume contraction market state."""
    selection = selector.select(MarketState.VOLUME_CONTRACTION)

    # Should select defensive factors with reduced weights
    assert "low_volatility" in selection.active_factors
    assert "low_turnover" in selection.active_factors
    assert "high_dividend" in selection.active_factors

    # low_volatility and low_turnover should have reduced weights (0.5x)
    assert selection.preset_weights["low_volatility"] < selection.preset_weights["high_dividend"]
    assert selection.preset_weights["low_turnover"] < selection.preset_weights["high_dividend"]

    # Weights should sum to 1.0
    assert sum(selection.preset_weights.values()) == pytest.approx(1.0)


def test_select_sentiment_cautious_factors(selector):
    """Test factor selection for cautious sentiment market state."""
    selection = selector.select(MarketState.SENTIMENT_CAUTIOUS)

    # Should select fundamental factors per implementation
    assert "roe" in selection.active_factors
    assert "sue" in selection.active_factors
    assert "macro_trend" in selection.active_factors

    # Weights should sum to 1.0
    assert sum(selection.preset_weights.values()) == pytest.approx(1.0)


def test_select_all_states_have_factors(selector):
    """Test that all market states have factor selections."""
    for state in MarketState:
        selection = selector.select(state)

        # Every state should have at least one factor
        assert len(selection.active_factors) > 0

        # Every state should have preset weights
        assert len(selection.preset_weights) == len(selection.active_factors)

        # Weights should sum to 1.0
        assert sum(selection.preset_weights.values()) == pytest.approx(1.0)


def test_select_factors_exist_in_registry(selector, registry):
    """Test that all selected factors exist in registry."""
    for state in MarketState:
        selection = selector.select(state)

        for factor_id in selection.active_factors:
            # Factor should be registered
            factor_def = registry.get(factor_id)
            assert factor_def is not None
            assert factor_def.id == factor_id


def test_select_consistent_across_calls(selector):
    """Test that factor selection is consistent across multiple calls."""
    state = MarketState.UPTREND

    selection1 = selector.select(state)
    selection2 = selector.select(state)

    # Should return same factors
    assert selection1.active_factors == selection2.active_factors

    # Should return same weights
    assert selection1.preset_weights == selection2.preset_weights


def test_select_different_states_different_factors(selector):
    """Test that different market states select different factors."""
    uptrend_selection = selector.select(MarketState.UPTREND)
    downtrend_selection = selector.select(MarketState.DOWNTREND)

    # Uptrend and downtrend should have different factors
    assert set(uptrend_selection.active_factors) != set(downtrend_selection.active_factors)


def test_select_factor_weights_positive(selector):
    """Test that all factor weights are positive."""
    for state in MarketState:
        selection = selector.select(state)

        for weight in selection.preset_weights.values():
            assert weight > 0


def test_select_factor_weights_reasonable(selector):
    """Test that factor weights are in reasonable range."""
    for state in MarketState:
        selection = selector.select(state)

        for weight in selection.preset_weights.values():
            # No single factor should dominate (> 80%)
            assert weight <= 0.8

            # No factor should be too small (< 5%)
            assert weight >= 0.05


def test_select_uptrend_vs_downtrend_opposite(selector):
    """Test that uptrend and downtrend select opposite factor types."""
    uptrend = selector.select(MarketState.UPTREND)
    downtrend = selector.select(MarketState.DOWNTREND)

    # Uptrend should have momentum/growth factors
    assert any(f in uptrend.active_factors for f in ["momentum_20d", "high_beta", "growth_yoy"])

    # Downtrend should have defensive factors
    assert any(f in downtrend.active_factors for f in ["low_volatility", "low_turnover", "high_dividend"])

    # They should not overlap significantly
    overlap = set(uptrend.active_factors) & set(downtrend.active_factors)
    assert len(overlap) == 0  # No overlap expected


def test_select_volume_contraction_weight_adjustment(selector):
    """Test that volume contraction adjusts weights correctly."""
    normal_downtrend = selector.select(MarketState.DOWNTREND)
    volume_contraction = selector.select(MarketState.VOLUME_CONTRACTION)

    # Both should have defensive factors
    assert "low_volatility" in normal_downtrend.active_factors
    assert "low_volatility" in volume_contraction.active_factors

    # Volume contraction should reduce low_vol/low_turnover weights
    if "low_volatility" in normal_downtrend.active_factors and "low_volatility" in volume_contraction.active_factors:
        # Volume contraction should have lower weight for low_volatility
        assert volume_contraction.preset_weights["low_volatility"] < normal_downtrend.preset_weights["low_volatility"]


def test_select_state_to_factors_mapping_complete(selector):
    """Test that state-to-factors mapping covers all states."""
    state_map = selector._default_state_map()

    # All market states should be in the mapping
    for state in MarketState:
        assert state in state_map

        # Each state should have at least one factor
        assert len(state_map[state]) > 0


def test_select_custom_state_map(registry):
    """Test factor selector with custom state-to-factors mapping."""
    custom_map = {
        MarketState.UPTREND: ["momentum_20d"],
        MarketState.DOWNTREND: ["low_volatility"],
        MarketState.OSCILLATION: ["reversal_5d"],
        MarketState.STRUCTURAL: ["roe"],
        MarketState.VOLUME_CONTRACTION: ["low_turnover"],
        MarketState.SENTIMENT_CAUTIOUS: ["macro_trend"],
    }

    selector = FactorSelector(registry, state_map=custom_map)

    # Should use custom mapping
    selection = selector.select(MarketState.UPTREND)
    assert selection.active_factors == ["momentum_20d"]


def test_select_handles_missing_factors_gracefully(registry):
    """Test that selector raises error for missing factors."""
    # Create mapping with non-existent factor
    custom_map = {
        MarketState.UPTREND: ["nonexistent_factor"],
        MarketState.DOWNTREND: ["low_volatility"],
        MarketState.OSCILLATION: ["reversal_5d"],
        MarketState.STRUCTURAL: ["roe"],
        MarketState.VOLUME_CONTRACTION: ["low_turnover"],
        MarketState.SENTIMENT_CAUTIOUS: ["macro_trend"],
    }

    selector = FactorSelector(registry, state_map=custom_map)

    # Should raise ValueError for missing factor
    with pytest.raises(ValueError, match="Factor .* not registered"):
        selector.select(MarketState.UPTREND)
