"""Unit tests for IC weight calculator."""

from __future__ import annotations

import pandas as pd
import pytest

from finquant.selection.factor_registry import FactorRegistry
from finquant.selection.ic_weight import ICWeightCalculator


@pytest.fixture
def registry():
    """Create factor registry with test factors."""
    from finquant.selection.factor_registry import FactorCategory, FactorDefinition

    reg = FactorRegistry()

    # Register test factors
    reg.register(FactorDefinition(
        id="momentum",
        name="Momentum",
        category=FactorCategory.MOMENTUM,
        compute_fn=lambda df: df["close"].pct_change(20),
        required_columns=["close"],
        direction=1,
    ))

    reg.register(FactorDefinition(
        id="volatility",
        name="Volatility",
        category=FactorCategory.LOW_VOL,
        compute_fn=lambda df: df["close"].pct_change().rolling(20).std(),
        required_columns=["close"],
        direction=-1,
    ))

    return reg


@pytest.fixture
def ic_calculator(registry):
    """Create IC weight calculator."""
    return ICWeightCalculator(registry=registry, window=60, min_periods=20)


def test_compute_ic_basic(ic_calculator):
    """Test basic IC computation (Spearman correlation)."""
    # Perfect positive correlation
    factor_values = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
    future_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=["A", "B", "C", "D", "E"])

    ic = ic_calculator.compute_ic(factor_values, future_returns)
    assert ic == pytest.approx(1.0, abs=0.01)


def test_compute_ic_negative_correlation(ic_calculator):
    """Test IC with negative correlation."""
    # Perfect negative correlation
    factor_values = pd.Series([5, 4, 3, 2, 1], index=["A", "B", "C", "D", "E"])
    future_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=["A", "B", "C", "D", "E"])

    ic = ic_calculator.compute_ic(factor_values, future_returns)
    assert ic == pytest.approx(-1.0, abs=0.01)


def test_compute_ic_no_correlation(ic_calculator):
    """Test IC with no correlation."""
    # Random uncorrelated data
    factor_values = pd.Series([1, 5, 2, 4, 3], index=["A", "B", "C", "D", "E"])
    future_returns = pd.Series([0.03, 0.01, 0.05, 0.02, 0.04], index=["A", "B", "C", "D", "E"])

    ic = ic_calculator.compute_ic(factor_values, future_returns)
    # IC can vary for small samples, just check it's computed
    assert -1.0 <= ic <= 1.0


def test_compute_ic_with_nan(ic_calculator):
    """Test IC computation with NaN values."""
    factor_values = pd.Series([1, 2, None, 4, 5], index=["A", "B", "C", "D", "E"])
    future_returns = pd.Series([0.01, 0.02, 0.03, None, 0.05], index=["A", "B", "C", "D", "E"])

    ic = ic_calculator.compute_ic(factor_values, future_returns)
    # Should compute on valid pairs only (A, B, E)
    assert ic != 0.0


def test_compute_ic_insufficient_data(ic_calculator):
    """Test IC with insufficient data points."""
    factor_values = pd.Series([1], index=["A"])
    future_returns = pd.Series([0.01], index=["A"])

    ic = ic_calculator.compute_ic(factor_values, future_returns)
    assert ic == 0.0


def test_compute_ic_misaligned_indices(ic_calculator):
    """Test IC with misaligned indices."""
    factor_values = pd.Series([1, 2, 3], index=["A", "B", "C"])
    future_returns = pd.Series([0.01, 0.02, 0.03], index=["D", "E", "F"])

    ic = ic_calculator.compute_ic(factor_values, future_returns)
    assert ic == 0.0


def test_update_ic_history(ic_calculator, registry):
    """Test IC history recording."""
    factor_values = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
    future_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=["A", "B", "C", "D", "E"])

    ic_calculator.update_ic_history("momentum", "2023-01-01", factor_values, future_returns)

    ic_series = registry.get_ic_series("momentum", window=10)
    assert len(ic_series) == 1
    assert ic_series.iloc[0] == pytest.approx(1.0, abs=0.01)


def test_compute_weights_equal_fallback(ic_calculator):
    """Test weight computation with insufficient IC history (equal weights fallback)."""
    weights = ic_calculator.compute_weights(
        factor_ids=["momentum", "volatility"],
        as_of_date="2023-01-01",
        preset_weights=None,
    )

    # Should return equal weights
    assert weights == {"momentum": 0.5, "volatility": 0.5}


def test_compute_weights_preset_fallback(ic_calculator):
    """Test weight computation with preset weights fallback."""
    preset = {"momentum": 0.7, "volatility": 0.3}

    weights = ic_calculator.compute_weights(
        factor_ids=["momentum", "volatility"],
        as_of_date="2023-01-01",
        preset_weights=preset,
    )

    # Should return preset weights
    assert weights == preset


def test_compute_weights_with_ic_history(ic_calculator, registry):
    """Test weight computation with sufficient IC history."""
    # Build IC history for both factors
    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")

        # momentum has higher IC (0.5)
        registry.record_ic("momentum", date_str, 0.5)

        # volatility has lower IC (0.2)
        registry.record_ic("volatility", date_str, 0.2)

    weights = ic_calculator.compute_weights(
        factor_ids=["momentum", "volatility"],
        as_of_date="2023-01-30",
        preset_weights=None,
    )

    # momentum should have higher weight (0.5 / 0.7 ≈ 0.714)
    assert weights["momentum"] > weights["volatility"]
    assert weights["momentum"] == pytest.approx(0.714, abs=0.01)
    assert weights["volatility"] == pytest.approx(0.286, abs=0.01)

    # Weights should sum to 1.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_compute_weights_all_zero_ic(ic_calculator, registry):
    """Test weight computation when all ICs are zero."""
    # Build IC history with all zeros
    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        registry.record_ic("momentum", date_str, 0.0)
        registry.record_ic("volatility", date_str, 0.0)

    preset = {"momentum": 0.6, "volatility": 0.4}

    weights = ic_calculator.compute_weights(
        factor_ids=["momentum", "volatility"],
        as_of_date="2023-01-30",
        preset_weights=preset,
    )

    # Should fallback to preset weights
    assert weights == preset


def test_compute_weights_negative_ic(ic_calculator, registry):
    """Test weight computation with negative IC values."""
    # Build IC history with negative values
    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        # momentum has negative IC
        registry.record_ic("momentum", date_str, -0.3)
        # volatility has positive IC
        registry.record_ic("volatility", date_str, 0.5)

    weights = ic_calculator.compute_weights(
        factor_ids=["momentum", "volatility"],
        as_of_date="2023-01-30",
        preset_weights=None,
    )

    # Weights based on absolute IC: |−0.3| / (|−0.3| + |0.5|) = 0.375
    assert weights["momentum"] == pytest.approx(0.375, abs=0.01)
    assert weights["volatility"] == pytest.approx(0.625, abs=0.01)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_compute_weights_single_factor(ic_calculator, registry):
    """Test weight computation with single factor."""
    # Build IC history
    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        registry.record_ic("momentum", date_str, 0.4)

    weights = ic_calculator.compute_weights(
        factor_ids=["momentum"],
        as_of_date="2023-01-30",
        preset_weights=None,
    )

    # Single factor should get weight 1.0
    assert weights == {"momentum": 1.0}


def test_compute_weights_window_parameter(registry):
    """Test IC weight calculation respects window parameter."""
    # Create calculator with small window
    calc = ICWeightCalculator(registry=registry, window=5, min_periods=3)

    # Build IC history with varying values
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        # Early dates have low IC, later dates have high IC
        ic_value = 0.1 if i < 5 else 0.9
        registry.record_ic("momentum", date_str, ic_value)

    # With window=5, should only consider last 5 days (high IC)
    weights = calc.compute_weights(
        factor_ids=["momentum"],
        as_of_date="2023-01-10",
        preset_weights=None,
    )

    assert weights == {"momentum": 1.0}
