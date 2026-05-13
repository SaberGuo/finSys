"""Unit tests for factor registry."""

from __future__ import annotations

import pandas as pd
import pytest

from finquant.selection.factor_registry import (
    FactorCategory,
    FactorDefinition,
    FactorRegistry,
    MissingStrategy,
)


class TestFactorDefinition:
    """Test FactorDefinition dataclass."""

    def test_factor_definition_creation(self):
        """Test creating a factor definition."""
        factor = FactorDefinition(
            id="test_factor",
            name="Test Factor",
            category=FactorCategory.MOMENTUM,
            compute_fn=lambda df: df["close"],
            required_columns=["close"],
            direction=1,
        )
        assert factor.id == "test_factor"
        assert factor.category == FactorCategory.MOMENTUM
        assert factor.direction == 1

    def test_factor_definition_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be 1 or -1"):
            FactorDefinition(
                id="test_factor",
                name="Test Factor",
                category=FactorCategory.MOMENTUM,
                compute_fn=lambda df: df["close"],
                required_columns=["close"],
                direction=0,  # Invalid
            )


class TestFactorRegistry:
    """Test FactorRegistry."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = FactorRegistry()
        assert len(registry.all_ids()) == 0

    def test_register_factor(self):
        """Test registering a factor."""
        registry = FactorRegistry()
        factor = FactorDefinition(
            id="momentum_20d",
            name="20日动量",
            category=FactorCategory.MOMENTUM,
            compute_fn=lambda df: df["close"].pct_change(20),
            required_columns=["close"],
            direction=1,
        )
        registry.register(factor)

        assert "momentum_20d" in registry.all_ids()
        assert registry.get("momentum_20d") == factor

    def test_register_duplicate_factor(self):
        """Test that registering duplicate factor raises error."""
        registry = FactorRegistry()
        factor = FactorDefinition(
            id="momentum_20d",
            name="20日动量",
            category=FactorCategory.MOMENTUM,
            compute_fn=lambda df: df["close"].pct_change(20),
            required_columns=["close"],
            direction=1,
        )
        registry.register(factor)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(factor)

    def test_get_factor(self):
        """Test retrieving a factor."""
        registry = FactorRegistry()
        factor = FactorDefinition(
            id="momentum_20d",
            name="20日动量",
            category=FactorCategory.MOMENTUM,
            compute_fn=lambda df: df["close"].pct_change(20),
            required_columns=["close"],
            direction=1,
        )
        registry.register(factor)

        retrieved = registry.get("momentum_20d")
        assert retrieved.id == "momentum_20d"
        assert retrieved.name == "20日动量"

    def test_get_nonexistent_factor(self):
        """Test that getting nonexistent factor raises error."""
        registry = FactorRegistry()

        with pytest.raises(KeyError, match="not found in registry"):
            registry.get("nonexistent_factor")

    def test_list_by_category(self):
        """Test listing factors by category."""
        registry = FactorRegistry()

        momentum_factor = FactorDefinition(
            id="momentum_20d",
            name="20日动量",
            category=FactorCategory.MOMENTUM,
            compute_fn=lambda df: df["close"].pct_change(20),
            required_columns=["close"],
            direction=1,
        )
        growth_factor = FactorDefinition(
            id="growth_yoy",
            name="成长YoY",
            category=FactorCategory.GROWTH,
            compute_fn=lambda df: df["revenue_growth_pct"],
            required_columns=["revenue_growth_pct"],
            direction=1,
        )

        registry.register(momentum_factor)
        registry.register(growth_factor)

        momentum_factors = registry.list_by_category(FactorCategory.MOMENTUM)
        assert len(momentum_factors) == 1
        assert momentum_factors[0].id == "momentum_20d"

        growth_factors = registry.list_by_category(FactorCategory.GROWTH)
        assert len(growth_factors) == 1
        assert growth_factors[0].id == "growth_yoy"

    def test_record_ic(self):
        """Test recording IC values."""
        registry = FactorRegistry()
        factor = FactorDefinition(
            id="momentum_20d",
            name="20日动量",
            category=FactorCategory.MOMENTUM,
            compute_fn=lambda df: df["close"].pct_change(20),
            required_columns=["close"],
            direction=1,
        )
        registry.register(factor)

        registry.record_ic("momentum_20d", "2023-07-01", 0.15)
        registry.record_ic("momentum_20d", "2023-07-02", 0.18)

        ic_series = registry.get_ic_series("momentum_20d", window=10)
        assert len(ic_series) == 2
        assert ic_series.iloc[0] == 0.15
        assert ic_series.iloc[1] == 0.18

    def test_record_ic_nonexistent_factor(self):
        """Test that recording IC for nonexistent factor raises error."""
        registry = FactorRegistry()

        with pytest.raises(KeyError, match="not found in registry"):
            registry.record_ic("nonexistent_factor", "2023-07-01", 0.15)

    def test_get_ic_series_with_window(self):
        """Test getting IC series with window limit."""
        registry = FactorRegistry()
        factor = FactorDefinition(
            id="momentum_20d",
            name="20日动量",
            category=FactorCategory.MOMENTUM,
            compute_fn=lambda df: df["close"].pct_change(20),
            required_columns=["close"],
            direction=1,
        )
        registry.register(factor)

        # Record 10 IC values
        for i in range(10):
            registry.record_ic("momentum_20d", f"2023-07-{i+1:02d}", 0.1 + i * 0.01)

        # Get last 5
        ic_series = registry.get_ic_series("momentum_20d", window=5)
        assert len(ic_series) == 5
        assert ic_series.iloc[0] == pytest.approx(0.15)  # 6th value
        assert ic_series.iloc[-1] == pytest.approx(0.19)  # 10th value

    def test_get_ic_series_empty(self):
        """Test getting IC series when no history."""
        registry = FactorRegistry()
        factor = FactorDefinition(
            id="momentum_20d",
            name="20日动量",
            category=FactorCategory.MOMENTUM,
            compute_fn=lambda df: df["close"].pct_change(20),
            required_columns=["close"],
            direction=1,
        )
        registry.register(factor)

        ic_series = registry.get_ic_series("momentum_20d", window=10)
        assert len(ic_series) == 0


class TestFactorRegistryDefaults:
    """Test default factor registry."""

    def test_from_defaults(self):
        """Test creating registry with default factors."""
        registry = FactorRegistry.from_defaults()

        # Should have 11 built-in factors
        assert len(registry.all_ids()) == 11

        # Check specific factors exist
        assert "momentum_20d" in registry.all_ids()
        assert "high_beta" in registry.all_ids()
        assert "growth_yoy" in registry.all_ids()
        assert "low_volatility" in registry.all_ids()
        assert "low_turnover" in registry.all_ids()
        assert "high_dividend" in registry.all_ids()
        assert "small_cap" in registry.all_ids()
        assert "reversal_5d" in registry.all_ids()
        assert "roe" in registry.all_ids()
        assert "sue" in registry.all_ids()
        assert "macro_trend" in registry.all_ids()

    def test_default_factors_have_correct_categories(self):
        """Test that default factors have correct categories."""
        registry = FactorRegistry.from_defaults()

        momentum = registry.get("momentum_20d")
        assert momentum.category == FactorCategory.MOMENTUM

        growth = registry.get("growth_yoy")
        assert growth.category == FactorCategory.GROWTH

        fundamental = registry.get("roe")
        assert fundamental.category == FactorCategory.FUNDAMENTAL

    def test_default_factors_compute_fn(self):
        """Test that default factors compute functions work."""
        registry = FactorRegistry.from_defaults()

        # Test momentum_20d computation
        df = pd.DataFrame(
            {
                "tic": ["000001.SZ"] * 30,
                "close": list(range(100, 130)),
            }
        )

        momentum_factor = registry.get("momentum_20d")
        result = momentum_factor.compute_fn(df)

        assert isinstance(result, pd.Series)
        assert len(result) == 30
        # First 20 values should be 0 (fillna), rest should be positive
        assert result.iloc[20:].sum() > 0

    def test_default_factors_direction(self):
        """Test that default factors have correct direction."""
        registry = FactorRegistry.from_defaults()

        # Positive direction factors
        assert registry.get("momentum_20d").direction == 1
        assert registry.get("growth_yoy").direction == 1
        assert registry.get("roe").direction == 1

        # Factors with negative values but positive direction (already negated in compute_fn)
        assert registry.get("low_volatility").direction == 1
        assert registry.get("small_cap").direction == 1
