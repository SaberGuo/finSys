"""Unit tests for IndicatorSet configuration and registry."""
from __future__ import annotations

import pytest

from finquant.config.settings import IndicatorSetConfig
from finquant.features.indicator_sets import IndicatorSet, IndicatorSetRegistry


class TestIndicatorSetConfig:
    """T007: IndicatorSetConfig validation."""

    def test_valid_config(self) -> None:
        cfg = IndicatorSetConfig(
            id="test_set",
            name="Test Set",
            frequency="5min",
            indicators=["macd", "rsi_30"],
            window_params={"rsi_period": 1440},
            target="future_return",
        )
        assert cfg.frequency == "5min"

    def test_invalid_frequency(self) -> None:
        with pytest.raises(ValueError, match="frequency must be"):
            IndicatorSetConfig(id="bad", frequency="1min")

    def test_invalid_target_type(self) -> None:
        from finquant.config.settings import TargetConfig

        with pytest.raises(ValueError, match="target type must be"):
            TargetConfig(type="invalid")


class TestIndicatorSetRegistry:
    """T007: registry lookup and loading."""

    def test_register_and_get(self) -> None:
        registry = IndicatorSetRegistry()
        iset = IndicatorSet(
            id="trend_momentum_5min",
            name="Trend + Momentum",
            frequency="5min",
            indicators=["macd", "rsi_30", "close_30_sma"],
        )
        registry.register(iset)
        assert registry.get("trend_momentum_5min").name == "Trend + Momentum"

    def test_get_missing_raises(self) -> None:
        registry = IndicatorSetRegistry()
        with pytest.raises(KeyError):
            registry.get("missing")

    def test_from_configs(self) -> None:
        configs = [
            IndicatorSetConfig(id="a", indicators=["macd"]),
            IndicatorSetConfig(id="b", indicators=["rsi_30"]),
        ]
        registry = IndicatorSetRegistry.from_configs(configs)
        assert set(registry.list_ids()) == {"a", "b"}
