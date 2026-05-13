"""Unit tests for indicator set computation."""
from __future__ import annotations

import pandas as pd
import pytest

from finquant.config.settings import IndicatorSetConfig
from finquant.features.indicator_sets import IndicatorSet, IndicatorSetRegistry
from finquant.features.technical import compute_indicators


class TestIndicatorSetCompute:
    """T025: validate each predefined set computes correct columns."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": ["2024-01-02"] * 10,
            "time": ["0930", "0935", "0940", "0945", "0950", "0955", "1000", "1005", "1010", "1015"],
            "tic": ["600000.SH"] * 10,
            "open": [10.0 + i * 0.1 for i in range(10)],
            "high": [10.5 + i * 0.1 for i in range(10)],
            "low": [9.8 + i * 0.1 for i in range(10)],
            "close": [10.2 + i * 0.1 for i in range(10)],
            "volume": [10000 - i * 100 for i in range(10)],
        })

    def test_trend_momentum_set(self, sample_df: pd.DataFrame) -> None:
        iset = IndicatorSet(
            id="trend_momentum_5min",
            name="Trend + Momentum",
            frequency="5min",
            indicators=["macd", "rsi_30", "close_30_sma"],
        )
        result = iset.compute(sample_df)
        assert "macd" in result.columns
        assert "rsi_30" in result.columns
        assert "close_30_sma" in result.columns

    def test_volatility_reversal_set(self, sample_df: pd.DataFrame) -> None:
        iset = IndicatorSet(
            id="volatility_reversal_5min",
            name="Volatility + Reversal",
            frequency="5min",
            indicators=["boll_ub", "boll_lb", "dx_30", "rsi_30"],
        )
        result = iset.compute(sample_df)
        assert "boll_ub" in result.columns
        assert "boll_lb" in result.columns
        assert "dx_30" in result.columns
        assert "rsi_30" in result.columns

    def test_volume_price_set(self, sample_df: pd.DataFrame) -> None:
        iset = IndicatorSet(
            id="volume_price_5min",
            name="Volume + Price",
            frequency="5min",
            indicators=["close_30_sma", "close_60_sma", "volume_ratio"],
        )
        result = iset.compute(sample_df)
        assert "close_30_sma" in result.columns
        assert "close_60_sma" in result.columns
        assert "volume_ratio" in result.columns

    def test_registry_from_configs(self, sample_df: pd.DataFrame) -> None:
        configs = [
            IndicatorSetConfig(id="a", indicators=["macd"]),
            IndicatorSetConfig(id="b", indicators=["rsi_30"]),
        ]
        registry = IndicatorSetRegistry.from_configs(configs)
        result_a = registry.get("a").compute(sample_df)
        assert "macd" in result_a.columns
        result_b = registry.get("b").compute(sample_df)
        assert "rsi_30" in result_b.columns
