"""Unit tests for multi-indicator-set dataset output."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finquant.config.settings import AppConfig, DatesConfig, IndicatorSetConfig, TargetConfig
from finquant.data.dataset import TrainingDatasetBuilder
from finquant.features.indicator_sets import IndicatorSetRegistry


class TestMultiIndicatorOutput:
    """T026: validate 3 indicator sets produce different column sets."""

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

    @pytest.fixture
    def config(self) -> AppConfig:
        return AppConfig(
            stocks=["600000.SH"],
            dates=DatesConfig(
                train_start="2024-01-01",
                train_end="2024-12-31",
                test_start="2025-01-01",
                test_end="2025-06-30",
            ),
            indicator_sets=[
                IndicatorSetConfig(id="set_a", indicators=["macd", "rsi_30"]),
                IndicatorSetConfig(id="set_b", indicators=["boll_ub", "boll_lb", "dx_30"]),
                IndicatorSetConfig(id="set_c", indicators=["close_30_sma", "close_60_sma", "volume_ratio"]),
            ],
            target=TargetConfig(type="future_return", horizon=1),
        )

    @pytest.fixture
    def registry(self, config: AppConfig) -> IndicatorSetRegistry:
        return IndicatorSetRegistry.from_configs(config.indicator_sets)

    def test_different_column_sets(self, sample_df: pd.DataFrame, config: AppConfig, registry: IndicatorSetRegistry, tmp_path: Path) -> None:
        builder = TrainingDatasetBuilder(config, registry=registry)

        out_a = builder.build(sample_df.copy(), indicator_set_id="set_a", output_path=tmp_path / "a.parquet")
        out_b = builder.build(sample_df.copy(), indicator_set_id="set_b", output_path=tmp_path / "b.parquet")
        out_c = builder.build(sample_df.copy(), indicator_set_id="set_c", output_path=tmp_path / "c.parquet")

        df_a = pd.read_parquet(out_a)
        df_b = pd.read_parquet(out_b)
        df_c = pd.read_parquet(out_c)

        cols_a = set(df_a.columns) - {"date", "time", "tic", "open", "high", "low", "close", "volume", "target"}
        cols_b = set(df_b.columns) - {"date", "time", "tic", "open", "high", "low", "close", "volume", "target"}
        cols_c = set(df_c.columns) - {"date", "time", "tic", "open", "high", "low", "close", "volume", "target"}

        assert cols_a != cols_b
        assert cols_b != cols_c
        assert "macd" in cols_a
        assert "boll_ub" in cols_b
        assert "volume_ratio" in cols_c
