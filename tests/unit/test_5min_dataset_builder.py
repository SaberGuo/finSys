"""Unit tests for TrainingDatasetBuilder."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finquant.config.settings import AppConfig, DatesConfig, IndicatorSetConfig, TargetConfig
from finquant.data.dataset import TrainingDatasetBuilder
from finquant.features.indicator_sets import IndicatorSetRegistry


class TestTrainingDatasetBuilder:
    """T017: validate TrainingDatasetBuilder.build() output schema and NaN handling."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "time": ["0930", "0935", "0930", "0935"],
            "tic": ["600000.SH", "600000.SH", "600000.SH", "600000.SH"],
            "open": [10.0, 10.2, 10.3, 10.4],
            "high": [10.5, 10.4, 10.6, 10.5],
            "low": [9.8, 10.1, 10.2, 10.3],
            "close": [10.2, 10.3, 10.4, 10.5],
            "volume": [10000, 8000, 9000, 7000],
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
                IndicatorSetConfig(
                    id="trend_momentum_5min",
                    indicators=["macd", "rsi_30", "close_30_sma"],
                ),
            ],
            target=TargetConfig(type="future_return", horizon=1),
        )

    @pytest.fixture
    def registry(self, config: AppConfig) -> IndicatorSetRegistry:
        from finquant.features.indicator_sets import IndicatorSetRegistry
        return IndicatorSetRegistry.from_configs(config.indicator_sets)

    def test_build_output_schema(self, sample_df: pd.DataFrame, config: AppConfig, registry: IndicatorSetRegistry, tmp_path: Path) -> None:
        builder = TrainingDatasetBuilder(config, registry=registry)
        out = builder.build(sample_df, indicator_set_id="trend_momentum_5min", output_path=tmp_path / "test.parquet")
        assert out.exists()
        result = pd.read_parquet(out)
        assert "macd" in result.columns
        assert "rsi_30" in result.columns
        assert "close_30_sma" in result.columns
        assert "target" in result.columns

    def test_no_nan_indicators(self, sample_df: pd.DataFrame, config: AppConfig, registry: IndicatorSetRegistry, tmp_path: Path) -> None:
        builder = TrainingDatasetBuilder(config, registry=registry)
        out = builder.build(sample_df, indicator_set_id="trend_momentum_5min", output_path=tmp_path / "test.parquet")
        result = pd.read_parquet(out)
        indicator_cols = [c for c in result.columns if c not in ["date", "time", "tic", "open", "high", "low", "close", "volume", "target"]]
        for col in indicator_cols:
            assert result[col].notna().all(), f"NaN found in {col}"

    def test_sorted_by_date_time_tic(self, sample_df: pd.DataFrame, config: AppConfig, registry: IndicatorSetRegistry, tmp_path: Path) -> None:
        builder = TrainingDatasetBuilder(config, registry=registry)
        out = builder.build(sample_df, indicator_set_id="trend_momentum_5min", output_path=tmp_path / "test.parquet")
        result = pd.read_parquet(out)
        assert result.equals(result.sort_values(["date", "time", "tic"]).reset_index(drop=True))

    def test_missing_indicator_set_raises(self, sample_df: pd.DataFrame, config: AppConfig, tmp_path: Path) -> None:
        builder = TrainingDatasetBuilder(config, registry=None)
        with pytest.raises(ValueError):
            builder.build(sample_df, indicator_set_id="missing", output_path=tmp_path / "test.parquet")
