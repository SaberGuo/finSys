from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finquant.config.settings import AppConfig, DataConfig, DatesConfig
from finquant.data.pipeline import DataPipeline, DataSourceExhaustedError


@pytest.fixture
def mock_config() -> AppConfig:
    return AppConfig(
        stocks=["000001.SZ"],
        dates=DatesConfig(
            train_start="2025-01-01",
            train_end="2025-01-10",
            test_start="2025-01-11",
            test_end="2025-01-20",
        ),
        data=DataConfig(source_priority=["akshare"]),
        indicators=["macd"],
    )


class TestDataPipelineFetch:
    def test_fetch_success(self, mock_config: AppConfig) -> None:
        pipeline = DataPipeline(mock_config)
        # Generate 35 rows so 30-period indicators (rsi_30, dx_30) compute without NaN
        dates = pd.date_range("2025-01-02", periods=35, freq="B").strftime("%Y-%m-%d").tolist()
        mock_df = pd.DataFrame(
            {
                "date": dates,
                "tic": ["000001.SZ"] * 35,
                "open": [10.0 + i * 0.01 for i in range(35)],
                "high": [10.2 + i * 0.01 for i in range(35)],
                "low": [9.9 + i * 0.01 for i in range(35)],
                "close": [10.1 + i * 0.01 for i in range(35)],
                "volume": [1000.0 + i * 100 for i in range(35)],
            }
        )

        with patch.object(
            pipeline.adapters["akshare"], "download", return_value=mock_df
        ) as mock_download:
            result = pipeline.fetch()

        mock_download.assert_called_once_with(
            symbols=["000001.SZ"],
            start_date="2025-01-01",
            end_date="2025-01-20",
        )
        assert isinstance(result, pd.DataFrame)
        assert "macd" in result.columns
        # Indicators need a warm-up window; verify core OHLCV + macd are present
        assert not result[["date", "tic", "open", "high", "low", "close", "volume", "macd"]].isnull().any().any()

    def test_fetch_failover(self, mock_config: AppConfig) -> None:
        mock_config.data.source_priority = ["xtquant", "akshare"]
        pipeline = DataPipeline(mock_config)
        mock_df = pd.DataFrame(
            {
                "date": ["2025-01-02"],
                "tic": ["000001.SZ"],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "volume": [1000.0],
            }
        )

        with patch.object(pipeline.adapters["xtquant"], "download", side_effect=ConnectionError("xtquant down")):
            with patch.object(
                pipeline.adapters["akshare"], "download", return_value=mock_df
            ) as mock_download:
                result = pipeline.fetch()

        mock_download.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_fetch_all_sources_fail(self, mock_config: AppConfig) -> None:
        mock_config.data.source_priority = ["akshare"]
        pipeline = DataPipeline(mock_config)

        with patch.object(pipeline.adapters["akshare"], "download", side_effect=RuntimeError("network")):
            with pytest.raises(DataSourceExhaustedError, match="all data sources failed"):
                pipeline.fetch()


class TestDataPipelineFetchAndSave:
    def test_fetch_and_save_writes_parquet(self, mock_config: AppConfig, tmp_path: Path) -> None:
        pipeline = DataPipeline(mock_config)
        mock_df = pd.DataFrame(
            {
                "date": ["2025-01-02"],
                "tic": ["000001.SZ"],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "volume": [1000.0],
            }
        )

        with patch.object(pipeline.adapters["akshare"], "download", return_value=mock_df):
            out_path = pipeline.fetch_and_save(output_dir=str(tmp_path))

        assert out_path.exists()
        assert out_path.suffix == ".parquet"
        read_back = pd.read_parquet(out_path)
        assert len(read_back) == 1
