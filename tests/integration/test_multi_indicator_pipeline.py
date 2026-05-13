"""Integration test for multi-indicator-set dataset generation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from finquant.config.settings import AppConfig, DatesConfig, IndicatorSetConfig, TargetConfig
from finquant.data.dataset import TrainingDatasetBuilder
from finquant.data.sources.db_5min import Db5MinDataSource
from finquant.features.indicator_sets import IndicatorSetRegistry


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "zz500_data.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE minute_data (
            code TEXT, date TEXT, time TEXT,
            open REAL, high REAL, low REAL, close REAL,
            volume INTEGER, amount REAL
        )
        """
    )
    rows = [
        ("sh.600000", "2024-01-02", "20240102093000000", 10.0, 10.5, 9.8, 10.2, 10000, 102000.0),
        ("sh.600000", "2024-01-02", "20240102093500000", 10.2, 10.4, 10.1, 10.3, 8000, 82400.0),
        ("sh.600000", "2024-01-03", "20240102093000000", 10.3, 10.6, 10.2, 10.4, 9000, 93600.0),
        ("sh.600000", "2024-01-03", "20240102093500000", 10.4, 10.5, 10.3, 10.5, 7000, 73500.0),
    ]
    conn.executemany("INSERT INTO minute_data VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


class TestMultiIndicatorPipeline:
    """T027: end-to-end multi-set dataset generation."""

    def test_multi_set_pipeline(self, tmp_db: Path, tmp_path: Path) -> None:
        src = Db5MinDataSource(db_path=tmp_db)
        raw = src.download(["600000.SH"], "2024-01-01", "2024-12-31")

        config = AppConfig(
            stocks=["600000.SH"],
            dates=DatesConfig(
                train_start="2024-01-01",
                train_end="2024-12-31",
                test_start="2025-01-01",
                test_end="2025-06-30",
            ),
            indicator_sets=[
                IndicatorSetConfig(id="trend_momentum_5min", indicators=["macd", "rsi_30", "close_30_sma"]),
                IndicatorSetConfig(id="volatility_reversal_5min", indicators=["boll_ub", "boll_lb", "dx_30", "rsi_30"]),
            ],
            target=TargetConfig(type="future_return", horizon=1),
        )
        registry = IndicatorSetRegistry.from_configs(config.indicator_sets)
        builder = TrainingDatasetBuilder(config, registry=registry)

        out_dir = tmp_path / "training"
        out_dir.mkdir(parents=True, exist_ok=True)

        for iset_id in registry.list_ids():
            out = out_dir / f"{iset_id}_dataset.parquet"
            builder.build(raw.copy(), indicator_set_id=iset_id, output_path=out)

        # Verify distinct outputs
        df1 = pd.read_parquet(out_dir / "trend_momentum_5min_dataset.parquet")
        df2 = pd.read_parquet(out_dir / "volatility_reversal_5min_dataset.parquet")

        assert "macd" in df1.columns
        assert "boll_ub" in df2.columns
        assert set(df1.columns) != set(df2.columns)
