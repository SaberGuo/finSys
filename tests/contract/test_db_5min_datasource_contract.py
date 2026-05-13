"""Contract tests for Db5MinDataSource.download() schema compliance."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from finquant.data.sources.base import REQUIRED_COLUMNS
from finquant.data.sources.db_5min import Db5MinDataSource


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
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
        ("sz.000001", "2024-01-02", "20240102093000000", 12.0, 12.2, 11.9, 12.1, 5000, 60500.0),
    ]
    conn.executemany("INSERT INTO minute_data VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


class TestDb5MinContract:
    """T008: validate download() returns correct columns and types."""

    def test_required_columns_present(self, populated_db: Path) -> None:
        src = Db5MinDataSource(db_path=populated_db)
        df = src.download(["600000.SH", "000001.SZ"], "2024-01-01", "2024-12-31")
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"missing required column: {col}"

    def test_time_column_present(self, populated_db: Path) -> None:
        src = Db5MinDataSource(db_path=populated_db)
        df = src.download(["600000.SH"], "2024-01-01", "2024-12-31")
        assert "time" in df.columns

    def test_no_empty_result(self, populated_db: Path) -> None:
        src = Db5MinDataSource(db_path=populated_db)
        df = src.download(["600000.SH"], "2024-01-01", "2024-12-31")
        assert not df.empty

    def test_price_integrity(self, populated_db: Path) -> None:
        src = Db5MinDataSource(db_path=populated_db)
        df = src.download(["600000.SH"], "2024-01-01", "2024-12-31")
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["close"] > 0).all()
