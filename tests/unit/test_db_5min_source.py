"""Unit tests for Db5MinDataSource ticker conversion and DB reading."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from finquant.data.sources.db_5min import Db5MinDataSource


class TestTickerConversion:
    """T006: ticker format conversion sh.600000 ↔ 600000.SH."""

    def test_ticker_to_db_code_sh(self) -> None:
        assert Db5MinDataSource._ticker_to_db_code("600000.SH") == "sh.600000"

    def test_ticker_to_db_code_sz(self) -> None:
        assert Db5MinDataSource._ticker_to_db_code("000001.SZ") == "sz.000001"

    def test_db_code_to_ticker_sh(self) -> None:
        assert Db5MinDataSource._db_code_to_ticker("sh.600000") == "600000.SH"

    def test_db_code_to_ticker_sz(self) -> None:
        assert Db5MinDataSource._db_code_to_ticker("sz.000001") == "000001.SZ"

    def test_roundtrip(self) -> None:
        original = "603000.SH"
        db_code = Db5MinDataSource._ticker_to_db_code(original)
        back = Db5MinDataSource._db_code_to_ticker(db_code)
        assert back == original


class TestDb5MinDataSourceDownload:
    """Contract-level tests for download() with an in-memory DB."""

    @pytest.fixture
    def tmp_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "test.db"
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
            ("sz.000001", "2024-01-02", "20240102093000000", 12.0, 12.2, 11.9, 12.1, 5000, 60500.0),
        ]
        conn.executemany(
            "INSERT INTO minute_data VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        conn.commit()
        conn.close()
        return db

    def test_download_returns_required_columns(self, tmp_db: Path) -> None:
        src = Db5MinDataSource(db_path=tmp_db)
        df = src.download(["600000.SH"], "2024-01-01", "2024-12-31")
        assert "tic" in df.columns
        assert "date" in df.columns
        assert "time" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_download_ticker_normalized(self, tmp_db: Path) -> None:
        src = Db5MinDataSource(db_path=tmp_db)
        df = src.download(["600000.SH", "000001.SZ"], "2024-01-01", "2024-12-31")
        tickers = df["tic"].unique().tolist()
        assert "600000.SH" in tickers
        assert "000001.SZ" in tickers
        assert "sh.600000" not in tickers

    def test_download_date_filter(self, tmp_db: Path) -> None:
        src = Db5MinDataSource(db_path=tmp_db)
        df = src.download(["600000.SH"], "2024-01-02", "2024-01-02")
        assert len(df) == 2

    def test_download_missing_db(self, tmp_path: Path) -> None:
        src = Db5MinDataSource(db_path=tmp_path / "no_such.db")
        with pytest.raises(FileNotFoundError):
            src.download(["600000.SH"], "2024-01-01", "2024-12-31")
