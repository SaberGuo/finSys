"""SQLite daily data source adapter for zz500_data.db."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from finquant.data.sources.base import DataSource


class DbDailyDataSource(DataSource):
    """Read daily K-line data from zz500_data.db."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else Path("data/processed/zz500_data.db")

    # ------------------------------------------------------------------
    # Ticker format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _ticker_to_db_code(ticker: str) -> str:
        """Convert ``600000.SH`` → ``600000.SH`` (already in correct format)."""
        return ticker

    @staticmethod
    def _db_code_to_ticker(db_code: str) -> str:
        """Convert ``600000.SH`` → ``600000.SH`` (already in correct format)."""
        return db_code

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def download(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Query ``daily_data`` table and return standardised DataFrame."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.db_path}")

        db_codes = [self._ticker_to_db_code(t) for t in symbols]
        placeholders = ",".join("?" for _ in db_codes)

        query = f"""
            SELECT code, date, open, high, low, close, volume, amount
            FROM daily_data
            WHERE code IN ({placeholders})
              AND date >= ? AND date <= ?
            ORDER BY code, date
        """

        with sqlite3.connect(self.db_path) as conn:
            params = (*db_codes, start_date, end_date)
            df = pd.read_sql_query(query, conn, params=params)

        # Rename code to tic
        df = df.rename(columns={"code": "tic"})

        return df
