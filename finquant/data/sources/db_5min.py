"""SQLite 5-minute data source adapter for zz500_data.db."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from finquant.data.sources.base import DataSource


class Db5MinDataSource(DataSource):
    """Read 5-minute K-line data from zz500_data.db."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else Path("data/processed/zz500_data.db")

    # ------------------------------------------------------------------
    # Ticker format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _ticker_to_db_code(ticker: str) -> str:
        """Convert ``600000.SH`` → ``sh.600000``."""
        code, exchange = ticker.upper().split(".")
        prefix = exchange.lower()
        return f"{prefix}.{code}"

    @staticmethod
    def _db_code_to_ticker(db_code: str) -> str:
        """Convert ``sh.600000`` → ``600000.SH``."""
        prefix, code = db_code.lower().split(".")
        exchange = prefix.upper()
        return f"{code}.{exchange}"

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def download(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Query ``minute_data`` table and return standardised DataFrame.

        Note: Database stores codes in standard format (e.g., 600000.SH, 000001.SZ),
        not in xtquant format (sh.600000, sz.000001).
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.db_path}")

        # Database uses standard ticker format (600000.SH), no conversion needed
        placeholders = ",".join("?" for _ in symbols)

        query = f"""
            SELECT code, date, time, open, high, low, close, volume, amount
            FROM minute_data
            WHERE code IN ({placeholders})
              AND date >= ? AND date <= ?
            ORDER BY code, date, time
        """

        with sqlite3.connect(self.db_path) as conn:
            params = (*symbols, start_date, end_date)
            df = pd.read_sql_query(query, conn, params=params)

        # Rename code column to tic (no format conversion needed)
        df = df.rename(columns={"code": "tic"})
        return df
