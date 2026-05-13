import sqlite3
from pathlib import Path

import pandas as pd


class DBReader:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get_daily_data(
        self,
        code: str,
        end_date: str,
        n_days: int = 65,
    ) -> pd.DataFrame:
        """Return up to n_days rows of daily OHLCV data ending at end_date (inclusive)."""
        sql = """
            SELECT date, open, high, low, close, volume, amount
            FROM daily_data
            WHERE code = ? AND date <= ?
            ORDER BY date DESC
            LIMIT ?
        """
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=(code, end_date, n_days))
        if df.empty:
            return df
        df = df.sort_values("date").reset_index(drop=True)
        # Exclude rows with null prices
        price_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=price_cols).reset_index(drop=True)
        return df

    def get_future_close(self, code: str, anchor_date: str, horizon: int) -> pd.DataFrame:
        """Return up to horizon rows of daily data strictly after anchor_date."""
        sql = """
            SELECT date, close
            FROM daily_data
            WHERE code = ? AND date > ?
            ORDER BY date ASC
            LIMIT ?
        """
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=(code, anchor_date, horizon))
        return df

    def get_all_codes(self) -> list[str]:
        """Return sorted list of all unique stock codes."""
        with self._connect() as conn:
            cur = conn.execute("SELECT DISTINCT code FROM daily_data ORDER BY code")
            return [row[0] for row in cur.fetchall()]

    def get_date_range(self, code: str) -> tuple[str, str]:
        """Return (min_date, max_date) for the given code."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT MIN(date), MAX(date) FROM daily_data WHERE code = ?", (code,)
            )
            row = cur.fetchone()
        if row[0] is None:
            raise KeyError(f"Code not found in database: {code}")
        return row[0], row[1]

    def get_tradeable_dates(self, code: str, start_date: str, end_date: str) -> list[str]:
        """Return sorted list of trading dates for a code within [start_date, end_date]."""
        sql = """
            SELECT DISTINCT date FROM daily_data
            WHERE code = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """
        with self._connect() as conn:
            cur = conn.execute(sql, (code, start_date, end_date))
            return [row[0] for row in cur.fetchall()]
