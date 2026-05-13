"""Unit tests for 5-minute data preprocessing."""
from __future__ import annotations

import pandas as pd
import pytest

from finquant.data.preprocessor import preprocess_5min_data


class Test5MinPreprocessor:
    """T016: validate 5min preprocessing."""

    @pytest.fixture
    def raw_5min(self) -> pd.DataFrame:
        return pd.DataFrame({
            "code": ["sh.600000", "sh.600000", "sz.000001"],
            "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "time": ["20240102093000000", "20240102093500000", "20240102093000000"],
            "open": [10.0, 10.2, 12.0],
            "high": [10.5, 10.4, 12.2],
            "low": [9.8, 10.1, 11.9],
            "close": [10.2, 10.3, 12.1],
            "volume": [10000, 8000, 5000],
            "amount": [102000.0, 82400.0, 60500.0],
        })

    def test_time_column_parsed(self, raw_5min: pd.DataFrame) -> None:
        df = preprocess_5min_data(raw_5min.rename(columns={"code": "tic"}))
        assert "time" in df.columns
        assert "datetime" in df.columns

    def test_ticker_normalized(self, raw_5min: pd.DataFrame) -> None:
        df = preprocess_5min_data(raw_5min.rename(columns={"code": "tic"}))
        tickers = df["tic"].unique().tolist()
        assert "600000.SH" in tickers
        assert "000001.SZ" in tickers
        assert "sh.600000" not in tickers

    def test_alignment_fills_missing(self, raw_5min: pd.DataFrame) -> None:
        # 600000 has 2 rows, 000001 has 1 row → alignment should create 3 rows
        df = preprocess_5min_data(raw_5min.rename(columns={"code": "tic"}))
        # After alignment we have 2 times × 2 tickers = 4 rows
        assert len(df) == 4

    def test_no_nan_prices(self, raw_5min: pd.DataFrame) -> None:
        df = preprocess_5min_data(raw_5min.rename(columns={"code": "tic"}))
        assert df["close"].notna().all()
        assert df["open"].notna().all()

    def test_missing_required_columns(self) -> None:
        bad = pd.DataFrame({"date": ["2024-01-02"]})
        with pytest.raises(ValueError, match="missing required columns"):
            preprocess_5min_data(bad)

    def test_sort_order(self, raw_5min: pd.DataFrame) -> None:
        df = preprocess_5min_data(raw_5min.rename(columns={"code": "tic"}))
        assert df.equals(df.sort_values(["date", "time", "tic"]).reset_index(drop=True))
