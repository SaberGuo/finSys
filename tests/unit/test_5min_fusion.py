"""Unit tests for 5-minute data fusion."""
from __future__ import annotations

import pandas as pd
import pytest

from finquant.features.fusion import fuse_dataframes


class Test5MinFusion:
    """T043: validate fuse_dataframes preserves 5min row count."""

    @pytest.fixture
    def market_5min(self) -> pd.DataFrame:
        rows = []
        for t in ["0930", "0935", "0940"]:
            rows.append({
                "date": "2024-01-02",
                "time": t,
                "tic": "600000.SH",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 10000,
            })
        return pd.DataFrame(rows)

    @pytest.fixture
    def sentiment_daily(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": ["2024-01-02"],
            "tic": ["600000.SH"],
            "sentiment_score": [0.5],
            "event_count": [3],
        })

    def test_row_count_preserved(self, market_5min: pd.DataFrame, sentiment_daily: pd.DataFrame) -> None:
        result = fuse_dataframes(market_5min, sentiment_daily)
        assert len(result) == len(market_5min)

    def test_fusion_columns_filled(self, market_5min: pd.DataFrame, sentiment_daily: pd.DataFrame) -> None:
        result = fuse_dataframes(market_5min, sentiment_daily)
        assert "sentiment_score" in result.columns
        assert "event_count" in result.columns
        assert result["sentiment_score"].notna().all()

    def test_time_column_preserved(self, market_5min: pd.DataFrame, sentiment_daily: pd.DataFrame) -> None:
        result = fuse_dataframes(market_5min, sentiment_daily)
        assert "time" in result.columns
