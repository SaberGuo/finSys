"""Contract tests for EnhancedDataset schema (T052)."""
from __future__ import annotations

import pandas as pd
import pytest

from finquant.features.fusion import FUSION_COLUMNS, fuse_dataframes
from finquant.training.env import OBS_DIM_PER_STOCK, compute_obs_dim


ENHANCED_OBS_DIM_PER_STOCK = OBS_DIM_PER_STOCK + len(FUSION_COLUMNS)  # 9 + 7 = 16


@pytest.fixture()
def base_market() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
    tickers = ["000001.SZ", "000002.SZ"]
    rows = []
    for d in dates:
        for tic in tickers:
            rows.append(
                {
                    "date": d,
                    "tic": tic,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "volume": 1_000_000.0,
                    "macd": 0.1,
                    "boll_ub": 11.0,
                    "boll_lb": 9.0,
                    "rsi_30": 50.0,
                    "dx_30": 25.0,
                    "close_30_sma": 10.0,
                    "close_60_sma": 10.0,
                }
            )
    return pd.DataFrame(rows)


class TestEnhancedDatasetContract:
    def test_schema_has_all_columns(self, base_market: pd.DataFrame) -> None:
        empty_sentiment = pd.DataFrame(
            columns=[
                "date", "tic", "sentiment_score", "event_count",
                "has_positive_event", "has_negative_event",
                "revenue_growth_pct", "net_profit_margin", "debt_ratio",
            ]
        )
        result = fuse_dataframes(base_market, empty_sentiment)
        for col in FUSION_COLUMNS:
            assert col in result.columns

    def test_no_nans(self, base_market: pd.DataFrame) -> None:
        empty_sentiment = pd.DataFrame(
            columns=[
                "date", "tic", "sentiment_score", "event_count",
                "has_positive_event", "has_negative_event",
                "revenue_growth_pct", "net_profit_margin", "debt_ratio",
            ]
        )
        result = fuse_dataframes(base_market, empty_sentiment)
        assert not result.isnull().any().any()

    def test_row_count_equals_market(self, base_market: pd.DataFrame) -> None:
        empty_sentiment = pd.DataFrame(
            columns=[
                "date", "tic", "sentiment_score", "event_count",
                "has_positive_event", "has_negative_event",
                "revenue_growth_pct", "net_profit_margin", "debt_ratio",
            ]
        )
        result = fuse_dataframes(base_market, empty_sentiment)
        assert len(result) == len(base_market)

    def test_enhanced_obs_dim_formula(self) -> None:
        N = 5
        # obs_dim for enhanced = 1 + 16*N
        enhanced_dim = 1 + ENHANCED_OBS_DIM_PER_STOCK * N
        assert enhanced_dim == 1 + 16 * N

    def test_unique_date_tic_pairs(self, base_market: pd.DataFrame) -> None:
        empty_sentiment = pd.DataFrame(
            columns=[
                "date", "tic", "sentiment_score", "event_count",
                "has_positive_event", "has_negative_event",
                "revenue_growth_pct", "net_profit_margin", "debt_ratio",
            ]
        )
        result = fuse_dataframes(base_market, empty_sentiment)
        assert not result.duplicated(subset=["date", "tic"]).any()
