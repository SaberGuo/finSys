"""Unit tests for 5-minute FinRL environment handling."""
from __future__ import annotations

import pandas as pd
import pytest

from finquant.training.env import _prepare_env_df, compute_obs_dim


class Test5MinEnv:
    """T033: validate _prepare_env_df() with 5min data uses correct (date, time) index."""

    @pytest.fixture
    def daily_df(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-02", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
        rows = []
        for d in dates:
            for tic in ["000001.SZ", "600519.SH"]:
                rows.append({
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
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def min5_df(self) -> pd.DataFrame:
        dates = ["2024-01-02"] * 10
        times = ["0930", "0935", "0940", "0945", "0950", "0955", "1000", "1005", "1010", "1015"]
        rows = []
        for d, t in zip(dates, times):
            for tic in ["000001.SZ", "600519.SH"]:
                rows.append({
                    "date": d,
                    "time": t,
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
                })
        return pd.DataFrame(rows)

    def test_daily_index_factorize(self, daily_df: pd.DataFrame) -> None:
        result = _prepare_env_df(daily_df)
        # Daily: index should be 0..4 (5 unique dates)
        assert result.index.nunique() == 5

    def test_5min_index_factorize(self, min5_df: pd.DataFrame) -> None:
        result = _prepare_env_df(min5_df)
        # 5min: index should be 0..9 (10 unique date_time combinations)
        assert result.index.nunique() == 10

    def test_5min_obs_dim_computation(self) -> None:
        dim = compute_obs_dim(2, indicators=["macd", "rsi_30", "volume_ratio"])
        assert dim == 11
