"""Unit tests for finquant.training.env (T026)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from finquant.training.env import (
    INDICATORS,
    OBS_DIM_PER_STOCK,
    build_env,
    compute_obs_dim,
)


@pytest.fixture()
def tiny_dataset() -> pd.DataFrame:
    """2 stocks × 20 days with required columns."""
    dates = pd.date_range("2024-01-02", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
    rows = []
    for d in dates:
        for tic in ["000001.SZ", "600519.SH"]:
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


class TestComputeObsDim:
    def test_formula(self) -> None:
        # obs_dim = 1 + OBS_DIM_PER_STOCK * N
        assert compute_obs_dim(5) == 1 + OBS_DIM_PER_STOCK * 5

    def test_single_stock(self) -> None:
        assert compute_obs_dim(1) == 1 + OBS_DIM_PER_STOCK

    def test_zero_stocks_raises(self) -> None:
        with pytest.raises(ValueError, match="stock_dim"):
            compute_obs_dim(0)

    def test_obs_dim_per_stock_matches_indicators(self) -> None:
        # OBS_DIM_PER_STOCK = close + volume + len(INDICATORS)
        assert OBS_DIM_PER_STOCK == 2 + len(INDICATORS)


class TestBuildEnv:
    def test_returns_env_object(self, tiny_dataset: pd.DataFrame) -> None:
        env = build_env(tiny_dataset, stock_dim=2)
        assert env is not None

    def test_observation_space_shape(self, tiny_dataset: pd.DataFrame) -> None:
        env = build_env(tiny_dataset, stock_dim=2)
        expected = compute_obs_dim(2)
        # FinRL StockTradingEnv observation_space is Box
        assert env.observation_space.shape[0] == expected

    def test_action_space_shape(self, tiny_dataset: pd.DataFrame) -> None:
        env = build_env(tiny_dataset, stock_dim=2)
        # action_space is Box of shape (stock_dim,)
        assert env.action_space.shape[0] == 2

    def test_custom_initial_amount(self, tiny_dataset: pd.DataFrame) -> None:
        env = build_env(tiny_dataset, stock_dim=2, initial_amount=500_000)
        obs, _ = env.reset()
        assert obs[0] == pytest.approx(500_000.0, rel=0.01)

    def test_env_resets_without_error(self, tiny_dataset: pd.DataFrame) -> None:
        env = build_env(tiny_dataset, stock_dim=2)
        obs, info = env.reset()
        assert obs.shape[0] == compute_obs_dim(2)

    def test_dimension_mismatch_raises(self, tiny_dataset: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="stock_dim"):
            build_env(tiny_dataset, stock_dim=5)  # only 2 unique tickers

    def test_custom_indicators(self, tiny_dataset: pd.DataFrame) -> None:
        custom = ["macd", "rsi_30"]
        env = build_env(tiny_dataset, stock_dim=2, indicators=custom)
        expected = 1 + (2 + len(custom)) * 2
        assert env.observation_space.shape[0] == expected
