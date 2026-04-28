"""Contract tests for FinRL environment interface (T028)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finquant.training.env import build_env, compute_obs_dim


@pytest.fixture()
def two_stock_dataset() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=30, freq="B").strftime("%Y-%m-%d").tolist()
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


class TestFinRLEnvContract:
    """Verify env satisfies Gymnasium interface contract."""

    def test_observation_space_is_box(self, two_stock_dataset: pd.DataFrame) -> None:
        from gymnasium.spaces import Box

        env = build_env(two_stock_dataset, stock_dim=2)
        assert isinstance(env.observation_space, Box)

    def test_action_space_is_box(self, two_stock_dataset: pd.DataFrame) -> None:
        from gymnasium.spaces import Box

        env = build_env(two_stock_dataset, stock_dim=2)
        assert isinstance(env.action_space, Box)

    def test_reset_returns_obs_and_info(self, two_stock_dataset: pd.DataFrame) -> None:
        env = build_env(two_stock_dataset, stock_dim=2)
        result = env.reset()
        assert isinstance(result, tuple)
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_returns_5_tuple(self, two_stock_dataset: pd.DataFrame) -> None:
        env = build_env(two_stock_dataset, stock_dim=2)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_obs_dim_matches_data(self, two_stock_dataset: pd.DataFrame) -> None:
        N = 2
        env = build_env(two_stock_dataset, stock_dim=N)
        expected_dim = compute_obs_dim(N)
        obs, _ = env.reset()
        assert obs.shape == (expected_dim,)

    def test_obs_contains_cash_as_first_element(
        self, two_stock_dataset: pd.DataFrame
    ) -> None:
        env = build_env(two_stock_dataset, stock_dim=2, initial_amount=1_000_000)
        obs, _ = env.reset()
        # First element should be initial cash (normalized by reward_scaling)
        assert obs[0] > 0

    def test_env_runs_full_episode(self, two_stock_dataset: pd.DataFrame) -> None:
        env = build_env(two_stock_dataset, stock_dim=2)
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 500:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            terminated = terminated or truncated
            steps += 1
        assert steps > 0
