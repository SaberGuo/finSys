"""Unit tests for StockScoringEnv."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finquant.training.scoring_env import StockScoringEnv, build_scoring_env


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample single-stock data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    data = {
        "date": dates.astype(str),
        "tic": ["600519.SH"] * 100,
        "close": 100 + np.cumsum(np.random.randn(100) * 2),
        "volume": np.random.randint(1000000, 10000000, 100),
        "macd": np.random.randn(100),
        "boll_ub": 105 + np.random.randn(100),
        "boll_lb": 95 + np.random.randn(100),
        "rsi_30": 50 + np.random.randn(100) * 10,
        "dx_30": 20 + np.random.randn(100) * 5,
        "close_30_sma": 100 + np.random.randn(100),
        "close_60_sma": 100 + np.random.randn(100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def indicators() -> list[str]:
    """Standard indicator list."""
    return ["macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"]


class TestStockScoringEnv:
    """Test suite for StockScoringEnv."""

    def test_init_valid(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test environment initialization with valid data."""
        env = StockScoringEnv(sample_data, indicators)

        assert env.ticker == "600519.SH"
        assert env.observation_space.shape == (9,)  # 2 + 7 indicators
        assert env.action_space.shape == (1,)
        assert env.max_steps == 99  # 100 - 1 (future_horizon)

    def test_init_multiple_stocks_raises(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test that multiple stocks raise ValueError."""
        multi_stock_data = sample_data.copy()
        multi_stock_data.loc[50:, "tic"] = "000001.SZ"

        with pytest.raises(ValueError, match="requires single stock"):
            StockScoringEnv(multi_stock_data, indicators)

    def test_init_missing_columns_raises(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test that missing columns raise ValueError."""
        incomplete_data = sample_data.drop(columns=["macd"])

        with pytest.raises(ValueError, match="Missing required columns"):
            StockScoringEnv(incomplete_data, indicators)

    def test_init_invalid_reward_type_raises(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test that invalid reward_type raises ValueError."""
        with pytest.raises(ValueError, match="reward_type must be"):
            StockScoringEnv(sample_data, indicators, reward_type="invalid")

    def test_reset(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test environment reset."""
        env = StockScoringEnv(sample_data, indicators)
        obs, info = env.reset()

        assert obs.shape == (9,)
        assert env.current_step == 0
        assert "date" in info
        assert "ticker" in info
        assert info["ticker"] == "600519.SH"
        assert "close" in info
        assert "volume" in info

    def test_step_daily_return(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test step with daily_return reward."""
        env = StockScoringEnv(sample_data, indicators, reward_type="daily_return")
        env.reset()

        action = np.array([0.5])  # Positive score
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (9,)
        assert isinstance(reward, float)
        assert not terminated
        assert not truncated
        assert info["score"] == 0.5
        assert env.current_step == 1

    def test_step_future_return(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test step with future_return reward."""
        env = StockScoringEnv(sample_data, indicators, reward_type="future_return", future_horizon=5)
        env.reset()

        action = np.array([-0.3])  # Negative score
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (9,)
        assert isinstance(reward, float)
        assert not terminated
        assert info["score"] == -0.3

    def test_episode_termination(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test that episode terminates at max_steps."""
        env = StockScoringEnv(sample_data, indicators)
        env.reset()

        # Step until termination
        for _ in range(env.max_steps - 1):
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            assert not terminated

        # Final step should terminate
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated

    def test_reward_calculation_daily_return(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test daily return reward calculation."""
        env = StockScoringEnv(sample_data, indicators, reward_type="daily_return")
        env.reset()

        # First step should have 0 reward (no previous day)
        _, reward, _, _, _ = env.step(np.array([0.0]))
        assert reward == 0.0

        # Second step should have non-zero reward
        close_prev = sample_data.iloc[0]["close"]
        close_curr = sample_data.iloc[1]["close"]
        expected_reward = (close_curr - close_prev) / close_prev

        _, reward, _, _, _ = env.step(np.array([0.0]))
        assert abs(reward - expected_reward) < 1e-6

    def test_reward_calculation_future_return(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test future return reward calculation."""
        env = StockScoringEnv(sample_data, indicators, reward_type="future_return", future_horizon=3)
        env.reset()

        close_curr = sample_data.iloc[0]["close"]
        close_future = sample_data.iloc[3]["close"]
        expected_reward = (close_future - close_curr) / close_curr

        _, reward, _, _, _ = env.step(np.array([0.0]))
        assert abs(reward - expected_reward) < 1e-6

    def test_observation_normalization(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test observation normalization."""
        env_normalized = StockScoringEnv(sample_data, indicators, normalize_obs=True)
        env_unnormalized = StockScoringEnv(sample_data, indicators, normalize_obs=False)

        obs_norm, _ = env_normalized.reset()
        obs_unnorm, _ = env_unnormalized.reset()

        # Normalized observations should have different values
        assert not np.allclose(obs_norm, obs_unnorm)

        # Normalized observations should have roughly zero mean and unit std
        # (not exact due to single sample, but should be different from unnormalized)
        assert abs(obs_norm.mean()) < abs(obs_unnorm.mean())

    def test_action_space_unbounded(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test that action space is unbounded."""
        env = StockScoringEnv(sample_data, indicators)

        assert env.action_space.low[0] == -np.inf
        assert env.action_space.high[0] == np.inf

    def test_observation_space_unbounded(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test that observation space is unbounded."""
        env = StockScoringEnv(sample_data, indicators)

        assert np.all(env.observation_space.low == -np.inf)
        assert np.all(env.observation_space.high == np.inf)

    def test_multiple_episodes(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test multiple episodes (reset after termination)."""
        env = StockScoringEnv(sample_data, indicators)

        # First episode
        obs1, _ = env.reset()
        for _ in range(10):
            env.step(np.array([0.0]))

        # Second episode
        obs2, _ = env.reset()
        assert env.current_step == 0
        assert np.allclose(obs1, obs2)  # Should start from same state

    def test_render_no_error(self, sample_data: pd.DataFrame, indicators: list[str]) -> None:
        """Test that render doesn't raise errors."""
        env = StockScoringEnv(sample_data, indicators)
        env.reset()

        # Should not raise
        env.render()
        env.step(np.array([0.0]))
        env.render()


class TestBuildScoringEnv:
    """Test suite for build_scoring_env helper function."""

    def test_build_with_defaults(self, sample_data: pd.DataFrame) -> None:
        """Test building environment with default parameters."""
        env = build_scoring_env(sample_data)

        assert isinstance(env, StockScoringEnv)
        assert env.observation_space.shape == (9,)  # 2 + 7 default indicators

    def test_build_with_custom_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test building environment with custom indicators."""
        custom_indicators = ["macd", "rsi_30"]
        env = build_scoring_env(sample_data, indicators=custom_indicators)

        assert env.observation_space.shape == (4,)  # 2 + 2 indicators

    def test_build_with_custom_reward_type(self, sample_data: pd.DataFrame) -> None:
        """Test building environment with custom reward type."""
        env = build_scoring_env(sample_data, reward_type="future_return", future_horizon=5)

        assert env.reward_type == "future_return"
        assert env.future_horizon == 5

    def test_build_without_normalization(self, sample_data: pd.DataFrame) -> None:
        """Test building environment without normalization."""
        env = build_scoring_env(sample_data, normalize_obs=False)

        assert not env.normalize_obs
        assert env._obs_mean is None
        assert env._obs_std is None
