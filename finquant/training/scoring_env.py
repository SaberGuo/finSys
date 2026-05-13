"""Stock scoring environment for RL training.

This module provides a Gym environment for training RL agents to score individual stocks
rather than directly trading them. The agent outputs a continuous score for each stock,
which is then used by a separate portfolio manager for trading decisions.

Key differences from StockTradingEnv:
- Single stock per environment (stock_dim=1)
- Action space: single continuous score (unbounded)
- Observation space: stock features only (no cash, no positions)
- Reward: daily return or future return
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class StockScoringEnv(gym.Env):
    """Single-stock scoring environment for RL training.

    The agent learns to output a score for a stock based on its features.
    The score represents the stock's attractiveness for trading.

    Observation Space
    -----------------
    Box(shape=(2 + len(indicators),), dtype=float32)
    - close: closing price
    - volume: trading volume
    - indicators: technical indicators (e.g., MACD, RSI, Bollinger Bands)

    Action Space
    ------------
    Box(low=-inf, high=inf, shape=(1,), dtype=float32)
    - Single continuous score (unbounded)
    - Positive score: bullish signal
    - Negative score: bearish signal
    - Magnitude: confidence level

    Reward
    ------
    - daily_return: (close_t - close_{t-1}) / close_{t-1}
    - future_return: (close_{t+N} - close_t) / close_t

    Parameters
    ----------
    df : pd.DataFrame
        Single-stock market data with columns: date, tic, close, volume, indicators
    indicators : list[str]
        List of technical indicator column names
    reward_type : str, default="daily_return"
        Reward calculation method: "daily_return" or "future_return"
    future_horizon : int, default=1
        Number of days ahead for future_return calculation
    normalize_obs : bool, default=True
        Whether to normalize observations using z-score
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        indicators: list[str],
        reward_type: str = "daily_return",
        future_horizon: int = 1,
        normalize_obs: bool = True,
    ) -> None:
        super().__init__()

        # Validate input
        if df["tic"].nunique() != 1:
            raise ValueError(
                f"StockScoringEnv requires single stock, got {df['tic'].nunique()} stocks"
            )

        required_cols = ["date", "tic", "close", "volume"] + indicators
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if reward_type not in ["daily_return", "future_return"]:
            raise ValueError(
                f"reward_type must be 'daily_return' or 'future_return', got {reward_type!r}"
            )

        # Store configuration
        self.df = df.sort_values("date").reset_index(drop=True)
        self.indicators = indicators
        self.reward_type = reward_type
        self.future_horizon = future_horizon
        self.normalize_obs = normalize_obs
        self.ticker = df["tic"].iloc[0]

        # Define spaces
        obs_dim = 2 + len(indicators)  # close + volume + indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )

        # Episode state
        self.current_step = 0
        self.max_steps = len(self.df) - future_horizon

        # Normalization statistics (computed on first reset)
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state.

        Returns
        -------
        observation : np.ndarray
            Initial observation
        info : dict
            Additional information (date, ticker, close)
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Compute normalization statistics on first reset
        if self.normalize_obs and self._obs_mean is None:
            self._compute_normalization_stats()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Parameters
        ----------
        action : np.ndarray
            Stock score (shape=(1,))

        Returns
        -------
        observation : np.ndarray
            Next observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode has ended
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information
        """
        # Store action (score) for info
        score = float(action[0])

        # Calculate reward
        reward = self._calculate_reward()

        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get next observation
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self._get_info()
        info["score"] = score

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector.

        Returns
        -------
        np.ndarray
            [close, volume, indicator_1, ..., indicator_N]
        """
        row = self.df.iloc[self.current_step]
        obs = np.array(
            [row["close"], row["volume"]] + [row[ind] for ind in self.indicators],
            dtype=np.float32,
        )

        # Normalize if enabled
        if self.normalize_obs and self._obs_mean is not None:
            obs = (obs - self._obs_mean) / (self._obs_std + 1e-8)

        return obs

    def _calculate_reward(self) -> float:
        """Calculate reward based on reward_type.

        Returns
        -------
        float
            Reward value
        """
        if self.reward_type == "daily_return":
            # Current day return: (close_t - close_{t-1}) / close_{t-1}
            if self.current_step == 0:
                return 0.0
            close_prev = self.df.iloc[self.current_step - 1]["close"]
            close_curr = self.df.iloc[self.current_step]["close"]
            return float((close_curr - close_prev) / close_prev)

        elif self.reward_type == "future_return":
            # Future return: (close_{t+N} - close_t) / close_t
            if self.current_step + self.future_horizon >= len(self.df):
                return 0.0
            close_curr = self.df.iloc[self.current_step]["close"]
            close_future = self.df.iloc[self.current_step + self.future_horizon]["close"]
            return float((close_future - close_curr) / close_curr)

        return 0.0

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about current state.

        Returns
        -------
        dict
            Information dictionary with date, ticker, close, volume
        """
        if self.current_step >= len(self.df):
            return {"date": None, "ticker": self.ticker, "close": 0.0, "volume": 0.0}

        row = self.df.iloc[self.current_step]
        return {
            "date": row["date"],
            "ticker": self.ticker,
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for observation normalization."""
        obs_list = []
        for i in range(len(self.df) - self.future_horizon):
            row = self.df.iloc[i]
            obs = np.array(
                [row["close"], row["volume"]] + [row[ind] for ind in self.indicators],
                dtype=np.float32,
            )
            obs_list.append(obs)

        obs_array = np.array(obs_list)
        self._obs_mean = np.mean(obs_array, axis=0)
        self._obs_std = np.std(obs_array, axis=0)

    def render(self) -> None:
        """Render environment state (optional)."""
        if self.current_step >= len(self.df):
            return
        row = self.df.iloc[self.current_step]
        print(f"Step {self.current_step}: {row['date']} | Close: {row['close']:.2f} | Volume: {row['volume']:.0f}")


def build_scoring_env(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
    reward_type: str = "daily_return",
    future_horizon: int = 1,
    normalize_obs: bool = True,
) -> StockScoringEnv:
    """Build a StockScoringEnv instance.

    Parameters
    ----------
    df : pd.DataFrame
        Single-stock market data
    indicators : list[str], optional
        Technical indicator column names. Defaults to standard 7 indicators.
    reward_type : str, default="daily_return"
        Reward calculation method
    future_horizon : int, default=1
        Number of days ahead for future_return
    normalize_obs : bool, default=True
        Whether to normalize observations

    Returns
    -------
    StockScoringEnv
        Configured scoring environment
    """
    from finquant.training.env import INDICATORS

    ind = indicators if indicators is not None else INDICATORS

    return StockScoringEnv(
        df=df,
        indicators=ind,
        reward_type=reward_type,
        future_horizon=future_horizon,
        normalize_obs=normalize_obs,
    )
