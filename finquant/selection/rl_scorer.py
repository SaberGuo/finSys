"""RL-based stock scorer for single-stock evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from finquant.training.env import build_env
from finquant.utils.logging import get_logger

logger = get_logger(__name__)


class RLStockScorer:
    """Score individual stocks using a single-stock RL model.

    The model is trained with stock_dim=1 and evaluates each stock
    independently. Actions are mapped to scores via sigmoid transformation.
    """

    def __init__(
        self,
        model_path: Path,
        indicators: list[str],
        score_mapping: str = "sigmoid",
    ):
        """Initialize scorer with trained RL model.

        Args:
            model_path: Path to trained .zip model file
            indicators: List of technical indicator names
            score_mapping: How to map action to score ("sigmoid", "linear", "direct")
        """
        from stable_baselines3 import PPO, SAC, TD3

        self.model_path = model_path
        self.indicators = indicators
        self.score_mapping = score_mapping

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Try loading with different algorithms
        for algo_cls in [PPO, SAC, TD3]:
            try:
                self.model = algo_cls.load(str(model_path))
                logger.info(f"Loaded {algo_cls.__name__} model from {model_path}")
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Failed to load model from {model_path}")

        # Validate model is single-stock (stock_dim=1)
        expected_obs_dim = 1 + (2 + len(indicators)) * 1
        actual_obs_dim = self.model.observation_space.shape[0]
        if actual_obs_dim != expected_obs_dim:
            raise ValueError(
                f"Model observation space mismatch: expected {expected_obs_dim} "
                f"(single-stock with {len(indicators)} indicators), got {actual_obs_dim}"
            )

    def score_stocks(
        self,
        market_df: pd.DataFrame,
        date: str,
    ) -> dict[str, float]:
        """Score all stocks for a given date.

        Args:
            market_df: DataFrame with columns [date, tic, close, volume, indicators...]
            date: Date to score (YYYY-MM-DD)

        Returns:
            Dict mapping ticker → score [0, 1]
        """
        date_df = market_df[market_df["date"] == date].copy()

        if date_df.empty:
            logger.warning(f"No data for date {date}")
            return {}

        scores = {}

        for ticker in date_df["tic"].unique():
            try:
                stock_df = date_df[date_df["tic"] == ticker].copy()

                env = build_env(
                    df=stock_df,
                    stock_dim=1,
                    initial_amount=1_000_000,
                    hmax=100,
                    indicators=self.indicators,
                )

                obs, _ = env.reset()
                action, _ = self.model.predict(obs, deterministic=True)
                score = self._map_action_to_score(action)
                scores[ticker] = float(score)

            except Exception as e:
                logger.warning(f"Failed to score {ticker}: {e}")
                scores[ticker] = 0.0

        return scores

    def _map_action_to_score(self, action: np.ndarray | float) -> float:
        """Map RL action to score [0, 1].

        Args:
            action: RL model output (continuous value)

        Returns:
            Score in [0, 1] range
        """
        if isinstance(action, np.ndarray):
            action = float(action[0])

        if self.score_mapping == "sigmoid":
            return 1.0 / (1.0 + np.exp(-action))

        elif self.score_mapping == "linear":
            return (action + 1.0) / 2.0

        elif self.score_mapping == "direct":
            return max(0.0, min(1.0, action))

        else:
            raise ValueError(f"Unknown score_mapping: {self.score_mapping}")
