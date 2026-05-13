"""Unit tests for Trainer with multi-indicator sets."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finquant.config.settings import (
    AppConfig,
    DatesConfig,
    EnvironmentConfig,
    TrainingConfig,
)
from finquant.training.trainer import Trainer


@pytest.fixture()
def tiny_config() -> AppConfig:
    return AppConfig(
        stocks=["000001.SZ", "600519.SH"],
        dates=DatesConfig(
            train_start="2024-01-01",
            train_end="2024-03-01",
            test_start="2024-03-02",
            test_end="2024-04-01",
        ),
        environment=EnvironmentConfig(
            initial_amount=100_000,
            hmax=10,
            buy_cost_pct=0.001,
            sell_cost_pct=0.001,
            reward_scaling=0.0001,
        ),
        training=TrainingConfig(algorithm="ppo", total_timesteps=10),
    )


@pytest.fixture()
def tiny_train_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
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
                "rsi_30": 50.0,
                "close_30_sma": 10.0,
            })
    return pd.DataFrame(rows)


class TestTrainerMultiIndicators:
    """T034: validate Trainer.train() with different indicator sets."""

    def test_train_with_indicator_set_id(self, tiny_config: AppConfig, tmp_path: Path) -> None:
        dates = pd.date_range("2024-01-02", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
        rows = []
        for d in dates:
            for tic in tiny_config.stocks:
                rows.append({
                    "date": d,
                    "tic": tic,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "volume": 1_000_000.0,
                    "macd": 0.1,
                    "rsi_30": 50.0,
                    "close_30_sma": 10.0,
                })
        train_df = pd.DataFrame(rows)

        fake_model = MagicMock()
        fake_model.save = MagicMock()
        fake_agent = MagicMock()
        fake_agent.get_model.return_value = fake_model
        fake_agent.train_model.return_value = fake_model

        fake_drl_module = MagicMock()
        fake_drl_module.DRLAgent.return_value = fake_agent

        fake_env = MagicMock()
        fake_env.observation_space.shape = (7,)

        with patch.dict(
            sys.modules,
            {
                "finrl": MagicMock(),
                "finrl.agents": MagicMock(),
                "finrl.agents.stablebaselines3": MagicMock(),
                "finrl.agents.stablebaselines3.models": fake_drl_module,
            },
        ):
            with patch("finquant.training.trainer.build_env", return_value=fake_env):
                trainer = Trainer(tiny_config)
                model_path = trainer.train(
                    train_df, output_dir=tmp_path, indicator_set_id="trend_momentum_5min"
                )
                assert "trend_momentum_5min" in str(model_path)
                fake_model.save.assert_called_once()

    def test_backtest_obs_dim_validation(self, tiny_config: AppConfig, tmp_path: Path) -> None:
        dates = pd.date_range("2024-03-02", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
        rows = []
        for d in dates:
            for tic in tiny_config.stocks:
                rows.append({
                    "date": d,
                    "tic": tic,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "volume": 1_000_000.0,
                    "macd": 0.1,
                    "rsi_30": 50.0,
                    "close_30_sma": 10.0,
                })
        test_df = pd.DataFrame(rows)

        fake_env = MagicMock()
        fake_env.observation_space.shape = (7,)  # Wrong dim deliberately

        with patch("finquant.training.trainer.build_env", return_value=fake_env):
            trainer = Trainer(tiny_config)
            model_path = tmp_path / "model.zip"
            model_path.write_text("dummy")
            with pytest.raises(ValueError, match="Observation dim mismatch"):
                trainer.backtest(model_path, test_df, expected_obs_dim=19)
