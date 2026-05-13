"""Unit tests for finquant.training.trainer."""
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
from finquant.training.trainer import (
    Trainer,
    _detect_fusion_indicators,
    _universe_hash,
)


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


class TestDetectFusionIndicators:
    def test_no_fusion_cols(self) -> None:
        df = pd.DataFrame({"date": ["2024-01-02"], "tic": ["000001.SZ"]})
        assert _detect_fusion_indicators(df) == []

    def test_some_fusion_cols(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "tic": ["000001.SZ"],
                "sentiment_score": [0.5],
                "revenue_growth_pct": [10.0],
            }
        )
        result = _detect_fusion_indicators(df)
        assert "sentiment_score" in result
        assert "revenue_growth_pct" in result


class TestUniverseHash:
    def test_deterministic(self) -> None:
        h1 = _universe_hash(["A", "B", "C"])
        h2 = _universe_hash(["C", "B", "A"])
        assert h1 == h2
        assert len(h1) == 8


class TestTrainerInit:
    def test_valid_algo(self, tiny_config: AppConfig) -> None:
        t = Trainer(tiny_config)
        assert t._algo == "ppo"

    def test_invalid_algo_raises(self, tiny_config: AppConfig) -> None:
        tiny_config.training.algorithm = "invalid"
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            Trainer(tiny_config)

    def test_algo_case_insensitive(self, tiny_config: AppConfig) -> None:
        tiny_config.training.algorithm = "PPO"
        t = Trainer(tiny_config)
        assert t._algo == "ppo"


class TestTrainerTrain:
    def test_train_saves_model(self, tiny_config: AppConfig, tmp_path: Path) -> None:
        # Build a minimal train DataFrame
        dates = pd.date_range("2024-01-02", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
        rows = []
        for d in dates:
            for tic in tiny_config.stocks:
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
        train_df = pd.DataFrame(rows)

        fake_model = MagicMock()
        fake_model.save = MagicMock()
        fake_agent = MagicMock()
        fake_agent.get_model.return_value = fake_model
        fake_agent.train_model.return_value = fake_model

        fake_drl_module = MagicMock()
        fake_drl_module.DRLAgent.return_value = fake_agent

        with patch.dict(
            sys.modules,
            {
                "finrl": MagicMock(),
                "finrl.agents": MagicMock(),
                "finrl.agents.stablebaselines3": MagicMock(),
                "finrl.agents.stablebaselines3.models": fake_drl_module,
            },
        ):
            trainer = Trainer(tiny_config)
            model_path = trainer.train(train_df, output_dir=tmp_path)
            assert model_path.exists() or model_path.name.endswith(".zip")
            fake_model.save.assert_called_once()


class TestTrainerBacktest:
    def test_backtest_returns_report(
        self, tiny_config: AppConfig, tmp_path: Path
    ) -> None:
        dates = pd.date_range("2024-03-02", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
        rows = []
        for d in dates:
            for tic in tiny_config.stocks:
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
        test_df = pd.DataFrame(rows)

        fake_model = MagicMock()
        fake_agent = MagicMock()
        fake_agent.get_model.return_value = fake_model

        fake_drl_module = MagicMock()
        fake_drl_module.DRLAgent.return_value = fake_agent
        fake_drl_module.DRLAgent.DRL_prediction.return_value = (
            pd.DataFrame({"date": dates, "account_value": [100_000.0] * len(dates)}),
            pd.DataFrame(0.0, index=dates, columns=tiny_config.stocks),
        )

        fake_env = MagicMock()
        fake_env.observation_space.shape = (len(tiny_config.stocks) * 16 + 1,)

        with patch.dict(
            sys.modules,
            {
                "finrl": MagicMock(),
                "finrl.agents": MagicMock(),
                "finrl.agents.stablebaselines3": MagicMock(),
                "finrl.agents.stablebaselines3.models": fake_drl_module,
            },
        ):
            fake_sb3 = MagicMock()
            fake_sb3.PPO.load.return_value = fake_model
            fake_sb3.SAC = MagicMock()
            fake_sb3.TD3 = MagicMock()
            with patch.dict(sys.modules, {"stable_baselines3": fake_sb3}):
                with patch("finquant.training.trainer.build_env", return_value=fake_env):
                    trainer = Trainer(tiny_config)
                    model_path = tmp_path / "model.zip"
                    model_path.write_text("dummy")
                    report = trainer.backtest(
                        model_path, test_df, output_dir=tmp_path
                    )
                    assert report is not None
                    assert hasattr(report, "sharpe")
