"""Integration test for multi-indicator-set training pipeline."""
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
from finquant.data.dataset import TrainingDatasetBuilder
from finquant.features.indicator_sets import IndicatorSetRegistry
from finquant.training.trainer import Trainer


@pytest.fixture()
def config_multi() -> AppConfig:
    from finquant.config.settings import IndicatorSetConfig

    return AppConfig(
        stocks=["000001.SZ"],
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
        indicator_sets=[
            IndicatorSetConfig(id="set_a", indicators=["macd", "rsi_30"]),
            IndicatorSetConfig(id="set_b", indicators=["boll_ub", "boll_lb", "dx_30"]),
        ],
    )


@pytest.fixture()
def train_df_multi() -> pd.DataFrame:
    dates = ["2024-01-02"] * 20
    times = [f"{h:02d}{m:02d}" for h in range(9, 15) for m in (30, 35)]
    rows = []
    for d, t in zip(dates, times):
        rows.append({
            "date": d,
            "time": t,
            "tic": "000001.SZ",
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 1_000_000.0,
            "macd": 0.1,
            "rsi_30": 50.0,
            "boll_ub": 11.0,
            "boll_lb": 9.0,
            "dx_30": 25.0,
        })
    return pd.DataFrame(rows)


class TestMultiIndicatorTraining:
    """T045: train multiple indicator sets and verify distinct model files."""

    def test_train_multiple_sets(self, config_multi: AppConfig, train_df_multi: pd.DataFrame, tmp_path: Path) -> None:
        fake_model = MagicMock()
        fake_model.save = MagicMock()
        fake_agent = MagicMock()
        fake_agent.get_model.return_value = fake_model
        fake_agent.train_model.return_value = fake_model

        fake_drl_module = MagicMock()
        fake_drl_module.DRLAgent.return_value = fake_agent
        fake_drl_module.DRLAgent.DRL_prediction.return_value = (
            pd.DataFrame({"date": ["2024-03-02"], "account_value": [100_000.0]}),
            pd.DataFrame(0.0, index=["2024-03-02"], columns=config_multi.stocks),
        )

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
            fake_sb3 = MagicMock()
            fake_sb3.PPO.load.return_value = fake_model
            fake_sb3.SAC = MagicMock()
            fake_sb3.TD3 = MagicMock()
            with patch.dict(sys.modules, {"stable_baselines3": fake_sb3}):
                with patch("finquant.training.trainer.build_env", return_value=fake_env):
                    registry = IndicatorSetRegistry.from_configs(config_multi.indicator_sets)
                    builder = TrainingDatasetBuilder(config_multi, registry=registry)

                    models: list[Path] = []
                    for iset_id in registry.list_ids():
                        dataset_path = builder.build(
                            train_df_multi.copy(),
                            indicator_set_id=iset_id,
                            output_path=tmp_path / f"{iset_id}_dataset.parquet",
                        )
                        df = pd.read_parquet(dataset_path)

                        trainer = Trainer(config_multi)
                        model_path = trainer.train(
                            train_df=df,
                            output_dir=tmp_path / iset_id / "models",
                            indicator_set_id=iset_id,
                        )
                        models.append(model_path)
                        assert model_path.name.endswith(".zip")
                        assert iset_id in str(model_path)

                    # Verify metadata files exist
                    for iset_id in registry.list_ids():
                        meta_files = list((tmp_path / iset_id / "models").glob("*_metadata.json"))
                        assert len(meta_files) > 0, f"missing metadata for {iset_id}"
