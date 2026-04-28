"""Integration tests for training pipeline (T029)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def sample_train_dataset() -> pd.DataFrame:
    """3 stocks × 60 days with indicators — minimal viable training dataset."""
    dates = pd.date_range("2023-01-03", periods=60, freq="B").strftime("%Y-%m-%d").tolist()
    tickers = ["000001.SZ", "000002.SZ", "600519.SH"]
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for tic in tickers:
            close = float(rng.uniform(9, 11))
            rows.append(
                {
                    "date": d,
                    "tic": tic,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": float(rng.integers(500_000, 2_000_000)),
                    "macd": float(rng.normal(0, 0.1)),
                    "boll_ub": close * 1.05,
                    "boll_lb": close * 0.95,
                    "rsi_30": float(rng.uniform(30, 70)),
                    "dx_30": float(rng.uniform(10, 50)),
                    "close_30_sma": close,
                    "close_60_sma": close,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def minimal_config(tmp_path: Path):
    """AppConfig with minimal training params for fast integration tests."""
    from finquant.config.settings import (
        AppConfig,
        DatesConfig,
        EnvironmentConfig,
        TrainingConfig,
    )

    return AppConfig(
        stocks=["000001.SZ", "000002.SZ", "600519.SH"],
        dates=DatesConfig(
            train_start="2023-01-03",
            train_end="2023-03-31",
            test_start="2023-04-01",
            test_end="2023-06-30",
        ),
        training=TrainingConfig(
            algorithm="ppo",
            total_timesteps=512,  # minimal for speed
            model_dir=str(tmp_path / "models"),
        ),
        environment=EnvironmentConfig(),
    )


class TestTrainingPipelineIntegration:
    def test_train_saves_model(
        self, sample_train_dataset: pd.DataFrame, minimal_config, tmp_path: Path
    ) -> None:
        from finquant.training.trainer import Trainer

        trainer = Trainer(minimal_config)
        model_path = trainer.train(
            train_df=sample_train_dataset,
            output_dir=tmp_path / "models",
        )
        assert model_path.exists()

    def test_backtest_produces_report(
        self, sample_train_dataset: pd.DataFrame, minimal_config, tmp_path: Path
    ) -> None:
        from finquant.training.backtest import BacktestReport
        from finquant.training.trainer import Trainer

        trainer = Trainer(minimal_config)
        model_path = trainer.train(
            train_df=sample_train_dataset,
            output_dir=tmp_path / "models",
        )
        report = trainer.backtest(
            model_path=model_path,
            test_df=sample_train_dataset,  # use same data for speed
            output_dir=tmp_path / "reports",
        )
        assert isinstance(report, BacktestReport)
        assert "sharpe" in report.to_dict()

    def test_backtest_report_html_generated(
        self, sample_train_dataset: pd.DataFrame, minimal_config, tmp_path: Path
    ) -> None:
        from finquant.training.trainer import Trainer

        trainer = Trainer(minimal_config)
        model_path = trainer.train(
            train_df=sample_train_dataset,
            output_dir=tmp_path / "models",
        )
        report = trainer.backtest(
            model_path=model_path,
            test_df=sample_train_dataset,
            output_dir=tmp_path / "reports",
        )
        html_files = list((tmp_path / "reports").glob("*.html"))
        assert len(html_files) >= 1

    def test_sc002_sharpe_calculated(
        self, sample_train_dataset: pd.DataFrame, minimal_config, tmp_path: Path
    ) -> None:
        """SC-002: backtest must report Sharpe ratio."""
        from finquant.training.trainer import Trainer

        trainer = Trainer(minimal_config)
        model_path = trainer.train(
            train_df=sample_train_dataset,
            output_dir=tmp_path / "models",
        )
        report = trainer.backtest(
            model_path=model_path,
            test_df=sample_train_dataset,
            output_dir=tmp_path / "reports",
        )
        d = report.to_dict()
        assert isinstance(d["sharpe"], float)
        assert isinstance(d["cagr"], float)
        assert isinstance(d["max_drawdown"], float)
