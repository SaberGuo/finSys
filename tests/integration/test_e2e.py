"""End-to-end integration test (T061)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def full_pipeline_config(tmp_path: Path):
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
            train_end="2023-04-30",
            test_start="2023-05-01",
            test_end="2023-06-30",
        ),
        training=TrainingConfig(
            algorithm="ppo",
            total_timesteps=256,
            model_dir=str(tmp_path / "models"),
        ),
        environment=EnvironmentConfig(),
    )


def _make_dataset(tickers: list[str], n_days: int, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2023-01-03", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(seed)
    rows = []
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


class TestEndToEndPipeline:
    def test_data_to_train_to_backtest(
        self, full_pipeline_config, tmp_path: Path
    ) -> None:
        """Full pipeline: dataset → train → backtest → BacktestReport."""
        from finquant.training.backtest import BacktestReport
        from finquant.training.trainer import Trainer

        tickers = full_pipeline_config.stocks
        train_df = _make_dataset(tickers, n_days=60)
        test_df = _make_dataset(tickers, n_days=30, seed=1)

        trainer = Trainer(full_pipeline_config)
        model_path = trainer.train(
            train_df=train_df,
            output_dir=tmp_path / "models",
        )
        assert model_path.exists(), f"Model not found at {model_path}"

        report = trainer.backtest(
            model_path=model_path,
            test_df=test_df,
            output_dir=tmp_path / "reports",
        )
        assert isinstance(report, BacktestReport)
        metrics = report.to_dict()
        assert "sharpe" in metrics
        assert "cagr" in metrics
        assert "max_drawdown" in metrics

        html_files = list((tmp_path / "reports").glob("*.html"))
        csv_files = list((tmp_path / "reports").glob("*.csv"))
        assert len(html_files) >= 1
        assert len(csv_files) >= 1

    def test_fusion_to_enhanced_train(
        self, full_pipeline_config, tmp_path: Path
    ) -> None:
        """Enhanced pipeline: market + sentiment → fuse → train."""
        from finquant.features.fusion import FUSION_COLUMNS, fuse_datasets
        from finquant.training.trainer import Trainer

        tickers = full_pipeline_config.stocks
        market_df = _make_dataset(tickers, n_days=60)

        # Create synthetic sentiment JSONL
        sentiment_records = [
            {
                "date": "2023-01-03",
                "tic": "000001.SZ",
                "sentiment_score": 0.7,
                "event_count": 1,
                "has_positive_event": 1,
                "has_negative_event": 0,
            }
        ]
        sent_file = tmp_path / "sentiment.jsonl"
        sent_file.write_text(
            "\n".join(json.dumps(r) for r in sentiment_records),
            encoding="utf-8",
        )

        enhanced_path = fuse_datasets(
            market_df=market_df,
            sentiment_file=sent_file,
            output_path=tmp_path / "enhanced.parquet",
        )
        assert enhanced_path.exists()
        enhanced_df = pd.read_parquet(enhanced_path)
        assert all(c in enhanced_df.columns for c in FUSION_COLUMNS)
        assert not enhanced_df.isnull().any().any()

        # Train on enhanced dataset with extended indicators
        all_indicators = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30",
            "close_30_sma", "close_60_sma",
        ] + FUSION_COLUMNS

        trainer = Trainer(full_pipeline_config)
        model_path = trainer.train(
            train_df=enhanced_df,
            output_dir=tmp_path / "models_enhanced",
            indicators=all_indicators,
        )
        assert model_path.exists()
