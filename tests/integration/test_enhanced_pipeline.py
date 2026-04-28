"""Integration tests for enhanced feature fusion pipeline (T053)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def market_parquet(tmp_path: Path) -> Path:
    dates = pd.date_range("2023-01-03", periods=40, freq="B").strftime("%Y-%m-%d").tolist()
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
    df = pd.DataFrame(rows)
    path = tmp_path / "market.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture()
def sentiment_jsonl(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-03", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
    records = []
    for d in dates:
        records.append(
            {
                "date": d,
                "tic": "000001.SZ",
                "sentiment_score": float(rng.uniform(-1, 1)),
                "event_count": int(rng.integers(0, 3)),
                "has_positive_event": int(rng.integers(0, 2)),
                "has_negative_event": int(rng.integers(0, 2)),
            }
        )
    path = tmp_path / "sentiment.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


class TestEnhancedPipelineIntegration:
    def test_fuse_datasets_end_to_end(
        self,
        market_parquet: Path,
        sentiment_jsonl: Path,
        tmp_path: Path,
    ) -> None:
        from finquant.features.fusion import FUSION_COLUMNS, fuse_datasets

        df = pd.read_parquet(market_parquet)
        out = fuse_datasets(
            market_df=df,
            sentiment_file=sentiment_jsonl,
            output_path=tmp_path / "enhanced.parquet",
        )
        assert out.exists()
        enhanced = pd.read_parquet(out)
        assert len(enhanced) == len(df)
        for col in FUSION_COLUMNS:
            assert col in enhanced.columns
        assert not enhanced.isnull().any().any()

    def test_sc004_enhanced_env_obs_dim(
        self,
        market_parquet: Path,
        sentiment_jsonl: Path,
        tmp_path: Path,
    ) -> None:
        """SC-004: Enhanced env obs_space = 1 + 16*N."""
        from finquant.features.fusion import FUSION_COLUMNS, fuse_datasets
        from finquant.training.env import build_env

        df = pd.read_parquet(market_parquet)
        out = fuse_datasets(
            market_df=df,
            sentiment_file=sentiment_jsonl,
            output_path=tmp_path / "enhanced.parquet",
        )
        enhanced = pd.read_parquet(out)

        N = enhanced["tic"].nunique()
        all_indicators = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30",
            "close_30_sma", "close_60_sma",
        ] + FUSION_COLUMNS

        env = build_env(
            enhanced,
            stock_dim=N,
            indicators=all_indicators,
        )
        expected_dim = 1 + (2 + len(all_indicators)) * N
        assert env.observation_space.shape[0] == expected_dim
