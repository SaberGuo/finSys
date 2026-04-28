"""Unit tests for finquant.features.fusion (T051)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finquant.features.fusion import (
    FUSION_COLUMNS,
    FUSION_FILL_VALUES,
    fuse_datasets,
    fuse_dataframes,
)


@pytest.fixture()
def market_df() -> pd.DataFrame:
    """3 stocks × 5 days market dataset with base columns."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
    tickers = ["000001.SZ", "600519.SH", "000002.SZ"]
    rows = []
    for d in dates:
        for tic in tickers:
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


@pytest.fixture()
def sentiment_records_df() -> pd.DataFrame:
    """Partial sentiment records (only some date×tic combinations)."""
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "tic": "000001.SZ",
                "sentiment_score": 0.8,
                "event_count": 2,
                "has_positive_event": 1,
                "has_negative_event": 0,
            },
            {
                "date": "2024-01-03",
                "tic": "600519.SH",
                "sentiment_score": -0.5,
                "event_count": 1,
                "has_positive_event": 0,
                "has_negative_event": 1,
            },
        ]
    )


class TestFusionColumns:
    def test_fusion_columns_defined(self) -> None:
        assert len(FUSION_COLUMNS) == 7

    def test_fill_values_match_columns(self) -> None:
        assert set(FUSION_FILL_VALUES.keys()) == set(FUSION_COLUMNS)


class TestFuseDataframes:
    def test_row_count_preserved(
        self, market_df: pd.DataFrame, sentiment_records_df: pd.DataFrame
    ) -> None:
        result = fuse_dataframes(market_df, sentiment_records_df)
        assert len(result) == len(market_df)

    def test_all_fusion_columns_present(
        self, market_df: pd.DataFrame, sentiment_records_df: pd.DataFrame
    ) -> None:
        result = fuse_dataframes(market_df, sentiment_records_df)
        for col in FUSION_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans_after_fusion(
        self, market_df: pd.DataFrame, sentiment_records_df: pd.DataFrame
    ) -> None:
        result = fuse_dataframes(market_df, sentiment_records_df)
        assert not result.isnull().any().any()

    def test_matched_rows_have_correct_values(
        self, market_df: pd.DataFrame, sentiment_records_df: pd.DataFrame
    ) -> None:
        result = fuse_dataframes(market_df, sentiment_records_df)
        row = result[(result["date"] == "2024-01-02") & (result["tic"] == "000001.SZ")]
        assert row["sentiment_score"].iloc[0] == pytest.approx(0.8)
        assert row["has_positive_event"].iloc[0] == 1

    def test_unmatched_rows_filled_with_defaults(
        self, market_df: pd.DataFrame, sentiment_records_df: pd.DataFrame
    ) -> None:
        result = fuse_dataframes(market_df, sentiment_records_df)
        row = result[(result["date"] == "2024-01-04") & (result["tic"] == "000002.SZ")]
        assert row["sentiment_score"].iloc[0] == pytest.approx(0.0)
        assert row["event_count"].iloc[0] == 0
        assert row["debt_ratio"].iloc[0] == pytest.approx(0.5)

    def test_empty_sentiment_gives_all_defaults(self, market_df: pd.DataFrame) -> None:
        empty_sentiment = pd.DataFrame(columns=list(sentiment_records_df_columns()))
        result = fuse_dataframes(market_df, empty_sentiment)
        assert len(result) == len(market_df)
        for col, fill in FUSION_FILL_VALUES.items():
            assert (result[col] == fill).all(), f"Col {col} not filled with {fill}"

    def test_primary_key_preserved(
        self, market_df: pd.DataFrame, sentiment_records_df: pd.DataFrame
    ) -> None:
        result = fuse_dataframes(market_df, sentiment_records_df)
        # No new (date, tic) pairs should appear
        expected_keys = set(zip(market_df["date"], market_df["tic"]))
        result_keys = set(zip(result["date"], result["tic"]))
        assert result_keys == expected_keys


def sentiment_records_df_columns():
    return [
        "date", "tic", "sentiment_score", "event_count",
        "has_positive_event", "has_negative_event",
    ]


class TestFuseDatasetsFunction:
    def test_saves_parquet(
        self,
        market_df: pd.DataFrame,
        sentiment_records_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        import json

        sentiment_file = tmp_path / "sentiment.jsonl"
        rows = sentiment_records_df.to_dict(orient="records")
        sentiment_file.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )

        out = fuse_datasets(
            market_df=market_df,
            sentiment_file=sentiment_file,
            output_path=tmp_path / "enhanced.parquet",
        )
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == len(market_df)
        for col in FUSION_COLUMNS:
            assert col in df.columns
