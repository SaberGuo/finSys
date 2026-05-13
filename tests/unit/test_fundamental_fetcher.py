"""Unit tests for finquant.data.fundamental_fetcher."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finquant.data.fundamental_fetcher import (
    align_to_daily,
    compute_fundamental_metrics,
    fetch_financial_reports,
)


class TestFetchFinancialReports:
    @patch("akshare.stock_financial_report_sina")
    def test_profit_and_balance(self, mock_sina: MagicMock) -> None:
        mock_sina.side_effect = [
            # Profit statement
            pd.DataFrame(
                [
                    {
                        "报告日": "20241231",
                        "营业总收入": 1_000_000_000.0,
                        "净利润": 100_000_000.0,
                        "归属于母公司所有者的净利润": 80_000_000.0,
                    },
                    {
                        "报告日": "20231231",
                        "营业总收入": 800_000_000.0,
                        "净利润": 80_000_000.0,
                        "归属于母公司所有者的净利润": 70_000_000.0,
                    },
                ]
            ),
            # Balance sheet
            pd.DataFrame(
                [
                    {
                        "报告日": "20241231",
                        "资产总计": 2_000_000_000.0,
                        "负债合计": 1_000_000_000.0,
                        "所有者权益(或股东权益)合计": 1_000_000_000.0,
                    },
                    {
                        "报告日": "20231231",
                        "资产总计": 1_800_000_000.0,
                        "负债合计": 900_000_000.0,
                        "所有者权益(或股东权益)合计": 900_000_000.0,
                    },
                ]
            ),
        ]
        profit_df, balance_df = fetch_financial_reports("000009.SZ")
        assert len(profit_df) == 2
        assert len(balance_df) == 2
        assert profit_df.loc[pd.to_datetime("2024-12-31"), "total_revenue"] == 1_000_000_000.0

    @patch("akshare.stock_financial_report_sina")
    def test_empty_response(self, mock_sina: MagicMock) -> None:
        mock_sina.return_value = pd.DataFrame()
        profit_df, balance_df = fetch_financial_reports("000009.SZ")
        assert profit_df.empty
        assert balance_df.empty

    @patch("akshare.stock_financial_report_sina")
    def test_api_error(self, mock_sina: MagicMock) -> None:
        mock_sina.side_effect = RuntimeError("fail")
        profit_df, balance_df = fetch_financial_reports("000009.SZ")
        assert profit_df.empty
        assert balance_df.empty


class TestComputeFundamentalMetrics:
    def test_basic_computation(self) -> None:
        profit = pd.DataFrame(
            {
                "total_revenue": [1_000_000_000.0, 800_000_000.0],
                "net_profit": [100_000_000.0, 80_000_000.0],
                "net_profit_parent": [80_000_000.0, 70_000_000.0],
            },
            index=pd.to_datetime(["2024-12-31", "2023-12-31"]),
        )
        balance = pd.DataFrame(
            {
                "total_assets": [2_000_000_000.0, 1_800_000_000.0],
                "total_liabilities": [1_000_000_000.0, 900_000_000.0],
                "total_equity": [1_000_000_000.0, 900_000_000.0],
            },
            index=pd.to_datetime(["2024-12-31", "2023-12-31"]),
        )
        metrics = compute_fundamental_metrics(profit, balance)
        assert len(metrics) == 2
        # Revenue growth for 2024: (1B - 0.8B) / 0.8B * 100 = 25%
        assert metrics.loc["2024-12-31", "revenue_growth_pct"] == pytest.approx(25.0, abs=0.1)
        # NPM for 2024: 100M / 1B * 100 = 10%
        assert metrics.loc["2024-12-31", "net_profit_margin"] == pytest.approx(10.0, abs=0.1)
        # Debt ratio for 2024: 1B / 2B = 0.5
        assert metrics.loc["2024-12-31", "debt_ratio"] == pytest.approx(0.5, abs=0.01)
        # ROE for 2024: 100M / 1B * 100 = 10%
        assert metrics.loc["2024-12-31", "roe"] == pytest.approx(10.0, abs=0.1)

    def test_empty_input(self) -> None:
        empty = pd.DataFrame()
        metrics = compute_fundamental_metrics(empty, empty)
        assert metrics.empty


class TestAlignToDaily:
    def test_forward_fill(self) -> None:
        metrics = pd.DataFrame(
            {
                "revenue_growth_pct": [25.0],
                "net_profit_margin": [10.0],
                "debt_ratio": [0.5],
                "roe": [10.0],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )
        trading_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        daily = align_to_daily(metrics, trading_dates, "000009.SZ")
        assert len(daily) == 3
        assert daily.loc[0, "revenue_growth_pct"] == pytest.approx(25.0)
        assert daily.loc[2, "revenue_growth_pct"] == pytest.approx(25.0)  # ffill

    def test_empty_metrics(self) -> None:
        empty = pd.DataFrame()
        trading_dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        daily = align_to_daily(empty, trading_dates, "000009.SZ")
        assert len(daily) == 2
        assert daily.loc[0, "debt_ratio"] == pytest.approx(0.5)  # default
