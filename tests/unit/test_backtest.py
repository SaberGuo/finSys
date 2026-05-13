"""Unit tests for finquant.training.backtest (T027)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finquant.training.backtest import (
    BacktestReport,
    _trade_rows_html,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_trade_stats,
)


@pytest.fixture()
def flat_returns() -> pd.Series:
    """Daily returns of zero — used for edge-case tests."""
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    return pd.Series(0.0, index=idx, name="portfolio_returns")


@pytest.fixture()
def positive_returns() -> pd.Series:
    """Steadily growing portfolio (+0.1% every day)."""
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    return pd.Series(0.001, index=idx, name="portfolio_returns")


@pytest.fixture()
def negative_drawdown_returns() -> pd.Series:
    """Returns that drop 20% then recover."""
    idx = pd.date_range("2024-01-02", periods=50, freq="B")
    values = [0.0] * 10 + [-0.02] * 10 + [0.0] * 10 + [0.02] * 10 + [0.0] * 10
    return pd.Series(values, index=idx, name="portfolio_returns")


class TestComputeSharpe:
    def test_zero_returns_gives_zero(self, flat_returns: pd.Series) -> None:
        sharpe = compute_sharpe(flat_returns, risk_free_rate=0.0)
        assert sharpe == pytest.approx(0.0, abs=1e-6)

    def test_positive_returns_positive_sharpe(self, positive_returns: pd.Series) -> None:
        sharpe = compute_sharpe(positive_returns, risk_free_rate=0.0)
        assert sharpe > 0.0

    def test_risk_free_adjustment(self, positive_returns: pd.Series) -> None:
        sharpe_no_rf = compute_sharpe(positive_returns, risk_free_rate=0.0)
        sharpe_with_rf = compute_sharpe(positive_returns, risk_free_rate=0.02)
        assert sharpe_no_rf > sharpe_with_rf


class TestComputeCAGR:
    def test_flat_returns_gives_zero(self, flat_returns: pd.Series) -> None:
        cagr = compute_cagr(flat_returns)
        assert cagr == pytest.approx(0.0, abs=1e-6)

    def test_positive_returns_positive_cagr(self, positive_returns: pd.Series) -> None:
        cagr = compute_cagr(positive_returns)
        assert cagr > 0.0

    def test_cagr_reasonable_range(self, positive_returns: pd.Series) -> None:
        # +0.1% daily → annual CAGR ≈ 28%
        cagr = compute_cagr(positive_returns)
        assert 0.10 < cagr < 0.60


class TestComputeMaxDrawdown:
    def test_flat_returns_zero_drawdown(self, flat_returns: pd.Series) -> None:
        mdd = compute_max_drawdown(flat_returns)
        assert mdd == pytest.approx(0.0, abs=1e-6)

    def test_drawdown_negative(self, negative_drawdown_returns: pd.Series) -> None:
        mdd = compute_max_drawdown(negative_drawdown_returns)
        assert mdd < 0.0

    def test_drawdown_bounded(self, negative_drawdown_returns: pd.Series) -> None:
        mdd = compute_max_drawdown(negative_drawdown_returns)
        assert -1.0 <= mdd <= 0.0


class TestBacktestReport:
    def test_from_portfolio_values(self) -> None:
        idx = pd.date_range("2024-01-02", periods=252, freq="B")
        portfolio = pd.Series(
            [1_000_000.0 * (1.001**i) for i in range(252)], index=idx
        )
        report = BacktestReport.from_portfolio_values(
            portfolio, benchmark=None, risk_free_rate=0.02
        )
        assert report.sharpe > 0.0
        assert report.cagr > 0.0
        assert -1.0 <= report.max_drawdown <= 0.0

    def test_to_dict_keys(self) -> None:
        idx = pd.date_range("2024-01-02", periods=50, freq="B")
        portfolio = pd.Series([1_000_000.0] * 50, index=idx)
        report = BacktestReport.from_portfolio_values(portfolio)
        d = report.to_dict()
        assert "sharpe" in d
        assert "cagr" in d
        assert "max_drawdown" in d
        assert "total_return" in d

    def test_save_html(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-02", periods=50, freq="B")
        portfolio = pd.Series(
            [1_000_000.0 * (1.001**i) for i in range(50)], index=idx
        )
        report = BacktestReport.from_portfolio_values(portfolio)
        out = tmp_path / "report.html"
        report.save_html(out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "sharpe" in content.lower() or "chart" in content.lower()

    def test_save_csv(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-02", periods=50, freq="B")
        portfolio = pd.Series(
            [1_000_000.0 * (1.001**i) for i in range(50)], index=idx
        )
        report = BacktestReport.from_portfolio_values(portfolio)
        out = tmp_path / "metrics.csv"
        report.save_csv(out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "sharpe" in df.columns or "metric" in df.columns

    def test_from_portfolio_values_with_benchmark(self) -> None:
        idx = pd.date_range("2024-01-02", periods=50, freq="B")
        portfolio = pd.Series([1_000_000.0 * (1.001**i) for i in range(50)], index=idx)
        benchmark = pd.Series([1_000_000.0] * 50, index=idx)
        report = BacktestReport.from_portfolio_values(portfolio, benchmark=benchmark)
        assert "benchmark_cagr" in report.extra
        assert "benchmark_sharpe" in report.extra

    def test_save_html_plain_fallback(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        portfolio = pd.Series([1_000_000.0] * 10, index=idx)
        report = BacktestReport.from_portfolio_values(portfolio)
        out = tmp_path / "report_plain.html"
        with patch("finquant.training.backtest.BacktestReport._save_html_pyecharts", side_effect=ImportError):
            report.save_html(out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Backtest Report" in content

    def test_to_dict_includes_trade_summary(self) -> None:
        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        portfolio = pd.Series([1_000_000.0] * 10, index=idx)
        report = BacktestReport.from_portfolio_values(portfolio)
        report.trade_summary = {"total_trades": 5, "win_rate": 0.6}
        d = report.to_dict()
        assert d["total_trades"] == 5
        assert d["win_rate"] == pytest.approx(0.6)


class TestComputeTradeStats:
    def test_buy_and_sell(self) -> None:
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        df_actions = pd.DataFrame(
            {
                "000001.SZ": [0.0, 10.0, 0.0],
            },
            index=dates,
        )
        price_df = pd.DataFrame(
            {
                "000001.SZ": [10.0, 11.0, 12.0],
            },
            index=dates,
        )
        trades_df, summary = compute_trade_stats(df_actions, price_df)
        assert len(trades_df) == 2
        assert summary["buy_count"] == 1
        assert summary["sell_count"] == 1
        assert summary["total_trades"] == 2

    def test_empty_actions(self) -> None:
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        df_actions = pd.DataFrame({"000001.SZ": [0.0, 0.0]}, index=dates)
        price_df = pd.DataFrame({"000001.SZ": [10.0, 11.0]}, index=dates)
        trades_df, summary = compute_trade_stats(df_actions, price_df)
        assert trades_df.empty
        assert summary["total_trades"] == 0
        assert summary["win_rate"] == 0.0

    def test_winning_trade(self) -> None:
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        df_actions = pd.DataFrame(
            {"000001.SZ": [0.0, 10.0, 0.0]},
            index=dates,
        )
        price_df = pd.DataFrame(
            {"000001.SZ": [10.0, 10.0, 12.0]},
            index=dates,
        )
        trades_df, summary = compute_trade_stats(df_actions, price_df)
        assert summary["winning_trades"] == 1
        assert summary["losing_trades"] == 0
        assert summary["win_rate"] == pytest.approx(1.0)


class TestTradeRowsHtml:
    def test_with_trades(self) -> None:
        trades_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "tic": "000001.SZ", "action": "BUY", "shares": 10, "price": 10.0, "amount": 100.0}
            ]
        )
        html = _trade_rows_html(trades_df)
        assert "<table" in html
        assert "000001.SZ" in html
        assert "BUY" in html

    def test_empty_trades(self) -> None:
        html = _trade_rows_html(pd.DataFrame())
        assert "No trades recorded" in html


class TestComputeCagrEdgeCases:
    def test_empty_series(self) -> None:
        assert compute_cagr(pd.Series([], dtype=float)) == pytest.approx(0.0)

    def test_zero_years(self) -> None:
        s = pd.Series([0.0], index=pd.date_range("2024-01-02", periods=1))
        assert compute_cagr(s) == pytest.approx(0.0)
