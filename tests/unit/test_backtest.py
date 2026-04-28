"""Unit tests for finquant.training.backtest (T027)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finquant.training.backtest import (
    BacktestReport,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
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
