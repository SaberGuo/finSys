import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_backtest_report_has_required_fields():
    from kline_vit.backtest.runner import BacktestReport
    report = BacktestReport(
        annual_return=0.15,
        max_drawdown=0.10,
        sharpe_ratio=1.5,
        win_rate=0.55,
        profit_factor=1.8,
        total_trades=42,
        benchmark_return=0.08,
        excess_return=0.07,
    )
    assert hasattr(report, "annual_return")
    assert hasattr(report, "max_drawdown")
    assert hasattr(report, "sharpe_ratio")
    assert hasattr(report, "win_rate")
    assert hasattr(report, "profit_factor")
    assert hasattr(report, "total_trades")
    assert hasattr(report, "benchmark_return")
    assert hasattr(report, "excess_return")


def test_backtest_report_numeric_fields():
    from kline_vit.backtest.runner import BacktestReport
    report = BacktestReport(
        annual_return=0.15,
        max_drawdown=0.10,
        sharpe_ratio=1.5,
        win_rate=0.55,
        profit_factor=1.8,
        total_trades=42,
        benchmark_return=0.08,
        excess_return=0.07,
    )
    assert isinstance(report.annual_return, float)
    assert isinstance(report.max_drawdown, float)
    assert isinstance(report.sharpe_ratio, float)
    assert isinstance(report.total_trades, int)
