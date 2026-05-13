import pytest
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_synthetic_db(tmp_path: Path, n_days: int = 60) -> str:
    db_path = tmp_path / "test_bt.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE daily_data (
            code TEXT, date TEXT, open REAL, high REAL,
            low REAL, close REAL, volume INTEGER, amount REAL
        )
    """)
    dates = pd.date_range("2025-01-02", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()
    price = 10.0
    rows = []
    for d in dates:
        price = max(1.0, price + np.random.randn() * 0.1)
        rows.append(("hk.00001", d, price, price * 1.01, price * 0.99, price, 100000, price * 100000))
    conn.executemany("INSERT INTO daily_data VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(db_path)


def make_mock_engine(buy_prob: float = 0.8):
    from kline_vit.model.types import InferenceResult
    engine = MagicMock()
    engine.predict_single.return_value = InferenceResult(
        image_path="dummy.png", buy_probability=buy_prob,
        label=1 if buy_prob >= 0.6 else 0, inference_time_ms=5.0
    )
    return engine


def test_backtest_runs_without_error(tmp_path):
    from kline_vit.backtest.runner import BacktestRunner
    db_path = make_synthetic_db(tmp_path)
    config = {
        "data": {"db_path": db_path, "image_dir": str(tmp_path / "images")},
        "backtest": {
            "initial_cash": 100000,
            "commission": 0.001,
            "signal_threshold": 0.6,
            "max_position_pct": 0.20,
            "stop_loss_pct": 0.08,
            "start_date": "2025-01-02",
            "end_date": "2025-04-30",
        }
    }
    engine = make_mock_engine(buy_prob=0.8)
    runner = BacktestRunner(config)
    report = runner.run(engine, db_path)
    assert report is not None


def test_backtest_report_fields_are_numeric(tmp_path):
    from kline_vit.backtest.runner import BacktestRunner
    db_path = make_synthetic_db(tmp_path)
    config = {
        "data": {"db_path": db_path, "image_dir": str(tmp_path / "images")},
        "backtest": {
            "initial_cash": 100000,
            "commission": 0.001,
            "signal_threshold": 0.6,
            "max_position_pct": 0.20,
            "stop_loss_pct": 0.08,
            "start_date": "2025-01-02",
            "end_date": "2025-04-30",
        }
    }
    engine = make_mock_engine(buy_prob=0.8)
    runner = BacktestRunner(config)
    report = runner.run(engine, db_path)
    assert isinstance(report.annual_return, float)
    assert isinstance(report.max_drawdown, float)
    assert isinstance(report.sharpe_ratio, float)
    assert isinstance(report.total_trades, int)
    assert report.max_drawdown >= 0
