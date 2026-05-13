import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_mock_engine(buy_prob: float = 0.8):
    engine = MagicMock()
    from kline_vit.model.types import InferenceResult
    engine.predict_single.return_value = InferenceResult(
        image_path="dummy.png",
        buy_probability=buy_prob,
        label=1 if buy_prob >= 0.6 else 0,
        inference_time_ms=10.0,
    )
    return engine


def make_config() -> dict:
    return {
        "data": {"image_dir": "data/images"},
        "backtest": {
            "signal_threshold": 0.6,
            "stop_loss_pct": 0.08,
            "max_position_pct": 0.20,
        }
    }


def test_strategy_buy_signal_when_probability_high():
    """Strategy should signal BUY when buy_probability >= threshold and not in position."""
    from kline_vit.backtest.strategy import _compute_signal
    engine = make_mock_engine(buy_prob=0.8)
    signal = _compute_signal(engine, "dummy.png", threshold=0.6, in_position=False)
    assert signal == 1  # BUY


def test_strategy_sell_signal_when_probability_low():
    """Strategy should signal SELL when buy_probability < threshold and in position."""
    from kline_vit.backtest.strategy import _compute_signal
    engine = make_mock_engine(buy_prob=0.3)
    signal = _compute_signal(engine, "dummy.png", threshold=0.6, in_position=True)
    assert signal == -1  # SELL


def test_strategy_hold_when_not_in_position_and_low_prob():
    from kline_vit.backtest.strategy import _compute_signal
    engine = make_mock_engine(buy_prob=0.3)
    signal = _compute_signal(engine, "dummy.png", threshold=0.6, in_position=False)
    assert signal == 0  # HOLD


def test_strategy_no_buy_when_in_position():
    """Strategy must not issue buy when already holding."""
    from kline_vit.backtest.strategy import _compute_signal
    engine = make_mock_engine(buy_prob=0.9)
    signal = _compute_signal(engine, "dummy.png", threshold=0.6, in_position=True)
    assert signal != 1  # no double buy


def test_strategy_hold_when_image_missing():
    """Strategy should return HOLD (0) when image file is missing."""
    from kline_vit.backtest.strategy import _compute_signal
    engine = MagicMock()
    engine.predict_single.side_effect = FileNotFoundError("image not found")
    signal = _compute_signal(engine, "missing.png", threshold=0.6, in_position=False)
    assert signal == 0  # HOLD, no crash
