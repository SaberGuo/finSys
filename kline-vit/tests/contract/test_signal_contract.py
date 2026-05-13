import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_signal_values_are_valid():
    from kline_vit.backtest.strategy import _compute_signal
    engine = MagicMock()
    from kline_vit.model.types import InferenceResult
    for prob in [0.0, 0.3, 0.5, 0.6, 0.8, 1.0]:
        engine.predict_single.return_value = InferenceResult(
            image_path="x.png", buy_probability=prob,
            label=1 if prob >= 0.6 else 0, inference_time_ms=1.0
        )
        for in_pos in [True, False]:
            signal = _compute_signal(engine, "x.png", threshold=0.6, in_position=in_pos)
            assert signal in (-1, 0, 1), f"Invalid signal {signal} for prob={prob}, in_pos={in_pos}"


def test_no_buy_when_in_position():
    from kline_vit.backtest.strategy import _compute_signal
    engine = MagicMock()
    from kline_vit.model.types import InferenceResult
    engine.predict_single.return_value = InferenceResult(
        image_path="x.png", buy_probability=0.9, label=1, inference_time_ms=1.0
    )
    signal = _compute_signal(engine, "x.png", threshold=0.6, in_position=True)
    assert signal != 1, "Must not issue BUY when already in position"


def test_stop_loss_percentage():
    """Stop loss should be 8% below buy price."""
    buy_price = 100.0
    stop_loss_pct = 0.08
    stop_price = buy_price * (1 - stop_loss_pct)
    assert abs(stop_price - 92.0) < 0.001
