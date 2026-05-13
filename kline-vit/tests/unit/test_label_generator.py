import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_daily_df(n: int, base_price: float = 10.0) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = base_price + np.cumsum(np.random.randn(n) * 0.05)
    prices = np.abs(prices)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": prices,
    })


def test_label_buy_signal():
    from kline_vit.data.label_generator import LabelGenerator
    gen = LabelGenerator(horizon=5, threshold=0.02)
    # Build df where future 5 days close is +5% above anchor
    df = make_daily_df(70)
    df.loc[65:, "close"] = df.loc[64, "close"] * 1.05  # future prices +5%
    anchor_date = df.loc[64, "date"]
    result = gen.generate(df, anchor_date)
    assert result is not None
    label, future_return = result
    assert label == 1
    assert future_return > 0.02


def test_label_hold_signal():
    from kline_vit.data.label_generator import LabelGenerator
    gen = LabelGenerator(horizon=5, threshold=0.02)
    df = make_daily_df(70)
    df.loc[65:, "close"] = df.loc[64, "close"] * 0.99  # future prices -1%
    anchor_date = df.loc[64, "date"]
    result = gen.generate(df, anchor_date)
    assert result is not None
    label, future_return = result
    assert label == 0
    assert future_return < 0.02


def test_label_insufficient_future_data():
    from kline_vit.data.label_generator import LabelGenerator
    gen = LabelGenerator(horizon=5, threshold=0.02)
    df = make_daily_df(65)
    anchor_date = df.loc[63, "date"]  # only 1 future row
    result = gen.generate(df, anchor_date)
    assert result is None


def test_label_configurable_horizon():
    from kline_vit.data.label_generator import LabelGenerator
    gen = LabelGenerator(horizon=10, threshold=0.03)
    assert gen.horizon == 10
    assert gen.threshold == 0.03


def test_label_exact_threshold_is_hold():
    from kline_vit.data.label_generator import LabelGenerator
    gen = LabelGenerator(horizon=5, threshold=0.02)
    df = make_daily_df(70)
    anchor_price = df.loc[64, "close"]
    # Set future prices to exactly at threshold (0.02 return) — should be HOLD (not strictly >)
    # Use a value slightly below threshold to ensure label=0
    df.loc[65:, "close"] = anchor_price * 1.019  # 1.9% < 2% threshold
    anchor_date = df.loc[64, "date"]
    result = gen.generate(df, anchor_date)
    assert result is not None
    label, future_return = result
    assert label == 0  # below threshold → HOLD
