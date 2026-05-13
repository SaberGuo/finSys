import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_ohlcv(n: int, start_price: float = 10.0) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = start_price + np.cumsum(np.random.randn(n) * 0.1)
    prices = np.abs(prices)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n),
        "amount": prices * np.random.randint(100000, 1000000, n),
    })


def test_renderer_output_is_png(tmp_path):
    from kline_vit.data.renderer import KlineRenderer
    renderer = KlineRenderer()
    daily_df = make_ohlcv(60)
    weekly_df = make_ohlcv(13)
    out_path = renderer.render("hk.00001", "2023-06-01", daily_df, weekly_df, output_dir=str(tmp_path))
    assert Path(out_path).exists()
    assert Path(out_path).suffix == ".png"


def test_renderer_output_dimensions(tmp_path):
    from kline_vit.data.renderer import KlineRenderer
    from PIL import Image
    renderer = KlineRenderer()
    daily_df = make_ohlcv(60)
    weekly_df = make_ohlcv(13)
    out_path = renderer.render("hk.00001", "2023-06-01", daily_df, weekly_df, output_dir=str(tmp_path))
    img = Image.open(out_path)
    w, h = img.size
    # 448x224 or close (mplfinance may vary slightly)
    assert w >= 224
    assert h >= 112


def test_renderer_dual_panel(tmp_path):
    """Image width should be roughly 2x height (dual panel)."""
    from kline_vit.data.renderer import KlineRenderer
    from PIL import Image
    renderer = KlineRenderer()
    daily_df = make_ohlcv(60)
    weekly_df = make_ohlcv(13)
    out_path = renderer.render("hk.00001", "2023-06-01", daily_df, weekly_df, output_dir=str(tmp_path))
    img = Image.open(out_path)
    w, h = img.size
    assert w > h  # wider than tall (dual panel)


def test_renderer_creates_subdirectory(tmp_path):
    from kline_vit.data.renderer import KlineRenderer
    renderer = KlineRenderer()
    daily_df = make_ohlcv(60)
    weekly_df = make_ohlcv(13)
    out_path = renderer.render("hk.00001", "2023-06-01", daily_df, weekly_df, output_dir=str(tmp_path))
    assert Path(out_path).parent.exists()
