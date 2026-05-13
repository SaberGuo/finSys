"""Unit tests for ADX and ATR technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finquant.features.technical import _adx, _atr, compute_indicators


class TestADX:
    """Test Average Directional Index (ADX) calculation."""

    def test_adx_basic_calculation(self):
        """Test ADX calculation with simple uptrend data."""
        # Create simple uptrend: prices increasing steadily
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        high = pd.Series(range(100, 130))
        low = pd.Series(range(98, 128))
        close = pd.Series(range(99, 129))

        adx = _adx(high, low, close, period=14)

        # ADX should be non-negative
        assert (adx >= 0).all()
        # ADX should increase in strong trend
        assert adx.iloc[-1] > adx.iloc[14]

    def test_adx_oscillation(self):
        """Test ADX with oscillating prices (weak trend)."""
        # Create oscillating pattern
        high = pd.Series([100 + 5 * np.sin(i * 0.5) for i in range(30)])
        low = pd.Series([98 + 5 * np.sin(i * 0.5) for i in range(30)])
        close = pd.Series([99 + 5 * np.sin(i * 0.5) for i in range(30)])

        adx = _adx(high, low, close, period=14)

        # ADX should be low in oscillation (< 20)
        assert adx.iloc[-1] < 25

    def test_adx_zero_range(self):
        """Test ADX with zero price range (edge case)."""
        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)
        close = pd.Series([100.0] * 30)

        adx = _adx(high, low, close, period=14)

        # ADX should be 0 when no movement
        assert (adx == 0).all()


class TestATR:
    """Test Average True Range (ATR) calculation."""

    def test_atr_basic_calculation(self):
        """Test ATR calculation with varying volatility."""
        high = pd.Series([105, 110, 108, 115, 112, 120, 118, 125, 122, 130] * 3)
        low = pd.Series([95, 100, 98, 105, 102, 110, 108, 115, 112, 120] * 3)
        close = pd.Series([100, 105, 103, 110, 107, 115, 113, 120, 117, 125] * 3)

        atr = _atr(high, low, close, period=14)

        # ATR should be non-negative
        assert (atr >= 0).all()
        # ATR should reflect volatility (roughly 10 in this case)
        assert 8 < atr.iloc[-1] < 12

    def test_atr_increasing_volatility(self):
        """Test ATR increases with increasing volatility."""
        # Low volatility period
        high1 = pd.Series([101, 102, 103, 104, 105] * 3)
        low1 = pd.Series([99, 100, 101, 102, 103] * 3)
        close1 = pd.Series([100, 101, 102, 103, 104] * 3)

        # High volatility period
        high2 = pd.Series([110, 120, 115, 130, 125] * 3)
        low2 = pd.Series([90, 100, 95, 110, 105] * 3)
        close2 = pd.Series([100, 110, 105, 120, 115] * 3)

        atr1 = _atr(high1, low1, close1, period=14)
        atr2 = _atr(high2, low2, close2, period=14)

        # High volatility should have higher ATR
        assert atr2.iloc[-1] > atr1.iloc[-1] * 2

    def test_atr_zero_range(self):
        """Test ATR with zero price range."""
        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)
        close = pd.Series([100.0] * 30)

        atr = _atr(high, low, close, period=14)

        # ATR should be 0 when no movement
        assert (atr == 0).all()


class TestComputeIndicatorsADXATR:
    """Test ADX and ATR integration in compute_indicators."""

    def test_compute_adx_atr_single_stock(self):
        """Test computing ADX and ATR for single stock."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=30, freq="D").strftime("%Y-%m-%d"),
                "tic": ["000001.SZ"] * 30,
                "open": range(100, 130),
                "high": range(102, 132),
                "low": range(98, 128),
                "close": range(100, 130),
                "volume": [1000000] * 30,
            }
        )

        result = compute_indicators(df, indicators=["adx_14", "atr_14"])

        assert "adx_14" in result.columns
        assert "atr_14" in result.columns
        assert len(result) == 30
        assert (result["adx_14"] >= 0).all()
        assert (result["atr_14"] >= 0).all()

    def test_compute_adx_atr_multiple_stocks(self):
        """Test computing ADX and ATR for multiple stocks."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D").strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "tic": ["000001.SZ"] * 20 + ["000002.SZ"] * 20,
                "open": list(range(100, 120)) + list(range(200, 220)),
                "high": list(range(102, 122)) + list(range(202, 222)),
                "low": list(range(98, 118)) + list(range(198, 218)),
                "close": list(range(100, 120)) + list(range(200, 220)),
                "volume": [1000000] * 40,
            }
        )

        result = compute_indicators(df, indicators=["adx_14", "atr_14"])

        assert "adx_14" in result.columns
        assert "atr_14" in result.columns
        assert len(result) == 40

        # Check per-stock calculation
        stock1 = result[result["tic"] == "000001.SZ"]
        stock2 = result[result["tic"] == "000002.SZ"]
        assert len(stock1) == 20
        assert len(stock2) == 20
        assert (stock1["adx_14"] >= 0).all()
        assert (stock2["atr_14"] >= 0).all()

    def test_compute_missing_high_low_columns(self):
        """Test ADX/ATR computation when high/low columns missing (fallback to close)."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=20, freq="D").strftime("%Y-%m-%d"),
                "tic": ["000001.SZ"] * 20,
                "close": range(100, 120),
                "volume": [1000000] * 20,
            }
        )

        result = compute_indicators(df, indicators=["adx_14", "atr_14"])

        # Should use close as fallback for high/low
        assert "adx_14" in result.columns
        assert "atr_14" in result.columns
        # With no range (high=low=close), ADX and ATR should be 0
        assert (result["adx_14"] == 0).all()
        assert (result["atr_14"] == 0).all()
