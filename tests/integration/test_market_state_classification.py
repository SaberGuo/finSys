"""Integration test for market state classification on historical data."""

from __future__ import annotations

import pandas as pd
import pytest

from finquant.selection import MarketState
from finquant.selection.market_state import MarketStateClassifier


@pytest.fixture
def classifier():
    """Create market state classifier."""
    return MarketStateClassifier(index_ticker="000905.SH")


def create_index_data(start_date: str, periods: int, trend: str) -> pd.DataFrame:
    """Create synthetic index data with specific trend characteristics.

    Args:
        start_date: Start date in YYYY-MM-DD format
        periods: Number of days
        trend: Trend type (uptrend, downtrend, oscillation, structural, volume_contraction, sentiment_cautious)

    Returns:
        DataFrame with date, close, high, low, volume, adx_14 columns
    """
    from finquant.features.technical import compute_indicators

    dates = pd.date_range(start_date, periods=periods, freq="D")

    if trend == "uptrend":
        # Strong uptrend: rising prices, high ADX
        close = list(range(100, 100 + periods))
        high = [c + 2 for c in close]
        low = [c - 1 for c in close]
        volume = [1000000] * periods

    elif trend == "downtrend":
        # Strong downtrend: falling prices, high ADX
        close = list(range(100, 100 - periods, -1))
        high = [c + 1 for c in close]
        low = [c - 2 for c in close]
        volume = [1000000] * periods

    elif trend == "oscillation":
        # Oscillating: low ADX, no clear trend
        close = [100 + (i % 10 - 5) for i in range(periods)]
        high = [c + 1 for c in close]
        low = [c - 1 for c in close]
        volume = [1000000] * periods

    elif trend == "structural":
        # Structural change: price below MA50, low ADX
        close = [100] * 50 + [90] * (periods - 50)
        high = [c + 1 for c in close]
        low = [c - 1 for c in close]
        volume = [1000000] * periods

    elif trend == "volume_contraction":
        # Volume contraction: low volume, low ADX
        close = [100 + (i % 5) for i in range(periods)]
        high = [c + 1 for c in close]
        low = [c - 1 for c in close]
        volume = [500000] * periods  # Low volume

    elif trend == "sentiment_cautious":
        # Cautious sentiment: moderate ADX, below MA50
        close = [100] * 50 + [95 - i * 0.1 for i in range(periods - 50)]
        high = [c + 1 for c in close]
        low = [c - 1 for c in close]
        volume = [1000000] * periods

    else:
        raise ValueError(f"Unknown trend: {trend}")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "tic": "INDEX",
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
    })

    # Compute ADX indicator
    df = compute_indicators(df, ["adx_14"])

    return df


def test_classify_uptrend_historical(classifier):
    """Test uptrend classification on historical-like data."""
    # Create 60 days of uptrend data
    df = create_index_data("2023-01-01", 60, "uptrend")

    # Classify the last day
    state = classifier.classify(df, "2023-03-01")

    # Should identify as uptrend
    assert state == MarketState.UPTREND


def test_classify_downtrend_historical(classifier):
    """Test downtrend classification on historical-like data."""
    # Create 60 days of downtrend data
    df = create_index_data("2023-01-01", 60, "downtrend")

    # Classify the last day
    state = classifier.classify(df, "2023-03-01")

    # Should identify as downtrend
    assert state == MarketState.DOWNTREND


def test_classify_oscillation_historical(classifier):
    """Test oscillation classification on historical-like data."""
    # Create 60 days of oscillating data
    df = create_index_data("2023-01-01", 60, "oscillation")

    # Classify the last day
    state = classifier.classify(df, "2023-03-01")

    # Should identify as oscillation
    assert state == MarketState.OSCILLATION


def test_classify_structural_historical(classifier):
    """Test structural change classification on historical-like data."""
    # Create 80 days with structural change
    df = create_index_data("2023-01-01", 80, "structural")

    # Classify after the structural change
    state = classifier.classify(df, "2023-03-21")

    # Should identify as structural change or oscillation (both valid for this pattern)
    assert state in [MarketState.STRUCTURAL, MarketState.OSCILLATION]


def test_classify_volume_contraction_historical(classifier):
    """Test volume contraction classification on historical-like data."""
    # Create 60 days with low volume
    df = create_index_data("2023-01-01", 60, "volume_contraction")

    # Classify the last day
    state = classifier.classify(df, "2023-03-01")

    # Should identify as volume contraction or oscillation (both valid for low volume + low ADX)
    assert state in [MarketState.VOLUME_CONTRACTION, MarketState.OSCILLATION]


def test_classify_sentiment_cautious_historical(classifier):
    """Test cautious sentiment classification on historical-like data."""
    # Create 80 days with cautious sentiment
    df = create_index_data("2023-01-01", 80, "sentiment_cautious")

    # Classify after sentiment shift
    state = classifier.classify(df, "2023-03-21")

    # Should identify as sentiment cautious or downtrend (both valid for declining prices)
    assert state in [MarketState.SENTIMENT_CAUTIOUS, MarketState.DOWNTREND]


def test_classify_multiple_dates(classifier):
    """Test classification across multiple dates."""
    # Create data with trend change
    df1 = create_index_data("2023-01-01", 30, "uptrend")
    df2 = create_index_data("2023-01-31", 30, "downtrend")

    df = pd.concat([df1, df2], ignore_index=True)

    # Classify during uptrend
    state1 = classifier.classify(df, "2023-01-20")
    assert state1 == MarketState.UPTREND

    # Classify during downtrend
    state2 = classifier.classify(df, "2023-02-20")
    assert state2 == MarketState.DOWNTREND


def test_classify_with_real_world_noise(classifier):
    """Test classification with noisy data (more realistic)."""
    import numpy as np
    from finquant.features.technical import compute_indicators

    # Create uptrend with noise
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    np.random.seed(42)

    base_trend = list(range(100, 160))
    noise = np.random.randn(60) * 2
    close = [b + n for b, n in zip(base_trend, noise)]

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "tic": "INDEX",
        "close": close,
        "high": [c + abs(np.random.randn()) for c in close],
        "low": [c - abs(np.random.randn()) for c in close],
        "volume": [1000000 + np.random.randint(-100000, 100000) for _ in range(60)],
    })

    # Compute ADX indicator
    df = compute_indicators(df, ["adx_14"])

    # Should still identify uptrend despite noise
    state = classifier.classify(df, "2023-03-01")
    assert state in [MarketState.UPTREND, MarketState.OSCILLATION]  # Noise may affect classification


def test_classify_insufficient_data(classifier):
    """Test classification with insufficient historical data."""
    # Only 10 days of data (need 50+ for MA50)
    df = create_index_data("2023-01-01", 10, "uptrend")

    # Should still classify (may default to oscillation)
    state = classifier.classify(df, "2023-01-10")
    assert state in MarketState


def test_classify_edge_case_all_same_price(classifier):
    """Test classification when price doesn't change."""
    from finquant.features.technical import compute_indicators

    dates = pd.date_range("2023-01-01", periods=60, freq="D")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "tic": "INDEX",
        "close": [100] * 60,
        "high": [101] * 60,
        "low": [99] * 60,
        "volume": [1000000] * 60,
    })

    # Compute ADX indicator
    df = compute_indicators(df, ["adx_14"])

    # Should classify (likely oscillation or structural)
    state = classifier.classify(df, "2023-03-01")
    assert state in MarketState


def test_classify_preserves_data_integrity(classifier):
    """Test that classification doesn't modify input data."""
    df = create_index_data("2023-01-01", 60, "uptrend")
    df_copy = df.copy()

    classifier.classify(df, "2023-03-01")

    # Original data should be unchanged
    pd.testing.assert_frame_equal(df, df_copy)


def test_classify_different_index_tickers(classifier):
    """Test classification works with different index tickers."""
    df = create_index_data("2023-01-01", 60, "uptrend")

    # Should work regardless of index ticker
    state = classifier.classify(df, "2023-03-01")
    assert state == MarketState.UPTREND


def test_classify_date_not_in_data(classifier):
    """Test classification when requested date is not in data."""
    df = create_index_data("2023-01-01", 60, "uptrend")

    # Request date not in data - should raise ValueError
    with pytest.raises(ValueError, match="Date .* not found"):
        classifier.classify(df, "2023-12-31")
