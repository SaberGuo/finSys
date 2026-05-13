"""Unit tests for market state classification."""

from __future__ import annotations

import pandas as pd
import pytest

from finquant.selection import MarketState
from finquant.selection.market_state import MarketStateClassifier, MarketStateRule


class TestMarketStateRule:
    """Test MarketStateRule dataclass."""

    def test_rule_creation(self):
        """Test creating a market state rule."""
        rule = MarketStateRule(
            state=MarketState.UPTREND,
            condition=lambda df: df["adx_14"] > 25,
            priority=10,
        )
        assert rule.state == MarketState.UPTREND
        assert rule.priority == 10


class TestMarketStateClassifier:
    """Test MarketStateClassifier."""

    def test_classifier_initialization(self):
        """Test classifier initialization with defaults."""
        classifier = MarketStateClassifier()
        assert classifier.index_ticker == "000905.SH"
        assert classifier.auto_optimize is False
        assert len(classifier.rules) > 0

    def test_classify_uptrend(self):
        """Test classification of uptrend market."""
        classifier = MarketStateClassifier()

        # Create index data with strong uptrend signals
        df = pd.DataFrame(
            {
                "date": ["2023-07-01"] * 60,
                "close": list(range(100, 160)),
                "adx_14": [30.0] * 60,
                "volume": [1000000] * 60,
            }
        )
        df["date"] = pd.date_range("2023-06-01", periods=60, freq="D").strftime("%Y-%m-%d")
        df["close"] = list(range(100, 160))

        state = classifier.classify(df, "2023-07-30")
        assert state == MarketState.UPTREND

    def test_classify_downtrend(self):
        """Test classification of downtrend market."""
        classifier = MarketStateClassifier()

        # Create index data with downtrend signals
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-06-01", periods=60, freq="D").strftime("%Y-%m-%d"),
                "close": list(range(160, 100, -1)),
                "adx_14": [30.0] * 60,
                "volume": [1000000] * 60,
            }
        )

        state = classifier.classify(df, "2023-07-30")
        assert state == MarketState.DOWNTREND

    def test_classify_oscillation(self):
        """Test classification of oscillating market."""
        classifier = MarketStateClassifier()

        # Create index data with low ADX (weak trend)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-06-01", periods=60, freq="D").strftime("%Y-%m-%d"),
                "close": [100 + 5 * (i % 10) for i in range(60)],
                "adx_14": [15.0] * 60,
                "volume": [1000000] * 60,
            }
        )

        state = classifier.classify(df, "2023-07-30")
        assert state == MarketState.OSCILLATION

    def test_classify_volume_contraction(self):
        """Test classification of volume contraction."""
        classifier = MarketStateClassifier()

        # Create index data with declining volume
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-06-01", periods=60, freq="D").strftime("%Y-%m-%d"),
                "close": list(range(100, 160)),
                "adx_14": [20.0] * 60,
                "volume": [1000000 - i * 10000 for i in range(60)],
            }
        )

        state = classifier.classify(df, "2023-07-30")
        # Should detect volume contraction
        assert state in (MarketState.VOLUME_CONTRACTION, MarketState.OSCILLATION)

    def test_classify_missing_date(self):
        """Test classification with missing date raises error."""
        classifier = MarketStateClassifier()

        df = pd.DataFrame(
            {
                "date": ["2023-07-01"],
                "close": [100.0],
                "adx_14": [25.0],
                "volume": [1000000],
            }
        )

        with pytest.raises(ValueError, match="Date .* not found"):
            classifier.classify(df, "2023-08-01")

    def test_classify_missing_columns(self):
        """Test classification with missing required columns raises error."""
        classifier = MarketStateClassifier()

        df = pd.DataFrame(
            {
                "date": ["2023-07-01"],
                "close": [100.0],
                # Missing adx_14
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            classifier.classify(df, "2023-07-01")

    def test_classify_series(self):
        """Test classifying multiple dates."""
        classifier = MarketStateClassifier()

        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-06-01", periods=60, freq="D").strftime("%Y-%m-%d"),
                "close": list(range(100, 160)),
                "adx_14": [30.0] * 60,
                "volume": [1000000] * 60,
            }
        )

        dates = ["2023-07-01", "2023-07-15", "2023-07-30"]
        results = classifier.classify_series(df, dates)

        assert len(results) == 3
        assert all(isinstance(state, MarketState) for state in results.values())

    def test_get_index_metrics(self):
        """Test getting index metrics snapshot."""
        classifier = MarketStateClassifier()

        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-06-01", periods=60, freq="D").strftime("%Y-%m-%d"),
                "close": list(range(100, 160)),
                "adx_14": [30.0] * 60,
                "volume": [1000000] * 60,
            }
        )

        metrics = classifier.get_index_metrics(df, "2023-07-30")

        assert "close" in metrics
        assert "adx" in metrics
        assert "ma50" in metrics
        assert "ma50_ratio" in metrics
        assert metrics["adx"] == 30.0


class TestMarketStateClassifierEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_rule_priority_ordering(self):
        """Test that rules are checked in priority order."""
        classifier = MarketStateClassifier()

        # Rules should be sorted by priority (descending)
        priorities = [rule.priority for rule in classifier.rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_fuzzy_classification_boundary(self):
        """Test classification at boundary conditions."""
        classifier = MarketStateClassifier()

        # ADX exactly at 25 threshold
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-06-01", periods=60, freq="D").strftime("%Y-%m-%d"),
                "close": list(range(100, 160)),
                "adx_14": [25.0] * 60,
                "volume": [1000000] * 60,
            }
        )

        # Should still classify (not raise error)
        state = classifier.classify(df, "2023-07-30")
        assert isinstance(state, MarketState)

    def test_ma50_computation_with_insufficient_data(self):
        """Test MA50 computation with less than 50 data points."""
        classifier = MarketStateClassifier()

        # Only 30 days of data
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-07-01", periods=30, freq="D").strftime("%Y-%m-%d"),
                "close": list(range(100, 130)),
                "adx_14": [30.0] * 30,
                "volume": [1000000] * 30,
            }
        )

        # Should still work with min_periods=1
        state = classifier.classify(df, "2023-07-30")
        assert isinstance(state, MarketState)

    def test_default_fallback_when_no_rules_match(self):
        """Test default fallback to OSCILLATION when no rules match."""
        classifier = MarketStateClassifier()

        # Create data that doesn't match any strong signals
        df = pd.DataFrame(
            {
                "date": ["2023-07-30"],
                "close": [100.0],
                "adx_14": [18.0],  # Low ADX
                "volume": [1000000],
            }
        )

        state = classifier.classify(df, "2023-07-30")
        assert state == MarketState.OSCILLATION
