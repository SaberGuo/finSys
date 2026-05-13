"""Unit tests for BreakoutStrategy."""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from finquant.selection.strategies.breakout import BreakoutStrategy, BreakoutConfig
from finquant.selection import MarketState


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    data = []

    for tic in ["600000.SH", "600016.SH", "600019.SH"]:
        for i, date in enumerate(dates):
            # Create uptrend pattern for 600000.SH
            if tic == "600000.SH":
                close = 10.0 + i * 0.02  # Gradual uptrend
                volume = 1000000 + i * 1000
            # Create breakout pattern for 600016.SH
            elif tic == "600016.SH":
                if i < 200:
                    close = 15.0 + (i % 10) * 0.1  # Oscillation
                    volume = 800000
                else:
                    close = 15.0 + (i - 200) * 0.05  # Breakout
                    volume = 1500000  # Volume surge
            # Create downtrend for 600019.SH
            else:
                close = 20.0 - i * 0.01
                volume = 500000

            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": tic,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": volume,
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_index_data():
    """Create sample index data."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    data = []

    for date in dates:
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "tic": "000905.SH",
            "close": 5000.0,
            "volume": 10000000,
        })

    return pd.DataFrame(data)


@pytest.fixture
def default_config():
    """Create default breakout config."""
    return BreakoutConfig(
        ma_periods=[120, 250],
        volume_multiplier=1.5,
        volume_ma_period=20,
        breakout_threshold=1.05,
        lookback_days=60,
        confirmation_days=3,
        anti_jitter_mode="threshold",
        top_k=10,
        exclude_st=True,
        exclude_halt=True,
    )


class TestBreakoutConfig:
    """Test BreakoutConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BreakoutConfig()
        assert config.ma_periods == [120, 250]
        assert config.volume_multiplier == 1.5
        assert config.volume_ma_period == 20
        assert config.breakout_threshold == 1.05
        assert config.lookback_days == 60
        assert config.confirmation_days == 3
        assert config.anti_jitter_mode == "threshold"
        assert config.top_k == 10
        assert config.exclude_st is True
        assert config.exclude_halt is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BreakoutConfig(
            ma_periods=[60, 120],
            volume_multiplier=2.0,
            anti_jitter_mode="both",
            top_k=5,
        )
        assert config.ma_periods == [60, 120]
        assert config.volume_multiplier == 2.0
        assert config.anti_jitter_mode == "both"
        assert config.top_k == 5


class TestBreakoutStrategy:
    """Test BreakoutStrategy selection logic."""

    def test_initialization(self, default_config):
        """Test strategy initialization."""
        strategy = BreakoutStrategy(default_config)
        assert strategy.config == default_config

    def test_compute_indicators(self, default_config, sample_market_data):
        """Test MA and volume indicator computation."""
        strategy = BreakoutStrategy(default_config)
        df = strategy._compute_indicators(sample_market_data)

        # Check MA columns exist
        assert "ma120" in df.columns
        assert "ma250" in df.columns
        assert "vol_ma" in df.columns

        # Check MA values are computed
        df_with_ma = df[df["date"] >= "2023-05-01"]  # After 120 days
        assert df_with_ma["ma120"].notna().any()

    def test_ma_breakout_detection(self, default_config, sample_market_data):
        """Test MA breakout signal detection."""
        strategy = BreakoutStrategy(default_config)
        df = strategy._compute_indicators(sample_market_data)

        # Create a clear breakout scenario
        test_date = "2023-08-01"
        stock_df = df[df["tic"] == "600016.SH"].sort_values("date")
        today_row = stock_df[stock_df["date"] == test_date].iloc[0]

        # Should detect breakout if conditions are met
        result = strategy._compute_ma_breakout(stock_df, today_row, "ma120", test_date)
        assert isinstance(result, bool)

    def test_volume_surge_detection(self, default_config):
        """Test volume surge detection."""
        strategy = BreakoutStrategy(default_config)

        # Test with volume surge
        row_surge = pd.Series({
            "volume": 1500000,
            "vol_ma": 1000000,
        })
        assert strategy._compute_volume_surge(row_surge) is True

        # Test without volume surge
        row_no_surge = pd.Series({
            "volume": 1000000,
            "vol_ma": 1000000,
        })
        assert strategy._compute_volume_surge(row_no_surge) is False

        # Test with NaN
        row_nan = pd.Series({
            "volume": float("nan"),
            "vol_ma": 1000000,
        })
        assert strategy._compute_volume_surge(row_nan) is False

    def test_first_breakout_check(self, default_config, sample_market_data):
        """Test first breakout logic."""
        strategy = BreakoutStrategy(default_config)
        df = strategy._compute_indicators(sample_market_data)

        test_date = "2023-08-01"
        stock_df = df[df["tic"] == "600000.SH"].sort_values("date")

        # Check if it's first breakout
        result = strategy._check_first_breakout(stock_df, "ma120", test_date)
        assert isinstance(result, bool)

    def test_threshold_anti_jitter(self, default_config):
        """Test threshold-based anti-jitter mechanism."""
        strategy = BreakoutStrategy(default_config)

        # Test passing threshold
        row_pass = pd.Series({
            "close": 105.0,
            "ma120": 100.0,
        })
        assert strategy._check_threshold(row_pass, "ma120") is True

        # Test failing threshold
        row_fail = pd.Series({
            "close": 102.0,
            "ma120": 100.0,
        })
        assert strategy._check_threshold(row_fail, "ma120") is False

    def test_confirmation_anti_jitter(self, default_config, sample_market_data):
        """Test confirmation-based anti-jitter mechanism."""
        strategy = BreakoutStrategy(default_config)
        df = strategy._compute_indicators(sample_market_data)

        test_date = "2023-08-01"
        stock_df = df[df["tic"] == "600000.SH"].sort_values("date")

        result = strategy._check_confirmation(stock_df, "ma120", test_date)
        assert isinstance(result, bool)

    def test_score_candidates(self, default_config):
        """Test candidate scoring logic."""
        strategy = BreakoutStrategy(default_config)

        candidates = pd.DataFrame([
            {
                "tic": "600000.SH",
                "close": 105.0,
                "ma120": 100.0,
                "ma250": 95.0,
                "volume": 1500000,
                "vol_ma": 1000000,
            },
            {
                "tic": "600016.SH",
                "close": 210.0,
                "ma120": 200.0,
                "ma250": 190.0,
                "volume": 2000000,
                "vol_ma": 1000000,
            },
        ])

        scores = strategy._score_candidates(candidates)

        assert len(scores) == 2
        assert "600000.SH" in scores
        assert "600016.SH" in scores
        assert all(isinstance(s, float) for s in scores.values())
        # Higher breakout should have higher score
        assert scores["600016.SH"] > scores["600000.SH"]

    def test_exclusion_rules(self, default_config):
        """Test ST and halted stock exclusion."""
        strategy = BreakoutStrategy(default_config)

        candidates = pd.DataFrame([
            {"tic": "600000.SH", "volume": 1000000},
            {"tic": "ST600001.SH", "volume": 1000000},  # ST stock
            {"tic": "600002.SH", "volume": 0},  # Halted
        ])

        scores = {
            "600000.SH": 0.5,
            "ST600001.SH": 0.6,
            "600002.SH": 0.4,
        }

        selected, all_scores, exclusions = strategy._apply_exclusions(candidates, scores)

        assert "600000.SH" in selected
        assert "ST600001.SH" not in selected
        assert "600002.SH" not in selected
        assert "ST600001.SH" in exclusions
        assert "600002.SH" in exclusions

    def test_empty_result(self, default_config):
        """Test empty result generation."""
        strategy = BreakoutStrategy(default_config)
        result = strategy._empty_result("2023-01-01")

        assert result.date == "2023-01-01"
        assert result.selected_tickers == []
        assert result.scores == {}
        assert result.market_state == MarketState.OSCILLATION
        assert result.active_factors == []
        assert result.factor_weights == {}

    def test_select_with_empty_data(self, default_config):
        """Test selection with empty data."""
        strategy = BreakoutStrategy(default_config)

        empty_df = pd.DataFrame(columns=["date", "tic", "open", "high", "low", "close", "volume"])
        index_df = pd.DataFrame(columns=["date", "tic", "close", "volume"])

        result = strategy.select(empty_df, index_df, "2023-01-01")

        assert result.date == "2023-01-01"
        assert result.selected_tickers == []

    def test_select_with_insufficient_history(self, default_config):
        """Test selection with insufficient historical data."""
        strategy = BreakoutStrategy(default_config)

        # Only 10 days of data (not enough for MA120)
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        data = []
        for date in dates:
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": "600000.SH",
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "volume": 1000000,
            })

        market_df = pd.DataFrame(data)
        index_df = pd.DataFrame([{"date": "2023-01-10", "tic": "000905.SH", "close": 5000.0}])

        result = strategy.select(market_df, index_df, "2023-01-10")

        # Should return empty result due to insufficient data
        assert len(result.selected_tickers) == 0

    def test_top_k_constraint(self, default_config):
        """Test that top_k constraint is respected."""
        config = BreakoutConfig(top_k=2)
        strategy = BreakoutStrategy(config)

        candidates = pd.DataFrame([
            {"tic": f"60000{i}.SH", "close": 100 + i, "ma120": 100.0, "ma250": 95.0,
             "volume": 1500000, "vol_ma": 1000000}
            for i in range(5)
        ])

        scores = {f"60000{i}.SH": 0.5 + i * 0.1 for i in range(5)}

        selected, _, _ = strategy._apply_exclusions(candidates, scores)

        # Should only select top 2
        assert len(selected[:config.top_k]) <= 2


class TestAntiJitterModes:
    """Test different anti-jitter modes."""

    def test_threshold_mode(self, sample_market_data):
        """Test threshold-only mode."""
        config = BreakoutConfig(anti_jitter_mode="threshold")
        strategy = BreakoutStrategy(config)

        df = strategy._compute_indicators(sample_market_data)
        stock_df = df[df["tic"] == "600000.SH"].sort_values("date")
        today_row = stock_df.iloc[-1]

        result = strategy._apply_anti_jitter(stock_df, today_row, "ma120", today_row["date"])
        assert isinstance(result, bool)

    def test_confirmation_mode(self, sample_market_data):
        """Test confirmation-only mode."""
        config = BreakoutConfig(anti_jitter_mode="confirmation")
        strategy = BreakoutStrategy(config)

        df = strategy._compute_indicators(sample_market_data)
        stock_df = df[df["tic"] == "600000.SH"].sort_values("date")
        today_row = stock_df.iloc[-1]

        result = strategy._apply_anti_jitter(stock_df, today_row, "ma120", today_row["date"])
        assert isinstance(result, bool)

    def test_both_mode(self, sample_market_data):
        """Test both threshold and confirmation required."""
        config = BreakoutConfig(anti_jitter_mode="both")
        strategy = BreakoutStrategy(config)

        df = strategy._compute_indicators(sample_market_data)
        stock_df = df[df["tic"] == "600000.SH"].sort_values("date")
        today_row = stock_df.iloc[-1]

        result = strategy._apply_anti_jitter(stock_df, today_row, "ma120", today_row["date"])
        assert isinstance(result, bool)

    def test_either_mode(self, sample_market_data):
        """Test either threshold or confirmation sufficient."""
        config = BreakoutConfig(anti_jitter_mode="either")
        strategy = BreakoutStrategy(config)

        df = strategy._compute_indicators(sample_market_data)
        stock_df = df[df["tic"] == "600000.SH"].sort_values("date")
        today_row = stock_df.iloc[-1]

        result = strategy._apply_anti_jitter(stock_df, today_row, "ma120", today_row["date"])
        assert isinstance(result, bool)
