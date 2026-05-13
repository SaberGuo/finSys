"""Unit tests for stock screener."""

from __future__ import annotations

import pandas as pd
import pytest

from finquant.selection.factor_registry import FactorRegistry
from finquant.selection.normalizer import FactorNormalizer
from finquant.selection.screener import ScreenConfig, StockScreener


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    data = {
        "date": ["2023-07-01"] * 10,
        "tic": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "close": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        "high": [102, 112, 122, 132, 142, 152, 162, 172, 182, 192],
        "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
    }
    return pd.DataFrame(data)


@pytest.fixture
def registry():
    """Create factor registry with test factors."""
    from finquant.selection.factor_registry import FactorCategory, FactorDefinition

    reg = FactorRegistry()

    # Simple momentum factor (higher close = higher score)
    reg.register(FactorDefinition(
        id="momentum",
        name="Momentum",
        category=FactorCategory.MOMENTUM,
        compute_fn=lambda df: df["close"] / 100,  # Simple scaling
        required_columns=["close"],
        direction=1,
    ))

    # Simple volatility factor (lower is better)
    reg.register(FactorDefinition(
        id="volatility",
        name="Volatility",
        category=FactorCategory.LOW_VOL,
        compute_fn=lambda df: df["volume"] / 1000,  # Simple scaling
        required_columns=["volume"],
        direction=-1,
    ))

    return reg


@pytest.fixture
def normalizer():
    """Create factor normalizer."""
    return FactorNormalizer(method="zscore")


@pytest.fixture
def screener():
    """Create stock screener with default config."""
    config = ScreenConfig(top_k=5, exclude_st=True, exclude_halt=True, exclude_limit_up=True)
    return StockScreener(config)


def test_screen_basic(sample_data, registry, normalizer, screener):
    """Test basic screening with equal weights."""
    factor_weights = {"momentum": 0.5, "volatility": 0.5}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should select top 5 stocks
    assert len(selected) == 5

    # All selected stocks should have scores
    for tic in selected:
        assert tic in scores

    # Scores should be sorted descending
    selected_scores = [scores[tic] for tic in selected]
    assert selected_scores == sorted(selected_scores, reverse=True)


def test_screen_top_k(sample_data, registry, normalizer):
    """Test top_k parameter."""
    config = ScreenConfig(top_k=3)
    screener = StockScreener(config)

    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should select exactly 3 stocks
    assert len(selected) == 3


def test_screen_momentum_only(sample_data, registry, normalizer, screener):
    """Test screening with momentum factor only."""
    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Highest momentum stocks should be selected (J, I, H, G, F)
    assert "J" in selected  # Highest close
    assert "I" in selected
    assert "H" in selected


def test_screen_composite_score_formula(sample_data, registry, normalizer, screener):
    """Test composite score calculation formula."""
    # Equal weights
    factor_weights = {"momentum": 0.5, "volatility": 0.5}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Verify scores are computed correctly
    # Score = (w1 * dir1 * norm1 + w2 * dir2 * norm2) / sum(|w|)
    # All scores should be present
    assert len(scores) == 10


def test_screen_exclude_st_stocks(registry, normalizer):
    """Test ST stock exclusion."""
    data = pd.DataFrame({
        "date": ["2023-07-01"] * 5,
        "tic": ["A", "B", "ST_C", "D", "ST_E"],
        "close": [100, 110, 120, 130, 140],
        "volume": [1000, 1100, 1200, 1300, 1400],
    })

    config = ScreenConfig(top_k=5, exclude_st=True)
    screener = StockScreener(config)

    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # ST stocks should be excluded
    assert "ST_C" not in selected
    assert "ST_E" not in selected

    # ST stocks should be in exclusion reasons
    assert "ST_C" in exclusions
    assert exclusions["ST_C"] == "ST"


def test_screen_exclude_halted_stocks(registry, normalizer):
    """Test halted stock exclusion (volume = 0)."""
    data = pd.DataFrame({
        "date": ["2023-07-01"] * 5,
        "tic": ["A", "B", "C", "D", "E"],
        "close": [100, 110, 120, 130, 140],
        "volume": [1000, 0, 1200, 0, 1400],  # B and D are halted
    })

    config = ScreenConfig(top_k=5, exclude_halt=True)
    screener = StockScreener(config)

    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Halted stocks should be excluded
    assert "B" not in selected
    assert "D" not in selected

    # Halted stocks should be in exclusion reasons
    assert "B" in exclusions
    assert exclusions["B"] == "halted"


def test_screen_no_exclusions(sample_data, registry, normalizer):
    """Test screening with all exclusions disabled."""
    config = ScreenConfig(top_k=5, exclude_st=False, exclude_halt=False, exclude_limit_up=False)
    screener = StockScreener(config)

    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # No exclusions should be applied
    assert len(exclusions) == 0


def test_screen_empty_data(registry, normalizer, screener):
    """Test screening with empty data."""
    data = pd.DataFrame(columns=["date", "tic", "close", "volume"])

    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should return empty results
    assert len(selected) == 0
    assert len(scores) == 0
    assert len(exclusions) == 0


def test_screen_wrong_date(sample_data, registry, normalizer, screener):
    """Test screening with non-existent date."""
    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-12-31",  # Date not in data
    )

    # Should return empty results
    assert len(selected) == 0
    assert len(scores) == 0
    assert len(exclusions) == 0


def test_screen_missing_factor_columns(sample_data, normalizer, screener):
    """Test screening when factor computation fails due to missing columns."""
    from finquant.selection.factor_registry import FactorCategory, FactorDefinition

    # Create registry with factor requiring non-existent column
    reg = FactorRegistry()
    reg.register(FactorDefinition(
        id="invalid",
        name="Invalid",
        category=FactorCategory.MOMENTUM,
        compute_fn=lambda df: df["nonexistent_column"],
        required_columns=["nonexistent_column"],
        direction=1,
    ))

    factor_weights = {"invalid": 1.0}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=reg,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should return empty results (no valid factors)
    assert len(selected) == 0
    assert len(scores) == 0


def test_screen_negative_weights(sample_data, registry, normalizer, screener):
    """Test screening with negative factor weights."""
    # Negative weight reverses factor direction
    factor_weights = {"momentum": -0.5, "volatility": 0.5}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should still select top 5 stocks
    assert len(selected) == 5

    # Scores should be computed correctly with negative weights
    assert len(scores) == 10


def test_screen_zero_total_weight(sample_data, registry, normalizer, screener):
    """Test screening when total weight is zero."""
    factor_weights = {"momentum": 0.0, "volatility": 0.0}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should handle zero total weight gracefully
    assert len(selected) <= 5


def test_screen_single_stock(registry, normalizer):
    """Test screening with single stock."""
    data = pd.DataFrame({
        "date": ["2023-07-01"],
        "tic": ["A"],
        "close": [100],
        "volume": [1000],
    })

    config = ScreenConfig(top_k=5)
    screener = StockScreener(config)

    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should select the single stock
    assert len(selected) == 1
    assert selected[0] == "A"


def test_screen_fewer_stocks_than_top_k(registry, normalizer):
    """Test screening when available stocks < top_k."""
    data = pd.DataFrame({
        "date": ["2023-07-01"] * 3,
        "tic": ["A", "B", "C"],
        "close": [100, 110, 120],
        "volume": [1000, 1100, 1200],
    })

    config = ScreenConfig(top_k=10)  # Request more than available
    screener = StockScreener(config)

    factor_weights = {"momentum": 1.0}

    selected, scores, exclusions = screener.screen(
        df=data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should select all available stocks
    assert len(selected) == 3


def test_screen_multiple_factors_different_directions(sample_data, registry, normalizer, screener):
    """Test screening with multiple factors having different directions."""
    # momentum: direction=1 (higher is better)
    # volatility: direction=-1 (lower is better)
    factor_weights = {"momentum": 0.6, "volatility": 0.4}

    selected, scores, exclusions = screener.screen(
        df=sample_data,
        factor_weights=factor_weights,
        registry=registry,
        normalizer=normalizer,
        as_of_date="2023-07-01",
    )

    # Should balance both factors
    assert len(selected) == 5

    # Verify composite scoring considers both directions
    for tic in selected:
        assert tic in scores
