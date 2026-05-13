"""Integration test for end-to-end selection pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from finquant.config.settings import AppConfig, SelectionConfig
from finquant.selection import MarketState
from finquant.selection.pipeline import SelectionPipeline


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    data = []
    for date in dates:
        for i, tic in enumerate(tickers):
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": tic,
                "close": 100 + i * 10 + (dates.get_loc(date) % 10),
                "high": 102 + i * 10 + (dates.get_loc(date) % 10),
                "low": 98 + i * 10 + (dates.get_loc(date) % 10),
                "volume": 1000000 + i * 100000,
                "turnover_rate": 0.05 + i * 0.01,
                "market_cap": 1e9 + i * 1e8,
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_index_data():
    """Create sample index data for testing."""
    from finquant.features.technical import compute_indicators

    dates = pd.date_range("2023-01-01", periods=60, freq="D")

    # Create uptrend index data
    data = []
    for i, date in enumerate(dates):
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "tic": "INDEX",
            "close": 100 + i,
            "high": 102 + i,
            "low": 98 + i,
            "volume": 1000000,
        })

    df = pd.DataFrame(data)

    # Compute ADX indicator
    df = compute_indicators(df, ["adx_14"])

    return df


@pytest.fixture
def config():
    """Create test configuration."""
    from finquant.config.settings import DatesConfig

    return AppConfig(
        stocks=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        dates=DatesConfig(
            train_start="2023-01-01",
            train_end="2023-02-28",
            test_start="2023-03-01",
            test_end="2023-03-31",
        ),
        selection=SelectionConfig(
            index_ticker="000905.SH",
            top_k=5,
            ic_window=60,
            ic_min_periods=20,
            normalizer="zscore",
            exclude_st=True,
            exclude_halt=True,
        ),
    )


@pytest.fixture
def pipeline(config):
    """Create selection pipeline."""
    return SelectionPipeline.from_config(config)


def test_pipeline_run_basic(pipeline, sample_market_data, sample_index_data):
    """Test basic pipeline execution."""
    result = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Should return SelectionResult
    assert result.date == "2023-03-01"
    assert len(result.selected_tickers) > 0
    assert len(result.selected_tickers) <= 5  # top_k=5

    # Should have scores for selected tickers
    assert len(result.scores) == len(result.selected_tickers)

    # Should have market state
    assert result.market_state in MarketState

    # Should have active factors
    assert len(result.active_factors) > 0

    # Should have factor weights
    assert len(result.factor_weights) == len(result.active_factors)


def test_pipeline_run_selects_top_k(pipeline, sample_market_data, sample_index_data):
    """Test that pipeline selects exactly top_k stocks."""
    result = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Should select exactly 5 stocks (top_k=5)
    assert len(result.selected_tickers) == 5


def test_pipeline_run_scores_sorted(pipeline, sample_market_data, sample_index_data):
    """Test that selected stocks are sorted by score descending."""
    result = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Extract scores in selection order
    scores = [result.scores[tic] for tic in result.selected_tickers]

    # Should be sorted descending
    assert scores == sorted(scores, reverse=True)


def test_pipeline_run_market_state_classification(pipeline, sample_market_data, sample_index_data):
    """Test that pipeline correctly classifies market state."""
    result = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Index data is uptrend, should classify as uptrend
    assert result.market_state == MarketState.UPTREND


def test_pipeline_run_factor_selection(pipeline, sample_market_data, sample_index_data):
    """Test that pipeline selects appropriate factors for market state."""
    result = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # For uptrend, should select momentum/growth factors
    assert any(f in result.active_factors for f in ["momentum_20d", "high_beta", "growth_yoy"])


def test_pipeline_run_ic_weighting(pipeline, sample_market_data, sample_index_data):
    """Test that pipeline computes IC weights."""
    result = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Should have weights for all active factors
    assert set(result.factor_weights.keys()) == set(result.active_factors)

    # Weights should sum to 1.0
    assert sum(result.factor_weights.values()) == pytest.approx(1.0)

    # All weights should be positive
    assert all(w > 0 for w in result.factor_weights.values())


def test_pipeline_run_exclusion_rules(sample_market_data, sample_index_data, config):
    """Test that pipeline applies exclusion rules."""
    # Add ST stock to market data
    st_data = sample_market_data.copy()
    st_rows = st_data[st_data["tic"] == "A"].copy()
    st_rows["tic"] = "ST_A"
    st_data = pd.concat([st_data, st_rows], ignore_index=True)

    pipeline = SelectionPipeline.from_config(config)

    result = pipeline.run(
        market_df=st_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # ST stock should be excluded
    assert "ST_A" not in result.selected_tickers

    # ST stock should be in exclusion reasons
    assert "ST_A" in result.exclusion_reasons
    assert result.exclusion_reasons["ST_A"] == "ST"


def test_pipeline_run_empty_market_data(pipeline, sample_index_data):
    """Test pipeline with empty market data."""
    empty_df = pd.DataFrame(columns=["date", "tic", "close", "volume"])

    result = pipeline.run(
        market_df=empty_df,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Should return empty selection
    assert len(result.selected_tickers) == 0
    assert len(result.scores) == 0


def test_pipeline_run_insufficient_stocks(sample_index_data, config):
    """Test pipeline when available stocks < top_k."""
    # Create data with only 3 stocks
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    tickers = ["A", "B", "C"]

    data = []
    for date in dates:
        for i, tic in enumerate(tickers):
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": tic,
                "close": 100 + i * 10,
                "volume": 1000000,
            })

    market_df = pd.DataFrame(data)

    pipeline = SelectionPipeline.from_config(config)

    result = pipeline.run(
        market_df=market_df,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Should select all available stocks (3 < top_k=5)
    assert len(result.selected_tickers) == 3


def test_pipeline_run_multiple_dates(pipeline, sample_market_data, sample_index_data):
    """Test pipeline execution across multiple dates."""
    dates = ["2023-02-01", "2023-02-15", "2023-03-01"]

    results = []
    for date in dates:
        result = pipeline.run(
            market_df=sample_market_data,
            index_df=sample_index_data,
            as_of_date=date,
        )
        results.append(result)

    # All results should be valid
    assert len(results) == 3

    # Each result should have selections
    for result in results:
        assert len(result.selected_tickers) > 0


def test_pipeline_run_index_metrics(pipeline, sample_market_data, sample_index_data):
    """Test that pipeline includes index metrics."""
    result = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Should have index metrics
    assert isinstance(result.index_metrics, dict)


def test_pipeline_run_serialization(pipeline, sample_market_data, sample_index_data):
    """Test that pipeline result can be serialized."""
    result = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Should be able to convert to dict
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert "date" in result_dict
    assert "selected_tickers" in result_dict
    assert "market_state" in result_dict


def test_pipeline_from_config(config):
    """Test pipeline creation from config."""
    pipeline = SelectionPipeline.from_config(config)

    # Should create all components
    assert pipeline.classifier is not None
    assert pipeline.registry is not None
    assert pipeline.selector is not None
    assert pipeline.ic_calculator is not None
    assert pipeline.normalizer is not None
    assert pipeline.screener is not None


def test_pipeline_from_config_missing_selection(config):
    """Test pipeline creation fails without selection config."""
    config.selection = None

    with pytest.raises(ValueError, match="config.selection is required"):
        SelectionPipeline.from_config(config)


def test_pipeline_run_consistent_results(pipeline, sample_market_data, sample_index_data):
    """Test that pipeline produces consistent results for same input."""
    result1 = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    result2 = pipeline.run(
        market_df=sample_market_data,
        index_df=sample_index_data,
        as_of_date="2023-03-01",
    )

    # Should produce same results
    assert result1.selected_tickers == result2.selected_tickers
    assert result1.market_state == result2.market_state
    assert result1.active_factors == result2.active_factors


def test_pipeline_run_different_normalizers(sample_market_data, sample_index_data, config):
    """Test pipeline with different normalization methods."""
    for method in ["zscore", "rank", "mad"]:
        config.selection.normalizer = method
        pipeline = SelectionPipeline.from_config(config)

        result = pipeline.run(
            market_df=sample_market_data,
            index_df=sample_index_data,
            as_of_date="2023-03-01",
        )

        # Should produce valid results with any normalizer
        assert len(result.selected_tickers) > 0
        assert result.market_state in MarketState
