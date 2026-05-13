"""Unit tests for factor normalizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finquant.selection.normalizer import FactorNormalizer


def test_normalizer_init_valid_methods():
    """Test normalizer initialization with valid methods."""
    for method in ["zscore", "rank", "mad"]:
        normalizer = FactorNormalizer(method=method)
        assert normalizer.method == method


def test_normalizer_init_invalid_method():
    """Test normalizer initialization with invalid method."""
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        FactorNormalizer(method="invalid")


def test_zscore_basic():
    """Test Z-Score normalization with basic data."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # Mean should be 0, std should be 1
    assert normalized.mean() == pytest.approx(0.0, abs=1e-10)
    assert normalized.std() == pytest.approx(1.0, abs=1e-10)


def test_zscore_with_nan():
    """Test Z-Score normalization with NaN values."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([1, 2, None, 4, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # NaN should be replaced with 0.0
    assert normalized.loc["C"] == 0.0

    # Valid values should be normalized
    assert normalized.loc["A"] < normalized.loc["B"]


def test_zscore_all_identical_fallback_to_rank():
    """Test Z-Score normalization fallback to rank when std=0."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([5, 5, 5, 5, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # All identical values should result in all zeros (rank fallback)
    assert (normalized == 0.0).all()


def test_zscore_empty_series():
    """Test Z-Score normalization with empty series."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([], dtype=float)

    normalized = normalizer.normalize(values)

    assert len(normalized) == 0


def test_zscore_all_nan():
    """Test Z-Score normalization with all NaN values."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([None, None, None], index=["A", "B", "C"])

    normalized = normalizer.normalize(values)

    # All NaN should result in all zeros
    assert (normalized == 0.0).all()


def test_rank_basic():
    """Test Rank normalization with basic data."""
    normalizer = FactorNormalizer(method="rank")
    values = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # Rank normalization: 2 * (rank_pct - 0.5)
    # Ranks: 0.2, 0.4, 0.6, 0.8, 1.0
    # Normalized: -0.6, -0.2, 0.2, 0.6, 1.0
    assert normalized.loc["A"] == pytest.approx(-0.6, abs=0.01)
    assert normalized.loc["C"] == pytest.approx(0.2, abs=0.01)
    assert normalized.loc["E"] == pytest.approx(1.0, abs=0.01)


def test_rank_with_ties():
    """Test Rank normalization with tied values."""
    normalizer = FactorNormalizer(method="rank")
    values = pd.Series([1, 2, 2, 4, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # Tied values (B, C) should have same normalized value
    assert normalized.loc["B"] == normalized.loc["C"]


def test_rank_all_identical():
    """Test Rank normalization with all identical values."""
    normalizer = FactorNormalizer(method="rank")
    values = pd.Series([5, 5, 5, 5, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # All identical should result in all zeros
    assert (normalized == 0.0).all()


def test_rank_with_nan():
    """Test Rank normalization with NaN values."""
    normalizer = FactorNormalizer(method="rank")
    values = pd.Series([1, None, 3, 4, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # NaN should be replaced with 0.0
    assert normalized.loc["B"] == 0.0

    # Valid values should be ranked
    assert normalized.loc["A"] < normalized.loc["C"]


def test_mad_basic():
    """Test MAD normalization with basic data."""
    normalizer = FactorNormalizer(method="mad")
    values = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # Median = 3, MAD = median(|values - 3|) = median([2, 1, 0, 1, 2]) = 1
    # Normalized = (values - 3) / 1
    assert normalized.loc["A"] == pytest.approx(-2.0, abs=0.01)
    assert normalized.loc["C"] == pytest.approx(0.0, abs=0.01)
    assert normalized.loc["E"] == pytest.approx(2.0, abs=0.01)


def test_mad_with_nan():
    """Test MAD normalization with NaN values."""
    normalizer = FactorNormalizer(method="mad")
    values = pd.Series([1, 2, None, 4, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # NaN should be replaced with 0.0
    assert normalized.loc["C"] == 0.0

    # Valid values should be normalized
    assert normalized.loc["A"] < normalized.loc["B"]


def test_mad_all_identical_fallback_to_rank():
    """Test MAD normalization fallback to rank when MAD=0."""
    normalizer = FactorNormalizer(method="mad")
    values = pd.Series([5, 5, 5, 5, 5], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # All identical values should result in all zeros (rank fallback)
    assert (normalized == 0.0).all()


def test_mad_outliers():
    """Test MAD normalization is robust to outliers."""
    normalizer_mad = FactorNormalizer(method="mad")
    normalizer_zscore = FactorNormalizer(method="zscore")

    # Data with outlier
    values = pd.Series([1, 2, 3, 4, 100], index=["A", "B", "C", "D", "E"])

    normalized_mad = normalizer_mad.normalize(values)
    normalized_zscore = normalizer_zscore.normalize(values)

    # MAD should be more robust (outlier has less impact)
    # For MAD: median=3, MAD=1, so outlier normalized to (100-3)/1 = 97
    # For Z-Score: mean=22, std≈43.5, so outlier normalized to (100-22)/43.5 ≈ 1.79

    # MAD outlier should be larger in absolute value
    assert abs(normalized_mad.loc["E"]) > abs(normalized_zscore.loc["E"])


def test_normalize_preserves_index():
    """Test that normalization preserves original index."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([1, 2, 3, 4, 5], index=["X", "Y", "Z", "W", "V"])

    normalized = normalizer.normalize(values)

    assert list(normalized.index) == ["X", "Y", "Z", "W", "V"]


def test_normalize_preserves_order():
    """Test that normalization preserves relative order."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([5, 1, 3, 2, 4], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # Relative order should be preserved
    assert normalized.loc["B"] < normalized.loc["D"] < normalized.loc["C"] < normalized.loc["E"] < normalized.loc["A"]


def test_normalize_range():
    """Test that normalized values are approximately in [-1, 1] range for typical data."""
    normalizer = FactorNormalizer(method="zscore")
    # Normal distribution data
    np.random.seed(42)
    values = pd.Series(np.random.randn(100))

    normalized = normalizer.normalize(values)

    # Most values should be in [-3, 3] range (99.7% for normal distribution)
    assert (normalized.abs() < 3).sum() > 95


def test_normalize_single_value():
    """Test normalization with single value."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([5], index=["A"])

    normalized = normalizer.normalize(values)

    # Single value should normalize to 0 (std=0 fallback to rank, which gives 0 for single value)
    assert normalized.loc["A"] == 0.0


def test_normalize_two_values():
    """Test normalization with two values."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([1, 5], index=["A", "B"])

    normalized = normalizer.normalize(values)

    # Two values: mean=3, std=sqrt(8)=2.828...
    # A: (1-3)/2.828 ≈ -0.707
    # B: (5-3)/2.828 ≈ 0.707
    assert normalized.loc["A"] == pytest.approx(-0.707, abs=0.01)
    assert normalized.loc["B"] == pytest.approx(0.707, abs=0.01)


def test_normalize_negative_values():
    """Test normalization with negative values."""
    normalizer = FactorNormalizer(method="zscore")
    values = pd.Series([-5, -3, -1, 1, 3], index=["A", "B", "C", "D", "E"])

    normalized = normalizer.normalize(values)

    # Mean = -1, std ≈ 3.16
    assert normalized.mean() == pytest.approx(0.0, abs=1e-10)
    assert normalized.std() == pytest.approx(1.0, abs=1e-10)
