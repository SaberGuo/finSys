"""Factor normalizer for cross-sectional standardization.

Normalizes factor values across the candidate pool to enable fair comparison
and composite scoring.
"""

from __future__ import annotations

import pandas as pd


class FactorNormalizer:
    """Normalize factor values cross-sectionally."""

    def __init__(self, method: str = "zscore"):
        """Initialize normalizer.

        Args:
            method: Normalization method ("zscore", "rank", "mad")

        Raises:
            ValueError: If method not supported
        """
        if method not in ("zscore", "rank", "mad"):
            raise ValueError(f"Unsupported normalization method: {method}")
        self.method = method

    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize factor values cross-sectionally.

        Args:
            values: Factor values (one value per stock)

        Returns:
            Normalized values in range approximately [-1, 1]

        Note:
            - zscore: (value - mean) / std, fallback to rank if std=0
            - rank: 2 * (rank_pct - 0.5), returns zeros if all identical
            - mad: (value - median) / MAD, fallback to rank if MAD=0
        """
        if len(values) == 0:
            return values

        # Remove NaN values for computation
        valid_mask = values.notna()
        if not valid_mask.any():
            return pd.Series(0.0, index=values.index)

        valid_values = values[valid_mask]

        if self.method == "zscore":
            return self._zscore(values, valid_values, valid_mask)
        elif self.method == "rank":
            return self._rank(values, valid_values, valid_mask)
        elif self.method == "mad":
            return self._mad(values, valid_values, valid_mask)

        raise ValueError(f"Unsupported method: {self.method}")

    def _zscore(
        self,
        values: pd.Series,
        valid_values: pd.Series,
        valid_mask: pd.Series,
    ) -> pd.Series:
        """Z-Score normalization: (value - mean) / std.

        Fallback to rank if std=0.
        """
        mean = valid_values.mean()
        std = valid_values.std()

        if std == 0 or pd.isna(std):
            # Fallback to rank normalization
            return self._rank(values, valid_values, valid_mask)

        normalized = (values - mean) / std
        normalized[~valid_mask] = 0.0
        return normalized

    def _rank(
        self,
        values: pd.Series,
        valid_values: pd.Series,
        valid_mask: pd.Series,
    ) -> pd.Series:
        """Rank normalization: 2 * (rank_pct - 0.5).

        Returns zeros if all values identical.
        """
        # Check if all values are identical
        if valid_values.nunique() == 1:
            return pd.Series(0.0, index=values.index)

        # Rank percentile (0 to 1)
        rank_pct = values.rank(pct=True, method="average")

        # Transform to [-1, 1] range
        normalized = 2 * (rank_pct - 0.5)
        normalized[~valid_mask] = 0.0
        return normalized

    def _mad(
        self,
        values: pd.Series,
        valid_values: pd.Series,
        valid_mask: pd.Series,
    ) -> pd.Series:
        """MAD normalization: (value - median) / MAD.

        Fallback to rank if MAD=0.
        """
        median = valid_values.median()
        mad = (valid_values - median).abs().median()

        if mad == 0 or pd.isna(mad):
            # Fallback to rank normalization
            return self._rank(values, valid_values, valid_mask)

        normalized = (values - median) / mad
        normalized[~valid_mask] = 0.0
        return normalized
