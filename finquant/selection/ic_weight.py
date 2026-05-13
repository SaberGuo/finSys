"""IC weight calculator for dynamic factor weighting.

Computes factor weights based on rolling Information Coefficient (IC) history.
IC measures the Spearman rank correlation between factor values and future returns.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from finquant.selection.factor_registry import FactorRegistry


class ICWeightCalculator:
    """Calculate factor weights based on rolling IC history."""

    def __init__(
        self,
        registry: FactorRegistry,
        window: int = 60,
        min_periods: int = 20,
    ):
        """Initialize IC weight calculator.

        Args:
            registry: Factor registry with IC history
            window: Rolling window size in trading days
            min_periods: Minimum periods required for IC calculation
        """
        self.registry = registry
        self.window = window
        self.min_periods = min_periods

    def compute_weights(
        self,
        factor_ids: list[str],
        as_of_date: str,
        preset_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute IC-weighted factor weights.

        Args:
            factor_ids: List of factor IDs to weight
            as_of_date: Date for IC calculation
            preset_weights: Fallback weights if IC history insufficient

        Returns:
            Dictionary of factor weights (sum to 1.0)

        Note:
            If IC history < min_periods, returns preset_weights.
            Weight formula: weight_i = |IC_i| / sum(|IC_j|)
        """
        # Get IC series for each factor
        ic_values = {}
        for fid in factor_ids:
            ic_series = self.registry.get_ic_series(fid, window=self.window)

            if len(ic_series) < self.min_periods:
                # Insufficient IC history, use preset weights
                if preset_weights is None:
                    # Equal weights fallback
                    weight = 1.0 / len(factor_ids)
                    return {fid: weight for fid in factor_ids}
                return preset_weights.copy()

            # Compute mean absolute IC
            mean_abs_ic = ic_series.abs().mean()
            ic_values[fid] = mean_abs_ic

        # Compute weights: |IC_i| / sum(|IC_j|)
        total_ic = sum(ic_values.values())
        if total_ic == 0:
            # All ICs are zero, use preset weights or equal weights
            if preset_weights is not None:
                return preset_weights.copy()
            weight = 1.0 / len(factor_ids)
            return {fid: weight for fid in factor_ids}

        weights = {fid: ic / total_ic for fid, ic in ic_values.items()}

        # Ensure weights sum to 1.0 (handle floating point errors)
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            weights = {fid: w / total for fid, w in weights.items()}

        return weights

    def compute_ic(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
    ) -> float:
        """Compute IC (Spearman rank correlation) between factor and returns.

        Args:
            factor_values: Factor values (cross-section)
            future_returns: Future returns (cross-section)

        Returns:
            IC value (Spearman correlation coefficient)

        Note:
            Returns 0.0 if insufficient data or computation fails.
        """
        # Align indices
        common_idx = factor_values.index.intersection(future_returns.index)
        if len(common_idx) < 2:
            return 0.0

        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = future_returns.loc[common_idx]

        # Remove NaN values
        valid_mask = factor_aligned.notna() & returns_aligned.notna()
        factor_clean = factor_aligned[valid_mask]
        returns_clean = returns_aligned[valid_mask]

        if len(factor_clean) < 2:
            return 0.0

        # Compute Spearman rank correlation
        try:
            corr, _ = spearmanr(factor_clean, returns_clean)
            return float(corr) if not pd.isna(corr) else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    def update_ic_history(
        self,
        factor_id: str,
        date: str,
        factor_values: pd.Series,
        future_returns: pd.Series,
    ) -> None:
        """Compute and record IC for a factor on a specific date.

        Args:
            factor_id: Factor identifier
            date: Date in "YYYY-MM-DD" format
            factor_values: Factor values (cross-section)
            future_returns: Future returns (cross-section)
        """
        ic = self.compute_ic(factor_values, future_returns)
        self.registry.record_ic(factor_id, date, ic)
