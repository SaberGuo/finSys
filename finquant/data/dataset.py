"""Training dataset builder for multi-indicator-set pipelines."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from finquant.config.settings import AppConfig, TargetConfig
from finquant.features.indicator_sets import IndicatorSet, IndicatorSetRegistry
from finquant.features.technical import compute_indicators

logger = logging.getLogger(__name__)


def _compute_target(
    df: pd.DataFrame,
    target_config: TargetConfig,
) -> pd.Series:
    """Compute target variable (future_return or direction)."""
    horizon = target_config.horizon
    target_type = target_config.type

    # Sort by tic and time to ensure correct shift
    sorted_df = df.sort_values(["tic", "date", "time"]).reset_index(drop=True)
    close = sorted_df["close"].astype(float)

    if target_type == "future_return":
        future = close.groupby(sorted_df["tic"]).shift(-horizon)
        target = (future - close) / close.replace(0, 1e-9)
    elif target_type == "direction":
        future = close.groupby(sorted_df["tic"]).shift(-horizon)
        change = (future - close) / close.replace(0, 1e-9)
        target = (change > target_config.threshold).astype(int)
    else:
        raise ValueError(f"unsupported target type: {target_type}")

    return target


class TrainingDatasetBuilder:
    """Orchestrate preprocessing → indicators → target → Parquet."""

    def __init__(
        self,
        config: AppConfig,
        registry: IndicatorSetRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry

    def build(
        self,
        df: pd.DataFrame,
        indicator_set_id: str | None = None,
        output_path: Path | str | None = None,
    ) -> Path:
        """Build a training dataset and save to Parquet.

        Parameters
        ----------
        df:
            Preprocessed 5-minute DataFrame with required columns.
        indicator_set_id:
            ID of the indicator set to use. If None, uses config.indicators.
        output_path:
            Destination Parquet path. If None, auto-generated.

        Returns
        -------
        Path
            Path to the saved Parquet file.
        """
        if output_path is None:
            output_path = Path("data/training") / f"{indicator_set_id or 'default'}_dataset.parquet"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = df.copy()
        logger.info(f"Building dataset from {len(data)} rows")

        # Compute indicators
        if indicator_set_id:
            if self.registry is None:
                raise ValueError(f"indicator_set_id '{indicator_set_id}' provided but no registry available")
            indicator_set = self.registry.get(indicator_set_id)
            data = indicator_set.compute(data)
            logger.info(f"Computed indicators for set '{indicator_set_id}': {indicator_set.indicators}")
        else:
            data = compute_indicators(data, self.config.indicators)
            logger.info(f"Computed default indicators: {self.config.indicators}")

        # Compute target
        target = _compute_target(data, self.config.target)
        data["target"] = target
        logger.info(f"Computed target: {self.config.target.type} (horizon={self.config.target.horizon})")

        # Drop rows where target is NaN (can't compute future value for last rows)
        before_drop = len(data)
        data = data.dropna(subset=["target"]).reset_index(drop=True)
        after_drop = len(data)
        if after_drop < before_drop:
            logger.warning(f"Dropped {before_drop - after_drop} rows with NaN target")

        # Validation
        self._validate(data, indicator_set_id)

        # Sort
        sort_cols = ["date", "time", "tic"] if "time" in data.columns else ["date", "tic"]
        data = data.sort_values(sort_cols).reset_index(drop=True)

        data.to_parquet(output_path, index=False)
        logger.info(f"Saved training dataset to {output_path} ({len(data)} rows)")
        return output_path

    def _validate(self, data: pd.DataFrame, indicator_set_id: str | None) -> None:
        """Validate dataset before saving."""
        # Check required columns
        required = ["date", "tic", "open", "high", "low", "close", "volume", "target"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")

        # Check no NaN in indicator columns (excluding target which is already handled)
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "target"]
        for col in numeric_cols:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count} NaN values")
                data[col] = data[col].fillna(0.0)

        # Check row count
        if len(data) == 0:
            raise ValueError("dataset is empty after processing")

        logger.info(f"Validation passed: {len(data)} rows, {len(numeric_cols)} numeric columns")
