"""Contract tests for 5min env observation space dimensions."""
from __future__ import annotations

import pandas as pd
import pytest

from finquant.training.env import compute_obs_dim


class Test5MinEnvContract:
    """T035: validate obs_dim computation matches expectation."""

    def test_obs_dim_with_fusion_indicators(self) -> None:
        indicators = ["macd", "rsi_30", "close_30_sma"]
        fusion = ["sentiment_score"]
        expected = compute_obs_dim(2, indicators=indicators, fusion_indicators=fusion)
        # 1 + (2 + 3 + 1) * 2 = 1 + 12 = 13
        assert expected == 13

    def test_obs_dim_without_fusion(self) -> None:
        indicators = ["macd", "volume_ratio"]
        expected = compute_obs_dim(2, indicators=indicators)
        # 1 + (2 + 2) * 2 = 1 + 8 = 9
        assert expected == 9

    def test_obs_dim_zero_stock_raises(self) -> None:
        with pytest.raises(ValueError, match="stock_dim must be > 0"):
            compute_obs_dim(0)
