from __future__ import annotations

import pandas as pd

from finquant.features.technical import DEFAULT_INDICATORS, compute_indicators


def test_compute_indicators_adds_expected_columns(sample_dataset: pd.DataFrame) -> None:
    out = compute_indicators(sample_dataset)
    for col in DEFAULT_INDICATORS:
        assert col in out.columns
    assert out[DEFAULT_INDICATORS].isna().sum().sum() == 0
