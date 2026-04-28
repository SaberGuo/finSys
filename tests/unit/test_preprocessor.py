from __future__ import annotations

import pandas as pd

from finquant.data.preprocessor import normalize_ticker, preprocess_market_data


def test_normalize_ticker_supports_multiple_formats() -> None:
    assert normalize_ticker("SZ000001") == "000001.SZ"
    assert normalize_ticker("600000") == "600000.SH"
    assert normalize_ticker("000001.SZ") == "000001.SZ"


def test_preprocess_market_data_removes_duplicates_and_nans(sample_dataset: pd.DataFrame) -> None:
    frame = sample_dataset.copy()
    frame.loc[0, "open"] = None
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    out = preprocess_market_data(frame)

    assert out.duplicated(subset=["date", "tic"]).sum() == 0
    assert out[["open", "high", "low", "close", "volume"]].isna().sum().sum() == 0
    assert out["tic"].str.contains(r"^\d{6}\.(SH|SZ)$", regex=True).all()
