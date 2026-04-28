from __future__ import annotations

import pandas as pd
import pytest

from finquant.data.sources.akshare import AkshareDataSource
from finquant.data.sources.baostock import BaostockDataSource
from finquant.data.sources.base import REQUIRED_COLUMNS
from finquant.data.sources.xtquant import XtquantDataSource


@pytest.mark.parametrize("source", [XtquantDataSource(), AkshareDataSource(), BaostockDataSource()])
def test_datasource_contract(source: object) -> None:
    frame = source.download(["000001.SZ"], "2025-01-01", "2025-01-10")
    assert isinstance(frame, pd.DataFrame)
    for col in REQUIRED_COLUMNS:
        assert col in frame.columns
    assert frame["tic"].str.contains(r"^\d{6}\.(SH|SZ)$", regex=True).all()
