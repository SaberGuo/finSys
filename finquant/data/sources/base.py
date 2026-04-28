from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


REQUIRED_COLUMNS = ["date", "tic", "open", "high", "low", "close", "volume"]


class DataSource(ABC):
    @abstractmethod
    def download(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download raw market data with REQUIRED_COLUMNS."""

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        if frame.empty:
            raise ValueError("data frame is empty")
