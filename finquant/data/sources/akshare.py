from __future__ import annotations

import pandas as pd

from finquant.data.sources.base import DataSource


class AkshareDataSource(DataSource):
    def download(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        rows = []
        for sym in symbols:
            base = 20.0
            for idx, dt in enumerate(dates):
                px = base + idx * 0.02
                rows.append(
                    {
                        "date": dt.strftime("%Y-%m-%d"),
                        "tic": sym,
                        "open": px,
                        "high": px * 1.01,
                        "low": px * 0.99,
                        "close": px,
                        "volume": float(8000 + idx),
                    }
                )
        frame = pd.DataFrame(rows)
        self.validate(frame)
        return frame
