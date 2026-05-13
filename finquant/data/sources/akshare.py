from __future__ import annotations

import warnings

import pandas as pd

from finquant.data.sources.base import DataSource


class AkshareDataSource(DataSource):
    def download(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        # akshare sina interface uses YYYYMMDD and symbol prefix like "sz000009" / "sh600004"
        _start = start_date.replace("-", "")
        _end = end_date.replace("-", "")

        all_frames: list[pd.DataFrame] = []
        for sym in symbols:
            # Map .SZ/.SH suffix to sina prefix
            code, exchange = sym.split(".")
            prefix = "sz" if exchange == "SZ" else "sh"
            sina_symbol = f"{prefix}{code}"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = ak.stock_zh_a_daily(
                    symbol=sina_symbol,
                    start_date=_start,
                    end_date=_end,
                    adjust="qfq",
                )
            if raw is None or raw.empty:
                raise RuntimeError(f"akshare returned no data for {sym}")

            # Columns are already in English: date, open, high, low, close, volume, ...
            df = raw.copy()
            df["tic"] = sym
            # Keep only required columns
            df = df[["date", "tic", "open", "high", "low", "close", "volume"]]
            all_frames.append(df)

        frame = pd.concat(all_frames, ignore_index=True)
        self.validate(frame)
        return frame
