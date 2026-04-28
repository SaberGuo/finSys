from __future__ import annotations

import re

import pandas as pd

from finquant.data.sources.base import REQUIRED_COLUMNS


_TICKER_RE = re.compile(r"^\d{6}\.(SH|SZ)$")


def normalize_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if _TICKER_RE.match(t):
        return t
    if t.startswith("SZ") or t.startswith("SH"):
        code = t[2:]
        exchange = t[:2]
        return f"{code}.{exchange}"
    if t.isdigit() and len(t) == 6:
        exchange = "SH" if t.startswith(("600", "601", "603", "605", "688")) else "SZ"
        return f"{t}.{exchange}"
    raise ValueError(f"invalid ticker format: {ticker}")


def preprocess_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    data["tic"] = data["tic"].map(normalize_ticker)
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

    data = data.sort_values(["date", "tic"]).drop_duplicates(subset=["date", "tic"], keep="last")

    # Suspension-day handling: close forward fill, volume defaults to 0.
    data["close"] = data.groupby("tic")["close"].ffill().bfill()
    for col in ["open", "high", "low"]:
        data[col] = data[col].fillna(data["close"])
    data["volume"] = data["volume"].fillna(0.0)

    # Enforce numeric columns and remove invalid rows.
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close", "volume"])

    if (data["close"] <= 0).any():
        raise ValueError("close price must be > 0")
    if (data["volume"] < 0).any():
        raise ValueError("volume must be >= 0")

    return data.reset_index(drop=True)
