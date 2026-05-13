from __future__ import annotations

import re

import pandas as pd

from finquant.data.sources.base import REQUIRED_COLUMNS


_TICKER_RE = re.compile(r"^\d{6}\.(SH|SZ)$")


def normalize_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if _TICKER_RE.match(t):
        return t
    # Handle sh.600000 / sz.000001 format
    if t.startswith("SH.") or t.startswith("SZ."):
        return t[3:] + "." + t[:2]
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

    # Handle empty DataFrame
    if data.empty:
        return data

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    data["tic"] = data["tic"].map(normalize_ticker)
    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values(["date", "tic"]).drop_duplicates(subset=["date", "tic"], keep="last")

    # Enforce numeric columns and remove invalid rows.
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close", "volume"])

    if (data["close"] <= 0).any():
        raise ValueError("close price must be > 0")
    if (data["volume"] < 0).any():
        raise ValueError("volume must be >= 0")

    # Compute turnover_rate and market_cap if not present
    if "turnover_rate" not in data.columns:
        # turnover_rate = volume / total_shares (approximation: use volume/close as proxy)
        # In real implementation, need actual total_shares data
        data["turnover_rate"] = 0.0  # Placeholder - requires actual shares outstanding data

    if "market_cap" not in data.columns:
        # market_cap = close * total_shares (approximation)
        # In real implementation, need actual shares outstanding data
        data["market_cap"] = 0.0  # Placeholder - requires actual shares outstanding data

    # Align data so every ticker has a row for every trading day (FinRL requirement).
    all_dates = data["date"].unique()
    all_tickers = data["tic"].unique()
    index = pd.MultiIndex.from_product(
        [all_dates, all_tickers], names=["date", "tic"]
    )
    data = (
        data.set_index(["date", "tic"])
        .reindex(index)
        .reset_index()
    )
    # Forward-fill all numeric columns per ticker, then back-fill leading NAs (e.g. pre-IPO).
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        data[col] = data.groupby("tic")[col].transform(lambda s: s.ffill().bfill())

    if data.select_dtypes(include=["number"]).isnull().any().any():
        raise ValueError("failed to align market data: missing values remain after fill")

    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return data.sort_values(["date", "tic"]).reset_index(drop=True)


def preprocess_5min_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Preprocess 5-minute market data with ``time`` column support.

    Parses ``time`` (``YYYYMMDDHHMMSSmmm``) into a ``datetime`` column,
    normalises tickers, and aligns per-ticker per-session.
    """
    data = frame.copy()

    required = ["date", "tic", "time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    data["tic"] = data["tic"].map(normalize_ticker)
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

    # Parse full datetime from time string (YYYYMMDDHHMMSSmmm)
    time_str = data["time"].astype(str)
    data["datetime"] = pd.to_datetime(
        time_str.str[:14],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    # Fallback: if time is already HHMM or HH:MM
    if data["datetime"].isna().any():
        alt = pd.to_datetime(
            data["date"] + " " + time_str.str[:4],
            format="%Y-%m-%d %H%M",
            errors="coerce",
        )
        data["datetime"] = data["datetime"].fillna(alt)

    # Enforce numeric columns and remove invalid rows.
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close", "volume"])

    if (data["close"] <= 0).any():
        raise ValueError("close price must be > 0")
    if (data["volume"] < 0).any():
        raise ValueError("volume must be >= 0")

    # Align so every (date, time) has every ticker (FinRL requirement).
    # For 5min data the full Cartesian product is too large for memory,
    # so we reindex date-by-date to keep each chunk small.
    all_dates = data["date"].unique()
    all_tickers = data["tic"].unique()

    if len(all_dates) == 0 or len(all_tickers) == 0:
        raise ValueError(
            f"No data to preprocess: {len(all_dates)} dates, {len(all_tickers)} tickers. "
            "Check that your data source has data for the configured date range and stocks."
        )

    grouped = data.set_index(["date", "time", "tic"])
    aligned_chunks: list[pd.DataFrame] = []
    for d in all_dates:
        day_data = grouped.xs(d, level="date", drop_level=False)
        day_times = day_data.index.get_level_values("time").unique()
        day_index = pd.MultiIndex.from_product(
            [[d], day_times, all_tickers], names=["date", "time", "tic"]
        )
        aligned_chunks.append(day_data.reindex(day_index))

    if not aligned_chunks:
        raise ValueError("No data chunks to concatenate after alignment")

    data = pd.concat(aligned_chunks).reset_index()

    # Forward-fill all numeric columns per ticker, then back-fill leading NAs.
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        data[col] = data.groupby("tic")[col].transform(lambda s: s.ffill().bfill())

    if data.select_dtypes(include=["number"]).isnull().any().any():
        raise ValueError("failed to align 5min data: missing values remain after fill")

    return data.sort_values(["date", "time", "tic"]).reset_index(drop=True)
