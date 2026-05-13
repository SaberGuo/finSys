"""Dividend yield data fetcher for high dividend factor.

dividend_yield = annual_dividend / current_price
"""

from __future__ import annotations

import pandas as pd


def fetch_dividend_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch dividend yield data.

    Args:
        tickers: List of stock tickers
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format

    Returns:
        DataFrame with columns: date, tic, dividend_yield

    Note:
        This is a placeholder implementation. In production, integrate with:
        - Wind API for Chinese market dividend data
        - Tushare Pro for dividend history
        - akshare dividend interface
    """
    # Placeholder: return empty DataFrame with correct schema
    # Real implementation would fetch from Wind/Tushare/akshare
    return pd.DataFrame(columns=["date", "tic", "dividend_yield"])


def compute_dividend_yield(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dividend yield from dividend and price data.

    Args:
        df: DataFrame with columns: date, tic, close, annual_dividend

    Returns:
        DataFrame with added 'dividend_yield' column
    """
    data = df.copy()

    if "annual_dividend" not in data.columns:
        # If dividend data not available, return zeros
        data["dividend_yield"] = 0.0
        return data

    # dividend_yield = annual_dividend / close
    data["dividend_yield"] = (
        data["annual_dividend"] / data["close"].replace(0, 1e-9)
    ).fillna(0.0)

    return data
