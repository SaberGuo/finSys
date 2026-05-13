"""A-share fundamental data fetching from Sina Finance via akshare."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finquant.utils.logging import get_logger

logger = get_logger("finquant.data.fundamentals")

# Default fill values matching fusion.py schema
_DEFAULT_FILLS: dict[str, float] = {
    "revenue_growth_pct": 0.0,
    "net_profit_margin": 0.0,
    "debt_ratio": 0.5,
    "roe": 0.0,
}


def _to_float(val: Any) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extract_code(ticker: str) -> str:
    """Return 6-digit numeric code from dot-notation ticker."""
    return ticker.split(".")[0]


def fetch_financial_reports(
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch profit statement and balance sheet for a single stock.

    Parameters
    ----------
    ticker:
        Dot-notation ticker, e.g. ``000009.SZ``.

    Returns
    -------
    profit_df, balance_df
        DataFrames indexed by ``report_date`` (datetime).  May be empty
        if the API returns no data.
    """
    import akshare as ak

    code = _extract_code(ticker)
    profit_records: list[dict] = []
    balance_records: list[dict] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            raw_profit = ak.stock_financial_report_sina(
                stock=code, symbol="利润表"
            )
        except Exception as exc:
            logger.warning(f"profit statement fetch failed for {ticker}: {exc}")
            raw_profit = pd.DataFrame()

        try:
            raw_balance = ak.stock_financial_report_sina(
                stock=code, symbol="资产负债表"
            )
        except Exception as exc:
            logger.warning(f"balance sheet fetch failed for {ticker}: {exc}")
            raw_balance = pd.DataFrame()

    if not raw_profit.empty and "报告日" in raw_profit.columns:
        for _, row in raw_profit.iterrows():
            rd = str(row.get("报告日", ""))
            if not rd or len(rd) != 8:
                continue
            profit_records.append(
                {
                    "report_date": pd.to_datetime(rd, format="%Y%m%d"),
                    "total_revenue": _to_float(row.get("营业总收入")),
                    "net_profit": _to_float(row.get("净利润")),
                    "net_profit_parent": _to_float(
                        row.get("归属于母公司所有者的净利润")
                    ),
                }
            )

    if not raw_balance.empty and "报告日" in raw_balance.columns:
        for _, row in raw_balance.iterrows():
            rd = str(row.get("报告日", ""))
            if not rd or len(rd) != 8:
                continue
            balance_records.append(
                {
                    "report_date": pd.to_datetime(rd, format="%Y%m%d"),
                    "total_assets": _to_float(row.get("资产总计")),
                    "total_liabilities": _to_float(row.get("负债合计")),
                    "total_equity": _to_float(
                        row.get("所有者权益(或股东权益)合计")
                    ),
                }
            )

    profit_df = pd.DataFrame(profit_records)
    if "report_date" in profit_df.columns:
        profit_df = profit_df.set_index("report_date").sort_index()
    else:
        profit_df.index = pd.DatetimeIndex([], name="report_date")

    balance_df = pd.DataFrame(balance_records)
    if "report_date" in balance_df.columns:
        balance_df = balance_df.set_index("report_date").sort_index()
    else:
        balance_df.index = pd.DatetimeIndex([], name="report_date")

    return profit_df, balance_df


def compute_fundamental_metrics(
    profit_df: pd.DataFrame,
    balance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-report fundamental metrics from raw financial statements.

    Parameters
    ----------
    profit_df:
        Profit statement indexed by ``report_date``.
    balance_df:
        Balance sheet indexed by ``report_date``.

    Returns
    -------
    pd.DataFrame
        Index = ``report_date``.  Columns:
        ``revenue_growth_pct``, ``net_profit_margin``,
        ``debt_ratio``, ``roe``.
    """
    # Merge on report_date index
    merged = profit_df.join(balance_df, how="outer").sort_index()

    metrics: dict[str, list] = {
        "report_date": [],
        "revenue_growth_pct": [],
        "net_profit_margin": [],
        "debt_ratio": [],
        "roe": [],
    }

    for date, row in merged.iterrows():
        metrics["report_date"].append(date)

        # YoY revenue growth: compare with ~4 quarters ago
        revenue_growth = _DEFAULT_FILLS["revenue_growth_pct"]
        curr_rev = row.get("total_revenue")
        if curr_rev is not None and curr_rev > 0:
            # Look for a report roughly 1 year ago (350–380 days)
            past = merged.loc[: date - pd.Timedelta(days=350)]
            if not past.empty:
                past_rev = past.iloc[-1].get("total_revenue")
                if past_rev is not None and past_rev > 0:
                    revenue_growth = (curr_rev - past_rev) / past_rev * 100
        metrics["revenue_growth_pct"].append(round(revenue_growth, 2))

        # Net profit margin
        npm = _DEFAULT_FILLS["net_profit_margin"]
        net_profit = row.get("net_profit")
        total_revenue = row.get("total_revenue")
        if (
            net_profit is not None
            and total_revenue is not None
            and total_revenue > 0
        ):
            npm = net_profit / total_revenue * 100
        metrics["net_profit_margin"].append(round(npm, 2))

        # Debt ratio
        dr = _DEFAULT_FILLS["debt_ratio"]
        total_liab = row.get("total_liabilities")
        total_assets = row.get("total_assets")
        if (
            total_liab is not None
            and total_assets is not None
            and total_assets > 0
        ):
            dr = total_liab / total_assets
        metrics["debt_ratio"].append(round(dr, 4))

        # ROE
        roe = _DEFAULT_FILLS["roe"]
        net_profit = row.get("net_profit")
        equity = row.get("total_equity")
        if equity is not None and equity > 0 and net_profit is not None:
            roe = net_profit / equity * 100
        metrics["roe"].append(round(roe, 2))

    return pd.DataFrame(metrics).set_index("report_date").sort_index()


def align_to_daily(
    metrics_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    ticker: str,
) -> pd.DataFrame:
    """Forward-fill quarterly metrics to daily trading-date frequency.

    Parameters
    ----------
    metrics_df:
        Quarterly metrics indexed by ``report_date``.
    trading_dates:
        Complete set of trading days to align to.
    ticker:
        Ticker symbol for the ``tic`` column.

    Returns
    -------
    pd.DataFrame
        One row per trading date with columns
        ``date``, ``tic``, ``revenue_growth_pct``, ``net_profit_margin``,
        ``debt_ratio``, ``roe``.
    """
    if metrics_df.empty:
        # Return all-default rows
        df = pd.DataFrame(
            {
                "date": trading_dates.strftime("%Y-%m-%d"),
                "tic": ticker,
                "revenue_growth_pct": _DEFAULT_FILLS["revenue_growth_pct"],
                "net_profit_margin": _DEFAULT_FILLS["net_profit_margin"],
                "debt_ratio": _DEFAULT_FILLS["debt_ratio"],
                "roe": _DEFAULT_FILLS["roe"],
            }
        )
        return df

    # Reindex to trading dates: for each trading date, use the latest
    # report whose report_date <= trading_date.
    daily = metrics_df.reindex(trading_dates, method="ffill")
    daily = daily.reset_index().rename(columns={"index": "date"})
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily["tic"] = ticker

    # Ensure all columns exist
    for col in ["revenue_growth_pct", "net_profit_margin", "debt_ratio", "roe"]:
        if col not in daily.columns:
            daily[col] = _DEFAULT_FILLS[col]
        else:
            daily[col] = daily[col].fillna(_DEFAULT_FILLS[col])

    return daily[["date", "tic", "revenue_growth_pct", "net_profit_margin", "debt_ratio", "roe"]]


def fetch_fundamentals_for_universe(
    tickers: list[str],
    trading_dates: list[str],
    output_dir: str | Path,
    verbose: bool = False,
) -> Path:
    """Fetch fundamental data for all tickers and save to JSONL.

    Parameters
    ----------
    tickers:
        List of dot-notation tickers.
    trading_dates:
        List of ``YYYY-MM-DD`` trading dates to align metrics to.
    output_dir:
        Destination directory.
    verbose:
        Print progress.

    Returns
    -------
    Path
        Path to the saved JSONL file (``fundamentals.jsonl``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_index = pd.to_datetime(trading_dates)
    all_records: list[dict[str, Any]] = []

    for i, ticker in enumerate(tickers):
        if verbose:
            logger.info(f"fetching fundamentals {i + 1}/{len(tickers)}: {ticker}")

        profit_df, balance_df = fetch_financial_reports(ticker)
        metrics_df = compute_fundamental_metrics(profit_df, balance_df)
        daily_df = align_to_daily(metrics_df, date_index, ticker)

        for _, row in daily_df.iterrows():
            all_records.append(
                {
                    "date": row["date"],
                    "tic": row["tic"],
                    "revenue_growth_pct": float(row["revenue_growth_pct"]),
                    "net_profit_margin": float(row["net_profit_margin"]),
                    "debt_ratio": float(row["debt_ratio"]),
                    "roe": float(row["roe"]),
                }
            )

    out_path = output_dir / "fundamentals.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"saved {len(all_records)} fundamental records to {out_path}")
    return out_path


def fetch_sue_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SUE (Standardized Unexpected Earnings) data.

    Args:
        tickers: List of stock tickers
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format

    Returns:
        DataFrame with columns: date, tic, actual_eps, expected_eps, std_eps, sue

    Note:
        This is a placeholder implementation. In production, integrate with:
        - Wind API for Chinese market earnings data
        - Tushare Pro for earnings estimates
        - Or custom database with analyst consensus data
    """
    # Placeholder: return empty DataFrame with correct schema
    # Real implementation would fetch from Wind/Tushare/database
    return pd.DataFrame(
        columns=["date", "tic", "actual_eps", "expected_eps", "std_eps", "sue"]
    )


def compute_sue(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SUE from earnings data.

    Args:
        df: DataFrame with columns: date, tic, actual_eps, expected_eps, std_eps

    Returns:
        DataFrame with added 'sue' column
    """
    data = df.copy()

    if "actual_eps" not in data.columns or "expected_eps" not in data.columns:
        # If earnings data not available, return zeros
        data["sue"] = 0.0
        return data

    # Compute SUE per ticker
    data["sue"] = 0.0
    for tic in data["tic"].unique():
        mask = data["tic"] == tic
        tic_data = data.loc[mask].copy()

        if "std_eps" in tic_data.columns and (tic_data["std_eps"] > 0).any():
            # Use provided std_eps
            std_eps = tic_data["std_eps"].replace(0, 1e-9)
        else:
            # Compute rolling std of earnings surprises
            surprise = tic_data["actual_eps"] - tic_data["expected_eps"]
            std_eps = surprise.rolling(8, min_periods=1).std().replace(0, 1e-9)

        sue = (tic_data["actual_eps"] - tic_data["expected_eps"]) / std_eps
        data.loc[mask, "sue"] = sue.fillna(0.0)

    return data
