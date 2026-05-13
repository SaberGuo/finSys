"""A-share news fetching from East Money via akshare."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from finquant.data.preprocessor import normalize_ticker
from finquant.utils.logging import get_logger

logger = get_logger("finquant.data.news")

_NEWS_COL_MAP: dict[str, str] = {
    "关键词": "keyword",
    "新闻标题": "title",
    "新闻内容": "content",
    "发布时间": "datetime",
    "文章来源": "source",
    "新闻链接": "url",
}


def _extract_code(ticker: str) -> str:
    """Return 6-digit numeric code from dot-notation ticker."""
    return ticker.split(".")[0]


def fetch_stock_news(
    ticker: str,
    max_pages: int = 5,
    sleep_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """Fetch East Money news for a single stock.

    Parameters
    ----------
    ticker:
        Dot-notation ticker, e.g. ``000009.SZ``.
    max_pages:
        Maximum pages to fetch (each page ~10 items).  Akshare's
        ``stock_news_em`` currently returns the first page only; this
        parameter reserves room for future pagination support.
    sleep_seconds:
        Delay between API calls to avoid throttling.

    Returns
    -------
    list[dict]
        Each dict has keys ``date``, ``tic``, ``title``, ``content``.
    """
    import akshare as ak

    code = _extract_code(ticker)
    records: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # akshare stock_news_em returns the latest ~10 news items.
                raw = ak.stock_news_em(symbol=code)
        except Exception as exc:
            logger.warning(f"akshare news fetch failed for {ticker} page {page}: {exc}")
            break

        if raw is None or raw.empty:
            break

        df = raw.rename(columns=_NEWS_COL_MAP)
        for _, row in df.iterrows():
            dt_str = str(row.get("datetime", ""))
            # datetime format is "YYYY-MM-DD HH:MM:SS"
            date = dt_str[:10] if len(dt_str) >= 10 else dt_str
            title = str(row.get("title", "")).strip()
            content = str(row.get("content", "")).strip()
            if not title and not content:
                continue
            records.append(
                {
                    "date": date,
                    "tic": ticker,
                    "title": title,
                    "content": content,
                }
            )

        if page < max_pages:
            time.sleep(sleep_seconds)

    return records


def fetch_news_for_universe(
    tickers: list[str],
    output_dir: str | Path,
    max_pages: int = 5,
    sleep_seconds: float = 0.5,
    verbose: bool = False,
) -> Path:
    """Fetch news for all tickers and save to a single JSONL file.

    Parameters
    ----------
    tickers:
        List of dot-notation tickers.
    output_dir:
        Destination directory.
    max_pages:
        Pages per ticker (see :func:`fetch_stock_news`).
    sleep_seconds:
        Throttle delay between tickers.
    verbose:
        Print progress.

    Returns
    -------
    Path
        Path to the saved JSONL file (``news.jsonl``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()  # (date, tic, title) dedup

    for i, ticker in enumerate(tickers):
        if verbose:
            logger.info(f"fetching news {i + 1}/{len(tickers)}: {ticker}")

        records = fetch_stock_news(ticker, max_pages=max_pages, sleep_seconds=sleep_seconds)
        for rec in records:
            key = (rec["date"], rec["tic"], rec["title"])
            if key not in seen:
                seen.add(key)
                all_records.append(rec)

        if i < len(tickers) - 1:
            time.sleep(sleep_seconds)

    out_path = output_dir / "news.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"saved {len(all_records)} news records to {out_path}")
    return out_path
