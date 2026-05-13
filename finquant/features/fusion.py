"""Feature fusion: left-join MarketDataset + SentimentRecord by (date, tic) (T054)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------
# Schema definition for fused columns
# -----------------------------------------------------------------------

FUSION_COLUMNS: list[str] = [
    "sentiment_score",
    "event_count",
    "has_positive_event",
    "has_negative_event",
    "revenue_growth_pct",
    "net_profit_margin",
    "debt_ratio",
]

FUSION_FILL_VALUES: dict[str, float | int] = {
    "sentiment_score": 0.0,
    "event_count": 0,
    "has_positive_event": 0,
    "has_negative_event": 0,
    "revenue_growth_pct": 0.0,
    "net_profit_margin": 0.0,
    "debt_ratio": 0.5,
}


def fuse_dataframes(
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join *market_df* with *sentiment_df* on ``(date, tic)``.

    Missing sentiment rows are filled with :data:`FUSION_FILL_VALUES`.
    Row count is always equal to ``len(market_df)`` (left join semantics).

    Parameters
    ----------
    market_df:
        MarketDataset. Must have ``date`` and ``tic`` columns.
    sentiment_df:
        Sentiment / fundamental feature DataFrame.  Must have ``date`` and
        ``tic`` columns; any subset of :data:`FUSION_COLUMNS` is accepted.

    Returns
    -------
    pd.DataFrame
        Enhanced dataset with all :data:`FUSION_COLUMNS` appended.
        No NaN values.
    """
    # Identify which fusion columns are present in sentiment_df
    available_cols = [c for c in FUSION_COLUMNS if c in sentiment_df.columns]

    if available_cols:
        right = sentiment_df[["date", "tic"] + available_cols].copy()
        merged = market_df.merge(right, on=["date", "tic"], how="left")
    else:
        merged = market_df.copy()

    # Ensure all fusion columns exist and fill NaN
    for col in FUSION_COLUMNS:
        if col not in merged.columns:
            merged[col] = FUSION_FILL_VALUES[col]
        else:
            merged[col] = merged[col].fillna(FUSION_FILL_VALUES[col])

    # Enforce integer types for flag/count columns
    for col in ["event_count", "has_positive_event", "has_negative_event"]:
        merged[col] = merged[col].astype(int)

    return merged.reset_index(drop=True)


def _load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame."""
    records: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return pd.DataFrame(records) if records else pd.DataFrame()


def fuse_datasets(
    market_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    fundamental_file: Path | str | None = None,
    output_path: Path | str = "data/enhanced/dataset.parquet",
    frequency: str = "daily",
    verbose: bool = False,
) -> Path:
    """Load optional sentiment / fundamental JSONLs, fuse with *market_df*, and save to Parquet.

    Parameters
    ----------
    market_df:
        MarketDataset DataFrame.
    sentiment_file:
        Optional path to sentiment JSONL.
    fundamental_file:
        Optional path to fundamental JSONL.
    output_path:
        Destination Parquet path.
    frequency:
        Data frequency ("daily" or "5min"). For 5min, merge is still on ``(date, tic)``
        since sentiment/fundamental data is daily-granularity.
    verbose:
        Print row counts if True.

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load optional auxiliary dataframes
    sentiment_df = _load_jsonl(Path(sentiment_file)) if sentiment_file else pd.DataFrame()
    fundamental_df = _load_jsonl(Path(fundamental_file)) if fundamental_file else pd.DataFrame()

    # Merge auxiliary frames on (date, tic) if both present
    if not sentiment_df.empty and not fundamental_df.empty:
        aux_df = sentiment_df.merge(fundamental_df, on=["date", "tic"], how="outer")
    elif not sentiment_df.empty:
        aux_df = sentiment_df
    elif not fundamental_df.empty:
        aux_df = fundamental_df
    else:
        aux_df = pd.DataFrame()

    if verbose:
        print(
            f"Fusing {len(market_df)} market rows "
            f"with {len(sentiment_df)} sentiment + {len(fundamental_df)} fundamental records ..."
        )

    enhanced = fuse_dataframes(market_df, aux_df)
    enhanced.to_parquet(output_path, index=False)

    if verbose:
        print(f"Enhanced dataset saved to {output_path} ({len(enhanced)} rows)")

    return output_path
