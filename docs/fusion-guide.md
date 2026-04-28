# Feature Fusion Guide

## Overview

Feature fusion combines MarketDataset (OHLCV + technical indicators) with sentiment features extracted by Qwen2.5-7B-Instruct to create an **EnhancedDataset** for FinRL training.

## Fused Columns

| Column | Source | Default Fill |
|--------|--------|-------------|
| `sentiment_score` | Qwen sentiment | `0.0` (neutral) |
| `event_count` | Qwen sentiment | `0` |
| `has_positive_event` | Qwen sentiment | `0` |
| `has_negative_event` | Qwen sentiment | `0` |
| `revenue_growth_pct` | Fundamental | `0.0` |
| `net_profit_margin` | Fundamental | `0.0` |
| `debt_ratio` | Fundamental | `0.5` (neutral) |

## Join Logic

- **Left join** on `(date, tic)` — market data is the primary table.
- Row count in EnhancedDataset = row count in MarketDataset (no rows added or removed).
- Missing sentiment for a `(date, tic)` pair → fill with defaults above.
- No NaN values permitted in output.

## Usage

```bash
# Step 1: Run sentiment analysis
finsys sentiment analyze \
  --input data/news/news_2024.jsonl \
  --output data/sentiment/

# Step 2: Fuse market data with sentiment
finsys fuse \
  --market data/processed/train.parquet \
  --sentiment data/sentiment/sentiment_records.jsonl \
  --output data/enhanced/enhanced_train.parquet

# Step 3: Train on enhanced dataset
finsys train \
  --data-file data/enhanced/enhanced_train.parquet \
  --algo ppo \
  --output models/enhanced/
```

## Observation Space

| Dataset | obs_dim formula | Example (N=30) |
|---------|----------------|----------------|
| Baseline | `1 + 9 * N` | 271 |
| Enhanced | `1 + 16 * N` | 481 |

The 16 per-stock features = `close` + `volume` + 7 indicators + 7 fusion columns.

## Python API

```python
from finquant.features.fusion import fuse_datasets, fuse_dataframes
import pandas as pd

# From files
out_path = fuse_datasets(
    market_df=pd.read_parquet("data/processed/train.parquet"),
    sentiment_file="data/sentiment/sentiment_records.jsonl",
    output_path="data/enhanced/train.parquet",
    verbose=True,
)

# In-memory
enhanced_df = fuse_dataframes(market_df, sentiment_df)
```
