# Data Pipeline Guide

## Overview

`finsys data fetch` downloads A-share market data using source priority
`xtquant -> akshare -> baostock`, preprocesses it, adds technical indicators,
and saves a FinRL-ready Parquet file.

## Command

```bash
finsys data fetch --config config/default.yaml.example --output data/processed
```

## Output Schema

Required columns:
- `date`, `tic`, `open`, `high`, `low`, `close`, `volume`
- default indicators: `macd`, `boll_ub`, `boll_lb`, `rsi_30`, `dx_30`, `close_30_sma`, `close_60_sma`

## Common Errors

- Source unavailable: pipeline logs warning and tries next source.
- Invalid ticker format: preprocessing raises `ValueError`.
- Empty dataset: adapter validation raises `ValueError`.
