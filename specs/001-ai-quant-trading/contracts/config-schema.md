# Contract: Configuration Schema (config.yaml)

**Module**: All modules read `config.yaml`  
**Date**: 2026-04-28

---

## Full Schema

```yaml
# config/default.yaml
# Full annotated schema. All fields shown with their defaults.

# ─────────────────────────────────────────────────
# Stock Universe
# ─────────────────────────────────────────────────
stocks:
  - "000001.SZ"   # Required: list of tickers in FinRL dot-notation (6-digit.SH|SZ)
  # - "600000.SH"
  # ...

# ─────────────────────────────────────────────────
# Date Ranges
# ─────────────────────────────────────────────────
dates:
  train_start: "2020-01-01"   # Required: YYYY-MM-DD
  train_end:   "2024-12-31"   # Required: YYYY-MM-DD
  test_start:  "2025-01-01"   # Required: YYYY-MM-DD
  test_end:    "2025-12-31"   # Required: YYYY-MM-DD

# ─────────────────────────────────────────────────
# Data Sources
# ─────────────────────────────────────────────────
data:
  primary_source: "xtquant"         # xtquant | akshare | baostock
  fallback_source: "akshare"        # akshare | baostock | null
  output_dir: "data/processed/"
  raw_dir: "data/raw/"
  xtquant:
    credentials_env: "XTQUANT_PATH" # Env var pointing to QMT client install dir
  akshare: {}                       # No auth required
  baostock:
    user_id: ""                     # Leave empty if not using baostock
    password_env: "BAOSTOCK_PWD"

# ─────────────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────────────
indicators:
  - "macd"
  - "boll_ub"
  - "boll_lb"
  - "rsi_30"
  - "dx_30"
  - "close_30_sma"
  - "close_60_sma"

# ─────────────────────────────────────────────────
# Trading Environment
# ─────────────────────────────────────────────────
environment:
  initial_amount: 1000000     # Initial cash in RMB
  hmax: 100                   # Max shares per trade action
  buy_cost_pct: 0.001         # 0.1% commission
  sell_cost_pct: 0.001        # 0.1% commission
  reward_scaling: 0.0001      # Scale portfolio change to reward

# ─────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────
training:
  algorithm: "ppo"            # ppo | sac | td3
  total_timesteps: 100000
  model_dir: "models/"
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
  sac:
    learning_rate: 0.0001
    batch_size: 256
    gamma: 0.99
    tau: 0.005
  td3:
    learning_rate: 0.001
    batch_size: 256
    gamma: 0.99
    tau: 0.005

# ─────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────
backtest:
  report_dir: "reports/"
  benchmark: "000300.SH"     # CSI 300 benchmark ticker

# ─────────────────────────────────────────────────
# Sentiment Analysis
# ─────────────────────────────────────────────────
sentiment:
  enabled: false              # Set true to use sentiment features
  news_file: ""               # Path to input JSONL file
  output_dir: "data/sentiment/"
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  quantize_4bit: true         # Enable NF4 4-bit quantization
  max_new_tokens: 512
  batch_size: 8

# ─────────────────────────────────────────────────
# Fundamentals
# ─────────────────────────────────────────────────
fundamentals:
  enabled: false              # Set true to extract fundamental features
  reports_dir: "data/reports_text/"  # Directory of report text files (JSONL)
  output_dir: "data/fundamentals/"

# ─────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────
logging:
  level: "INFO"               # DEBUG | INFO | WARNING | ERROR
  log_dir: "logs/"
  structured: true            # JSON-formatted logs

# ─────────────────────────────────────────────────
# Reserved (not implemented in this spec)
# ─────────────────────────────────────────────────
live_trading: false           # Reserved for future 002-live-trading spec
```

---

## Validation Rules

| Key path | Rule |
|----------|------|
| `stocks` | Non-empty list; each element matches `^\d{6}\.(SH\|SZ)$` |
| `dates.*` | All 4 date fields required; `train_start < train_end < test_start < test_end` |
| `data.primary_source` | Must be one of `['xtquant', 'akshare', 'baostock']` |
| `indicators` | Non-empty list; must match column names computed by stockstats |
| `training.algorithm` | Must be one of `['ppo', 'sac', 'td3']` |
| `sentiment.news_file` | Required if `sentiment.enabled = true`; must be readable JSONL |

---

## Sensitive Values

Credentials MUST NOT be stored directly in `config.yaml`. Use environment variables:

| Env Var | Used By | Notes |
|---------|---------|-------|
| `XTQUANT_PATH` | xtquant fetcher | Path to QMT client installation directory |
| `BAOSTOCK_PWD` | baostock fetcher | Only required if using baostock |
