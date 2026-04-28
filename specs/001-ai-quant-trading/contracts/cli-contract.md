# Contract: CLI Interface

**Module**: `src/cli/main.py`  
**Tool**: Click-based CLI  
**Date**: 2026-04-28

---

## Commands

### `finsys data fetch`

Download and preprocess A-share historical data.

```
finsys data fetch [OPTIONS]

Options:
  --config PATH       Path to config YAML  [default: config/default.yaml]
  --output PATH       Output directory for processed data  [default: data/processed/]
  --start TEXT        Start date YYYY-MM-DD  [overrides config]
  --end TEXT          End date YYYY-MM-DD  [overrides config]
  --dry-run           Validate config without downloading
  --verbose           Show per-stock download progress

Exit codes:
  0  Success — all stocks fetched and processed
  1  Partial success — some stocks failed, see logs
  2  Configuration error
  3  All data fetches failed (no data downloaded)
```

**Stdout contract** (success, one line per completed stock):
```
2026-04-28 10:00:01 [INFO] Fetched 000001.SZ: 750 trading days
2026-04-28 10:00:02 [INFO] Fetched 600000.SH: 750 trading days
...
2026-04-28 10:05:00 [INFO] Saved to data/processed/20230101_20251231_abc123.parquet
```

---

### `finsys train`

Train a FinRL RL agent on preprocessed data.

```
finsys train [OPTIONS]

Options:
  --config PATH       Path to config YAML  [default: config/default.yaml]
  --data PATH         Path to .parquet dataset (overrides config)
  --algo TEXT         Algorithm: ppo|sac|td3  [default from config]
  --enhanced          Use enhanced (sentiment+fundamental) dataset
  --output PATH       Directory to save model  [default: models/]

Exit codes:
  0  Model trained and saved
  1  Training failed (check logs)
  2  Dataset not found or invalid schema
```

**Stdout contract** (one line per epoch milestone):
```
2026-04-28 11:00:00 [INFO] Training started: PPO, 30 stocks, obs_dim=271
2026-04-28 11:00:30 [INFO] Step 10000/100000 | reward_mean=-0.002
2026-04-28 11:05:00 [INFO] Training complete. Model saved to models/ppo_20260428_abc123/
```

---

### `finsys backtest`

Run backtest on a trained model.

```
finsys backtest [OPTIONS]

Options:
  --config PATH       Path to config YAML  [default: config/default.yaml]
  --model PATH        Path to trained model directory  [required]
  --data PATH         Path to test .parquet dataset  [required]
  --output PATH       Report output directory  [default: reports/]
  --format TEXT       Output format: html|csv|both  [default: html]

Exit codes:
  0  Backtest complete, report saved
  1  Backtest failed
  2  Model or data not found
```

**Stdout contract**:
```
2026-04-28 12:00:00 [INFO] Backtest period: 2025-01-01 to 2025-12-31
2026-04-28 12:00:05 [INFO] Sharpe Ratio: 0.82 | Annual Return: 12.4% | Max Drawdown: -8.3%
2026-04-28 12:00:05 [INFO] Report saved to reports/bt_20260428_abc123/
```

---

### `finsys sentiment analyze`

Run Qwen sentiment analysis on a JSONL text file.

```
finsys sentiment analyze [OPTIONS]

Options:
  --config PATH       Path to config YAML  [default: config/default.yaml]
  --input PATH        Input JSONL file  [required]
  --output PATH       Output JSONL file  [default: data/sentiment/{hash}.jsonl]
  --model TEXT        HuggingFace model ID  [default: Qwen/Qwen2.5-7B-Instruct]
  --quantize          Enable 4-bit NF4 quantization  [default: true]

Exit codes:
  0  Analysis complete
  1  Analysis failed (model load error or inference error)
  2  Input file not found or invalid format
```

---

### `finsys run`

Run full end-to-end pipeline (fetch → train → backtest).

```
finsys run [OPTIONS]

Options:
  --config PATH       Path to config YAML  [default: config/default.yaml]
  --mode TEXT         baseline|enhanced|both  [default: baseline]
  --skip-fetch        Skip data fetch (use existing data)
  --skip-sentiment    Skip sentiment analysis step

Exit codes:
  0  Pipeline complete
  1  Pipeline failed at a step (check logs)
```
