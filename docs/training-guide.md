# Training Guide

## Overview

`finSys` uses [FinRL](https://github.com/AI4Finance-Foundation/FinRL) with [Stable-Baselines3](https://stable-baselines3.readthedocs.io) to train reinforcement-learning agents (PPO / SAC / TD3) on A-share market data.

## Prerequisites

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare market data: `finsys data fetch --config config/default.yaml.example`

## Train an Agent

```bash
finsys train \
  --config config/default.yaml.example \
  --data-file data/processed/train_data.parquet \
  --algo ppo \
  --timesteps 200000 \
  --output models/
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | from config | `ppo`, `sac`, or `td3` |
| `--timesteps` | from config | Override `training.total_timesteps` |
| `--output` | `models/` | Directory for saved model `.zip` |
| `--dry-run` | off | Validate config without training |

## Run Backtest

```bash
finsys backtest \
  --config config/default.yaml.example \
  --model models/ppo_20240131_abc12345.zip \
  --data-file data/processed/test_data.parquet \
  --output reports/ \
  --risk-free-rate 0.02
```

Output files:
- `reports/ppo_YYYYMMDD_report.html` — interactive pyecharts chart
- `reports/ppo_YYYYMMDD_metrics.csv` — scalar metrics table

## Metrics

| Metric | Description |
|--------|-------------|
| `sharpe` | Annualised Sharpe ratio |
| `cagr` | Compound Annual Growth Rate |
| `max_drawdown` | Maximum drawdown (negative value) |
| `total_return` | Total period return |

## Algorithm Configuration

Set hyperparameters in `config.yaml`:

```yaml
training:
  algorithm: ppo
  total_timesteps: 200000
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
  sac:
    learning_rate: 0.0003
    buffer_size: 1000000
  td3:
    learning_rate: 0.001
    buffer_size: 1000000
```

## Observation Space

The environment observation vector has dimension `1 + 9 * N` where:
- `1` = current cash balance
- `9 * N` = per-stock features: `close`, `volume`, + 7 technical indicators

For enhanced training (with sentiment features): `1 + 16 * N`.

## Full Pipeline Shortcut

```bash
finsys run \
  --config config/default.yaml.example \
  --output runs/latest/ \
  --algo ppo \
  --timesteps 200000
```

This runs: fetch → train → backtest in sequence.
