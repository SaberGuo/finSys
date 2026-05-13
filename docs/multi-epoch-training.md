# Multi-Epoch Random RL Training

## Overview

The `random_rl_train.py` script now supports multi-epoch training where each epoch randomly selects different stocks and time periods. This improves model generalization by exposing it to diverse market conditions.

## Features

- **Incremental Training**: Model accumulates knowledge across epochs
- **Random Sampling**: Each epoch uses different stocks and time periods
- **Efficient Data Loading**: Pre-fetches all data once, filters per epoch
- **Progress Tracking**: Real-time ETA and per-epoch metrics
- **Comprehensive Results**: Per-epoch and aggregated statistics
- **Backward Compatible**: Default behavior unchanged (epochs=1)

## Usage

### Multi-Epoch Training (Recommended)

Train for 100 epochs with 10 random stocks per epoch:

```bash
python scripts/random_rl_train.py \
  --epochs 100 \
  --stocks-per-epoch 10 \
  --timesteps-per-epoch 10000 \
  --train-days 60 \
  --test-days 30 \
  --algo ppo \
  --seed 42
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 1)
- `--stocks-per-epoch`: Number of stocks to randomly select per epoch (default: 10)
- `--timesteps-per-epoch`: Training steps per epoch (default: 10000)
- `--train-days`: Training period length in days (default: 60)
- `--test-days`: Testing period length in days (default: 30)
- `--algo`: RL algorithm - ppo, sac, or td3 (default: ppo)
- `--seed`: Random seed for reproducibility (optional)

### Single-Epoch Training (Original Behavior)

```bash
python scripts/random_rl_train.py \
  --stock-count 10 \
  --timesteps 100000 \
  --train-days 60 \
  --test-days 30
```

## Output Structure

### Multi-Epoch Mode

```
runs/random/TIMESTAMP/
├── epoch_0/
│   ├── ppo_epoch0.zip              # Model weights
│   ├── ppo_epoch0_metadata.json    # Training metadata
│   └── epoch_0_info.json           # Epoch info (stocks, dates, metrics)
├── epoch_1/
│   └── ...
├── epoch_99/
│   └── ...
├── all_epochs_metrics.csv          # All epochs metrics in CSV
├── aggregated_summary.json         # Mean/std/min/max statistics
└── final_model.zip                 # Final trained model
```

### Epoch Info JSON

Each `epoch_N_info.json` contains:

```json
{
  "epoch": 0,
  "stocks": ["600000.SH", "600009.SH", ...],
  "train_start": "2023-05-15",
  "train_end": "2023-07-14",
  "test_start": "2023-07-15",
  "test_end": "2023-08-14",
  "timesteps": 10000,
  "metrics": {
    "annual_return": 0.15,
    "max_drawdown": -0.08,
    "sharpe_ratio": 1.2,
    "win_rate": 0.55,
    "total_trades": 45
  }
}
```

### Aggregated Summary JSON

```json
{
  "total_epochs": 100,
  "total_timesteps": 1000000,
  "metrics": {
    "annual_return": {
      "mean": 0.12,
      "std": 0.08,
      "min": -0.05,
      "max": 0.35
    },
    "sharpe_ratio": {
      "mean": 1.15,
      "std": 0.45,
      "min": 0.2,
      "max": 2.3
    },
    ...
  }
}
```

## Training Strategy

### Incremental Training

The model continues training from the previous epoch, accumulating knowledge:

```
Epoch 0: Train on stocks A, B, C (time period 1)
Epoch 1: Continue training on stocks D, E, F (time period 2)
Epoch 2: Continue training on stocks G, H, I (time period 3)
...
```

This approach:
- ✅ Improves generalization across different stocks
- ✅ Learns patterns from diverse market conditions
- ✅ Reduces overfitting to specific stocks or time periods
- ✅ Produces more robust trading strategies

### Data Pre-fetching

For efficiency, all stock data is loaded once before the epoch loop:

1. **Pre-fetch**: Load all stocks from database (one-time cost)
2. **Filter**: Each epoch filters the pre-fetched data by selected stocks and time range
3. **Train**: Train on filtered data

This avoids 100 database queries and significantly speeds up training.

## Example Output

```
============================================================
Multi-epoch training mode: 100 epochs
============================================================
Pre-fetching all 343 stocks from 2020-01-02 to 2026-04-30
Raw data: 24002166 rows
After cleaning: 23985404 rows, 343 valid tickers
Pre-fetched data: 25222848 rows, 343 tickers

============================================================
[Epoch 1/100] Starting...
============================================================
[Epoch 1/100] Stocks: 600837.SH, 002572.SZ, 000921.SZ... (10 total)
[Epoch 1/100] Time: train=2022-07-04~2022-08-03, test=2022-08-04~2022-08-14
[Epoch 1/100] Training for 10000 steps...
[Epoch 1/100] Backtest: annual_return=12.50%, sharpe=1.35
[Epoch 1/100] Elapsed: 180.5s, ETA: 298.3min

============================================================
[Epoch 2/100] Starting...
============================================================
...

多轮训练完成! (100 epochs)
============================================================

总训练步数: 1,000,000

聚合指标:
  annual_return:
    mean: 0.1234
    std:  0.0567
    min:  -0.0234
    max:  0.3456
  sharpe_ratio:
    mean: 1.2345
    std:  0.4567
    min:  0.2345
    max:  2.3456
  ...

输出目录: runs/random/20260502_143000
```

## Performance Considerations

### Memory Usage

Pre-fetching all 343 stocks requires ~2-4GB RAM. If memory is constrained:
- Reduce the date range (e.g., last 2 years instead of 6 years)
- Use fewer stocks in the database
- Implement per-epoch fetching (slower but less memory)

### Training Time

Estimated time for 100 epochs:
- **Per epoch**: ~2-3 minutes (10k steps)
- **Total**: ~3-5 hours (100 epochs)
- **Pre-fetching**: ~10-15 minutes (one-time)

### Disk Space

Each epoch saves:
- Model weights: ~1MB
- Metadata: ~10KB
- Reports: ~100KB

Total for 100 epochs: ~100-200MB

## Tips

1. **Start Small**: Test with 3-5 epochs first to verify everything works
2. **Use Seeds**: Set `--seed` for reproducible experiments
3. **Monitor Progress**: Check ETA and per-epoch metrics during training
4. **Analyze Results**: Use `all_epochs_metrics.csv` to plot training curves
5. **Compare Models**: Test single-epoch vs multi-epoch on unseen data

## Troubleshooting

### "No valid data after cleaning"

- Check database path is correct
- Verify database contains data for the specified date range
- Ensure stock codes are in correct format (e.g., `600000.SH`)

### "Memory error during pre-fetching"

- Reduce date range with custom min/max dates
- Use fewer stocks
- Increase system RAM or use swap space

### "Training too slow"

- Reduce `--timesteps-per-epoch` (e.g., 5000 instead of 10000)
- Use fewer epochs for initial testing
- Check CPU/GPU utilization

## Configuration

Multi-epoch parameters can also be set in config files:

```yaml
# config/multi_epoch.yaml
training:
  algorithm: ppo
  epochs: 100
  stocks_per_epoch: 10
  timesteps_per_epoch: 10000
```

Then load with:
```bash
python scripts/random_rl_train.py --config config/multi_epoch.yaml
```

## Implementation Details

See the [implementation plan](../C:/Users/gx/.claude/plans/random-rl-train-config-stocks-epoch-10-1-hidden-kay.md) for technical details.

### Key Changes

1. **Config**: Added `epochs`, `stocks_per_epoch`, `timesteps_per_epoch` to `TrainingConfig`
2. **Functions**: Added `fetch_all_stocks_data()`, `filter_epoch_data()`, `save_epoch_results()`, `aggregate_epoch_metrics()`
3. **Training**: Modified `train_and_backtest()` to support incremental training
4. **Main Loop**: Added epoch loop with progress tracking and result aggregation
