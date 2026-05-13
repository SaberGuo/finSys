# MA Breakout Selection Strategy Guide

## Overview

The MA Breakout Selection Strategy is a technical analysis-based stock selection approach that identifies stocks breaking through key moving averages with strong volume confirmation. This strategy complements the existing factor-based selection strategy in finSys.

## Strategy Logic

The strategy selects stocks that meet ALL of the following conditions:

### 1. MA Breakout Signal
- **Condition**: Current close crosses above MA (either MA120 or MA250)
- **Formula**: `close_t > MA_t AND close_{t-1} <= MA_{t-1}`
- **Rationale**: Identifies the exact moment when price breaks through a key support/resistance level

### 2. Volume Surge
- **Condition**: Trading volume exceeds threshold
- **Formula**: `volume_t >= K * vol_ma20_t`
- **Default K**: 1.5 (150% of 20-day average volume)
- **Rationale**: Confirms breakout with strong buying interest

### 3. First Breakout
- **Condition**: No breakout in the past N days
- **Default N**: 60 trading days
- **Rationale**: Ensures this is a fresh breakout, not a continuation

### 4. Anti-Jitter Mechanism
Prevents false signals from price oscillations around the MA. Four modes available:

- **threshold**: Price must be at least 5% above MA (`close > MA * 1.05`)
- **confirmation**: Price stays above MA for 3 consecutive days
- **both**: Both threshold AND confirmation required (most strict)
- **either**: Threshold OR confirmation sufficient (most lenient)

## Configuration

### Basic Configuration

Create a YAML config file (e.g., `config/selection_breakout.yaml`):

```yaml
stocks:
  - "600000.SH"
  - "600016.SH"
  # ... more stocks

dates:
  train_start: "2023-01-01"
  train_end: "2023-06-30"
  test_start: "2023-07-01"
  test_end: "2023-12-31"

data:
  source_priority: ["db_daily", "akshare", "baostock"]

selection:
  strategy_type: "breakout"
  
  breakout:
    ma_periods: [120, 250]           # Half-year and year MAs
    volume_multiplier: 1.5           # 1.5x volume surge
    volume_ma_period: 20             # 20-day volume MA
    breakout_threshold: 1.05         # 5% above MA
    lookback_days: 60                # Check 60 days history
    confirmation_days: 3             # 3 days confirmation
    anti_jitter_mode: "threshold"    # threshold | confirmation | both | either
    top_k: 10                        # Select top 10 stocks
    exclude_st: true                 # Exclude ST stocks
    exclude_halt: true               # Exclude halted stocks
```

### Parameter Tuning

#### MA Periods
- **[120, 250]**: Standard (half-year and year lines)
- **[60, 120]**: More sensitive, catches shorter-term breakouts
- **[250]**: Conservative, only year-line breakouts

#### Volume Multiplier
- **1.5**: Moderate volume surge requirement
- **2.0**: Strong volume surge (more conservative)
- **1.2**: Weak volume surge (more aggressive)

#### Breakout Threshold
- **1.05**: 5% above MA (balanced)
- **1.03**: 3% above MA (more signals)
- **1.10**: 10% above MA (fewer but stronger signals)

#### Lookback Days
- **30**: Short-term breakouts
- **60**: Medium-term (default)
- **120**: Long-term, ensures very fresh breakouts

#### Anti-Jitter Mode
- **threshold**: Fast signals, may have false positives
- **confirmation**: Delayed but more reliable
- **both**: Most conservative, fewest signals
- **either**: Most aggressive, most signals

## Usage

### Command Line

Run selection for a date range:

```bash
finquant selection run \
  --config config/selection_breakout.yaml \
  --start 2023-07-01 \
  --end 2023-12-31 \
  --output-dir data/selection_breakout \
  --verbose
```

Output: Daily selection results saved as JSON files in `data/selection_breakout/`

### Python API

```python
from finquant.config.settings import AppConfig
from finquant.selection import create_strategy
import pandas as pd

# Load config
config = AppConfig.from_yaml("config/selection_breakout.yaml")

# Create strategy
strategy = create_strategy(config)

# Prepare data
market_df = pd.read_csv("data/market_data.csv")
index_df = pd.read_csv("data/index_data.csv")

# Run selection
result = strategy.select(market_df, index_df, "2023-08-01")

# Access results
print(f"Selected {len(result.selected_tickers)} stocks:")
for tic in result.selected_tickers:
    print(f"  {tic}: score={result.scores[tic]:.4f}")
```

## Comparing Strategies

To compare breakout strategy with factor-based strategy:

### 1. Create Two Configs

**config/selection_breakout.yaml**:
```yaml
selection:
  strategy_type: "breakout"
  breakout:
    # ... breakout params
```

**config/selection_factor_based.yaml**:
```yaml
selection:
  strategy_type: "factor_based"
  index_ticker: "000905.SH"
  # ... factor-based params
```

### 2. Run Both Strategies

```bash
# Run breakout strategy
finquant selection run \
  --config config/selection_breakout.yaml \
  --start 2023-07-01 \
  --end 2023-12-31 \
  --output-dir data/selection_breakout

# Run factor-based strategy
finquant selection run \
  --config config/selection_factor_based.yaml \
  --start 2023-07-01 \
  --end 2023-12-31 \
  --output-dir data/selection_factor
```

### 3. Compare Results

Analyze selection overlap, forward returns, and portfolio performance:

```python
import json
from pathlib import Path

# Load results
breakout_results = []
factor_results = []

for date_file in Path("data/selection_breakout").glob("*.json"):
    with open(date_file) as f:
        breakout_results.append(json.load(f))

for date_file in Path("data/selection_factor").glob("*.json"):
    with open(date_file) as f:
        factor_results.append(json.load(f))

# Compare overlap
for br, fr in zip(breakout_results, factor_results):
    breakout_set = set(br["selected_tickers"])
    factor_set = set(fr["selected_tickers"])
    overlap = breakout_set & factor_set
    
    print(f"{br['date']}: {len(overlap)} stocks in common")
```

## Evaluation Metrics

The strategy can be evaluated using:

### Selection Quality Metrics
- **Forward Returns**: Average return of selected stocks over next N days
- **Win Rate**: Percentage of selected stocks with positive returns
- **Excess Return**: Selected stocks' return vs benchmark index

### Portfolio Performance Metrics
- **Sharpe Ratio**: Risk-adjusted return
- **CAGR**: Compound annual growth rate
- **Max Drawdown**: Largest peak-to-trough decline

## Best Practices

### 1. Data Requirements
- Minimum 250 trading days of history (for MA250)
- Clean OHLCV data without gaps
- Accurate volume data

### 2. Parameter Selection
- Start with default parameters
- Backtest on historical data
- Adjust based on market conditions

### 3. Risk Management
- Don't rely solely on breakout signals
- Combine with other analysis (fundamentals, sentiment)
- Use stop-loss orders
- Diversify across multiple stocks

### 4. Market Conditions
- **Bull Market**: Use lower thresholds (1.03), shorter lookback (30 days)
- **Bear Market**: Use higher thresholds (1.10), longer lookback (120 days)
- **Sideways Market**: Use confirmation mode to reduce false signals

## Troubleshooting

### No Stocks Selected
- **Cause**: Criteria too strict or no breakouts in period
- **Solution**: Lower `breakout_threshold`, increase `ma_periods` options, or change `anti_jitter_mode` to "either"

### Too Many Stocks Selected
- **Cause**: Criteria too loose
- **Solution**: Increase `breakout_threshold`, use "both" anti-jitter mode, or increase `volume_multiplier`

### False Breakouts
- **Cause**: Price quickly falls back below MA
- **Solution**: Use "confirmation" or "both" anti-jitter mode, increase `confirmation_days`

### Missing Historical Data
- **Cause**: Insufficient data for MA calculation
- **Solution**: Ensure at least 250 days of history, or use shorter MA periods like [60, 120]

## Technical Details

### Implementation
- **File**: `finquant/selection/strategies/breakout.py`
- **Class**: `BreakoutStrategy`
- **Interface**: Implements `SelectionStrategy` ABC

### Data Flow
1. Compute MAs and volume MA for all stocks
2. Filter stocks meeting breakout criteria
3. Score candidates by breakout strength
4. Normalize scores to [-1.0, 1.0] range using tanh
5. Apply exclusion rules (ST, halted)
6. Select top-k by score

### Scoring Logic

Selected stocks are scored based on:

1. **Breakout Strength** (60% weight): `(close - MA) / MA`
2. **Volume Surge Strength** (40% weight): `(volume - vol_ma) / vol_ma`

The raw score is: `raw_score = 0.6 * breakout_strength + 0.4 * volume_strength`

**Score Normalization**: Raw scores are normalized to the range [-1.0, 1.0] using the hyperbolic tangent function:
```
normalized_score = tanh(raw_score * 5)
```

This ensures all scores remain within the valid range while preserving relative rankings. Typical breakout values (5-50%) map to scores between 0.24 and 0.99.

Stocks are ranked by normalized score in descending order, and the top K are selected.

### Performance
- Vectorized pandas operations for efficiency
- Handles missing data gracefully
- Scales to hundreds of stocks

## Examples

### Conservative Setup (Fewer Signals)
```yaml
breakout:
  ma_periods: [250]
  volume_multiplier: 2.0
  breakout_threshold: 1.10
  lookback_days: 120
  anti_jitter_mode: "both"
  top_k: 5
```

### Aggressive Setup (More Signals)
```yaml
breakout:
  ma_periods: [60, 120, 250]
  volume_multiplier: 1.2
  breakout_threshold: 1.03
  lookback_days: 30
  anti_jitter_mode: "either"
  top_k: 20
```

### Balanced Setup (Default)
```yaml
breakout:
  ma_periods: [120, 250]
  volume_multiplier: 1.5
  breakout_threshold: 1.05
  lookback_days: 60
  anti_jitter_mode: "threshold"
  top_k: 10
```

## References

- Technical Analysis: Moving Average Crossover Strategies
- Volume Analysis: Volume Confirmation in Breakouts
- Risk Management: Anti-Jitter Mechanisms in Trading Systems
