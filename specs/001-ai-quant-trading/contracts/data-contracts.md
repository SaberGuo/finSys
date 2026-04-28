# Contract: Module Data Interface Contracts

**Feature**: `001-ai-quant-trading`  
**Date**: 2026-04-28  
**Purpose**: Defines the exact input/output data contracts at each module boundary. These contracts are verified by tests in `tests/contract/`.

---

## Contract 1: DataFetcher → PreProcessor (Raw Data Output)

**Module boundary**: `src/data/fetchers/*.py` → `src/data/pipeline/preprocessor.py`

**Output schema** (raw, before preprocessing):

```python
# Each fetcher MUST return a pd.DataFrame with exactly these columns.
# Values may be NaN at this stage — preprocessing cleans them.
RAW_DATA_SCHEMA = {
    "date":   "str",     # format: "YYYY-MM-DD"
    "tic":    "str",     # FinRL dot-notation: "000001.SZ"
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
    "volume": "float64",
}
```

**Contract assertions** (verified in `tests/contract/test_fetcher_contract.py`):

```python
def assert_raw_data_contract(df: pd.DataFrame, expected_tickers: list[str]):
    assert set(df.columns) >= set(RAW_DATA_SCHEMA.keys())
    assert df["date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    assert df["tic"].str.match(r"^\d{6}\.(SH|SZ)$").all()
    assert set(df["tic"].unique()) == set(expected_tickers)
    for col in ["open", "high", "low", "close", "volume"]:
        assert df[col].dtype == "float64"
```

---

## Contract 2: PreProcessor → MarketDataset (Processed Output)

**Module boundary**: `src/data/pipeline/preprocessor.py` → `src/envs/stock_env.py`

**Output schema** (fully clean, FinRL-ready):

```python
# MarketDataset MUST satisfy ALL assertions below.
PROCESSED_DATA_SCHEMA = {
    "date":          "str",
    "tic":           "str",
    "open":          "float64",
    "high":          "float64",
    "low":           "float64",
    "close":         "float64",
    "volume":        "float64",
    # Plus all indicator columns from config:
    "macd":          "float64",
    "boll_ub":       "float64",
    "boll_lb":       "float64",
    "rsi_30":        "float64",
    "dx_30":         "float64",
    "close_30_sma":  "float64",
    "close_60_sma":  "float64",
}

def assert_processed_data_contract(df: pd.DataFrame, indicators: list[str]):
    # No NaN values permitted
    assert df.isnull().sum().sum() == 0
    # Required columns present
    required = list(PROCESSED_DATA_SCHEMA.keys()) + indicators
    assert set(required) <= set(df.columns)
    # Sorted correctly
    assert df.equals(df.sort_values(["date", "tic"]).reset_index(drop=True))
    # Unique (date, tic) pairs
    assert df.duplicated(subset=["date", "tic"]).sum() == 0
    # Positive price values
    assert (df["close"] > 0).all()
    assert (df["volume"] >= 0).all()  # 0 on suspension days
```

---

## Contract 3: SentimentAnalyzer Output Schema

**Module boundary**: `src/sentiment/analyzer.py` → `src/features/fusion.py`

**Output schema** (one record per input text):

```python
SENTIMENT_RECORD_SCHEMA = {
    "date":             str,    # "YYYY-MM-DD"
    "tic":              str,    # "000001.SZ"
    "source":           str,    # e.g. "user_file"
    "text_hash":        str,    # MD5 hex string
    "sentiment_score":  float,  # in [-1.0, 1.0]
    "event_tags":       list,   # list of str, may be []
    "summary":          str,    # max 200 chars
    "model_id":         str,    # "Qwen/Qwen2.5-7B-Instruct"
}

def assert_sentiment_record_contract(record: dict):
    assert -1.0 <= record["sentiment_score"] <= 1.0
    assert isinstance(record["event_tags"], list)
    assert len(record["summary"]) <= 200
    assert record["date"] matches r"^\d{4}-\d{2}-\d{2}$"
    assert record["tic"] matches r"^\d{6}\.(SH|SZ)$"
```

---

## Contract 4: FeatureFusion → EnhancedDataset Output

**Module boundary**: `src/features/fusion.py` → `src/envs/stock_env.py`

**Contract assertions**:

```python
def assert_enhanced_dataset_contract(
    enhanced_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    indicators: list[str]
):
    # Row count MUST match baseline (left join, no new rows)
    assert len(enhanced_df) == len(baseline_df)
    # Must contain all baseline columns
    assert set(baseline_df.columns) <= set(enhanced_df.columns)
    # Additional sentiment columns present
    for col in ["sentiment_score", "event_count", "has_positive_event", "has_negative_event"]:
        assert col in enhanced_df.columns
    # No NaN in enhanced columns (filled with defaults)
    sentiment_cols = ["sentiment_score", "event_count",
                      "has_positive_event", "has_negative_event",
                      "revenue_growth_pct", "net_profit_margin", "debt_ratio"]
    assert enhanced_df[sentiment_cols].isnull().sum().sum() == 0
```

---

## Contract 5: TradingEnvironment ↔ RLAgent (FinRL gym interface)

**Module boundary**: `src/envs/stock_env.py` ↔ `src/agents/trainer.py`

**Observation space contract**:

```python
def assert_env_obs_space_contract(env, df: pd.DataFrame, indicators: list[str]):
    N = df["tic"].nunique()
    expected_dim = 1 + 2 * N + len(indicators) * N
    assert env.observation_space.shape == (expected_dim,)
    assert env.action_space.shape == (N,)  # continuous action per stock

def assert_backtest_output_contract(stats: dict):
    required_keys = ["sharpe_ratio", "annual_return_pct", "max_drawdown_pct",
                     "cumulative_return_pct"]
    for key in required_keys:
        assert key in stats
        assert isinstance(stats[key], float)
        assert not math.isnan(stats[key])
```

---

## Contract 6: BacktestReport Storage Schema

**Module boundary**: `src/backtest/reporter.py` → file system

**`metrics.json` schema**:

```json
{
  "run_id": "string",
  "model_path": "string",
  "dataset_type": "baseline | enhanced",
  "test_start": "YYYY-MM-DD",
  "test_end": "YYYY-MM-DD",
  "sharpe_ratio": 0.0,
  "annual_return_pct": 0.0,
  "max_drawdown_pct": 0.0,
  "cumulative_return_pct": 0.0,
  "baseline_sharpe_ratio": 0.0,
  "sharpe_improvement_pct": 0.0,
  "generated_at": "ISO-8601 datetime"
}
```
