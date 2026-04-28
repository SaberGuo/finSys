# Data Model: A股AI量化交易系统

**Feature**: `001-ai-quant-trading`  
**Phase**: 1 — Design  
**Date**: 2026-04-28  
**Source**: [spec.md](spec.md) Key Entities + [research.md](research.md)

---

## Entity Overview

```
StockUniverse ──(1:N)──> MarketDataset ──(1:1)──> TradingEnvironment
                                                         │
                                                    RLAgent ──> BacktestReport
                                                    
StockUniverse ──(1:N)──> SentimentRecord
StockUniverse ──(1:N)──> FundamentalFeature

MarketDataset + SentimentRecord + FundamentalFeature ──> EnhancedDataset ──> TradingEnvironment
```

---

## Entity 1: StockUniverse（股票池）

**Description**: Target A-share stocks. Primary key for all data operations.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `tic` | `str` | NOT NULL, UNIQUE, format `\d{6}\.(SH\|SZ)` | FinRL ticker format: `000001.SZ` |
| `name` | `str` | NOT NULL | Short Chinese name e.g. `平安银行` |
| `industry` | `str` | OPTIONAL | CSRC Level-1 industry classification |
| `exchange` | `str` | NOT NULL, enum `['SH', 'SZ']` | Derived from ticker suffix |

**State transitions**: None (static configuration)  
**Validation rules**:
- `tic` must match `^\d{6}\.(SH|SZ)$`
- No duplicates allowed within a single universe configuration

**Source**: Loaded from `config.yaml::stocks` list on system startup  
**Sample record**: `{"tic": "000001.SZ", "name": "平安银行", "industry": "银行", "exchange": "SZ"}`

---

## Entity 2: MarketDataset（行情数据集）

**Description**: Standardized multi-stock daily time-series. Direct input to FinRL `StockTradingEnv`.

**Schema** (one row = one stock × one trading day):

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `date` | `str` | NOT NULL, format `YYYY-MM-DD` | Trading date (A-share calendar) |
| `tic` | `str` | NOT NULL, FK → StockUniverse.tic | Ticker in dot-notation |
| `open` | `float` | NOT NULL, > 0 | Opening price (adjusted) |
| `high` | `float` | NOT NULL, >= open | Intraday high |
| `low` | `float` | NOT NULL, <= open, > 0 | Intraday low |
| `close` | `float` | NOT NULL, > 0 | Closing price (adjusted) |
| `volume` | `float` | NOT NULL, >= 0 | Trading volume (shares). 0 on suspension days |
| `macd` | `float` | NOT NULL | MACD signal, computed by stockstats |
| `boll_ub` | `float` | NOT NULL | Bollinger Band upper |
| `boll_lb` | `float` | NOT NULL | Bollinger Band lower |
| `rsi_30` | `float` | NOT NULL, range [0, 100] | 30-period RSI |
| `dx_30` | `float` | NOT NULL, range [0, 100] | 30-period Directional Index |
| `close_30_sma` | `float` | NOT NULL, > 0 | 30-period SMA of close |
| `close_60_sma` | `float` | NOT NULL, > 0 | 60-period SMA of close |

**Invariants**:
- No `NaN` values permitted in any column (suspension days → forward-fill close, volume=0)
- Rows sorted by `(date ASC, tic ASC)`
- Each `(date, tic)` pair is UNIQUE
- Indicator columns must match `INDICATORS` list in `config.yaml` exactly

**Derived property**: `observation_space_dim = 1 + 9 * N` where N = `len(StockUniverse)`

**Storage**: Parquet file at `data/processed/{start_date}_{end_date}_{universe_hash}.parquet`

---

## Entity 3: TradingEnvironment（交易环境）

**Description**: FinRL `StockTradingEnv` instance. Not persisted — created at runtime from `MarketDataset`.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `df` | `pd.DataFrame` | — | MarketDataset slice (train or test) |
| `stock_dim` | `int` | `len(universe)` | Number of stocks N |
| `hmax` | `int` | `100` | Max shares per trade action |
| `initial_amount` | `float` | `1_000_000` | Initial cash (RMB) |
| `buy_cost_pct` | `list[float]` | `[0.001] * N` | 0.1% commission per stock |
| `sell_cost_pct` | `list[float]` | `[0.001] * N` | 0.1% commission + stamp duty equivalent |
| `reward_scaling` | `float` | `1e-4` | Scales portfolio value change to reward |
| `tech_indicator_list` | `list[str]` | 7-indicator list | Must match MarketDataset columns |

**State transitions** (per trading step):
1. `RESET` → initial cash + zero positions
2. `STEP(action)` → compute new positions → update portfolio value → compute reward
3. `TERMINAL` → episode ends at last trading date

---

## Entity 4: RLAgent（强化学习智能体）

**Description**: Trained policy network. Persisted as stable-baselines3 model file.

| Field | Type | Notes |
|-------|------|-------|
| `algorithm` | `str` | `ppo` \| `sac` \| `td3` |
| `model_path` | `str` | `models/{algorithm}_{date}_{universe_hash}/` |
| `train_start` | `str` | Training period start date |
| `train_end` | `str` | Training period end date |
| `hyperparams` | `dict` | Sourced from `config.yaml::training.{algorithm}` |
| `total_timesteps` | `int` | Total env steps during training |

**State transitions**:
- `INITIALIZED` → `TRAINING` (on `agent.train()`) → `TRAINED` (model saved)
- `TRAINED` → `BACKTESTING` (on `agent.predict()`) → `EVALUATED`

**Storage**: `models/{algorithm}_{YYYYMMDD}_{hash}/` (zip file from SB3 `model.save()`)

---

## Entity 5: SentimentRecord（舆情记录）

**Description**: Single financial text analysis result from Qwen2.5-7B-Instruct.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `date` | `str` | NOT NULL, format `YYYY-MM-DD` | Publication date (used for daily join) |
| `tic` | `str` | NOT NULL, FK → StockUniverse.tic | Target stock ticker |
| `source` | `str` | NOT NULL | e.g. `eastmoney`, `sina`, `user_file` |
| `text_hash` | `str` | NOT NULL, MD5 | Deduplication key for input text |
| `sentiment_score` | `float` | NOT NULL, range [-1.0, 1.0] | −1=very negative, 0=neutral, +1=very positive |
| `event_tags` | `list[str]` | NOT NULL, may be empty `[]` | e.g. `["业绩超预期", "监管风险"]` |
| `summary` | `str` | NOT NULL, max 200 chars | Qwen-generated Chinese summary |
| `model_id` | `str` | NOT NULL | e.g. `Qwen/Qwen2.5-7B-Instruct` |

**Daily aggregation** (for feature fusion):
- `daily_sentiment_score = mean(sentiment_score)` for all records on same `(date, tic)`
- `daily_event_count = len(event_tags aggregated)` across all records

**Storage**: `data/sentiment/{tic}_{start_date}_{end_date}.jsonl`

---

## Entity 6: FundamentalFeature（基本面特征）

**Description**: Structured fundamental indicators extracted from financial report text.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `date` | `str` | NOT NULL, format `YYYY-MM-DD` | Report disclosure date |
| `tic` | `str` | NOT NULL, FK → StockUniverse.tic | Target stock |
| `report_type` | `str` | enum `['annual', 'semi', 'q1', 'q3']` | Report period |
| `revenue_growth_pct` | `float` | OPTIONAL | YoY revenue growth rate (%) |
| `net_profit_margin` | `float` | OPTIONAL | Net profit / Revenue |
| `debt_ratio` | `float` | OPTIONAL | Total debt / Total assets |
| `roe` | `float` | OPTIONAL | Return on equity |
| `pe_qualitative` | `str` | OPTIONAL | Descriptive PE range: e.g. `"undervalued"` |
| `extraction_confidence` | `float` | NOT NULL, range [0, 1] | Qwen confidence score for extraction |

**Join logic**: Left-joined to MarketDataset on `(tic)`; a single quarterly report applies to all trading days in that quarter. Missing quarters → forward-fill from last known.

**Storage**: `data/fundamentals/{tic}.jsonl`

---

## Entity 7: EnhancedDataset（增强数据集）

**Description**: Wide-table merge of MarketDataset + daily sentiment + fundamental features. Direct input to enhanced FinRL training.

**Schema**: All columns from MarketDataset PLUS:

| Column | Type | Source | Fill rule if missing |
|--------|------|--------|---------------------|
| `sentiment_score` | `float` | SentimentRecord | `0.0` (neutral) |
| `event_count` | `int` | SentimentRecord | `0` |
| `has_positive_event` | `int` | SentimentRecord | `0` (0/1 flag) |
| `has_negative_event` | `int` | SentimentRecord | `0` (0/1 flag) |
| `revenue_growth_pct` | `float` | FundamentalFeature | `0.0` |
| `net_profit_margin` | `float` | FundamentalFeature | `0.0` |
| `debt_ratio` | `float` | FundamentalFeature | `0.5` (neutral default) |

**Invariants**:
- Primary key: `(date, tic)` — same as MarketDataset
- Row count MUST equal MarketDataset row count (left join, no new rows)
- No `NaN` permitted (all fills applied before saving)

**Derived property**: `observation_space_dim = 1 + (9 + 7) * N = 1 + 16*N`  
For N=30: state_space = 481 vs baseline 271.

**Storage**: `data/enhanced/{start_date}_{end_date}_{universe_hash}_enhanced.parquet`

---

## Entity 8: BacktestReport（回测报告）

**Description**: Complete performance record for a single model evaluation run.

| Field | Type | Notes |
|-------|------|-------|
| `model_path` | `str` | FK → RLAgent.model_path |
| `test_start` | `str` | Test period start |
| `test_end` | `str` | Test period end |
| `universe_hash` | `str` | Stock universe fingerprint |
| `dataset_type` | `str` | `baseline` \| `enhanced` |
| `sharpe_ratio` | `float` | Annualized Sharpe (risk-free = 0) |
| `annual_return_pct` | `float` | Annualized return (%) |
| `max_drawdown_pct` | `float` | Maximum drawdown (%) |
| `cumulative_return_pct` | `float` | Total return over test period |
| `daily_portfolio_value` | `pd.Series` | Indexed by date; daily NAV in RMB |
| `trade_log` | `pd.DataFrame` | Columns: date, tic, action, shares, price |

**Storage**:
- Summary JSON: `reports/{run_id}/metrics.json`
- HTML report: `reports/{run_id}/report.html` (plotly charts)
- Trade log CSV: `reports/{run_id}/trades.csv`
