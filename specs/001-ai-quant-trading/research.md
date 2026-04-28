# Research: A股AI量化交易系统技术选型决策

**Feature**: `001-ai-quant-trading`  
**Phase**: 0 — Pre-Design Research  
**Date**: 2026-04-28  
**Status**: Complete — all NEEDS CLARIFICATION items resolved

---

## Decision 1: Technical Indicator Default Set (NC-001)

**Decision**: Default indicator set = `['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma']` (7 indicators, matches `finrl.config.INDICATORS` canonical default). User can override via `config.yaml::indicators` field.

**Rationale**: FinRL's official examples and `finrl.config` module ship this exact set. Using it avoids `observation_space` mismatch between FinRL environment construction and data preprocessing. The 7-indicator set provides momentum (MACD, RSI), volatility (Bollinger Bands), trend (DX, SMA 30/60) coverage — standard for baseline A-share RL trading tasks.

**Alternatives considered**:
- `ta-lib` with custom 15-indicator set: rejected because `ta-lib` requires C compilation, adds OS-level dependency; stockstats is pure Python.
- `pandas-ta` with 20 indicators: rejected because wider feature space without validation inflates observation_space and risks slow convergence; not standard in FinRL examples.

**Observation space formula (with 7 indicators, N stocks)**:
```
state_space = 1 + 2*N + 7*N  =  1 + 9*N
```
For N=30 stocks: state_space = 271.

---

## Decision 2: News / Announcement Text Input Source (NC-002)

**Decision**: Sentiment module accepts user-provided JSONL files. Each line: `{"date": "YYYY-MM-DD", "tic": "000001.SZ", "text": "新闻正文"}`. File path configured via `config.yaml::sentiment.news_file`. Direct URL scraping is NOT in scope per Assumption 5 in spec.

**Rationale**: Spec Assumption 5 explicitly states "text collection is not in scope." Requiring a defined JSONL contract keeps the sentiment module self-contained, testable in isolation with fixture files, and decoupled from scraping infrastructure that varies per data provider.

**Alternatives considered**:
- Real-time HTTP scraping (东方财富 API, Sina Finance RSS): rejected for this spec — introduces authentication, rate limiting, and Terms-of-Service complexity that requires a separate feature.
- Pre-bundled financial news dataset (e.g., CCF-BDCI): rejected — distributing third-party financial data is a licensing risk.

---

## Decision 3: Live Trading API Reservation (NC-003)

**Decision**: Live trading is NOT in scope for this spec. The `config.yaml` includes a reserved key `live_trading: false`. xtquant is used for data download only (via `xtdata.get_market_data_ex()`). A placeholder `LiveTradingAdapter` interface stub is documented in contracts but not implemented. This becomes `002-live-trading` if needed.

**Rationale**: Spec Assumption 4 confirms "A-share data is for historical backtesting only." Implementing a live order execution path requires QMT account setup, risk controls, and regulatory awareness that are separate concerns from model training and research.

---

## Decision 4: FinRL Package Version and Environment Class

**Decision**: Use `finrl>=0.3.7` (PyPI: `finrl`). Environment class: `finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv`. RL algorithms via `finrl.agents.stablebaselines3.models.DRLAgent`. Use date-based train/test split via `finrl.meta.preprocessor.preprocessors.data_split()`.

**Rationale**: Version 0.3.7 is current stable as of April 2026. The `DRLAgent.get_model("ppo"|"sac"|"td3")` API abstracts stable-baselines3 and is the official FinRL interface for algorithm selection. Date-based splitting (not random) is mandatory for time-series financial data to avoid look-ahead bias.

**Alternatives considered**:
- FinRL-X / FinRL-Trading (next-gen, not yet stable for research workflows): deferred for future evaluation once stable.
- Direct stable-baselines3 without FinRL wrapper: rejected — adds boilerplate for gym environment setup that FinRL already handles correctly.

---

## Decision 5: Qwen Model for Sentiment Analysis

**Decision**: `Qwen/Qwen2.5-7B-Instruct` (HuggingFace). Load with bitsandbytes NF4 4-bit quantization (`bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=bfloat16`). Structured JSON output via prompt engineering (system prompt enforces JSON schema).

**Rationale**: Qwen2.5-7B-Instruct has 131K context window, supports Chinese financial text, runs on 8 GB VRAM with 4-bit quantization. No official "Qwen-Finance" variant exists as of April 2026 — the 7B-Instruct general model with prompt engineering is the community-recommended approach for financial sentiment.

**Alternatives considered**:
- Qwen2.5-32B-Instruct: rejected for baseline — requires 20+ GB VRAM without quantization; use as stretch goal after baseline validation.
- FinBERT (Chinese): rejected — classification-only (3 classes), cannot extract event_tags or summaries.
- GPT-4o API: rejected — requires cloud API key, adds cost and latency; against open-source requirement.

---

## Decision 6: Fallback Data Interface

**Decision**: `akshare` as primary fallback. Function: `ak.stock_zh_a_hist(symbol=..., period="daily", ...)`. akshare returns Chinese-language column names — preprocessing layer renames to FinRL format. xtquant uses `SZ/SH` prefix codes; akshare uses bare 6-digit codes; preprocessing normalizes to FinRL's `000001.SZ` dot-notation.

**Rationale**: akshare is actively maintained (18k+ GitHub stars, weekly releases as of April 2026), requires no authentication, and covers A-shares broadly. baostock requires login and is less actively maintained.

**Code format normalization**:
```
xtquant:  SZ000001  →  finrl: 000001.SZ
akshare:  000001    →  finrl: 000001.SZ  (append .SZ/.SH based on prefix range)
```
(Shanghai stocks: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx → `.SH`; rest → `.SZ`)

---

## Decision 7: Testing Framework and Strategy

**Decision**: `pytest` with `pytest-cov` for coverage, `pytest-mock` for mocking external interfaces. Test structure: `tests/unit/`, `tests/integration/`, `tests/contract/`. Minimum coverage target: 80% for `src/data/` and `src/features/` modules. FinRL environment and Qwen model interactions mocked in unit tests via fixtures.

**Rationale**: Matches constitution Principle I (test-first). Mocking xtquant and akshare HTTP calls avoids flaky tests dependent on live market data. Integration tests use small fixture datasets (5 stocks, 60 days) to run FinRL training in < 30 seconds.
