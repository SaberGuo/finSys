# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                     finSys CLI                          │
│  finsys data fetch | train | backtest | sentiment       │
│          analyze | fuse | compare | run                 │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐
    │   Data     │ │  Training  │ │    NLP     │
    │  Pipeline  │ │  Pipeline  │ │  Pipeline  │
    └─────┬──────┘ └─────┬──────┘ └────┬───────┘
          │              │              │
    ┌─────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐
    │ DataSource │ │   FinRL    │ │   Qwen     │
    │ (failover) │ │ + SB3 Env  │ │ 7B 4-bit   │
    └─────┬──────┘ └─────┬──────┘ └────┬───────┘
          │              │              │
     xtquant        StockTrading   SentimentRecord
     akshare         Env / PPO /
     baostock        SAC / TD3
          │              │              │
          └──────────────┼──────────────┘
                         │
                  ┌──────▼──────┐
                  │   Feature   │
                  │   Fusion    │
                  │  (US4 join) │
                  └──────┬──────┘
                         │
                  EnhancedDataset
                  (Parquet file)
```

## Module Map

| Module | Purpose |
|--------|---------|
| `finquant/config/settings.py` | Pydantic Settings — YAML + env-var config loader |
| `finquant/data/sources/` | xtquant / akshare / baostock adapters with failover |
| `finquant/data/pipeline.py` | Orchestrate source failover → preprocess → Parquet |
| `finquant/data/preprocessor.py` | Normalize, fill suspensions, validate FinRL schema |
| `finquant/features/technical.py` | stockstats MACD / RSI / BOLL / SMA indicators |
| `finquant/features/sentiment.py` | SentimentRecord + SentimentProcessor (batch Qwen) |
| `finquant/features/fusion.py` | Left-join MarketDataset + SentimentRecord |
| `finquant/training/env.py` | FinRL StockTradingEnv wrapper + obs_dim validation |
| `finquant/training/trainer.py` | DRLAgent train/backtest lifecycle |
| `finquant/training/backtest.py` | Sharpe / CAGR / MDD + HTML/CSV report |
| `finquant/nlp/model.py` | Qwen2.5-7B lazy loader with 4-bit BitsAndBytes |
| `finquant/nlp/analyzer.py` | Prompt templates + JSON response parsing |
| `finquant/cli/main.py` | Click CLI entry point (all commands) |
| `finquant/utils/logging.py` | Structured JSON logger |
| `finquant/utils/retry.py` | Exponential backoff decorator |

## Data Flow

```
config.yaml
    │
    ▼
AppConfig (Pydantic)
    │
    ├── DataPipeline.fetch_and_save()
    │       xtquant → akshare → baostock (failover)
    │       → preprocess_market_data()
    │       → compute_indicators()
    │       → MarketDataset.parquet
    │
    ├── SentimentProcessor.process_file()
    │       QwenModel.load() [4-bit NF4]
    │       SentimentAnalyzer.analyze(text)
    │       → SentimentRecord[]
    │       → sentiment_records.jsonl
    │
    ├── fuse_datasets()
    │       MarketDataset ⟕ SentimentRecord on (date, tic)
    │       → EnhancedDataset.parquet
    │
    └── Trainer.train()
            build_env(df, stock_dim=N)  obs_dim = 1+9N or 1+16N
            DRLAgent(PPO/SAC/TD3)
            → model.zip
            Trainer.backtest()
            → BacktestReport (Sharpe, CAGR, MDD)
            → report.html + metrics.csv
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 3-source data failover | xtquant requires paid QMT auth unavailable in CI |
| FinRL + SB3 framework | Battle-tested RL; avoids hand-rolling PPO/SAC/TD3 |
| Qwen 4-bit quantization | 8 GB VRAM constraint on consumer GPUs |
| Pydantic Settings | Type-safe config; catches malformed values at startup |
| Left join for fusion | Preserves market data integrity; no rows added/removed |
| Graceful NLP degradation | GPU OOM or model failure → neutral score, no crash |

## Deployment

Single-node workstation deployment:
- Python 3.10+ virtualenv
- CUDA GPU ≥ 8 GB VRAM recommended (for Qwen inference)
- No external services required (all local filesystem)
- No real-time trading (historical/research only)

See [quickstart.md](specs/001-ai-quant-trading/quickstart.md) for setup instructions.
