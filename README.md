# finSys — A股AI量化交易系统

A-share quantitative trading research pipeline using **FinRL** reinforcement learning and optional **Qwen2.5-7B-Instruct** sentiment analysis.

## Features

- **Data Pipeline** — Download A-share historical data via xtquant / akshare / baostock with automatic failover
- **RL Training** — Train PPO / SAC / TD3 agents using FinRL + stable-baselines3
- **Backtest Reports** — Sharpe ratio, CAGR, max drawdown with HTML + CSV export
- **Sentiment Analysis** — Qwen 7B 4-bit quantized inference for Chinese financial text
- **Feature Fusion** — Merge market data + sentiment + fundamentals for enhanced training

## Quickstart (5 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit config
cp config/default.yaml.example config/my_config.yaml
# Edit stocks, dates, and data source

# 3. Fetch data
finsys data fetch --config config/my_config.yaml

# 4. Train model
finsys train --config config/my_config.yaml --algo ppo

# 5. Backtest
finsys backtest --config config/my_config.yaml \
  --model models/ppo_YYYYMMDD_xxxxxx/ \
  --data data/processed/xxx.parquet
```

See [`specs/001-ai-quant-trading/quickstart.md`](specs/001-ai-quant-trading/quickstart.md) for detailed instructions.

## Project Structure

```
finquant/
├── config/settings.py      # Pydantic YAML + env-var config loader
├── data/sources/           # xtquant / akshare / baostock adapters
├── data/pipeline.py        # Failover orchestration
├── features/               # Technical indicators, sentiment, fusion
├── training/               # FinRL env, trainer, backtest reports
├── nlp/                    # Qwen 4-bit loader + analyzer
├── cli/main.py             # Click CLI entry point
└── utils/                  # Structured JSON logging, retry decorator

tests/
├── unit/                   # 54 unit tests
├── contract/               # 15 interface contract tests
└── integration/            # 8 end-to-end integration tests (opt-in)
```

## Tech Stack

| Component | Library |
|-----------|---------|
| RL Framework | FinRL ≥ 0.3.7, stable-baselines3 ≥ 2.0 |
| Data Sources | xtquant, akshare, baostock |
| Indicators | stockstats |
| LLM | transformers, torch, bitsandbytes (4-bit NF4) |
| Config | pydantic-settings, pyyaml |
| CLI | click |
| Testing | pytest, pytest-cov, pytest-mock |

## Testing

```bash
# Unit + contract tests (coverage ≥ 80%)
pytest tests/unit tests/contract

# Integration tests (requires external data / GPU)
RUN_INTEGRATION=1 pytest tests/integration
```

## Documentation

- [`docs/data-pipeline.md`](docs/data-pipeline.md) — Data fetch & preprocessing guide
- [`docs/training-guide.md`](docs/training-guide.md) — Training & backtest guide
- [`docs/sentiment-guide.md`](docs/sentiment-guide.md) — Qwen setup & VRAM requirements
- [`docs/fusion-guide.md`](docs/fusion-guide.md) — Feature fusion workflow
- [`docs/FAQ.md`](docs/FAQ.md) — Troubleshooting
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — System design & data flow

## Requirements

- Python ≥ 3.10
- CUDA ≥ 11.8 (optional, for GPU training / Qwen inference)
- ≥ 8 GB VRAM (for Qwen 7B 4-bit sentiment analysis)
- ≥ 5 GB disk space

## License

MIT


/speckit-specify 可根据不同情况先进行因子选择，如：上涨市场中，动量、高Beta和成长因子往往表现较好，市场关注"上涨弹性"；下跌或震荡市场中，低波动、低换手率、高股息等防御性因子通常更稳健；结构化行情扩大时，小市值因子超额收益显著，但基本面因子可能阶段性失灵。
按因子类别：小市值因子尤其在结构化行情扩大后表现出色；反转因子在各种极端情景中相对稳定；基本面因子（ROE、SUE）在市场情绪谨慎时效果最突出；低波、低换手率因子在市场成交萎缩后选股效果也会随之减弱。
在A股，高股息和低波因子长期表现一贯领先，有异于通常由动量因子领先的全球惯例。2025年全年，成长、低波、小市值因子整体表现较好，盈利、流动性因子一般，Beta和动量因子波动较大，呈现"先抑后扬"走势，反映出因子表现随行情切换的动态特征。选择因子后，再调用RL进行交易决策。