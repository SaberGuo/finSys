# Implementation Plan: A股AI量化交易系统（数据采集 + FinRL训练 + 舆情基本面融合）

**Branch**: `001-ai-quant-trading` | **Date**: 2026-04-28 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/001-ai-quant-trading/spec.md`

## Summary

Build a modular Python system for A-share quantitative trading research. xtquant (with akshare/baostock failover) provides data acquisition; FinRL (≥ 0.3.x, gymnasium-based) provides the reinforcement learning environment and algorithm layer (PPO/SAC/TD3); Qwen2.5-7B-Instruct (4-bit quantized) provides Chinese-language sentiment and fundamental feature extraction. The four delivery slices are P1 data pipeline → P2 FinRL training → P3 Qwen sentiment → P4 enhanced fusion training, each independently testable and deliverable. All configuration is file-driven (YAML); no source-code modification required to run.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: `finrl ≥ 0.3.6`, `stable-baselines3 ≥ 2.0`, `gymnasium`, `xtquant` (xtquant QMT), `akshare`, `baostock`, `transformers ≥ 4.40`, `torch ≥ 2.0`, `bitsandbytes`, `pandas`, `numpy`, `pyarrow`, `pydantic-settings`, `click`, `pyyaml`, `matplotlib`, `pyecharts`  
**Storage**: Local filesystem — Parquet for datasets (raw + processed), `.zip` for SB3 model weights, HTML/CSV for backtest reports; SQLite optional for metadata caching  
**Testing**: `pytest`, `pytest-cov` (≥ 80% line coverage on `finquant/`), `pytest-mock`; integration tests gated behind `RUN_INTEGRATION=1` env var  
**Target Platform**: Linux/Windows workstation; CUDA GPU ≥ 8 GB VRAM recommended; CPU-only fallback for data + training (slower)  
**Project Type**: Modular Python application (CLI-driven, config-file controlled)  
**Performance Goals**: Data acquisition for 50 stocks × 3 years ≤ 10 min (SC-001); FinRL single-GPU training run ≤ 30 min; Qwen 7B 4-bit inference ≤ 5 sec/text  
**Constraints**: Qwen runs in ≤ 8 GB VRAM via 4-bit (bitsandbytes); no hardcoded credentials (env vars or file path only); no real-time trading (historical/research only)  
**Scale/Scope**: 50–500 A-share stocks; 1–10 years historical data; single-node execution

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Test-first strategy is explicit**: Each story (P1–P4) has an Independent Test definition in spec.md. TS-001~TS-004 specify unit/contract/integration layers with failing-first sequence. pytest coverage ≥ 80% required on `finquant/`.
- [x] **Plan is executable**: Four priority slices (P1→P2→P3→P4) with explicit dependency chain; file-level targets listed in Project Structure above; measurable done criteria map to SC-001~SC-006.
- [x] **Requirement traceability exists**: FR-001~FR-012 each trace to acceptance scenarios in spec.md; FR-011 mandates executable decomposition; FR-012 mandates requirement-to-task traceability.
- [x] **Integration risks are identified**: (1) xtquant→akshare→baostock failover; (2) FinRL `observation_space` dimension mismatch on feature addition; (3) Qwen load failure → neutral-value degradation; (4) date/tic alignment NaN handling in fusion. Each has explicit contract/integration test coverage in `tests/contract/` and `tests/integration/`.
- [x] **Complexity is justified**: Qwen LLM additive (P3 non-blocking, research.md §Qwen); FinRL+SB3 vs hand-rolled RL (research.md §FinRL); 3-source failover required for CI (research.md §Failover); Pydantic Settings vs raw dict (Complexity Tracking above).

## Project Structure

### Documentation (this feature)

```text
specs/001-ai-quant-trading/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── cli-schema.md    # CLI command contracts
│   └── config-schema.md # config.yaml field contracts
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
finquant/
├── config/
│   ├── __init__.py
│   └── settings.py          # Pydantic Settings loader (config.yaml + env vars)
├── data/
│   ├── __init__.py
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract DataSource interface (download, validate)
│   │   ├── xtquant.py       # xtquant QMT adapter
│   │   ├── akshare.py       # akshare fallback adapter
│   │   └── baostock.py      # baostock fallback adapter
│   ├── pipeline.py          # Orchestrate failover download + preprocess
│   └── preprocessor.py      # Clean, align, standardize to FinRL schema
├── features/
│   ├── __init__.py
│   ├── technical.py         # MACD, RSI, BOLL, CCI, DX, SMA indicators
│   ├── sentiment.py         # Invoke Qwen, return SentimentRecord list
│   └── fusion.py            # Left-join MarketDataset + SentimentRecord by (date, tic)
├── training/
│   ├── __init__.py
│   ├── env.py               # Wrap FinRL StockTradingEnv; validate obs_space dims
│   ├── trainer.py           # Configure + run FinRL agent (PPO/SAC/TD3)
│   └── backtest.py          # Run backtest; compute sharpe, CAGR, MDD; export report
├── nlp/
│   ├── __init__.py
│   ├── model.py             # Qwen loader (4-bit BitsAndBytesConfig)
│   └── analyzer.py          # Sentiment + fundamental extraction prompt templates
├── cli/
│   ├── __init__.py
│   └── main.py              # Click CLI: fetch / train / analyze / fuse / report
└── utils/
    ├── __init__.py
    ├── logging.py           # Structured JSON logger
    └── retry.py             # Exponential backoff decorator for API calls

tests/
├── conftest.py              # Fixtures: sample_dataset, mock_data_source, sample_texts
├── unit/
│   ├── test_preprocessor.py
│   ├── test_technical.py
│   ├── test_sentiment.py
│   └── test_fusion.py
├── contract/
│   ├── test_datasource_contract.py   # All adapters satisfy base.DataSource interface
│   └── test_finrl_env_contract.py    # obs_space dims match dataset columns
└── integration/
    ├── test_data_pipeline.py         # download → preprocess → FinRL-loadable (opt-in)
    ├── test_training_pipeline.py     # dataset → train → backtest → report (opt-in)
    └── test_nlp_pipeline.py          # text → sentiment → SentimentRecord (opt-in)

config.yaml.example          # Template config (no secrets)
pyproject.toml               # Build + pinned dependencies
requirements.txt             # Pip-installable locked deps
README.md
```

**Structure Decision**: Single modular Python project (`finquant/` package). No monorepo split — all components are research-phase and co-located to simplify dependency management and testing. `cli/main.py` is the user-facing entry point; all other modules are importable independently for notebook/script use.

## Complexity Tracking

| Item | Why Needed | Simpler Alternative Rejected Because |
|------|-----------|--------------------------------------|
| 3-source data failover (xtquant→akshare→baostock) | xtquant requires paid QMT auth unavailable in CI/test environments (FR-002) | Single source breaks automated testing without credentials |
| Qwen 4-bit quantization via bitsandbytes | 8 GB VRAM constraint on consumer GPUs (spec Assumption 3) | Full-precision Qwen-7B requires ≥ 14 GB VRAM; eliminates most dev machines |
| FinRL + SB3 framework | Battle-tested RL for multi-asset portfolio; avoids hand-rolling PPO/SAC/TD3 | Hand-rolled RL would require 3× implementation effort for same quality and reproducibility |
| Pydantic Settings config loader | Type-safe YAML + env var config; catches malformed values at startup | Plain `yaml.safe_load()` dict access silently passes wrong types into training |
