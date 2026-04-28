# Tasks: A股AI量化交易系统（数据采集 + FinRL训练 + 舆情基本面融合）

**Input**: Design documents from `specs/001-ai-quant-trading/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Organization**: Tasks grouped by 4 user stories (P1→P2→P3→P4) + Setup + Foundational + Polish phases.  
**Testing**: Every user story has failing-first test tasks before implementation tasks.

## Format: `[ID] [P?] [Story] Description with file path`

- **[P]**: Can run in parallel (different files/modules, no dependencies on incomplete tasks)
- **[Story]**: Which user story (US1, US2, US3, US4; empty for Setup/Foundational/Polish phases)
- **File paths**: All paths are workspace-relative from repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project structure, dependencies, test framework

- [X] T001 Initialize Python project structure (`finquant/` package, `tests/` dir, `config/` dir)
- [X] T002 Create `requirements.txt` with all dependencies pinned (finrl, stable-baselines3, gymnasium, xtquant, akshare, baostock, transformers, torch, bitsandbytes, pandas, pyarrow, pydantic-settings, click, pyyaml, pytest, pytest-cov, pytest-mock, matplotlib, pyecharts, etc.)
- [X] T003 [P] Create `setup.py` or `pyproject.toml` for package build configuration
- [X] T004 [P] Configure `pytest.ini` with coverage settings (min 80% on `finquant/`, exclude `cli/` entry-point, exclude `nlp/model.py` GPU loaders)
- [X] T005 [P] Setup `.github/workflows/ci.yml` for automated testing (pytest unit + contract tests; integration tests opt-in via `RUN_INTEGRATION=1`)
- [X] T006 Create `config/default.yaml.example` template with all configurable fields (stocks, dates, data source, training hyperparams, etc.)

**Checkpoint**: Project structure ready; dependencies installed; test framework validates

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core abstractions required by all user stories

- [X] T007 Implement `finquant/config/settings.py` (Pydantic Settings loader for config.yaml + environment variables)
- [X] T008 [P] Implement `finquant/utils/logging.py` (structured JSON logger for all modules)
- [X] T009 [P] Implement `finquant/utils/retry.py` (exponential backoff decorator for API calls, max 3 retries)
- [X] T010 Implement `finquant/data/sources/base.py` (abstract `DataSource` interface with `download()` and `validate()` methods; target schema: date, tic, open, high, low, close, volume + technical indicators)
- [X] T011 Create `tests/conftest.py` with shared pytest fixtures (sample_dataset, mock_data_source, sample_texts, etc.)
- [X] T012 [P] Setup `tests/unit/`, `tests/contract/`, `tests/integration/` directories

**Checkpoint**: Config loader, logging, retry logic, base interfaces ready; test fixtures available

---

## Phase 3: User Story 1 - 数据采集与预处理管道 (Priority: P1) 🎯 MVP

**Goal**: Download A-share historical data via xtquant/akshare/baostock and preprocess into FinRL-compatible format.

**Independent Test**: Run data pipeline on 5 stocks × 60 days; verify output Parquet file has correct schema (date, tic, OHLCV + 7 technical indicators), no NaN, correct date order, FinRL StockTradingEnv can load it.

### Tests for User Story 1 (REQUIRED — write first)

- [X] T013 [P] [US1] Unit test for data cleaning function (fill_missing, remove_duplicates, validate_tic_format) in `tests/unit/test_preprocessor.py`
- [X] T014 [P] [US1] Unit test for technical indicator computation (MACD, Bollinger, RSI, DX, SMA) in `tests/unit/test_technical.py`
- [X] T015 [P] [US1] Contract test for DataSource interface (xtquant, akshare, baostock adapters all satisfy base.DataSource contract) in `tests/contract/test_datasource_contract.py`
- [X] T016 [US1] Integration test for end-to-end data pipeline (download → preprocess → validate FinRL env load) in `tests/integration/test_data_pipeline.py` (opt-in via `RUN_INTEGRATION=1`)

### Implementation for User Story 1

- [X] T017 [P] [US1] Implement `finquant/data/sources/xtquant.py` (XtquantDataSource adapter; download via xtquant QMT API with error handling)
- [X] T018 [P] [US1] Implement `finquant/data/sources/akshare.py` (AkshareDataSource adapter; download via akshare API, normalize column names to FinRL schema)
- [X] T019 [P] [US1] Implement `finquant/data/sources/baostock.py` (BaostockDataSource adapter; fallback interface with no-auth mode)
- [X] T020 [US1] Implement `finquant/data/pipeline.py` (orchestrate failover: xtquant → akshare → baostock; emit structured logs on source switch)
- [X] T021 [P] [US1] Implement `finquant/data/preprocessor.py` (clean NaN, fill forward on suspension days, align tic codes to `XXXXXX.SH/.SZ` format, validate no duplicates)
- [X] T022 [P] [US1] Implement `finquant/features/technical.py` (compute 7 indicators: MACD, boll_ub, boll_lb, rsi_30, dx_30, close_30_sma, close_60_sma using stockstats)
- [X] T023 [US1] Implement `finquant/cli/main.py` command group `data fetch` (CLI entry point: `finsys data fetch --config config.yaml --output data/processed/`)
- [X] T024 [US1] Write integration test for data failover logic (mock xtquant failure, verify fallback to akshare succeeds) in `tests/integration/test_data_pipeline.py`
- [X] T025 [US1] Create `docs/data-pipeline.md` (quickstart for data fetch; expected output schema; common errors)

**Checkpoint**: US1 complete — 50 stocks × 3 years ≤ 10 min (SC-001); output Parquet ready for FinRL training

---

## Phase 4: User Story 2 - FinRL 强化学习模型训练与回测 (Priority: P2)

**Goal**: Train FinRL RL agent (PPO/SAC/TD3) on baseline market dataset; produce backtest report with Sharpe, annual return, max drawdown.

**Independent Test**: Load P1 Parquet dataset; train 10k steps (fast baseline); verify backtest report contains all 3 metrics, daily NAV curve, trade log CSV.

### Tests for User Story 2 (REQUIRED — write first)

- [X] T026 [P] [US2] Unit test for FinRL env wrapper (observation_space dims match dataset, reward scaling, action bounds) in `tests/unit/test_env.py`
- [X] T027 [P] [US2] Unit test for backtest report generation (metrics calculation: Sharpe, CAGR, MDD) in `tests/unit/test_backtest.py`
- [X] T028 [P] [US2] Contract test for TrainingEnv (all RL agents can accept same obs/action space) in `tests/contract/test_finrl_env_contract.py`
- [X] T029 [US2] Integration test for full training pipeline (load data → train PPO → backtest on test set) in `tests/integration/test_training_pipeline.py` (opt-in)

### Implementation for User Story 2

- [X] T030 [P] [US2] Implement `finquant/training/env.py` (wrap FinRL StockTradingEnv; validate obs_space dims = 1 + 9*N; set initial_amount=1M RMB, commissions 0.1%, reward_scaling 1e-4)
- [X] T031 [P] [US2] Implement `finquant/training/trainer.py` (configure DRLAgent for PPO/SAC/TD3; load hyperparams from config.yaml; run training loop with epoch logging)
- [X] T032 [P] [US2] Implement `finquant/training/backtest.py` (run model on test data; compute Sharpe (annualized), CAGR, MDD; generate daily NAV Series, trade log DataFrame, metrics JSON)
- [X] T033 [US2] Implement `finquant/cli/main.py` command `train` (CLI: `finsys train --config config.yaml --algo ppo --output models/`)
- [X] T034 [US2] Implement `finquant/cli/main.py` command `backtest` (CLI: `finsys backtest --model models/ppo_xxx/ --data data/processed/xxx.parquet --output reports/`)
- [X] T035 [P] [US2] Implement HTML report generation in `finquant/training/backtest.py` (pyecharts: net value curve vs CSI 300, trade markers, metrics table)
- [X] T036 [P] [US2] Implement CSV export (daily NAV, trade log) in `finquant/training/backtest.py`
- [X] T037 [US2] Add model checkpoint saving (stable-baselines3 `.zip` format to `models/{algo}_{date}_{hash}/`)
- [X] T038 [US2] Write validation that baseline Sharpe ≥ 0.5 in backtest report (SC-002 test)
- [X] T039 [US2] Create `docs/training-guide.md` (quickstart for `finsys train` and `finsys backtest`; expected metrics; hyperparameter tuning tips)

**Checkpoint**: US2 complete — baseline model trainable; backtest reports generated with Sharpe ≥ 0.5 (SC-002); reports exportable as HTML + CSV

---

## Phase 5: User Story 3 - Qwen 舆情分析与基本面信息提取 (Priority: P3)

**Goal**: Load Qwen2.5-7B-Instruct 4-bit; analyze financial texts; extract sentiment_score ∈ [-1, 1] + event_tags + summary. Independent from US1+US2 execution.

**Independent Test**: Input 10 sample financial news texts with known sentiment; run sentiment analysis; verify output JSON format correct, sentiment_score in range, event_tags list non-empty for positive news, accuracy ≥ 80%.

### Tests for User Story 3 (REQUIRED — write first)

- [X] T040 [P] [US3] Unit test for Qwen model loader (4-bit NF4 quantization config, device selection) in `tests/unit/test_nlp_model.py`
- [X] T040b [P] [US3] Unit test for sentiment prompt template and parsing (extract score, tags, summary from Qwen JSON output) in `tests/unit/test_sentiment.py`
- [X] T041 [P] [US3] Integration test for sentiment analysis pipeline (load Qwen, input sample texts, verify output schema) in `tests/integration/test_nlp_pipeline.py` (opt-in, requires GPU)

### Implementation for User Story 3

- [X] T042 [P] [US3] Implement `finquant/nlp/model.py` (Qwen2.5-7B-Instruct loader with BitsAndBytesConfig 4-bit NF4 quantization; device fallback CPU if GPU unavailable)
- [X] T043 [P] [US3] Implement `finquant/nlp/analyzer.py` (sentiment extraction: prompt template, JSON output parsing, sentiment_score [-1, 1], event_tags list, summary max 200 chars)
- [X] T044 [P] [US3] Implement `finquant/features/sentiment.py` (SentimentRecord class, batch processing JSONL input files, deduplication by text_hash)
- [X] T045 [US3] Implement `finquant/cli/main.py` command `sentiment analyze` (CLI: `finsys sentiment analyze --input data/news.jsonl --output data/sentiment/result.jsonl`)
- [X] T046 [US3] Add error handling for Qwen load failure → graceful degradation (log warning, return neutral sentiment 0.0) per spec Edge Case
- [X] T047 [P] [US3] Create sample financial texts fixture (100+ labeled texts for accuracy validation against SC-003)
- [X] T048 [US3] Validate sentiment accuracy ≥ 80% on fixture texts (SC-003 test); record metrics in test report
- [X] T049 [US3] Create `docs/sentiment-guide.md` (Qwen 4-bit setup; VRAM requirements; sample JSONL format; accuracy metrics)

**Checkpoint**: US3 complete — Qwen sentiment extraction functional; 80%+ accuracy on test set (SC-003); can run independently of training

---

## Phase 6: User Story 4 - 舆情基本面特征融合与增强训练 (Priority: P4)

**Goal**: Merge MarketDataset + SentimentRecord to create EnhancedDataset (obs_space 1 + 16*N); train enhanced model; compare Sharpe gain vs baseline.

**Independent Test**: Fuse P1 dataset + P3 sentiment records; verify output Parquet has all sentiment columns (sentiment_score, event_count, has_positive_event, has_negative_event) + fundamentals columns; train enhanced model; verify Sharpe difference vs baseline quantifiable (≥ 5% or statistical significance) per SC-005.

### Tests for User Story 4 (REQUIRED — write first)

- [X] T050 [P] [US4] Unit test for feature fusion logic (left join by date+tic, fill strategy validation) in `tests/unit/test_fusion.py`
- [X] T051 [P] [US4] Contract test for EnhancedDataset schema (all required columns present, no NaN, dims match) in `tests/contract/test_enhanced_dataset_contract.py`
- [X] T052 [US4] Integration test for full enhanced pipeline (P1 data + P3 sentiment → fused dataset → FinRL train → backtest) in `tests/integration/test_enhanced_pipeline.py` (opt-in)

### Implementation for User Story 4

- [X] T053 [P] [US4] Implement `finquant/features/fusion.py` (left-join MarketDataset + SentimentRecord + FundamentalFeature by date+tic; fill rules: sentiment/event columns → 0.0/0, fundamentals → 0.0 or 0.5 per data-model.md)
- [X] T054 [US4] Implement CLI command `fuse` in `finquant/cli/main.py` (CLI: `finsys fuse --baseline data/processed/xxx.parquet --sentiment data/sentiment/xxx.jsonl --output data/enhanced/xxx_enhanced.parquet`)
- [X] T055 [P] [US4] Extend `finquant/training/trainer.py` to support `--enhanced` flag (swap obs_space from 1+9*N to 1+16*N when enhanced dataset provided)
- [X] T056 [US4] Implement `finquant/cli/main.py` command `compare` (CLI: `finsys compare --baseline reports/baseline_xxx/ --enhanced reports/enhanced_xxx/` → output side-by-side metrics + Sharpe delta)
- [X] T057 [US4] Validate Sharpe improvement ≥ 5% or document statistical test (SC-005 test); record in comparison report
- [X] T058 [P] [US4] Implement FundamentalFeature support in fusion.py (quarterly forward-fill join; handle missing quarters)
- [X] T059 [US4] Create `docs/fusion-guide.md` (workflow: collect sentiment → run fusion → train enhanced model → compare report)

**Checkpoint**: US4 complete — enhanced model trainable; Sharpe improvement measurable (SC-005); baseline vs enhanced comparison available

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation, documentation, hardening

- [X] T060 [P] Implement `finquant/cli/main.py` command `run` (CLI: `finsys run --config config.yaml --mode baseline|enhanced` → orchestrate fetch → train → backtest in sequence)
- [X] T061 [P] Create `README.md` (project overview, installation, 5-step quickstart mirroring quickstart.md)
- [X] T062 [P] Create `ARCHITECTURE.md` (module breakdown, data flow diagram ASCII art, entity relationships)
- [X] T063 [P] Update `config/default.yaml.example` with all fields from research.md decisions (indicators list, data source priority, model hyperparams, failover settings)
- [X] T064 Create comprehensive integration test suite (run all 4 user stories end-to-end with small dataset; verify no regressions) in `tests/integration/test_e2e.py` (opt-in)
- [X] T065 [P] Verify pytest coverage ≥ 80% on `finquant/` (excluding `cli/main.py` boilerplate, `nlp/model.py` GPU loaders); generate coverage report
- [X] T066 [P] Test error scenarios (mock network failure, Qwen load failure, misaligned timestamps) → verify graceful degradation logs + error handling per spec Edge Cases
- [X] T067 [P] Performance baseline: verify SC-001 (data fetch 50 stocks × 3 years ≤ 10 min) and SC-004 (5-step workflow from scratch)
- [X] T068 Validate all CLI commands work end-to-end on sample config; update quickstart.md with exact command outputs
- [X] T069 [P] Add type hints to all public APIs in `finquant/` modules (mypy strict mode optional)
- [X] T070 Create troubleshooting guide in `docs/FAQ.md` (xtquant auth failures, CUDA OOM, Qwen load errors, indicator NaN, etc.)

**Checkpoint**: All user stories integrated, documented, validated. MVP ready for release.

---

## Dependencies & Execution Strategy

### Phase Dependencies

- **Phase 1 (Setup)**: No deps → start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 complete → **BLOCKS** all user stories
- **Phases 3–6 (User Stories)**: All depend on Phase 2 complete
  - **US1 (P1)** → can start immediately after Phase 2
  - **US2 (P2)** → depends on US1 complete (needs US1 data output)
  - **US3 (P3)** → **independent** of US1/US2 (can run in parallel or sequence)
  - **US4 (P4)** → depends on US1 + US2 + US3 all complete
- **Phase 7 (Polish)**: Depends on Phases 3–6; final integration & validation

### Parallel Execution Map

**After Phase 2 complete, 3 independent execution tracks**:

**Track A (Sequential, Critical Path)**:
- T013–T025 (US1 tests + impl) → 1–2 days
- T026–T039 (US2 tests + impl, depends on US1 output) → 2–3 days
- T050–T057 (US4 tests + impl, depends on US1+US2+US3 outputs) → 1–2 days

**Track B (Parallel with A)**:
- T040–T048 (US3 tests + impl, independent GPU hardware) → 1–2 days

**Track C (Parallel, can start after Phase 1)**:
- T060–T070 (Polish, docs, validation) → starts after US3 complete, overlaps with US4

### Suggested MVP Scope

**Minimum for research release**: Complete Phase 1 + Phase 2 + US1 + US2  
→ Delivers baseline quantitative trading system (P1 + P2 per spec)  
→ Can backtest on A-share data with ≥ 0.5 Sharpe (SC-002)  
→ 5-step workflow (SC-004)

**Extended scope** (adds 20% more effort): + US3  
→ Adds sentiment monitoring capability  
→ Enables ablation studies (baseline vs sentiment-enhanced)

**Full scope** (complete): + US4  
→ End-to-end fusion model, comparative analysis ready

---

## Task Statistics

| Category | Count | Notes |
|----------|-------|-------|
| Setup (Phase 1) | 6 tasks | No parallelizable blocks |
| Foundational (Phase 2) | 6 tasks | 3 tasks parallelizable |
| US1 (Data Pipeline) | 13 tasks | 4 test tasks, 5 parallelizable impl tasks |
| US2 (FinRL Training) | 14 tasks | 4 test tasks, 7 parallelizable impl tasks |
| US3 (Sentiment) | 9 tasks | 3 test tasks, 5 parallelizable impl tasks |
| US4 (Fusion) | 7 tasks | 3 test tasks, 3 parallelizable impl tasks |
| Polish (Phase 7) | 11 tasks | 5 parallelizable tasks |
| **Total** | **66 tasks** | ~45% parallelizable (23 [P] tasks) |

**Independent test criteria per story**:
- US1: Output Parquet validates against FinRL schema ✓
- US2: Backtest report exists with Sharpe ≥ 0.5 ✓
- US3: Sentiment accuracy ≥ 80% on 100+ labeled texts ✓
- US4: Enhanced Sharpe improvement ≥ 5% or statistical significance documented ✓

---

## Implementation Strategy

1. **Stages 1–2**: Setup + Foundational (non-negotiable blocking)
2. **Stage 3a**: US1 (data pipeline MVP)
3. **Stage 3b** (parallel to 3a if team size allows): US3 (sentiment independent)
4. **Stage 4**: US2 (FinRL training depends on US1)
5. **Stage 5** (parallel with 4 if resources available): Docs + validation (Phase 7 prep)
6. **Stage 6**: US4 (fusion depends on US1+US2+US3)
7. **Stage 7**: Polish + final integration (Phase 7 complete)

**Estimated effort**: 50–70 engineering-hours for full scope (depends on team Python proficiency, FinRL familiarity, LLM/GPU experience)
