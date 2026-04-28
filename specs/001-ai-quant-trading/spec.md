# Feature Specification: A股AI量化交易系统（数据采集 + FinRL训练 + 舆情基本面融合）

**Feature Branch**: `001-ai-quant-trading`
**Created**: 2026-04-28
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

<!--
  User stories are PRIORITIZED as independent delivery slices.
  Each story delivers standalone value and can be tested in isolation.
-->

### User Story 1 - 数据采集与预处理管道 (Priority: P1)

量化研究员希望通过统一接口（xtquant 等）批量下载 A 股历史行情数据（OHLCV、资金流、成交量），
并将原始数据清洗、对齐、标准化，生成符合 FinRL 标准数据格式的 DataSet 对象，以便直接送入强化学习环境。

**Why this priority**: 数据管道是整个系统的基础；若无干净可用的数据，后续所有模型训练和预测均无法进行。

**Independent Test**: 可独立测试：运行数据采集脚本后，验证输出的 CSV/Parquet 文件包含指定股票、指定时间范围、
正确字段（date, tic, open, high, low, close, volume, ...），无缺失值或异常值，且 FinRL 的 `StockTradingEnv` 能成功加载该数据集。

**Acceptance Scenarios**:

1. **Given** 用户配置了目标股票列表和日期范围，**When** 执行数据采集命令，**Then** 系统通过 xtquant（或备用接口）完整下载指定范围内每只股票的 OHLCV 日线数据，缺数据自动补齐或标记。
2. **Given** 下载完成的原始数据，**When** 运行预处理流水线，**Then** 输出的 DataFrame 满足 FinRL MultiStockTradingEnv 所要求的列格式，且无 NaN、无重复时间戳、股票代码格式统一（6 位数字.SH/.SZ）。
3. **Given** 网络不可用或数据接口返回异常，**When** 采集任务运行，**Then** 系统记录错误日志、保留已成功下载的数据，不因单只股票失败而中断整批任务。

---

### User Story 2 - FinRL 强化学习模型训练与回测 (Priority: P2)

量化研究员希望使用 P1 输出的标准化数据集配置 FinRL 强化学习训练环境，选择交易算法（如 PPO、SAC、TD3），
完成模型训练，并在历史数据上运行回测，获得夏普比率、累计收益、最大回撤等绩效指标。

**Why this priority**: FinRL 训练是核心业务价值所在；P1 完成后即可独立交付此故事的 MVP。

**Independent Test**: 可独立测试：使用 P1 生成的标准化数据集，运行 FinRL 训练脚本后，
验证模型权重文件已保存、回测结果 DataFrame 包含每日净值和交易记录、绩效报告可生成。

**Acceptance Scenarios**:

1. **Given** 标准化的 A 股数据集，**When** 用户配置算法类型（PPO/SAC/TD3）和训练参数后执行训练，**Then** 系统启动 FinRL 的 `StockTradingEnv`，模型训练完成，权重和训练日志保存到指定目录。
2. **Given** 训练完毕的模型权重，**When** 用户在独立测试集时段执行回测，**Then** 系统输出每日持仓、交易记录、累计收益曲线，并计算年化收益率、夏普比率、最大回撤。
3. **Given** 回测结束，**When** 用户请求绩效报告，**Then** 系统生成可视化图表（净值曲线 vs 沪深300基准）和指标摘要表，支持导出为 HTML 或 CSV。

---

### User Story 3 - Qwen 舆情分析与基本面信息提取 (Priority: P3)

量化研究员希望对指定 A 股标的，利用 Qwen 等开源大语言模型，自动分析新闻、公告、财务摘要等文本，
提取情感极性（正面/负面/中性）、关键事件（业绩预期、监管变化、并购消息等）和基本面摘要，生成结构化特征向量。

**Why this priority**: 舆情基本面特征能提升模型对异常行情的预判能力，但系统在 P1+P2 完成后已独立可用，P3 为增强层。

**Independent Test**: 可独立测试：输入若干条真实财经新闻，运行舆情分析接口，验证输出包含 sentiment_score、
event_tags、summary 字段，且结构化 JSON 格式可被后续特征工程模块直接消费。

**Acceptance Scenarios**:

1. **Given** 一批历史财经新闻文本（或实时抓取的新闻 URL 列表），**When** 触发舆情分析任务，**Then** Qwen 模型对每条文本输出情感得分（−1 到 1）、关键事件标签列表、200 字以内摘要。
2. **Given** 某只股票的季度财报 PDF 或文字摘录，**When** 触发基本面提取，**Then** 系统输出结构化基本面指标（营收增速、利润率、负债率等估值描述性字段），准确率与人工标注对比一致率 ≥ 80%。
3. **Given** 舆情分析服务不可用（模型加载失败或超时），**When** 训练流程请求舆情特征，**Then** 系统以默认中性值填充该特征，并记录告警日志，不中断整体训练流程。

---

### User Story 4 - 舆情基本面特征融合与增强训练 (Priority: P4)

量化研究员希望将 P3 生成的舆情 / 基本面特征向量，与 P1 的行情技术特征合并，作为增强输入送入 FinRL 环境，
重新训练模型，对比纯技术因子基线的绩效提升。

**Why this priority**: 特征融合是端到端系统的最终目标，验证舆情信息是否能带来超额收益，依赖 P1~P3 全部就绪。

**Independent Test**: 可独立测试：构造一个包含技术 + 舆情特征的合并数据集，验证 FinRL 的 `observation_space`
维度与新特征数量一致，训练正常启动，回测绩效可与 P2 基线对比。

**Acceptance Scenarios**:

1. **Given** P1 行情数据集和 P3 舆情特征数据集（相同股票、相同日期），**When** 执行特征融合，**Then** 输出数据集的每行包含原始技术指标 + 舆情得分 + 关键事件 one-hot 编码，日期和股票代码对齐，无缺失。
2. **Given** 融合后的增强数据集，**When** 运行 FinRL 训练，**Then** 训练可正常完成，回测输出结果与 P2 基线同维度可比。
3. **Given** 基线模型（P2）和增强模型（P4）的回测结果，**When** 用户请求对比报告，**Then** 系统展示两者在夏普比率、年化收益、最大回撤上的差异，并给出统计显著性说明。

---

### Edge Cases

- 股票停牌区间的数据处理：停牌日价格应填充前收盘价，成交量填 0，不得直接删除该行（会导致时间序列断裂）。
- 新股上市不足 60 个交易日：系统应过滤或提示，避免数据量不足导致特征计算失效。
- xtquant 授权失效或 quota 耗尽：系统自动切换到备用数据接口（如 akshare、baostock），并通知用户。
- Qwen 模型显存不足：支持 4-bit 量化推理模式，保证在 8 GB VRAM 机器上可运行。
- 特征融合时舆情数据存在但行情数据缺失（或反之）：以行情数据为主键 left join，舆情缺失用中性值填充。
- 强化学习训练发散（reward 持续为负）：提供超参数检查清单和早停机制，训练异常自动保存 checkpoint。

---

## Test Strategy & Evidence *(mandatory)*

- **TS-001**: 数据层（P1）：单元测试验证数据清洗函数（缺值填充、异常过滤）；集成测试验证 xtquant 接口端到端下载后格式符合 FinRL 规范。
- **TS-002**: 训练层（P2）：契约测试验证 `StockTradingEnv` 的 `observation_space` 与数据集列数一致；回测集成测试验证绩效指标计算函数输出有效数值。
- **TS-003**: 舆情层（P3）：单元测试验证 sentiment_score 在 [−1, 1] 范围内；集成测试输入已知情感倾向的新闻样本验证标注准确率 ≥ 80%。
- **TS-004**: 每个用户故事的"Given-When-Then"场景对应一个可执行的 pytest 测试用例，在 CI 中自动运行；集成风险最高处在数据接口切换逻辑和特征融合对齐逻辑，需专项回归用例。

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 支持通过 xtquant 接口下载 A 股日线 OHLCV 数据，支持批量多股票、自定义日期范围。
- **FR-002**: 系统 MUST 提供至少一个备用数据接口（如 akshare 或 baostock），在主接口不可用时自动切换。
- **FR-003**: 系统 MUST 将原始数据标准化为 FinRL 兼容格式（包含 date, tic, open, high, low, close, volume 及可扩展技术指标列）。
- **FR-004**: 系统 MUST 支持配置并运行 FinRL 内置算法（PPO、SAC、TD3 至少之一）对 A 股多资产投资组合进行强化学习训练。
- **FR-005**: 系统 MUST 在测试集上执行回测并输出夏普比率、年化收益率、最大回撤三项核心绩效指标。
- **FR-006**: 系统 MUST 支持调用 Qwen（或同类开源 LLM）对金融文本进行情感分析，输出 sentiment_score 和 event_tags。
- **FR-007**: 系统 MUST 提供特征融合模块，将舆情/基本面特征与行情技术特征按日期、股票代码对齐后合并。
- **FR-008**: 系统 MUST 支持融合后的增强数据集重新送入 FinRL 训练，且与纯技术因子基线可对比。
- **FR-009**: 系统 MUST 在关键节点（数据下载完成、训练 epoch 结束、回测完成）写入结构化日志，便于问题追踪。
- **FR-010**: 系统 MUST 提供配置文件（YAML/TOML）统一管理股票列表、日期范围、模型超参数、接口凭据路径，避免硬编码。
- **FR-011**: 规格 MUST 包含可执行的分解计划（阶段划分、依赖关系、独立交付切片）。
- **FR-012**: 每条需求 MUST 可追溯到对应验收场景和测试用例。

### Key Entities

- **StockUniverse（股票池）**: 目标 A 股标的集合，包含股票代码、名称、所属行业；是数据采集和特征生成的主键。
- **MarketDataset（行情数据集）**: 标准化后的多股票日线时间序列，包含 OHLCV + 技术指标列；FinRL 的直接输入。
- **TradingEnvironment（交易环境）**: FinRL 的 gym 环境实例，封装了行情数据集、交易规则（手续费、滑点）和资金账户状态。
- **RLAgent（强化学习智能体）**: 使用选定算法（PPO/SAC/TD3）训练的策略网络，输出每只股票的仓位动作。
- **SentimentRecord（舆情记录）**: 单条金融文本的分析结果，包含文本源、发布时间、股票代码关联、sentiment_score、event_tags、summary。
- **FundamentalFeature（基本面特征）**: 从财务报告中提取的结构化指标，如营收增速、净利润率、市盈率估算等描述字段。
- **EnhancedDataset（增强数据集）**: MarketDataset 与 SentimentRecord/FundamentalFeature 按日期和股票代码对齐合并的宽表，作为融合训练的输入。
- **BacktestReport（回测报告）**: 模型在测试集上的完整绩效记录，包含每日净值、交易流水、绩效指标和可视化图表。

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 数据采集任务对 50 只 A 股 3 年历史日线数据的完整下载 + 预处理，在普通宽带环境下完成时间 ≤ 10 分钟。
- **SC-002**: FinRL 基线模型（纯技术特征）在 A 股测试集上的年化夏普比率 ≥ 0.5（高于持有沪深300指数的基准）。
- **SC-003**: 舆情分析模块对已知情感倾向的财经新闻样本（100 条以上）的准确率 ≥ 80%。
- **SC-004**: 从新环境启动到完成第一次完整训练（数据下载 + 预处理 + FinRL 训练 + 回测），全流程操作步骤 ≤ 5 步（通过配置文件驱动，无需修改源码）。
- **SC-005**: 融合舆情特征后的增强模型，在同等测试集上的夏普比率较纯技术基线提升 ≥ 5%（或差异在统计上可量化）。
- **SC-006**: 系统在依赖外部接口出现超时或错误时，自动降级并继续运行，不因单点失败导致全流程崩溃，故障转移成功率 ≥ 95%。

---

## Assumptions

- 假设运行环境为 Python 3.10+ 及 CUDA 可用（GPU 训练），CPU-only 模式作为降级支持但不保证训练速度。
- 假设 xtquant 已安装并持有有效的迅投 QMT 账户授权；项目文档提供备用接口（akshare/baostock）的无授权配置方式。
- 假设 Qwen 模型（如 Qwen2.5-7B-Instruct）可通过 HuggingFace 或本地路径加载；支持 4-bit 量化以适配消费级 GPU（≥ 8 GB VRAM）。
- 假设 A 股数据仅面向历史回测和研究目的，不涉及实盘自动下单（实盘对接为后续独立功能，不在本规格范围内）。
- 假设新闻/公告文本通过爬虫、RSS 或第三方 API 已完成采集并以结构化格式存储（文本采集本身不在本规格范围内）。
- 假设用户具备基本的 Python 和命令行使用能力；系统提供完整的 quickstart 文档，无需深入了解 FinRL 内部实现即可运行。
- 假设 FinRL 使用其最新稳定版本（>= 0.3.x），所用接口遵循其官方 README；版本锁定于 requirements.txt。
