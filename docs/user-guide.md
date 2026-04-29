# finSys 操作手册

> 从数据抓取到模型训练的全流程指南

---

## 目录

1. [环境准备](#环境准备)
2. [配置文件详解](#配置文件详解)
3. [数据抓取](#数据抓取)
4. [模型训练](#模型训练)
5. [回测验证](#回测验证)
6. [一键全流程](#一键全流程)
7. [舆情分析（可选）](#舆情分析可选)
8. [特征融合与对比（可选）](#特征融合与对比可选)
9. [结果解读](#结果解读)
10. [常见问题](#常见问题)

---

## 环境准备

### 系统要求

| 项目 | 要求 | 说明 |
|------|------|------|
| Python | >= 3.10 | 推荐 3.11 |
| 操作系统 | Windows / Linux / macOS | Windows 需 PowerShell |
| 磁盘空间 | >= 5 GB | 数据、模型、日志 |
| GPU（可选） | CUDA >= 11.8，VRAM >= 8 GB | 训练加速 + Qwen 推理 |
| 内存 | >= 16 GB | 数据处理 |

### 安装依赖

```bash
# 克隆仓库
git clone <repo-url> finSys
cd finSys

# 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

> **注意**：`bitsandbytes`（Qwen 4-bit 量化）在 Windows 上可能安装失败。如不启用舆情分析，可跳过。

### 验证安装

```bash
# 查看 CLI 帮助
finsys --help

# 验证配置加载（dry-run）
finsys data fetch --config config/default.yaml.example --dry-run
```

---

## 配置文件详解

复制默认配置模板并按需修改：

```bash
cp config/default.yaml.example config/my_config.yaml
```

### 最小可用配置

```yaml
# 股票池 — 必须为 6位数字.SH 或 6位数字.SZ 格式
stocks:
  - "000001.SZ"   # 平安银行
  - "600000.SH"   # 浦发银行

# 日期范围
dates:
  train_start: "2021-01-01"
  train_end:   "2024-06-30"
  test_start:  "2024-07-01"
  test_end:    "2025-06-30"

# 数据源（无需迅投授权时设为 akshare）
data:
  source_priority:
    - "akshare"      # 免费，无需认证
    - "baostock"     # 备用
```

### 完整配置字段说明

| 配置段 | 关键字段 | 默认值 | 说明 |
|--------|---------|--------|------|
| `stocks` | 列表 | — | 股票代码，格式 `\d{6}\.(SH\|SZ)` |
| `dates` | `train_start/end` | — | 训练集起止日期 |
| `dates` | `test_start/end` | — | 测试集起止日期 |
| `data.source_priority` | 列表 | `[xtquant, akshare, baostock]` | 数据源优先级，逐个尝试 |
| `indicators` | 列表 | 7 个技术指标 | MACD、BOLL、RSI、DX、SMA |
| `environment` | `initial_amount` | 1,000,000 | 初始资金（人民币） |
| `environment` | `buy/sell_cost_pct` | 0.001 | 手续费 0.1% |
| `training` | `algorithm` | `ppo` | `ppo` / `sac` / `td3` |
| `training` | `total_timesteps` | 100,000 | 训练步数 |
| `sentiment` | `enabled` | `false` | 是否启用舆情分析 |
| `sentiment` | `model_id` | Qwen2.5-7B | 模型 ID |
| `sentiment` | `quantize_4bit` | `true` | 4-bit 量化 |

> **没有迅投账号？** 将 `data.source_priority` 首项设为 `"akshare"`，无需任何认证即可下载 A 股数据。

---

## 数据抓取

### 基本命令

```bash
finsys data fetch --config config/my_config.yaml
```

### 完整参数

```bash
finsys data fetch \
  --config config/my_config.yaml      # 配置文件路径 \
  --output data/processed              # 输出目录 \
  --start 2021-01-01                   # 覆盖起始日期 \
  --end 2025-06-30                     # 覆盖结束日期 \
  --verbose                            # 显示详细进度
```

### 预期输出

```text
Saved to data/processed/20210101_20250630_dataset.parquet
```

### 输出文件验证

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/20210101_20250630_dataset.parquet')
print(f'形状: {df.shape}')           # (行数, 列数)
print(f'股票数: {df.tic.nunique()}')  # 应等于配置中的股票数
print(f'NaN 检查: {df.isnull().sum().sum()}')  # 应为 0
print(f'列名: {list(df.columns)}')
"
```

### 数据源自动故障转移

系统按 `source_priority` 顺序尝试数据源：

1. **xtquant** — 需迅投 QMT 客户端已登录
2. **akshare** — 免费，无需认证
3. **baostock** — 备用，可匿名使用

如果 `xtquant` 连接失败，日志会记录警告并自动切换到 `akshare`：

```text
{"timestamp": "...", "level": "WARNING", "message": "source xtquant failed: ..."}
{"timestamp": "...", "level": "INFO", "message": "downloaded from akshare"}
```

### 输出数据 Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `date` | str | 交易日 `YYYY-MM-DD` |
| `tic` | str | 股票代码 `000001.SZ` |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `volume` | float | 成交量 |
| `macd` | float | MACD 指标 |
| `boll_ub` | float | 布林带上轨 |
| `boll_lb` | float | 布林带下轨 |
| `rsi_30` | float | 30 日 RSI |
| `dx_30` | float | 30 日方向指标 |
| `close_30_sma` | float | 30 日收盘均价 |
| `close_60_sma` | float | 60 日收盘均价 |

---

## 模型训练

### 基本命令

```bash
finsys train \
  --config config/my_config.yaml \
  --data-file data/processed/20210101_20250630_dataset.parquet \
  --algo ppo
```

### 完整参数

```bash
finsys train \
  --config config/my_config.yaml       # 配置文件 \
  --data-file data/processed/xxx.parquet \
  --algo ppo                           # ppo / sac / td3 \
  --timesteps 200000                   # 覆盖训练步数 \
  --output models/                     # 模型保存目录 \
  --verbose
```

### 预期输出

```text
Training ppo on 2 stocks x 860 days ...
Model saved to models/ppo_20250630_a1b2c3d4.zip
```

### 训练环境参数

系统在训练时会构建 FinRL `StockTradingEnv`，关键参数来自配置：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `initial_amount` | 1,000,000 | 初始现金 |
| `hmax` | 100 | 单次交易最大股数 |
| `buy_cost_pct` | 0.001 | 买入手续费 |
| `sell_cost_pct` | 0.001 | 卖出手续费 |
| `reward_scaling` | 1e-4 | 奖励缩放系数 |

### Observation Space

对于 `N` 只股票，状态向量维度 = `1 + 9 * N`：
- `1` = 当前现金余额
- `9 * N` = 每只股票：close + volume + 7 个技术指标

### 切换算法

```bash
# PPO（默认，适合大多数场景）
finsys train ... --algo ppo

# SAC（连续动作空间，样本效率更高）
finsys train ... --algo sac

# TD3（处理过估计问题）
finsys train ... --algo td3
```

### 超参数调优

在 `config.yaml` 中修改：

```yaml
training:
  ppo:
    learning_rate: 0.0003   # 学习率
    n_steps: 2048           # 每次更新收集的步数
    batch_size: 64          # 批次大小
    n_epochs: 10            # 每次数据迭代轮数
    gamma: 0.99             # 折扣因子
```

> **Tip**: 如果训练 reward 不收敛，尝试降低 `learning_rate` 或增加 `total_timesteps`。

---

## 回测验证

### 基本命令

```bash
finsys backtest \
  --config config/my_config.yaml \
  --model models/ppo_20250630_a1b2c3d4.zip \
  --data-file data/processed/20210101_20250630_dataset.parquet
```

### 完整参数

```bash
finsys backtest \
  --config config/my_config.yaml       # 配置文件 \
  --model models/ppo_xxx.zip           # 训练好的模型 \
  --data-file data/processed/xxx.parquet \
  --output reports/                    # 报告输出目录 \
  --risk-free-rate 0.02                # 无风险利率 \
  --verbose
```

### 预期输出

```json
{
  "sharpe_ratio": 0.73,
  "annual_return_pct": 9.8,
  "max_drawdown_pct": -6.4,
  "cumulative_return_pct": 9.2
}
```

同时生成：
- `reports/ppo_YYYYMMDD_report.html` — 交互式净值曲线图
- `reports/ppo_YYYYMMDD_metrics.csv` — 指标表格

### 回测原理

回测时系统会：
1. 加载训练好的 SB3 模型
2. 在测试集上逐日执行交易决策
3. 记录每日账户净值和交易动作
4. 计算风险收益指标

---

## 一键全流程

如果希望自动执行 `fetch → train → backtest`：

```bash
finsys run \
  --config config/my_config.yaml \
  --output runs/latest \
  --algo ppo \
  --timesteps 100000 \
  --verbose
```

### 输出结构

```text
runs/latest/
├── data/
│   └── 20210101_20250630_dataset.parquet
├── models/
│   └── ppo_20250630_a1b2c3d4.zip
└── reports/
    ├── ppo_20250630_report.html
    └── ppo_20250630_metrics.csv
```

---

## 舆情分析（可选）

> **前提**：需要 GPU >= 8 GB VRAM，且已安装 `bitsandbytes`

### 准备输入数据

创建 JSONL 文件（每行一条记录）：

```jsonl
{"date": "2024-07-01", "tic": "000001.SZ", "text": "平安银行发布半年报，净利润同比增长15%"}
{"date": "2024-07-02", "tic": "600000.SH", "text": "浦发银行收到监管函，股价承压"}
```

### 运行分析

```bash
finsys sentiment analyze \
  --config config/my_config.yaml \
  --input data/news.jsonl \
  --output data/sentiment \
  --verbose
```

### 输出格式

每条记录包含：
- `sentiment_score`: [-1.0, 1.0] 情感得分
- `event_tags`: 事件标签列表，如 `["业绩超预期"]`
- `summary`: 200 字以内的摘要
- `text_hash`: MD5 去重标识

###  graceful degradation

如果 Qwen 加载失败（如显存不足），系统会：
- 记录警告日志
- 返回中性得分 `0.0`
- 不中断后续流程

---

## 特征融合与对比（可选）

### 融合市场数据 + 舆情数据

```bash
finsys fuse \
  --config config/my_config.yaml \
  --market data/processed/xxx.parquet \
  --sentiment data/sentiment/result.jsonl \
  --output data/enhanced/dataset.parquet \
  --verbose
```

增强后的数据集会增加以下列：
- `sentiment_score` — 当日平均情感得分
- `event_count` — 事件数量
- `has_positive_event` — 是否有正面事件
- `has_negative_event` — 是否有负面事件

### 使用增强数据训练

```bash
finsys train \
  --config config/my_config.yaml \
  --data-file data/enhanced/dataset.parquet \
  --algo ppo
```

此时 observation space 变为 `1 + 16 * N`（增加 7 个情感/基本面特征）。

### 对比基线与增强模型

```bash
finsys compare \
  --baseline-report reports/baseline_metrics.csv \
  --enhanced-report reports/enhanced_metrics.csv
```

输出 JSON 格式的指标对比表，包含 `delta`（增强 - 基线）。

---

## 结果解读

### 核心指标

| 指标 | 含义 | 参考值 |
|------|------|--------|
| **Sharpe Ratio** | 夏普比率，单位风险超额收益 | >= 0.5 为合格（项目目标） |
| **Annual Return** | 年化收益率 | 越高越好 |
| **Max Drawdown** | 最大回撤，负数表示亏损幅度 | 绝对值越小越好，建议 < -10% |
| **Cumulative Return** | 累计收益率 | 总回报 |

### Sharpe Ratio 解读

- `< 0`: 策略不如无风险资产
- `0 ~ 0.5`: 风险收益比一般
- `0.5 ~ 1.0`: 较好（本项目最低目标）
- `> 1.0`: 优秀

### 净值曲线分析

打开 `reports/ppo_xxx_report.html`：
- **蓝线**: 策略净值
- **橙线**: 沪深300 基准
- 如果策略净值持续高于基准，说明有超额收益（alpha）

---

## 常见问题

### Q1: xtquant 连接失败

**症状**: `source xtquant failed: ...`

**解决**:
1. 确认迅投 QMT 客户端已启动并登录
2. 或改用 `akshare` 作为首选数据源

### Q2: 训练时 CUDA out of memory

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
1. 减少 `stocks` 数量（建议从 5~10 只开始）
2. 降低 `batch_size`
3. 使用 CPU 训练：`export CUDA_VISIBLE_DEVICES=""`

### Q3: 技术指标出现 NaN

**症状**: `AssertionError: NaN values found`

**原因**: 数据时间窗口不足（如 rsi_30 需要至少 30 个交易日）

**解决**: 确保 `train_start` 到首个有效日期 >= 60 个交易日

### Q4: Qwen 加载失败

**症状**: `ImportError: bitsandbytes`

**解决**:
```bash
# Linux
pip install bitsandbytes --upgrade

# Windows（bitsandbytes 官方不支持 Windows）
# 如不需要舆情分析，可忽略此错误
```

### Q5: 训练 reward 不收敛

**症状**: reward_mean 始终为负或波动剧烈

**解决**:
1. 检查数据是否有停牌日未处理（volume = 0 是正常的）
2. 降低 `learning_rate`（如从 3e-4 降到 1e-4）
3. 增加 `total_timesteps`（如从 100k 增加到 500k）
4. 尝试 SAC 或 TD3 算法

### Q6: 回测 Sharpe 为负数

**症状**: `sharpe_ratio: -0.2`

**原因**: 模型在测试集上表现差于随机策略

**解决**:
1. 检查训练/测试集划分是否合理（避免数据泄漏）
2. 延长训练时间
3. 尝试不同的随机种子或超参数

---

## 完整工作流示例

```bash
# Step 1: 配置
cp config/default.yaml.example config/my_config.yaml
# 编辑 my_config.yaml，设置股票和日期

# Step 2: 抓取数据
finsys data fetch --config config/my_config.yaml --verbose

# Step 3: 训练模型
finsys train \
  --config config/my_config.yaml \
  --data-file data/processed/20210101_20250630_dataset.parquet \
  --algo ppo \
  --timesteps 200000 \
  --verbose

# Step 4: 回测
finsys backtest \
  --config config/my_config.yaml \
  --model models/ppo_20250630_a1b2c3d4.zip \
  --data-file data/processed/20210101_20250630_dataset.parquet \
  --verbose

# Step 5: 查看报告
start reports/ppo_20250630_report.html   # Windows
open reports/ppo_20250630_report.html    # macOS
xdg-open reports/ppo_20250630_report.html # Linux
```

或使用一键命令：

```bash
finsys run --config config/my_config.yaml --algo ppo --verbose
```

---

> 更多技术细节请参阅 [`ARCHITECTURE.md`](../ARCHITECTURE.md) 和 [`quickstart.md`](../specs/001-ai-quant-trading/quickstart.md)。
