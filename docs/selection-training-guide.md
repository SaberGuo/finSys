# 因子选股与RL训练集成指南

本文档介绍如何将因子选股结果与强化学习训练结合，实现"选股+交易"双阶段流水线。

## 概述

### 工作流程

```
1. 因子选股 (日频)
   ↓
2. 获取候选股票池
   ↓
3. 获取5分钟数据
   ↓
4. RL训练与回测
```

### 核心概念

- **选股模块**: 基于市场状态和多因子评分，每日筛选出top-k候选股票
- **RL交易模块**: 在候选股票池上进行5分钟级别的买卖决策
- **数据对齐**: 选股使用日频数据，交易使用5分钟数据，需要时间对齐

## 快速开始

### 1. 运行因子选股

首先生成选股结果：

```bash
# 对2024年全年进行选股
python -m finquant.cli.main selection run \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --verbose

# 结果保存在 data/selection/ 目录
# 每个交易日生成一个 YYYY-MM-DD_selection.json 文件
```

### 2. 查看选股结果

```bash
# 查看某日的选股结果
cat data/selection/2024-01-15_selection.json
```

选股结果包含：
- `selected_tickers`: 选中的股票列表
- `scores`: 每只股票的综合评分
- `market_state`: 市场状态（上涨/下跌/震荡等）
- `active_factors`: 当前使用的因子
- `factor_weights`: 因子权重

### 3. 基于选股结果训练RL模型

#### 方式一：使用单日选股结果

```bash
# 使用2024-01-15的选股结果训练
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --test-days 10 \
  --algo ppo \
  --timesteps 100000
```

**参数说明**：
- `--selection-date`: 选股日期，使用该日的候选股票池
- `--train-days`: 训练天数（从选股日期往前推）
- `--test-days`: 测试天数（从选股日期往后推）
- `--algo`: RL算法（ppo/sac/td3）
- `--timesteps`: 训练步数

**训练流程**：
1. 加载选股结果，获取候选股票池（如10只股票）
2. 从数据库获取这些股票的5分钟数据
3. 训练RL agent在这个固定股票池上进行交易
4. 在测试集上回测并输出报告

#### 方式二：使用随机股票池（对比基准）

```bash
# 随机选择10只股票训练（不使用选股结果）
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 60 \
  --test-days 30 \
  --algo ppo \
  --timesteps 100000
```

## 详细说明

### 选股结果格式

每个选股结果文件（`YYYY-MM-DD_selection.json`）包含：

```json
{
  "version": "1.0.0",
  "date": "2024-01-15",
  "selected_tickers": [
    "600000.SH",
    "600016.SH",
    "000001.SZ",
    ...
  ],
  "scores": {
    "600000.SH": 0.5080,
    "600016.SH": 0.3687,
    "000001.SZ": 0.0189,
    ...
  },
  "market_state": "oscillation",
  "active_factors": [
    "low_volatility",
    "low_turnover",
    "high_dividend"
  ],
  "factor_weights": {
    "low_volatility": 0.3333,
    "low_turnover": 0.3333,
    "high_dividend": 0.3333
  },
  "index_metrics": {
    "close": 12.61,
    "adx": 100.0,
    "ma50": 12.365,
    "ma50_ratio": 1.0198,
    "volume_ma20_ratio": 1.0009
  },
  "exclusion_reasons": {}
}
```

### 数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 选股阶段 (日频)                                           │
├─────────────────────────────────────────────────────────────┤
│ 输入: 日频行情数据 (data/processed/zz500_data.db)           │
│ 处理:                                                        │
│   - 计算技术指标 (ADX, ATR, MA等)                           │
│   - 识别市场状态 (上涨/下跌/震荡/...)                       │
│   - 选择因子组合 (根据市场状态)                             │
│   - 计算因子权重 (IC滚动窗口)                               │
│   - 综合评分并筛选top-k                                     │
│ 输出: 候选股票池 + 评分 (data/selection/*.json)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 数据准备阶段                                              │
├─────────────────────────────────────────────────────────────┤
│ 输入: 候选股票池                                             │
│ 处理:                                                        │
│   - 从数据库获取5分钟数据                                   │
│   - 预处理 (对齐、填充缺失值)                               │
│   - 计算技术指标 (MACD, BOLL, RSI等)                        │
│ 输出: 5分钟级训练数据集                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. RL训练阶段 (5分钟频)                                      │
├─────────────────────────────────────────────────────────────┤
│ 输入: 5分钟数据集 + 候选股票池                              │
│ 处理:                                                        │
│   - 构建StockTradingEnv (stock_dim=候选池大小)              │
│   - 训练RL agent (PPO/SAC/TD3)                              │
│   - 学习买卖时机和仓位管理                                  │
│ 输出: 训练好的模型 (.zip)                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 回测阶段                                                  │
├─────────────────────────────────────────────────────────────┤
│ 输入: 训练模型 + 测试数据集                                 │
│ 处理:                                                        │
│   - 在测试集上运行策略                                      │
│   - 计算收益、回撤、夏普比率等指标                          │
│ 输出: 回测报告 (HTML + CSV)                                 │
└─────────────────────────────────────────────────────────────┘
```

### 时间对齐策略

**问题**: 选股使用日频数据，交易使用5分钟数据，如何对齐？

**解决方案**:

1. **固定候选池模式** (当前实现)
   - 选择某日的选股结果
   - 使用该日的候选股票池
   - 在前后时间段的5分钟数据上训练
   - 适用于：验证选股效果、单策略回测

2. **动态候选池模式** (未来扩展)
   - 每日更新候选股票池
   - RL agent需要处理股票池变化
   - 在每日开盘前（9:30前）执行选股
   - 当日9:30-15:00使用固定候选池交易
   - 适用于：实盘模拟、长期回测

## 使用示例

### 示例1: 验证选股效果

比较使用选股结果 vs 随机选股的训练效果：

```bash
# 1. 使用2024-01-15的选股结果
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --test-days 10 \
  --algo ppo \
  --timesteps 100000 \
  --output-dir runs/selection_test

# 2. 随机选择10只股票（对比基准）
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 30 \
  --test-days 10 \
  --algo ppo \
  --timesteps 100000 \
  --output-dir runs/random_test

# 3. 比较两者的回测结果
# 查看 runs/selection_test/*/metadata.json
# 查看 runs/random_test/*/metadata.json
```

### 示例2: 不同市场状态下的训练

```bash
# 找出不同市场状态的日期
python -c "
import json
from pathlib import Path
from collections import defaultdict

results = defaultdict(list)
for f in Path('data/selection').glob('*_selection.json'):
    with open(f) as fp:
        data = json.load(fp)
        results[data['market_state']].append(data['date'])

for state, dates in results.items():
    print(f'{state}: {dates[:3]}...')
"

# 在震荡市场训练
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --algo ppo

# 在上涨市场训练
python scripts/selection_rl_train.py \
  --selection-date 2024-03-20 \
  --train-days 30 \
  --algo ppo
```

### 示例3: 不同算法对比

```bash
# PPO
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --algo ppo \
  --timesteps 100000

# SAC
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --algo sac \
  --timesteps 100000

# TD3
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --algo td3 \
  --timesteps 100000
```

## 配置说明

### 选股配置 (config/selection_test.yaml)

```yaml
stocks:
  - "000001.SZ"
  - "000002.SZ"
  # ... 更多股票

dates:
  train_start: "2023-01-01"
  train_end: "2023-12-31"
  test_start: "2024-01-01"
  test_end: "2024-12-31"

data:
  source_priority:
    - "db_daily"      # 优先使用本地数据库
    - "xtquant"       # 备用数据源
    - "akshare"
    - "baostock"

selection:
  index_ticker: "000300.SH"  # 沪深300指数
  top_k: 10                   # 选择前10只股票
  min_score: -999             # 最低评分阈值
  
  # 市场状态配置
  market_state:
    adx_threshold: 25         # ADX阈值
    ma_threshold: 0.02        # MA偏离阈值
    volume_threshold: 0.8     # 成交量阈值
  
  # 因子配置
  factors:
    # 上涨市场因子
    uptrend:
      - momentum
      - high_beta
      - growth_yoy
    # 下跌市场因子
    downtrend:
      - low_volatility
      - high_dividend
      - value_pb
    # 震荡市场因子
    oscillation:
      - low_volatility
      - low_turnover
      - high_dividend
    # ... 其他市场状态
  
  # IC权重计算
  ic_weight:
    window: 20                # 滚动窗口大小
    min_periods: 10           # 最小样本数
```

### RL训练配置

训练参数在脚本中指定，也可以通过配置文件：

```yaml
environment:
  initial_amount: 1000000     # 初始资金
  hmax: 100                   # 单只股票最大持仓
  buy_cost_pct: 0.001         # 买入手续费
  sell_cost_pct: 0.001        # 卖出手续费
  reward_scaling: 1e-4        # 奖励缩放

training:
  algorithm: "ppo"            # ppo/sac/td3
  total_timesteps: 100000     # 训练步数
  
  # PPO超参数
  ppo:
    n_steps: 2048
    batch_size: 128
    n_epochs: 10
    learning_rate: 0.00025
    ent_coef: 0.01
```

## 输出结果

### 目录结构

```
runs/selection/
└── 2024-01-15_143052/          # 训练时间戳
    ├── ppo_20240115_abc123.zip # 训练模型
    ├── ppo_20240115_abc123_metadata.json  # 模型元数据
    ├── metadata.json            # 训练元数据
    └── reports/                 # 回测报告
        ├── backtest_report.html
        ├── account_value.csv
        └── actions.csv
```

### 元数据示例

```json
{
  "selection_date": "2024-01-15",
  "selection_result": {
    "date": "2024-01-15",
    "selected_tickers": ["600000.SH", "600016.SH", ...],
    "market_state": "oscillation",
    "active_factors": ["low_volatility", "low_turnover", "high_dividend"]
  },
  "train_days": 30,
  "test_days": 10,
  "algorithm": "ppo",
  "timesteps": 100000,
  "report": {
    "annual_return": 0.1523,
    "max_drawdown": -0.0856,
    "sharpe_ratio": 1.78,
    "win_rate": 0.6234,
    "total_trades": 145
  }
}
```

## 常见问题

### Q1: 选股结果为空怎么办？

**原因**: 
- 配置的股票池太小
- 评分阈值太高
- 数据缺失

**解决**:
```bash
# 检查选股结果
python -c "
import json
from pathlib import Path

for f in sorted(Path('data/selection').glob('*_selection.json'))[:5]:
    with open(f) as fp:
        data = json.load(fp)
        print(f\"{data['date']}: {len(data['selected_tickers'])} stocks\")
"

# 调整配置
# 1. 增加股票池大小
# 2. 降低 min_score 阈值
# 3. 检查数据完整性
```

### Q2: 5分钟数据不足怎么办？

**原因**: 
- 数据库中没有对应股票的5分钟数据
- 时间范围超出数据库范围

**解决**:
```bash
# 检查数据库
python -c "
import sqlite3
from pathlib import Path

db = Path('data/processed/zz500_data.db')
conn = sqlite3.connect(db)
cursor = conn.cursor()

# 检查5分钟数据
cursor.execute('SELECT MIN(date), MAX(date), COUNT(DISTINCT code) FROM minute_data')
print('5分钟数据:', cursor.fetchone())

# 检查特定股票
cursor.execute(\"SELECT COUNT(*) FROM minute_data WHERE code = '600000.SH'\")
print('600000.SH 数据量:', cursor.fetchone()[0])
"

# 如果数据不足，需要重新下载
python scripts/download_zz500_data.py
```

### Q3: 训练时间太长怎么办？

**优化建议**:
1. 减少训练步数: `--timesteps 50000`
2. 减少股票数量: 选择top-5而不是top-10
3. 缩短训练时间段: `--train-days 15`
4. 使用更快的算法: SAC通常比PPO快

### Q4: 如何评估选股效果？

**方法**:
1. 对比实验: 选股 vs 随机选股
2. 不同市场状态下的表现
3. 长期回测（多个选股周期）
4. 因子贡献分析

```bash
# 批量测试
for date in 2024-01-15 2024-02-15 2024-03-15; do
  python scripts/selection_rl_train.py \
    --selection-date $date \
    --train-days 30 \
    --test-days 10 \
    --output-dir runs/batch_test/$date
done

# 汇总结果
python -c "
import json
from pathlib import Path

for d in Path('runs/batch_test').iterdir():
    meta = json.load(open(d / 'metadata.json'))
    print(f\"{meta['selection_date']}: {meta['report']['annual_return']:.2%}\")
"
```

## 下一步

1. **实现动态候选池模式**: 每日更新股票池，模拟实盘场景
2. **集成到CLI**: `finsys selection-trading --start --end`
3. **添加更多因子**: SUE、基本面因子、舆情因子
4. **优化IC权重计算**: 机器学习权重、自适应窗口
5. **风控模块**: 止损、止盈、仓位管理

## 参考

- [因子选股模块文档](../specs/004-factor-selection-trading/spec.md)
- [RL训练模块文档](./user-guide.md)
- [数据模型文档](../specs/004-factor-selection-trading/data-model.md)
