# ZZ500 单股票RL选股交易系统 - 快速入门

## 系统概述

本系统使用强化学习（RL）对中证500（ZZ500）指数成分股进行评分和选股，维护一个5只股票的投资组合。

### 核心特点

- **单股票RL模型**：每次评估1只股票（观察空间仅9维），训练快速
- **批量评分**：对所有343只ZZ500股票独立评分
- **智能选股**：筛选评分>0.9的股票，选择前5名
- **评分加权**：根据RL评分分配持仓权重
- **风险管理**：止损5%，止盈20%，RL评分<-0.2时退出

## 数据验证

数据库状态：
- 股票数量：343只
- 数据范围：2020-01-02 至 2026-04-30
- 位置：`data/processed/zz500_data.db`

## 使用流程

### 1. 训练RL模型

```bash
python scripts/train_zz500_single_stock.py \
    --config config/zz500_rl_single_stock.yaml \
    --db-path data/processed/zz500_data.db \
    --output-dir models/zz500_single_stock
```

**训练参数**：
- 训练期：2023-01-01 至 2024-06-30
- 算法：PPO
- 训练步数：200,000
- 观察空间：10维（1现金 + 9股票特征）
- 动作空间：1维（单股票连续动作）

**输出**：
- 模型文件：`models/zz500_single_stock/ppo_zz500_single_YYYYMMDD_HASH.zip`
- 元数据：`models/zz500_single_stock/training_metadata.json`

### 2. 回测策略

```bash
python scripts/backtest_zz500_single_stock.py \
    --model models/zz500_single_stock/ppo_zz500_single_20240630_abc123.zip \
    --config config/zz500_rl_single_stock.yaml \
    --start 2024-07-01 \
    --end 2024-12-31 \
    --output-dir runs/zz500_backtest
```

**回测逻辑**：
1. 每日对343只股票评分
2. 筛选评分>0.9的股票
3. 选择前5名
4. 按评分加权分配100万资金
5. 记录每日持仓和收益

**输出**：
- 每日记录：`runs/zz500_backtest/daily_records.csv`
- 终端显示：总收益率、最终资产等

## 配置说明

配置文件：`config/zz500_rl_single_stock.yaml`

### 关键参数

```yaml
environment:
  stock_dim: 1  # 单股票模型
  initial_amount: 1000000  # 初始资金100万

training:
  algorithm: "ppo"
  total_timesteps: 200000

zz500_selection:
  portfolio_size: 5  # 持仓5只股票
  score_threshold: 0.9  # 评分阈值
  score_mapping: "sigmoid"  # Action映射方法
  position_sizing: "score_weighted"  # 评分加权
  
  # 风险管理
  stop_loss_pct: -0.05  # 止损-5%
  take_profit_pct: 0.20  # 止盈+20%
  rl_exit_threshold: -0.2  # RL退出阈值
```

## 核心组件

### 1. 股票列表加载器
**文件**：`finquant/data/sources/zz500_loader.py`

```python
from finquant.data.sources.zz500_loader import load_zz500_stocks

stocks = load_zz500_stocks("data/processed/zz500_data.db")
# 返回：['000001.SZ', '000002.SZ', ..., '603999.SH']
```

### 2. RL评分器
**文件**：`finquant/selection/rl_scorer.py`

```python
from finquant.selection.rl_scorer import RLStockScorer

scorer = RLStockScorer(
    model_path=Path("models/zz500_single_stock/ppo_zz500_single.zip"),
    indicators=["macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"],
    score_mapping="sigmoid"
)

scores = scorer.score_stocks(market_df, date="2024-07-01")
# 返回：{'000001.SZ': 0.85, '000002.SZ': 0.92, ...}
```

### 3. Action到评分映射

**Sigmoid映射**（推荐）：
```python
score = 1 / (1 + exp(-action))
```

- action > 2.2 → score > 0.9（强买入信号）
- action = 0 → score = 0.5（中性）
- action < -2.2 → score < 0.1（强卖出信号）

## 架构优势

### 为什么选择单股票模型？

**传统多股票模型**：
- 观察空间：1 + (2+7) × 343 = 3088维
- 训练慢，收敛困难

**单股票模型**：
- 观察空间：1 + (2+7) × 1 = 10维
- 训练快，易收敛
- 可扩展到任意数量股票

### 评分加权示例

假设选出5只股票，评分为：
- 股票A：0.95
- 股票B：0.92
- 股票C：0.91
- 股票D：0.90
- 股票E：0.88

总评分：4.56

权重分配：
- 股票A：0.95/4.56 = 20.8% → 20.8万
- 股票B：0.92/4.56 = 20.2% → 20.2万
- 股票C：0.91/4.56 = 20.0% → 20.0万
- 股票D：0.90/4.56 = 19.7% → 19.7万
- 股票E：0.88/4.56 = 19.3% → 19.3万

## 验证步骤

### 1. 验证模型训练
```bash
# 检查模型文件
ls -lh models/zz500_single_stock/

# 查看元数据
cat models/zz500_single_stock/training_metadata.json
```

### 2. 测试评分器
```python
from pathlib import Path
from finquant.selection.rl_scorer import RLStockScorer
from finquant.data.sources.db_daily import DbDailyDataSource
from finquant.features.technical import compute_indicators

# 加载数据
data_source = DbDailyDataSource(db_path=Path("data/processed/zz500_data.db"))
df = data_source.download(["000001.SZ"], "2024-07-01", "2024-07-01")
df = compute_indicators(df, ["macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"])

# 初始化评分器
scorer = RLStockScorer(
    model_path=Path("models/zz500_single_stock/ppo_zz500_single.zip"),
    indicators=["macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"],
    score_mapping="sigmoid"
)

# 评分
scores = scorer.score_stocks(df, "2024-07-01")
print(scores)  # {'000001.SZ': 0.XX}
```

### 3. 验证回测结果
```bash
# 查看每日记录
head runs/zz500_backtest/daily_records.csv

# 分析收益
python -c "
import pandas as pd
df = pd.read_csv('runs/zz500_backtest/daily_records.csv')
print(df[['date', 'cash', 'selected_tickers']].head(10))
"
```

## 下一步优化

1. **增强风险管理**：
   - 使用5分钟数据进行盘中监控
   - 添加追踪止损
   - 基于波动率的动态仓位

2. **模型改进**：
   - 添加基本面特征（PE、ROE等）
   - 融合市场状态分类
   - 集成多个模型

3. **回测增强**：
   - 交易成本建模
   - 滑点模拟
   - 真实订单执行

4. **生产部署**：
   - 实时数据管道
   - 自动交易执行
   - 监控和告警

## 常见问题

### Q1: 为什么数据库只有343只股票而不是355只？
A: 部分股票可能已退市或暂停交易，343只是当前可用的ZZ500成分股数量。

### Q2: 如果没有股票评分>0.9怎么办？
A: 根据配置，系统会保持现有持仓数量（可能<5只），不会强制买入低评分股票。

### Q3: 如何调整评分阈值？
A: 修改配置文件中的`zz500_selection.score_threshold`参数，如改为0.85。

### Q4: 训练需要多长时间？
A: 取决于硬件配置，单股票模型通常在CPU上训练20-30分钟，GPU上5-10分钟。

### Q5: 如何使用不同的RL算法？
A: 修改配置文件中的`training.algorithm`为"sac"或"td3"。

## 技术支持

如有问题，请查看：
- 计划文档：`.claude/plans/finsys-train-rl-zz500-db-5-zz500-db-rl-fluffy-octopus.md`
- 日志文件：`logs/zz500_rl/`
- 项目文档：`docs/`
