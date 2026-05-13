# 动态回测指南：每日选股 + RL交易

本文档介绍如何使用动态回测功能，模拟真实交易场景：每天早上选股，盘中使用RL模型进行交易。

## 概述

### 动态回测流程

```
每个交易日：
  09:00 - 执行因子选股，获取当日候选股票池
    ↓
  09:30-15:00 - 使用训练好的RL模型在候选池上交易
    ↓
  15:00 - 记录当日持仓、收益等信息
    ↓
  次日重复
```

### 与静态回测的区别

| 特性 | 静态回测 | 动态回测 |
|------|---------|---------|
| 股票池 | 固定不变 | 每日更新 |
| 选股时机 | 一次性 | 每日开盘前 |
| 真实性 | 较低 | 更接近实盘 |
| 计算量 | 较小 | 较大 |

## 快速开始

### 步骤1: 训练RL模型

首先需要一个训练好的RL模型。可以使用以下任一方式：

#### 方式A: 使用选股结果训练

```bash
# 基于2024-01-15的选股结果训练
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --test-days 10 \
  --algo ppo \
  --timesteps 100000

# 模型保存在: runs/selection/2024-01-15_*/ppo_*.zip
```

#### 方式B: 使用随机股票池训练

```bash
# 随机选择10只股票训练
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 60 \
  --test-days 30 \
  --algo ppo \
  --timesteps 100000

# 模型保存在: runs/random/*/ppo_*.zip
```

### 步骤2: 运行动态回测

```bash
# 使用训练好的模型进行动态回测
python -m finquant.cli.main selection backtest \
  --model runs/selection/2024-01-15_143052/ppo_20240115_abc123.zip \
  --start 2024-02-01 \
  --end 2024-02-29 \
  --verbose
```

**参数说明**：
- `--model`: 训练好的RL模型路径（.zip文件）
- `--start`: 回测开始日期
- `--end`: 回测结束日期
- `--config`: 配置文件（默认: config/selection_test.yaml）
- `--output-dir`: 输出目录（默认: runs/dynamic_backtest）
- `--verbose`: 显示详细信息

### 步骤3: 查看回测结果

回测完成后会显示：

```
============================================================
Dynamic Backtest Results
============================================================

Period: 2024-02-01 to 2024-02-29
Trading days: 20

Initial amount: ¥1,000,000.00
Final value: ¥1,045,230.00
Total return: 4.52%
Annual return: 68.34%
Max drawdown: -3.21%
Sharpe ratio: 2.15
Win rate: 65.00%
Avg daily return: 0.23%
Volatility: 1.45%
Total trades: 156

Report saved to: runs/dynamic_backtest/backtest_2024-02-01_2024-02-29.json
```

## 详细说明

### 回测报告格式

回测报告保存为JSON格式，包含以下信息：

```json
{
  "start_date": "2024-02-01",
  "end_date": "2024-02-29",
  "initial_amount": 1000000.0,
  "final_value": 1045230.0,
  "total_return": 0.0452,
  "annual_return": 0.6834,
  "max_drawdown": -0.0321,
  "sharpe_ratio": 2.15,
  "total_trades": 156,
  "win_rate": 0.65,
  "avg_daily_return": 0.0023,
  "volatility": 0.0145,
  "daily_records": [
    {
      "date": "2024-02-01",
      "market_state": "oscillation",
      "selected_tickers": ["600000.SH", "600016.SH", ...],
      "selection_scores": {
        "600000.SH": 0.5080,
        "600016.SH": 0.3687,
        ...
      },
      "start_value": 1000000.0,
      "end_value": 1002340.0,
      "daily_return": 0.00234,
      "positions": {
        "600000.SH": 1000,
        "600016.SH": 500
      },
      "cash": 980000.0,
      "trades": [...]
    },
    ...
  ]
}
```

### 每日记录说明

每个交易日的记录包含：

- **选股信息**:
  - `market_state`: 市场状态
  - `selected_tickers`: 选中的股票
  - `selection_scores`: 股票评分

- **交易统计**:
  - `start_value`: 开盘时账户价值
  - `end_value`: 收盘时账户价值
  - `daily_return`: 当日收益率

- **持仓信息**:
  - `positions`: 持仓股票及数量
  - `cash`: 现金余额

- **交易动作**:
  - `trades`: 当日所有交易动作

## 使用场景

### 场景1: 验证选股策略

比较使用选股 vs 不使用选股的效果：

```bash
# 1. 训练两个模型
# 模型A: 基于选股结果训练
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --timesteps 100000

# 模型B: 随机股票池训练
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 30 \
  --timesteps 100000

# 2. 使用模型A进行动态回测（每日选股）
python -m finquant.cli.main selection backtest \
  --model runs/selection/*/ppo_*.zip \
  --start 2024-02-01 \
  --end 2024-03-31

# 3. 使用模型B进行静态回测（固定股票池）
# 注意：模型B无法用于动态回测，因为它是在固定股票池上训练的
```

### 场景2: 测试不同市场周期

在不同市场环境下测试策略表现：

```bash
# 上涨市场（假设2024-03-01到2024-03-31是上涨期）
python -m finquant.cli.main selection backtest \
  --model runs/selection/*/ppo_*.zip \
  --start 2024-03-01 \
  --end 2024-03-31

# 下跌市场（假设2024-08-01到2024-08-31是下跌期）
python -m finquant.cli.main selection backtest \
  --model runs/selection/*/ppo_*.zip \
  --start 2024-08-01 \
  --end 2024-08-31

# 震荡市场
python -m finquant.cli.main selection backtest \
  --model runs/selection/*/ppo_*.zip \
  --start 2024-01-01 \
  --end 2024-01-31
```

### 场景3: 长期回测

测试策略的长期稳定性：

```bash
# 全年回测
python -m finquant.cli.main selection backtest \
  --model runs/selection/*/ppo_*.zip \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --verbose
```

## 重要说明

### ⚠️ 模型兼容性

**random_rl_train训练的模型可以用于动态回测吗？**

**答案：可以，但有限制**

1. **技术上可行**：
   - random_rl_train训练的模型是标准的RL模型（PPO/SAC/TD3）
   - 动态回测系统可以加载并使用这些模型

2. **实际效果可能不佳**：
   - random_rl_train在**固定股票池**上训练
   - 动态回测每天**更换股票池**
   - 模型可能无法适应新的股票

3. **最佳实践**：
   ```bash
   # 推荐：使用多个选股结果训练，提高泛化能力
   python scripts/selection_rl_train.py \
     --selection-date 2024-01-15 \
     --train-days 60 \
     --timesteps 200000
   
   # 然后用于动态回测
   python -m finquant.cli.main selection backtest \
     --model runs/selection/*/ppo_*.zip \
     --start 2024-03-01 \
     --end 2024-03-31
   ```

### 模型训练建议

为了在动态回测中获得更好的效果：

1. **使用更长的训练周期**：
   ```bash
   --train-days 60  # 而不是30
   ```

2. **使用更多的训练步数**：
   ```bash
   --timesteps 200000  # 而不是100000
   ```

3. **在多个选股结果上训练**（未来功能）：
   ```bash
   # 未来可能支持
   python scripts/selection_rl_train.py \
     --start 2024-01-01 \
     --end 2024-03-31 \
     --dynamic-pool
   ```

## 性能优化

### 加速回测

动态回测计算量较大，可以通过以下方式加速：

1. **缩短回测周期**：
   ```bash
   # 先测试一个月
   --start 2024-02-01 --end 2024-02-29
   ```

2. **减少股票数量**：
   ```yaml
   # config/selection_test.yaml
   selection:
     top_k: 5  # 从10减少到5
   ```

3. **使用更快的模型**：
   ```bash
   # SAC通常比PPO快
   --algo sac
   ```

### 内存优化

如果遇到内存不足：

1. **分段回测**：
   ```bash
   # 分成多个月份
   for month in 01 02 03; do
     python -m finquant.cli.main selection backtest \
       --model model.zip \
       --start 2024-${month}-01 \
       --end 2024-${month}-31
   done
   ```

2. **减少数据缓存**：
   - 关闭不必要的日志
   - 不保存中间结果

## 结果分析

### 分析脚本示例

```python
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 加载回测报告
report_path = Path("runs/dynamic_backtest/backtest_2024-02-01_2024-02-29.json")
with open(report_path) as f:
    report = json.load(f)

# 提取每日收益
daily_returns = [r["daily_return"] for r in report["daily_records"]]
dates = [r["date"] for r in report["daily_records"]]

# 计算累计收益
cumulative_returns = [(1 + r) for r in daily_returns]
for i in range(1, len(cumulative_returns)):
    cumulative_returns[i] *= cumulative_returns[i-1]

# 绘制收益曲线
plt.figure(figsize=(12, 6))
plt.plot(dates, cumulative_returns)
plt.title("Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_returns.png")

# 分析市场状态分布
states = [r["market_state"] for r in report["daily_records"]]
state_counts = pd.Series(states).value_counts()
print("\n市场状态分布:")
print(state_counts)

# 分析不同市场状态下的收益
state_returns = {}
for record in report["daily_records"]:
    state = record["market_state"]
    if state not in state_returns:
        state_returns[state] = []
    state_returns[state].append(record["daily_return"])

print("\n不同市场状态下的平均收益:")
for state, returns in state_returns.items():
    avg_return = sum(returns) / len(returns)
    print(f"  {state}: {avg_return:.4%}")
```

## 常见问题

### Q1: 动态回测很慢怎么办？

**原因**：每天都要执行选股和交易，计算量大

**解决**：
1. 先在短周期测试（1个月）
2. 减少候选股票数量
3. 使用更快的算法（SAC）

### Q2: 模型在动态回测中表现不佳？

**原因**：
- 模型在固定股票池上训练
- 每天更换股票池，模型无法适应

**解决**：
1. 使用更长的训练周期
2. 在多个不同的股票池上训练
3. 考虑使用迁移学习

### Q3: 如何比较不同策略？

**方法**：
```bash
# 策略A: 使用选股
python -m finquant.cli.main selection backtest \
  --model model_a.zip \
  --start 2024-02-01 \
  --end 2024-02-29 \
  --output-dir runs/strategy_a

# 策略B: 不使用选股（需要另外实现）
# 或者使用不同的选股配置

# 比较结果
python compare_strategies.py \
  runs/strategy_a/backtest_*.json \
  runs/strategy_b/backtest_*.json
```

## 下一步

1. **实现多模型集成**：使用多个模型投票决策
2. **添加风控模块**：止损、止盈、仓位限制
3. **优化选股策略**：更多因子、更好的权重
4. **实盘模拟**：连接实时数据源

## 相关文件

- `finquant/training/dynamic_backtest.py` - 动态回测核心模块
- `finquant/cli/main.py` - CLI命令（selection backtest）
- `scripts/selection_rl_train.py` - 训练脚本
- `docs/selection-training-guide.md` - 选股+训练指南
