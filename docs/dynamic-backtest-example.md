# 动态回测完整示例

本文档提供一个完整的端到端示例，展示如何：
1. 训练RL模型
2. 使用模型进行动态回测
3. 分析回测结果

## 完整流程示例

### 步骤1: 准备数据（已完成）

```bash
# 确认数据库存在
ls -lh data/processed/zz500_data.db

# 确认选股结果存在
ls data/selection/ | head -5
```

### 步骤2: 训练RL模型

#### 选项A: 基于选股结果训练（推荐用于动态回测）

```bash
# 使用2024-01-15的选股结果训练
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --test-days 10 \
  --algo ppo \
  --timesteps 100000 \
  --verbose

# 输出示例:
# ============================================================
# 训练完成!
# ============================================================
# 
# 选股日期: 2024-01-15
# 市场状态: oscillation
# 活跃因子: low_volatility, low_turnover, high_dividend
# 
# 股票池 (9只):
#   1. 600000.SH (score: 0.5080)
#   2. 600016.SH (score: 0.3687)
#   ...
# 
# 模型: PPO
# 训练步数: 100,000
# 
# 回测结果:
#   年化收益率: 15.23%
#   最大回撤: -8.56%
#   夏普比率: 1.78
#   胜率: 62.34%
#   总交易次数: 145
# 
# 输出目录: runs/selection/2024-01-15_143052
# 模型文件: runs/selection/2024-01-15_143052/ppo_20240115_abc123.zip
```

#### 选项B: 随机股票池训练

```bash
# 随机选择10只股票训练
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 60 \
  --test-days 30 \
  --algo ppo \
  --timesteps 100000 \
  --verbose

# 输出示例:
# ============================================================
# 训练完成!
# ============================================================
# 
# 股票池 (10只):
#   1. 600000.SH
#   2. 000001.SZ
#   ...
# 
# 时间范围:
#   训练: 2023-11-15 ~ 2024-01-15 (60天)
#   测试: 2024-01-16 ~ 2024-02-15 (30天)
# 
# 模型: PPO
# 训练步数: 100,000
# 
# 回测结果:
#   年化收益率: 12.45%
#   最大回撤: -10.23%
#   夏普比率: 1.52
#   胜率: 58.67%
#   总交易次数: 132
# 
# 输出目录: runs/random/20240502_143052
# 模型文件: runs/random/20240502_143052/ppo_20240115_xyz789.zip
```

### 步骤3: 运行动态回测

#### 使用选股训练的模型（推荐）

```bash
# 在2024年2月进行动态回测
python -m finquant.cli.main selection backtest \
  --model runs/selection/2024-01-15_143052/ppo_20240115_abc123.zip \
  --start 2024-02-01 \
  --end 2024-02-29 \
  --verbose

# 回测过程输出:
# Running dynamic backtest from 2024-02-01 to 2024-02-29...
# Using model: runs/selection/2024-01-15_143052/ppo_20240115_abc123.zip
# Fetching market data...
# Initializing backtester...
# Running backtest...
# 2024-02-01: oscillation, 9 stocks, value=1002340.00, return=0.23%
# 2024-02-02: oscillation, 8 stocks, value=1004567.00, return=0.22%
# 2024-02-05: oscillation, 9 stocks, value=1006234.00, return=0.17%
# ...
# 2024-02-29: oscillation, 9 stocks, value=1045230.00, return=0.31%
```

#### 使用随机训练的模型（对比）

```bash
# 同样可以使用random_rl_train训练的模型
python -m finquant.cli.main selection backtest \
  --model runs/random/20240502_143052/ppo_20240115_xyz789.zip \
  --start 2024-02-01 \
  --end 2024-02-29 \
  --verbose

# 注意：random_rl_train训练的模型在固定股票池上训练
# 在动态回测中每天更换股票池，可能表现不如专门训练的模型
```

### 步骤4: 查看回测结果

```bash
# 回测完成后会显示汇总结果:
# ============================================================
# Dynamic Backtest Results
# ============================================================
# 
# Period: 2024-02-01 to 2024-02-29
# Trading days: 20
# 
# Initial amount: ¥1,000,000.00
# Final value: ¥1,045,230.00
# Total return: 4.52%
# Annual return: 68.34%
# Max drawdown: -3.21%
# Sharpe ratio: 2.15
# Win rate: 65.00%
# Avg daily return: 0.23%
# Volatility: 1.45%
# Total trades: 156
# 
# Report saved to: runs/dynamic_backtest/backtest_2024-02-01_2024-02-29.json
```

### 步骤5: 分析回测报告

```bash
# 查看详细报告
cat runs/dynamic_backtest/backtest_2024-02-01_2024-02-29.json | python -m json.tool | head -50

# 或者使用Python分析
python -c "
import json
from pathlib import Path

# 加载报告
report_path = Path('runs/dynamic_backtest/backtest_2024-02-01_2024-02-29.json')
with open(report_path) as f:
    report = json.load(f)

# 显示前5天的详细记录
print('前5天的交易记录:')
for record in report['daily_records'][:5]:
    print(f\"\\n{record['date']}:\")
    print(f\"  市场状态: {record['market_state']}\")
    print(f\"  选中股票: {len(record['selected_tickers'])} 只\")
    print(f\"  开盘价值: ¥{record['start_value']:,.2f}\")
    print(f\"  收盘价值: ¥{record['end_value']:,.2f}\")
    print(f\"  当日收益: {record['daily_return']:.2%}\")
    print(f\"  持仓: {list(record['positions'].keys())}\")
    print(f\"  现金: ¥{record['cash']:,.2f}\")
"
```

## 关键问题解答

### Q: random_rl_train训练的模型可以用于动态回测吗？

**答：可以，但效果可能不理想**

#### 技术上可行

```bash
# random_rl_train训练的模型是标准的RL模型
# 可以直接用于动态回测
python -m finquant.cli.main selection backtest \
  --model runs/random/*/ppo_*.zip \
  --start 2024-02-01 \
  --end 2024-02-29
```

#### 为什么效果可能不好？

1. **训练环境 vs 回测环境不匹配**

   ```
   训练时（random_rl_train）:
   - 固定10只股票
   - 股票池不变
   - 模型学习这10只股票的特征
   
   回测时（动态回测）:
   - 每天选股，股票池变化
   - 可能出现训练时没见过的股票
   - 模型无法识别新股票的特征
   ```

2. **泛化能力问题**

   ```python
   # 训练时的observation
   # [cash, price1, holdings1, price2, holdings2, ..., indicators1, indicators2, ...]
   # 模型学习了特定股票的价格范围、波动特征等
   
   # 回测时遇到新股票
   # 价格范围、波动特征可能完全不同
   # 模型的决策可能不准确
   ```

#### 对比实验

```bash
# 实验1: 使用selection_rl_train训练的模型
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --timesteps 100000

python -m finquant.cli.main selection backtest \
  --model runs/selection/*/ppo_*.zip \
  --start 2024-02-01 \
  --end 2024-02-29

# 实验2: 使用random_rl_train训练的模型
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 30 \
  --timesteps 100000

python -m finquant.cli.main selection backtest \
  --model runs/random/*/ppo_*.zip \
  --start 2024-02-01 \
  --end 2024-02-29

# 比较两者的结果
# 预期: 实验1的效果更好，因为训练时也使用了选股结果
```

### 最佳实践建议

#### 1. 为动态回测专门训练模型

```bash
# 使用选股结果训练，更接近实际场景
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 60 \
  --timesteps 200000
```

#### 2. 使用更长的训练周期

```bash
# 更长的训练周期 = 更多样化的数据
--train-days 60  # 而不是30
```

#### 3. 增加训练步数

```bash
# 更多的训练步数 = 更好的泛化能力
--timesteps 200000  # 而不是100000
```

#### 4. 在多个选股结果上训练（未来功能）

```bash
# 理想情况：在多个不同的选股结果上训练
# 这样模型见过更多不同的股票组合
# 泛化能力更强

# 未来可能的实现:
python scripts/selection_rl_train.py \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --dynamic-pool \
  --timesteps 500000
```

## 完整对比实验

### 实验设计

```bash
#!/bin/bash
# 对比实验：selection vs random 训练的模型在动态回测中的表现

# 1. 训练selection模型
echo "Training selection model..."
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --timesteps 100000 \
  --output-dir runs/experiment/selection

# 2. 训练random模型
echo "Training random model..."
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 30 \
  --timesteps 100000 \
  --output-dir runs/experiment/random

# 3. 使用selection模型进行动态回测
echo "Backtesting with selection model..."
python -m finquant.cli.main selection backtest \
  --model runs/experiment/selection/*/ppo_*.zip \
  --start 2024-02-01 \
  --end 2024-02-29 \
  --output-dir runs/experiment/backtest_selection

# 4. 使用random模型进行动态回测
echo "Backtesting with random model..."
python -m finquant.cli.main selection backtest \
  --model runs/experiment/random/*/ppo_*.zip \
  --start 2024-02-01 \
  --end 2024-02-29 \
  --output-dir runs/experiment/backtest_random

# 5. 比较结果
echo "Comparing results..."
python -c "
import json
from pathlib import Path

# 加载两个回测报告
selection_report = json.load(open('runs/experiment/backtest_selection/backtest_2024-02-01_2024-02-29.json'))
random_report = json.load(open('runs/experiment/backtest_random/backtest_2024-02-01_2024-02-29.json'))

print('\\n' + '='*60)
print('对比结果')
print('='*60)

print(f\"\\n{'指标':<20} {'Selection模型':<20} {'Random模型':<20}\")
print('-'*60)
print(f\"{'总收益率':<20} {selection_report['total_return']:>18.2%} {random_report['total_return']:>18.2%}\")
print(f\"{'年化收益率':<20} {selection_report['annual_return']:>18.2%} {random_report['annual_return']:>18.2%}\")
print(f\"{'最大回撤':<20} {selection_report['max_drawdown']:>18.2%} {random_report['max_drawdown']:>18.2%}\")
print(f\"{'夏普比率':<20} {selection_report['sharpe_ratio']:>18.2f} {random_report['sharpe_ratio']:>18.2f}\")
print(f\"{'胜率':<20} {selection_report['win_rate']:>18.2%} {random_report['win_rate']:>18.2%}\")
print(f\"{'平均日收益':<20} {selection_report['avg_daily_return']:>18.4%} {random_report['avg_daily_return']:>18.4%}\")
print(f\"{'波动率':<20} {selection_report['volatility']:>18.4%} {random_report['volatility']:>18.4%}\")
print(f\"{'总交易次数':<20} {selection_report['total_trades']:>18} {random_report['total_trades']:>18}\")
"
```

### 预期结果

```
============================================================
对比结果
============================================================

指标                   Selection模型         Random模型         
------------------------------------------------------------
总收益率                          4.52%              2.31%
年化收益率                       68.34%             35.12%
最大回撤                         -3.21%             -5.67%
夏普比率                           2.15               1.23
胜率                             65.00%             52.00%
平均日收益                        0.23%              0.12%
波动率                            1.45%              2.34%
总交易次数                          156                142
```

**结论**：
- Selection模型在动态回测中表现更好
- Random模型也能工作，但效果较差
- 建议为动态回测专门训练模型

## 总结

### ✅ 可以使用random_rl_train训练的模型

- 技术上完全可行
- 模型格式兼容
- 可以正常运行

### ⚠️ 但效果可能不理想

- 训练环境与回测环境不匹配
- 泛化能力有限
- 建议使用selection_rl_train

### 🎯 最佳实践

1. 使用`selection_rl_train.py`训练模型
2. 使用更长的训练周期（60天）
3. 使用更多的训练步数（200000）
4. 在动态回测前先做短期测试

### 📚 相关文档

- [动态回测指南](dynamic-backtest-guide.md)
- [选股+训练指南](selection-training-guide.md)
- [快速开始](selection-training-quickstart.md)
