# 因子选股 + RL训练 快速开始

## 完整流程示例

### 步骤1: 运行因子选股

```bash
# 对2024年进行选股（已完成）
python -m finquant.cli.main selection run \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --verbose

# 结果: 242个交易日的选股结果保存在 data/selection/
```

### 步骤2: 查看选股结果

```bash
# 查看可用的选股结果
python -c "
import json
from pathlib import Path

files = sorted(Path('data/selection').glob('2024-*_selection.json'))
print(f'共有 {len(files)} 个选股结果')

# 查看某日详情
with open(files[10], 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f'\n日期: {data[\"date\"]}')
    print(f'市场状态: {data[\"market_state\"]}')
    print(f'选中股票: {len(data[\"selected_tickers\"])} 只')
    print(f'活跃因子: {data[\"active_factors\"]}')
    print(f'\nTop 5 股票及评分:')
    for ticker, score in list(data[\"scores\"].items())[:5]:
        print(f'  {ticker}: {score:.4f}')
"
```

### 步骤3: 基于选股结果训练RL模型

```bash
# 使用2024-01-15的选股结果训练
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --test-days 10 \
  --algo ppo \
  --timesteps 50000 \
  --verbose
```

**这个命令会**:
1. 加载2024-01-15的选股结果（约9只股票）
2. 获取这些股票在2023-12-16到2024-01-25的5分钟数据
3. 训练PPO模型（50000步）
4. 在测试集上回测
5. 输出报告到 `runs/selection/2024-01-15_*/`

### 步骤4: 查看训练结果

训练完成后会显示：

```
============================================================
训练完成!
============================================================

选股日期: 2024-01-15
市场状态: oscillation
活跃因子: low_volatility, low_turnover, high_dividend

股票池 (9只):
  1. 600000.SH (score: 0.5080)
  2. 600016.SH (score: 0.3687)
  3. 600036.SH (score: 0.1480)
  ...

模型: PPO
训练步数: 50,000

回测结果:
  年化收益率: 15.23%
  最大回撤: -8.56%
  夏普比率: 1.78
  胜率: 62.34%
  总交易次数: 145

输出目录: runs/selection/2024-01-15_143052
```

## 对比实验

### 实验1: 选股 vs 随机选股

```bash
# 1. 使用选股结果
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --test-days 10 \
  --algo ppo \
  --timesteps 50000

# 2. 随机选择10只股票
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 30 \
  --test-days 10 \
  --algo ppo \
  --timesteps 50000

# 3. 比较结果
# 查看 runs/selection/*/metadata.json
# 查看 runs/random/*/metadata.json
```

### 实验2: 不同算法对比

```bash
# PPO
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --algo ppo \
  --timesteps 50000

# SAC
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --algo sac \
  --timesteps 50000

# TD3
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --algo td3 \
  --timesteps 50000
```

## 批量测试

测试多个日期的选股效果：

```bash
# 创建批量测试脚本
cat > batch_test.sh << 'EOF'
#!/bin/bash

dates=(
  "2024-01-15"
  "2024-02-20"
  "2024-03-15"
  "2024-04-15"
  "2024-05-15"
)

for date in "${dates[@]}"; do
  echo "Testing $date..."
  python scripts/selection_rl_train.py \
    --selection-date $date \
    --train-days 30 \
    --test-days 10 \
    --algo ppo \
    --timesteps 50000 \
    --output-dir runs/batch_test/$date
done

# 汇总结果
python -c "
import json
from pathlib import Path

print('\n批量测试结果汇总:')
print('=' * 60)
for d in sorted(Path('runs/batch_test').iterdir()):
    if (d / 'metadata.json').exists():
        meta = json.load(open(d / 'metadata.json'))
        report = meta['report']
        print(f\"{meta['selection_date']}:\")
        print(f\"  年化收益: {report['annual_return']:.2%}\")
        print(f\"  最大回撤: {report['max_drawdown']:.2%}\")
        print(f\"  夏普比率: {report['sharpe_ratio']:.2f}\")
"
EOF

chmod +x batch_test.sh
./batch_test.sh
```

## 常用命令

### 查看选股统计

```bash
python -c "
import json
from pathlib import Path
from collections import Counter

files = sorted(Path('data/selection').glob('2024-*_selection.json'))

# 市场状态分布
states = []
for f in files:
    with open(f) as fp:
        states.append(json.load(fp)['market_state'])

print('市场状态分布:')
for state, count in Counter(states).items():
    print(f'  {state}: {count} 天 ({count/len(files)*100:.1f}%)')

# 最常被选中的股票
all_stocks = []
for f in files:
    with open(f) as fp:
        all_stocks.extend(json.load(fp)['selected_tickers'])

print('\n最常被选中的股票 (Top 10):')
for stock, count in Counter(all_stocks).most_common(10):
    print(f'  {stock}: {count} 次 ({count/len(files)*100:.1f}%)')
"
```

### 检查数据库状态

```bash
python -c "
import sqlite3
from pathlib import Path

db = Path('data/processed/zz500_data.db')
conn = sqlite3.connect(db)
cursor = conn.cursor()

# 日频数据
cursor.execute('SELECT COUNT(*), MIN(date), MAX(date) FROM daily_data')
count, min_d, max_d = cursor.fetchone()
print(f'日频数据: {count:,} 条, {min_d} ~ {max_d}')

# 5分钟数据
cursor.execute('SELECT COUNT(*), MIN(date), MAX(date) FROM minute_data')
count, min_d, max_d = cursor.fetchone()
print(f'5分钟数据: {count:,} 条, {min_d} ~ {max_d}')

# 股票数量
cursor.execute('SELECT COUNT(DISTINCT code) FROM daily_data')
print(f'日频股票数: {cursor.fetchone()[0]}')

cursor.execute('SELECT COUNT(DISTINCT code) FROM minute_data')
print(f'5分钟股票数: {cursor.fetchone()[0]}')
"
```

## 参数调优建议

### 快速测试（5-10分钟）
```bash
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 15 \
  --test-days 5 \
  --timesteps 10000
```

### 标准训练（30-60分钟）
```bash
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --test-days 10 \
  --timesteps 100000
```

### 深度训练（2-4小时）
```bash
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 60 \
  --test-days 20 \
  --timesteps 500000
```

## 故障排查

### 问题1: 找不到选股结果

```bash
# 检查选股结果目录
ls -lh data/selection/

# 如果为空，重新运行选股
python -m finquant.cli.main selection run \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --verbose
```

### 问题2: 5分钟数据不足

```bash
# 检查数据库
python -c "
import sqlite3
conn = sqlite3.connect('data/processed/zz500_data.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM minute_data WHERE code = \"600000.SH\"')
print(f'600000.SH 5分钟数据量: {cursor.fetchone()[0]:,}')
"

# 如果数据不足，重新下载
python scripts/download_zz500_data.py
```

### 问题3: 训练失败

```bash
# 使用更少的训练步数
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --timesteps 10000 \
  --verbose

# 检查日志
tail -f logs/selection_rl_train.log
```

## 下一步

1. **阅读详细文档**: [docs/selection-training-guide.md](../docs/selection-training-guide.md)
2. **查看因子选股规范**: [specs/004-factor-selection-trading/spec.md](../specs/004-factor-selection-trading/spec.md)
3. **实现动态候选池**: 每日更新股票池，模拟实盘
4. **添加更多因子**: 基本面、舆情、宏观因子
5. **优化训练策略**: 超参数调优、集成学习

## 相关文件

- `scripts/selection_rl_train.py` - 基于选股结果的RL训练脚本
- `scripts/random_rl_train.py` - 随机选股的RL训练脚本（对比基准）
- `finquant/selection/` - 因子选股模块
- `finquant/training/` - RL训练模块
- `data/selection/` - 选股结果存储目录
- `runs/selection/` - 训练输出目录
