# 选股详细输出功能说明

## 功能概述

在运行 `finquant selection run` 命令时，使用 `--verbose` 参数可以查看每只股票的详细筛选过程和选择原因。

## 使用方法

### 命令行

```bash
finquant selection run \
  --config config/selection_breakout.yaml \
  --start 2020-08-01 \
  --end 2020-12-31 \
  --output-dir data/selection_breakout \
  --verbose
```

### Python API

```python
from finquant.config.settings import load_config
from finquant.data.pipeline import DataPipeline
from finquant.selection import create_strategy
import logging

# 配置日志级别
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 加载配置
config = load_config("config/selection_breakout.yaml")

# 加载数据
pipeline = DataPipeline(config)
market_df = pipeline.fetch()
index_df = market_df[market_df['tic'] == config.stocks[0]].copy()

# 创建策略（verbose=True）
strategy = create_strategy(config, verbose=True)

# 运行选股
result = strategy.select(market_df, index_df, "2020-08-03")
```

## 输出内容

### 1. 配置参数

显示当前使用的策略参数：

```
======================================================================
MA突破选股 - 2020-08-03
======================================================================
配置参数:
  MA周期: [120, 250]
  放量倍数: 1.5x
  突破阈值: 1.05 (5%)
  回溯天数: 60
  确认天数: 3
  防抖模式: threshold
  选股数量: 10
```

### 2. 筛选过程

显示候选股票数和通过筛选的股票数：

```
候选股票数: 10
✓ 通过突破筛选: 3 只股票
```

如果没有股票通过筛选：

```
候选股票数: 10
✗ 无符合突破条件的股票
```

### 3. 排除原因

显示被排除的股票及原因：

```
排除股票:
  ST600001.SH: ST stock
  600002.SH: halted (volume=0)
```

### 4. 选中股票详情

显示每只选中股票的详细信息：

```
最终选中: 2 只股票
股票详情:

  600030.SH:
    评分: 0.8234
    收盘价: 125.50
    MA120: 120.30 (突破 +4.32%)
    成交量: 8,500,000 (1.7x 均量)

  000100.SZ:
    评分: 0.7156
    收盘价: 45.80
    MA250: 43.20 (突破 +6.02%)
    成交量: 12,300,000 (1.5x 均量)
```

## 输出字段说明

| 字段 | 说明 |
|------|------|
| 评分 | 综合评分（0-1范围），60%突破强度 + 40%放量强度 |
| 收盘价 | 当日收盘价 |
| MA120/MA250 | 突破的移动平均线及突破幅度 |
| 成交量 | 当日成交量及相对20日均量的倍数 |

## 评分计算

评分由两部分组成：

1. **突破强度** (60%权重)
   - 计算公式：`(收盘价 - MA) / MA`
   - 例如：收盘价125.50，MA120为120.30
   - 突破强度：`(125.50 - 120.30) / 120.30 = 0.0432` (4.32%)

2. **放量强度** (40%权重)
   - 计算公式：`(成交量 - 20日均量) / 20日均量`
   - 例如：成交量8,500,000，均量5,000,000
   - 放量强度：`(8,500,000 - 5,000,000) / 5,000,000 = 0.70` (70%)

3. **综合评分**
   - 原始分：`0.6 × 0.0432 + 0.4 × 0.70 = 0.306`
   - 归一化：`tanh(0.306 × 5) = 0.8234`

## 非Verbose模式

不使用 `--verbose` 参数时，只显示简洁的汇总信息：

```
2020-08-03: uptrend, 2 stocks selected
2020-08-04: uptrend, 1 stocks selected
2020-08-05: oscillation, 0 stocks selected
```

## 适用策略

目前 verbose 功能支持：

- ✅ **Breakout Strategy** (MA突破策略)
- ⚠️ **Factor-Based Strategy** (因子策略) - 暂不支持详细输出

## 调试建议

当选股结果不符合预期时，使用 `--verbose` 可以帮助诊断：

1. **无股票选中**
   - 检查配置参数是否过于严格
   - 查看候选股票数是否为0（数据问题）
   - 确认是否所有股票都被排除

2. **选中股票过多**
   - 检查突破阈值是否过低
   - 查看放量倍数是否过小
   - 考虑使用更严格的防抖模式

3. **评分异常**
   - 查看突破幅度和放量倍数
   - 确认数据质量（价格、成交量）
   - 检查MA计算是否正确

## 示例输出

完整的verbose输出示例：

```bash
$ finquant selection run \
    --config config/selection_breakout.yaml \
    --start 2020-08-03 \
    --end 2020-08-03 \
    --verbose

Running selection from 2020-08-03 to 2020-08-03...
Processing 1 trading days...

======================================================================
MA突破选股 - 2020-08-03
======================================================================
配置参数:
  MA周期: [120, 250]
  放量倍数: 1.5x
  突破阈值: 1.05 (5%)
  回溯天数: 60
  确认天数: 3
  防抖模式: threshold
  选股数量: 10

候选股票数: 10
✓ 通过突破筛选: 2 只股票

最终选中: 2 只股票
股票详情:

  600030.SH:
    评分: 0.8234
    收盘价: 125.50
    MA120: 120.30 (突破 +4.32%)
    成交量: 8,500,000 (1.7x 均量)

  000100.SZ:
    评分: 0.7156
    收盘价: 45.80
    MA250: 43.20 (突破 +6.02%)
    成交量: 12,300,000 (1.5x 均量)

Completed: 1 selection results saved to data/selection_breakout
```

## 相关文档

- [MA突破策略快速指南](breakout-strategy-quickstart.md)
- [MA突破策略完整文档](breakout-strategy-guide.md)
- [评分归一化说明](breakout-strategy-score-normalization.md)
