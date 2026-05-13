# 市场状态分类说明

## 概述

选股策略会根据指数的技术指标自动判断市场状态，不同的市场状态会影响选股结果的解读。

## 市场状态类型

### 1. UPTREND (上升趋势)

**判断条件**：
- ADX > 25（趋势强度高）
- 收盘价 > MA50 × 1.02（价格在均线上方2%以上）

**特征**：
- 市场处于明确的上升趋势
- 突破信号更可靠
- 适合追涨策略

### 2. DOWNTREND (下降趋势)

**判断条件**：
- 收盘价 < MA50（价格在均线下方）
- ADX > 25（趋势强度高）

**特征**：
- 市场处于明确的下降趋势
- 突破信号可能是反弹
- 需要谨慎对待

### 3. OSCILLATION (震荡)

**判断条件**：
- ADX < 20（趋势强度低）

**特征**：
- 市场没有明确方向
- 价格在区间内波动
- 突破信号容易失效

### 4. VOLUME_CONTRACTION (缩量)

**判断条件**：
- 成交量 < 20日均量 × 0.8

**特征**：
- 市场交投清淡
- 观望情绪浓厚
- 突破可能缺乏持续性

### 5. STRUCTURAL (结构性行情)

**判断条件**：
- 当前为占位符，需要进一步完善

**特征**：
- 板块轮动明显
- 个股分化严重

### 6. SENTIMENT_CAUTIOUS (情绪谨慎)

**判断条件**：
- 当前为占位符，需要情绪数据

**特征**：
- 市场情绪偏谨慎
- 风险偏好降低

## 判断优先级

市场状态按以下优先级判断（从高到低）：

1. UPTREND (优先级 10)
2. DOWNTREND (优先级 9)
3. VOLUME_CONTRACTION (优先级 8)
4. STRUCTURAL (优先级 7)
5. SENTIMENT_CAUTIOUS (优先级 6)
6. OSCILLATION (优先级 5，默认兜底)

## 技术指标说明

### ADX (Average Directional Index)

- **含义**：平均趋向指标，衡量趋势强度
- **范围**：0-100
- **解读**：
  - ADX < 20：弱趋势或无趋势
  - 20 ≤ ADX ≤ 25：趋势开始形成
  - ADX > 25：强趋势
  - ADX > 50：极强趋势

### MA50 (50日移动平均线)

- **含义**：过去50个交易日的平均价格
- **作用**：判断中期趋势方向
- **解读**：
  - 价格 > MA50：中期上升趋势
  - 价格 < MA50：中期下降趋势
  - 价格 ≈ MA50：趋势不明确

### Volume MA20 Ratio (成交量相对20日均量)

- **含义**：当日成交量 / 20日平均成交量
- **解读**：
  - 比率 > 1.5：放量
  - 比率 < 0.8：缩量
  - 0.8 ≤ 比率 ≤ 1.5：正常

## 在选股中的应用

### Breakout Strategy (MA突破策略)

突破策略会根据市场状态调整解读：

```python
# 示例输出
{
  "date": "2020-07-01",
  "selected_tickers": ["600048.SH"],
  "market_state": "oscillation",  # 震荡市
  "index_metrics": {
    "close": 13.12,
    "adx": 0.94,      # ADX很低，趋势弱
    "ma50": 13.26,
    "ma50_ratio": 0.9897  # 价格略低于MA50
  }
}
```

**解读**：
- 市场处于震荡状态（ADX=0.94 < 20）
- 价格略低于MA50（98.97%）
- 此时的突破信号可靠性较低
- 建议：谨慎对待，可能需要更严格的确认条件

### Factor-Based Strategy (因子策略)

因子策略会根据市场状态动态调整因子权重（未来功能）。

## Verbose模式查看

使用 `--verbose` 参数可以查看市场状态判断详情：

```bash
finquant selection run \
  --config config/selection_breakout.yaml \
  --start 2020-07-01 \
  --end 2020-07-31 \
  --verbose
```

输出示例：

```
======================================================================
MA突破选股 - 2020-07-01
======================================================================
配置参数:
  MA周期: [120, 250]
  放量倍数: 1.5x
  突破阈值: 1.05 (5%)
  ...

候选股票数: 10
✓ 通过突破筛选: 1 只股票

市场状态: oscillation
指数指标:
  收盘价: 13.12
  ADX: 0.94
  MA50: 13.26
  相对MA50: 98.97%

最终选中: 1 只股票
...
```

## 常见问题

### Q1: 为什么市场状态总是oscillation？

**A:** 可能的原因：
1. 指数数据缺少ADX指标
2. 当前市场确实处于震荡状态
3. ADX计算周期不匹配（使用dx_30作为adx_14的代理）

**解决方法**：
- 检查指数数据是否包含`adx_14`列
- 使用更长的历史数据计算ADX
- 调整市场状态判断规则

### Q2: 市场状态判断不准确怎么办？

**A:** 可以自定义判断规则：

```python
from finquant.selection.market_state import MarketStateClassifier

# 创建自定义分类器
classifier = MarketStateClassifier()

# 修改规则阈值
# 例如：降低UPTREND的ADX要求
# 需要修改 finquant/selection/market_state.py
```

### Q3: 不同策略的市场状态判断一样吗？

**A:** 是的，所有策略使用相同的`MarketStateClassifier`，基于指数数据判断。这确保了不同策略对市场环境的理解一致。

## 技术实现

### 代码位置

- **分类器**：`finquant/selection/market_state.py`
- **Breakout策略集成**：`finquant/selection/strategies/breakout.py`
- **Factor策略集成**：`finquant/selection/pipeline.py`

### 使用示例

```python
from finquant.selection.market_state import MarketStateClassifier
import pandas as pd

# 创建分类器
classifier = MarketStateClassifier(index_ticker="000905.SH")

# 准备指数数据（需要包含 date, close, adx_14 列）
index_df = pd.DataFrame({
    'date': ['2020-07-01'],
    'close': [13.12],
    'adx_14': [0.94],
    'volume': [1000000]
})

# 分类
state = classifier.classify(index_df, '2020-07-01')
print(f"Market state: {state.value}")  # oscillation

# 获取指标
metrics = classifier.get_index_metrics(index_df, '2020-07-01')
print(f"Metrics: {metrics}")
```

## 相关文档

- [选股详细输出功能](selection-verbose-output.md)
- [MA突破策略指南](breakout-strategy-guide.md)
- [因子选股策略指南](selection-training-guide.md)
