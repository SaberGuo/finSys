# MA突破选股策略 - 快速使用指南

## ✅ 策略已完全集成并可用

### 数据源配置

策略使用 `zz500_data.db` 数据库，配置如下：

```yaml
data:
  source_priority: ["db_daily"]  # 使用本地数据库

stocks:
  - "000001.SZ"  # 深圳股票格式
  - "600030.SH"  # 上海股票格式
  # ... 更多股票

selection:
  strategy_type: "breakout"
  index_ticker: "600030.SH"  # 使用一只股票作为伪指数（突破策略不需要真实指数）
```

### 使用方法

#### 1. 命令行使用（推荐）

```bash
# 使用测试配置（宽松参数）- 带详细输出
finquant selection run \
  --config config/selection_breakout_test.yaml \
  --start 2020-08-01 \
  --end 2020-12-31 \
  --output-dir data/selection_breakout \
  --verbose

# 使用标准配置 - 简洁输出
finquant selection run \
  --config config/selection_breakout.yaml \
  --start 2020-08-01 \
  --end 2020-12-31 \
  --output-dir data/selection_breakout
```

**Verbose模式说明**：

使用 `--verbose` 参数时，会显示每个交易日的详细选股过程：

```
======================================================================
MA突破选股 - 2020-08-03
======================================================================
配置参数:
  MA周期: [60, 120]
  放量倍数: 1.2x
  突破阈值: 1.02 (2%)
  回溯天数: 30
  确认天数: 2
  防抖模式: either
  选股数量: 10

候选股票数: 10
✓ 通过突破筛选: 2 只股票

排除股票:
  ST600001.SH: ST stock

最终选中: 2 只股票
股票详情:

  600030.SH:
    评分: 0.8234
    收盘价: 125.50
    MA60: 120.30 (突破 +4.32%)
    成交量: 8,500,000 (1.7x 均量)

  000100.SZ:
    评分: 0.7156
    收盘价: 45.80
    MA120: 43.20 (突破 +6.02%)
    成交量: 12,300,000 (1.5x 均量)
```

这样可以清楚地看到：
- 每只股票的筛选参数
- 突破的具体幅度
- 成交量放大倍数
- 为什么某些股票被排除

#### 2. Python API使用

```python
from finquant.config.settings import load_config
from finquant.data.pipeline import DataPipeline
from finquant.selection import create_strategy
import logging

# 配置日志以启用详细输出
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 加载配置
config = load_config("config/selection_breakout_test.yaml")

# 加载数据
pipeline = DataPipeline(config)
market_df = pipeline.fetch()

# 准备索引数据（使用第一只股票）
index_df = market_df[market_df['tic'] == config.stocks[0]].copy()

# 创建策略（verbose=True 启用详细输出）
strategy = create_strategy(config, verbose=True)

# 运行选股
result = strategy.select(market_df, index_df, "2020-08-03")

# 查看结果
print(f"选中 {len(result.selected_tickers)} 只股票")
for tic in result.selected_tickers:
    print(f"  {tic}: {result.scores[tic]:.4f}")
```

### 输出结果

每个交易日生成一个JSON文件，格式如下：

```json
{
  "version": "1.0.0",
  "date": "2020-08-03",
  "selected_tickers": ["600030.SH", "000100.SZ"],
  "scores": {
    "600030.SH": 0.0523,
    "000100.SZ": 0.0412
  },
  "market_state": "oscillation",
  "active_factors": ["ma_breakout", "volume_surge"],
  "factor_weights": {
    "ma_breakout": 0.6,
    "volume_surge": 0.4
  },
  "index_metrics": {},
  "exclusion_reasons": {
    "ST600001.SH": "ST stock",
    "600002.SH": "halted (volume=0)"
  }
}
```

### 配置参数说明

#### 宽松参数（更多信号）

```yaml
selection:
  breakout:
    ma_periods: [60, 120]          # 较短MA周期
    volume_multiplier: 1.2         # 较低放量要求
    breakout_threshold: 1.02       # 2%突破阈值
    lookback_days: 30              # 较短回溯期
    anti_jitter_mode: "either"     # 宽松防抖动
```

#### 标准参数（平衡）

```yaml
selection:
  breakout:
    ma_periods: [120, 250]         # 半年线、年线
    volume_multiplier: 1.5         # 1.5倍放量
    breakout_threshold: 1.05       # 5%突破阈值
    lookback_days: 60              # 60天回溯
    anti_jitter_mode: "threshold"  # 价格阈值
```

#### 严格参数（更少但更可靠的信号）

```yaml
selection:
  breakout:
    ma_periods: [250]              # 仅年线
    volume_multiplier: 2.0         # 2倍放量
    breakout_threshold: 1.10       # 10%突破阈值
    lookback_days: 120             # 120天回溯
    anti_jitter_mode: "both"       # 严格防抖动
```

### 常见问题

#### Q1: 为什么没有选中股票？

**A:** 这是正常现象。突破策略要求同时满足4个条件：
1. 价格从MA下方突破到上方（昨天≤MA，今天>MA）
2. 成交量放大
3. 首次突破
4. 防抖动

如果市场处于趋势延续阶段，可能没有新的突破发生。

**解决方法**：
- 使用更宽松的参数（如 `selection_breakout_test.yaml`）
- 增加测试的股票数量
- 扩大测试的时间范围

#### Q2: 如何增加更多股票？

**A:** 编辑配置文件的 `stocks` 列表：

```yaml
stocks:
  - "000001.SZ"
  - "000002.SZ"
  # ... 添加更多中证500成分股
```

确保股票代码在 `zz500_data.db` 中存在。

#### Q3: 如何与factor-based策略比较？

**A:** 运行两个策略并比较结果：

```bash
# 运行突破策略
finquant selection run \
  --config config/selection_breakout.yaml \
  --start 2020-08-01 --end 2020-12-31 \
  --output-dir data/selection_breakout

# 运行因子策略
finquant selection run \
  --config config/selection_factor_based.yaml \
  --start 2020-08-01 --end 2020-12-31 \
  --output-dir data/selection_factor

# 比较结果
python scripts/compare_strategies.py
```

### 验证测试

运行以下脚本验证策略工作正常：

```bash
# 完整演示
python scripts/demo_breakout_complete.py

# 诊断分析
python scripts/diagnose_breakout_fixed.py

# 验证集成
python scripts/verify_breakout_strategy.py
```

### 技术支持

- **用户指南**: `docs/breakout-strategy-guide.md`
- **实现总结**: `docs/breakout-strategy-implementation-summary.md`
- **代码示例**: `examples/breakout_strategy_example.py`

### 成功标志 ✅

- ✅ 数据从 `zz500_data.db` 加载成功
- ✅ 策略执行无错误
- ✅ 输出JSON文件格式正确
- ✅ CLI命令 `finquant selection run` 工作正常
- ✅ 可以通过配置文件切换策略

**策略已完全集成并可以使用！** 🎉
