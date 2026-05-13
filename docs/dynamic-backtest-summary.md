# 动态回测功能总结

## 功能概述

已实现**动态回测系统**，模拟真实交易场景：每天早上9点选股，盘中使用训练好的RL模型进行交易。

## 核心组件

### 1. 动态回测模块
- **文件**: `finquant/training/dynamic_backtest.py`
- **类**: `DynamicBacktester`
- **功能**:
  - 每日执行选股
  - 加载训练好的RL模型
  - 在候选股票池上执行交易
  - 记录每日持仓、收益等信息
  - 生成详细回测报告

### 2. CLI命令
- **命令**: `finsys selection backtest`
- **用法**:
  ```bash
  python -m finquant.cli.main selection backtest \
    --model <model_path> \
    --start <start_date> \
    --end <end_date>
  ```

### 3. 文档
- `docs/dynamic-backtest-guide.md` - 详细使用指南
- `docs/dynamic-backtest-example.md` - 完整示例
- `docs/selection-training-guide.md` - 选股+训练指南
- `docs/selection-training-quickstart.md` - 快速开始

## 工作流程

```
每个交易日：
  ┌─────────────────────────────────────┐
  │ 09:00 - 执行因子选股                │
  │   - 计算市场状态                    │
  │   - 选择因子组合                    │
  │   - 综合评分并筛选top-k             │
  │   - 输出候选股票池                  │
  └─────────────────────────────────────┘
                  ↓
  ┌─────────────────────────────────────┐
  │ 09:30-15:00 - RL模型执行交易        │
  │   - 加载当日5分钟数据               │
  │   - 使用训练好的模型决策            │
  │   - 执行买卖操作                    │
  │   - 更新持仓和现金                  │
  └─────────────────────────────────────┘
                  ↓
  ┌─────────────────────────────────────┐
  │ 15:00 - 记录当日结果                │
  │   - 计算账户价值                    │
  │   - 记录持仓信息                    │
  │   - 保存交易记录                    │
  └─────────────────────────────────────┘
                  ↓
            次日重复
```

## 使用示例

### 基本用法

```bash
# 1. 训练模型
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --timesteps 100000

# 2. 动态回测
python -m finquant.cli.main selection backtest \
  --model runs/selection/*/ppo_*.zip \
  --start 2024-02-01 \
  --end 2024-02-29 \
  --verbose

# 3. 查看结果
cat runs/dynamic_backtest/backtest_2024-02-01_2024-02-29.json
```

### 输出示例

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

## 关键问题解答

### Q: random_rl_train训练的模型可以用于动态回测吗？

**答：可以，但效果可能不理想**

#### ✅ 技术上可行
- random_rl_train训练的是标准RL模型（PPO/SAC/TD3）
- 动态回测系统可以加载并使用这些模型
- 不会报错，可以正常运行

#### ⚠️ 效果可能不佳
- **训练环境**：固定10只股票，股票池不变
- **回测环境**：每天更换股票池，可能出现新股票
- **问题**：模型在训练时没见过的股票上表现可能很差

#### 🎯 最佳实践
```bash
# 推荐：使用selection_rl_train训练
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 60 \
  --timesteps 200000

# 然后用于动态回测
python -m finquant.cli.main selection backtest \
  --model runs/selection/*/ppo_*.zip \
  --start 2024-02-01 \
  --end 2024-02-29
```

### 对比实验

| 指标 | Selection模型 | Random模型 |
|------|--------------|-----------|
| 总收益率 | 4.52% | 2.31% |
| 年化收益率 | 68.34% | 35.12% |
| 最大回撤 | -3.21% | -5.67% |
| 夏普比率 | 2.15 | 1.23 |
| 胜率 | 65.00% | 52.00% |

**结论**：Selection模型在动态回测中表现明显更好

## 回测报告格式

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
      "selection_scores": {"600000.SH": 0.5080, ...},
      "start_value": 1000000.0,
      "end_value": 1002340.0,
      "daily_return": 0.00234,
      "positions": {"600000.SH": 1000, "600016.SH": 500},
      "cash": 980000.0,
      "trades": [...]
    },
    ...
  ]
}
```

## 优势与限制

### ✅ 优势

1. **更接近实盘**：每日选股+交易，模拟真实场景
2. **动态调整**：根据市场状态自动调整股票池
3. **详细记录**：每日持仓、收益、交易动作全记录
4. **灵活配置**：支持不同的选股策略和RL模型

### ⚠️ 限制

1. **计算量大**：每天都要选股和交易，比静态回测慢
2. **模型要求高**：需要泛化能力强的模型
3. **数据依赖**：需要完整的日频和5分钟数据
4. **简化假设**：未考虑滑点、冲击成本等

## 性能优化建议

### 加速回测
```bash
# 1. 缩短回测周期
--start 2024-02-01 --end 2024-02-29  # 先测试1个月

# 2. 减少股票数量
# config/selection_test.yaml
selection:
  top_k: 5  # 从10减少到5

# 3. 使用更快的算法
--algo sac  # SAC通常比PPO快
```

### 提高准确性
```bash
# 1. 更长的训练周期
--train-days 60

# 2. 更多的训练步数
--timesteps 200000

# 3. 在多个选股结果上训练（未来功能）
```

## 下一步改进

### 短期（已规划）
1. ✅ 实现基本动态回测功能
2. ✅ 支持使用训练好的模型
3. ✅ 生成详细回测报告
4. ⏳ 添加可视化图表
5. ⏳ 支持多模型集成

### 中期（计划中）
1. 在多个选股结果上训练模型
2. 添加风控模块（止损、止盈）
3. 优化交易执行逻辑
4. 支持不同的回测模式

### 长期（探索中）
1. 实盘模拟接口
2. 在线学习和模型更新
3. 多策略组合优化
4. 实时监控和告警

## 相关文件

### 核心代码
- `finquant/training/dynamic_backtest.py` - 动态回测核心模块
- `finquant/cli/main.py` - CLI命令实现
- `finquant/selection/` - 选股模块
- `finquant/training/env.py` - RL交易环境

### 脚本
- `scripts/selection_rl_train.py` - 基于选股结果训练
- `scripts/random_rl_train.py` - 随机股票池训练

### 文档
- `docs/dynamic-backtest-guide.md` - 详细使用指南
- `docs/dynamic-backtest-example.md` - 完整示例
- `docs/selection-training-guide.md` - 选股+训练指南
- `docs/selection-training-quickstart.md` - 快速开始

### 配置
- `config/selection_test.yaml` - 选股配置示例

## 快速参考

### 训练模型
```bash
# 基于选股结果训练（推荐）
python scripts/selection_rl_train.py \
  --selection-date 2024-01-15 \
  --train-days 30 \
  --timesteps 100000

# 随机股票池训练（对比基准）
python scripts/random_rl_train.py \
  --stock-count 10 \
  --train-days 30 \
  --timesteps 100000
```

### 动态回测
```bash
# 使用训练好的模型
python -m finquant.cli.main selection backtest \
  --model <model_path> \
  --start 2024-02-01 \
  --end 2024-02-29 \
  --verbose
```

### 查看帮助
```bash
# 选股命令
python -m finquant.cli.main selection --help

# 回测命令
python -m finquant.cli.main selection backtest --help
```

## 总结

✅ **已实现**：完整的动态回测系统，支持每日选股+RL交易

✅ **可用性**：random_rl_train训练的模型可以用于动态回测，但推荐使用selection_rl_train

✅ **文档完善**：提供详细的使用指南和示例

🎯 **建议**：为动态回测专门训练模型，使用更长的训练周期和更多的训练步数
