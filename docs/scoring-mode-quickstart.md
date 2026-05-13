# 评分模式快速开始指南

## 概述

评分模式允许你训练一个模型来为每个股票打分，然后使用规则化的组合管理器进行交易。这比传统的多股票交易模式更灵活和可扩展。

## 核心概念

### 评分模式 vs 交易模式

| 特性 | 交易模式 | 评分模式 |
|------|---------|---------|
| 输入 | 所有股票特征 (2854维) | 单股票特征 (9维) |
| 输出 | 每股票买卖动作 | 单个评分值 |
| 训练 | 在所有股票上训练 | 在单股票上训练 |
| 扩展性 | 添加股票需重训练 | 可对任意股票评分 |
| 组合管理 | RL模型控制 | 规则引擎控制 |

### 工作流程

```
1. 训练阶段:
   单股票数据 → StockScoringEnv → RL模型 → 评分模型

2. 回测阶段:
   多股票数据 → 评分模型 → 股票评分 → PortfolioManager → 交易决策
```

## 快速开始

### 步骤 1: 准备配置文件

创建 `config/my_scoring.yaml`:

```yaml
stocks:
  - "600519.SH"  # 单股票用于训练

dates:
  train_start: "2024-01-01"
  train_end: "2024-06-30"
  test_start: "2024-07-01"
  test_end: "2024-12-31"

training:
  algorithm: "ppo"
  total_timesteps: 100000
  mode: "scoring"  # 关键: 设置为评分模式
  
  scoring:
    enabled: true
    reward_type: "daily_return"  # 或 "future_return"
    future_horizon: 1
    normalize_obs: true

portfolio:
  max_positions: 10
  stop_loss_pct: -0.05  # -5% 止损
  take_profit_pct: 0.20  # +20% 止盈
  score_threshold: 0.0  # 最低评分阈值
  position_sizing: "equal"

indicators:
  - "macd"
  - "boll_ub"
  - "boll_lb"
  - "rsi_30"
  - "dx_30"
  - "close_30_sma"
  - "close_60_sma"
```

### 步骤 2: 获取数据

```bash
# 获取单股票数据用于训练
finsys data fetch \
  --config config/my_scoring.yaml \
  --output data/processed
```

### 步骤 3: 训练评分模型

```bash
# 训练单股票评分模型
finsys train \
  --config config/my_scoring.yaml \
  --data-file data/processed/600519_SH.parquet \
  --mode scoring \
  --output models/scoring

# 输出: models/scoring/ppo_scoring_600519_SH_20240630.zip
```

### 步骤 4: 使用评分模型

#### 方法 A: Python 脚本

```python
from finquant.training.portfolio_manager import PortfolioManager
from finquant.config.settings import load_config
from stable_baselines3 import PPO
import pandas as pd
import numpy as np

# 1. 加载配置
config = load_config("config/my_scoring.yaml")

# 2. 加载评分模型
model = PPO.load("models/scoring/ppo_scoring_600519_SH_20240630")

# 3. 初始化组合管理器
pm = PortfolioManager(
    initial_cash=1_000_000,
    max_positions=config.portfolio.max_positions,
    stop_loss_pct=config.portfolio.stop_loss_pct,
    take_profit_pct=config.portfolio.take_profit_pct,
    score_threshold=config.portfolio.score_threshold,
)

# 4. 加载测试数据 (多股票)
test_data = pd.read_parquet("data/processed/test_data.parquet")
trading_days = sorted(test_data["date"].unique())

# 5. 每日回测循环
for date in trading_days:
    # 获取当日所有股票的数据
    day_data = test_data[test_data["date"] == date]
    
    # 为每个股票评分
    scores = {}
    prices = {}
    
    for _, row in day_data.iterrows():
        ticker = row["tic"]
        
        # 构建观察向量 [close, volume, indicators...]
        obs = np.array([
            row["close"],
            row["volume"],
            row["macd"],
            row["boll_ub"],
            row["boll_lb"],
            row["rsi_30"],
            row["dx_30"],
            row["close_30_sma"],
            row["close_60_sma"],
        ], dtype=np.float32)
        
        # 模型预测评分
        score, _ = model.predict(obs, deterministic=True)
        scores[ticker] = float(score[0])
        prices[ticker] = row["close"]
    
    # 更新组合
    result = pm.update(date, scores, prices)
    
    print(f"{date}: {result['num_positions']} positions, "
          f"value={result['total_value']:,.2f}, "
          f"return={result['daily_return']:.4f}")

# 6. 获取回测结果
summary = pm.get_summary()
print(f"\n=== 回测结果 ===")
print(f"初始资金: ¥{summary['initial_cash']:,.2f}")
print(f"最终价值: ¥{summary['final_value']:,.2f}")
print(f"总收益率: {summary['total_return']:.2%}")
print(f"总交易数: {summary['num_trades']}")

# 7. 保存结果
portfolio_series = pm.get_portfolio_series()
portfolio_series.to_csv("backtest_results.csv")

trades_df = pm.get_trades_df()
trades_df.to_csv("trades.csv", index=False)
```

#### 方法 B: 批量训练多个股票

```python
# train_all_stocks.py
from finquant.config.settings import load_config
from finquant.training.trainer import Trainer
from pathlib import Path
import pandas as pd

config = load_config("config/csi500_scoring.yaml")

# 所有要训练的股票
all_stocks = [
    "600519.SH", "000001.SZ", "600036.SH", 
    # ... 更多股票
]

for ticker in all_stocks:
    print(f"\n训练 {ticker}...")
    
    # 加载单股票数据
    data_file = f"data/processed/{ticker.replace('.', '_')}.parquet"
    df = pd.read_parquet(data_file)
    
    # 训练
    trainer = Trainer(config)
    model_path = trainer.train(
        train_df=df,
        output_dir=Path("models/scoring"),
        mode="scoring"
    )
    
    print(f"✓ {ticker} 模型已保存: {model_path}")
```

## 配置参数详解

### training.scoring

```yaml
scoring:
  enabled: true
  reward_type: "daily_return"  # 奖励类型
  future_horizon: 1            # 未来收益窗口
  normalize_obs: true          # 观察归一化
```

**reward_type**:
- `daily_return`: 当日收益率 `(close_t - close_{t-1}) / close_{t-1}`
- `future_return`: 未来收益率 `(close_{t+N} - close_t) / close_t`

**future_horizon**:
- 仅在 `reward_type="future_return"` 时使用
- 表示预测未来 N 天的收益

**normalize_obs**:
- `true`: 使用 z-score 归一化观察
- `false`: 使用原始值

### portfolio

```yaml
portfolio:
  max_positions: 10           # 最大持仓数
  stop_loss_pct: -0.05       # 止损百分比
  take_profit_pct: 0.20      # 止盈百分比
  score_threshold: 0.0       # 评分阈值
  position_sizing: "equal"   # 持仓大小策略
  transaction_cost_pct: 0.001 # 交易成本
```

**max_positions**:
- 同时持有的最大股票数量
- 建议: 5-20

**stop_loss_pct**:
- 止损阈值（负数）
- 例: -0.05 = 亏损 5% 时卖出

**take_profit_pct**:
- 止盈阈值（正数）
- 例: 0.20 = 盈利 20% 时卖出

**score_threshold**:
- 最低评分阈值
- 评分低于此值时卖出/不买入

**position_sizing**:
- `equal`: 等额分配资金
- `score_weighted`: 按评分加权分配（待实现）

## 常见问题

### Q1: 评分模式和交易模式哪个更好？

**评分模式优势**:
- ✅ 可扩展性强（添加新股票无需重训练）
- ✅ 训练速度快（单股票训练）
- ✅ 观察空间小（9维 vs 2854维）
- ✅ 规则易调整（无需重训练）

**交易模式优势**:
- ✅ 端到端学习（RL直接控制交易）
- ✅ 可能学到股票间关系

**建议**: 对于大规模股票池（>50只），使用评分模式。

### Q2: 如何选择 reward_type？

- **daily_return**: 适合短期交易，模型学习预测当日涨跌
- **future_return**: 适合中长期持有，模型学习预测未来趋势

建议从 `daily_return` 开始，然后尝试 `future_return` 并调整 `future_horizon`。

### Q3: 如何调整止损和止盈？

根据股票波动性调整:
- **高波动股票**: 放宽止损（如 -8%），提高止盈（如 30%）
- **低波动股票**: 收紧止损（如 -3%），降低止盈（如 15%）

建议先回测不同参数组合，选择夏普比率最高的。

### Q4: 训练需要多长时间？

- **单股票训练**: 约 10-30 分钟（100K timesteps）
- **317 只股票**: 约 50-160 小时（串行）

建议使用并行训练脚本加速。

### Q5: 如何验证模型质量？

```python
# 检查模型元数据
import json
with open("models/scoring/ppo_scoring_600519_SH_20240630_metadata.json") as f:
    meta = json.load(f)
    print(f"训练步数: {meta['total_timesteps']}")
    print(f"观察维度: {meta['obs_dim']}")
    print(f"奖励类型: {meta['reward_type']}")

# 回测验证
# 1. 在训练集上回测（应该表现良好）
# 2. 在测试集上回测（真实性能）
# 3. 比较夏普比率、最大回撤、年化收益
```

## 最佳实践

### 1. 数据准备
- ✅ 确保数据质量（无缺失值、异常值）
- ✅ 使用足够长的训练期（至少 6 个月）
- ✅ 保留独立的测试集

### 2. 模型训练
- ✅ 从小的 `total_timesteps` 开始（如 10K）快速验证
- ✅ 监控训练日志，确保收敛
- ✅ 保存训练曲线（TensorBoard）

### 3. 参数调优
- ✅ 先固定大部分参数，逐个调优
- ✅ 使用网格搜索或贝叶斯优化
- ✅ 在验证集上选择最佳参数

### 4. 回测验证
- ✅ 使用真实交易成本
- ✅ 考虑滑点和流动性
- ✅ 进行多次随机种子测试

### 5. 生产部署
- ✅ 定期重新训练模型（如每月）
- ✅ 监控模型性能衰减
- ✅ 设置风险控制阈值

## 下一步

1. **阅读完整文档**: [scoring-mode-implementation-summary.md](./scoring-mode-implementation-summary.md)
2. **查看示例**: `examples/scoring_mode_example.py`
3. **运行测试**: `pytest tests/unit/test_scoring_env.py tests/unit/test_portfolio_manager.py`
4. **加入讨论**: 在 GitHub Issues 中提问

## 参考资料

- [FinRL 文档](https://finrl.readthedocs.io/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
