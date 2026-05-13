# ZZ500 RL选股系统 - 使用说明

## 系统已完成实施 ✅

所有核心组件已创建并测试通过：

### 已创建文件
1. ✅ `finquant/data/sources/zz500_loader.py` - 股票列表加载器
2. ✅ `finquant/selection/rl_scorer.py` - RL评分器
3. ✅ `config/zz500_rl_single_stock.yaml` - 配置文件
4. ✅ `scripts/train_zz500_single_stock.py` - 训练脚本
5. ✅ `scripts/backtest_zz500_single_stock.py` - 回测脚本
6. ✅ `docs/zz500-rl-quickstart.md` - 详细文档

### 数据验证
- 股票数量：343只ZZ500成分股
- 数据范围：2020-01-02 至 2026-04-30
- 训练期：2023-01-01 至 2024-06-30
- 测试期：2024-07-01 至 2024-12-31

## 快速开始

### 1. 训练模型

```bash
python scripts/train_zz500_single_stock.py \
    --config config/zz500_rl_single_stock.yaml \
    --db-path data/processed/zz500_data.db \
    --output-dir models/zz500_single_stock
```

**训练特点**：
- 使用单股票模型（stock_dim=1，观察空间10维）
- 在数据最完整的股票上训练（000001.SZ）
- PPO算法，200,000训练步数
- 训练时间：约20-30分钟（CPU）

### 2. 回测策略

训练完成后，使用生成的模型进行回测：

```bash
python scripts/backtest_zz500_single_stock.py \
    --model models/zz500_single_stock/ppo_zz500_single_20240630_HASH.zip \
    --config config/zz500_rl_single_stock.yaml \
    --start 2024-07-01 \
    --end 2024-12-31 \
    --output-dir runs/zz500_backtest
```

**回测流程**：
1. 对343只股票逐一评分
2. 筛选评分>0.9的股票
3. 选择前5名
4. 按评分加权分配资金
5. 记录每日持仓和收益

## 核心设计

### 单股票RL架构

**为什么选择单股票模型？**

传统多股票模型问题：
- 观察空间：1 + (2+7) × 343 = 3088维
- 训练慢，收敛困难

单股票模型优势：
- 观察空间：1 + (2+7) × 1 = 10维
- 训练快速，易于收敛
- 可扩展到任意数量股票

### 评分机制

**Action到Score映射（Sigmoid）**：
```python
score = 1 / (1 + exp(-action))
```

- action > 2.2 → score > 0.9（强买入）
- action = 0 → score = 0.5（中性）
- action < -2.2 → score < 0.1（强卖出）

### 持仓分配

**评分加权示例**：

假设选出5只股票，评分为[0.95, 0.92, 0.91, 0.90, 0.88]，总资金100万：

- 总评分：4.56
- 股票A (0.95)：20.8% → 20.8万
- 股票B (0.92)：20.2% → 20.2万
- 股票C (0.91)：20.0% → 20.0万
- 股票D (0.90)：19.7% → 19.7万
- 股票E (0.88)：19.3% → 19.3万

### 风险管理

- **止损**：-5%
- **止盈**：+20%
- **RL退出**：评分<0.8（action<-1.39）

## 配置参数

关键参数在 `config/zz500_rl_single_stock.yaml`：

```yaml
zz500_selection:
  portfolio_size: 5          # 持仓股票数
  score_threshold: 0.9       # 评分阈值
  score_mapping: "sigmoid"   # 映射方法
  position_sizing: "score_weighted"  # 仓位分配
  
  stop_loss_pct: -0.05      # 止损-5%
  take_profit_pct: 0.20     # 止盈+20%
  rl_exit_threshold: -0.2   # RL退出阈值
```

## 输出文件

### 训练输出
- 模型：`models/zz500_single_stock/ppo_zz500_single_YYYYMMDD_HASH.zip`
- 元数据：`models/zz500_single_stock/training_metadata.json`

### 回测输出
- 每日记录：`runs/zz500_backtest/daily_records.csv`
- 包含：日期、选中股票、评分、现金、持仓

## 使用示例

### Python API

```python
from pathlib import Path
from finquant.selection.rl_scorer import RLStockScorer
from finquant.data.sources.db_daily import DbDailyDataSource
from finquant.features.technical import compute_indicators

# 加载数据
data_source = DbDailyDataSource(db_path=Path("data/processed/zz500_data.db"))
df = data_source.download(["000001.SZ", "000002.SZ"], "2024-07-01", "2024-07-01")
df = compute_indicators(df, ["macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"])

# 初始化评分器
scorer = RLStockScorer(
    model_path=Path("models/zz500_single_stock/ppo_zz500_single.zip"),
    indicators=["macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"],
    score_mapping="sigmoid"
)

# 评分
scores = scorer.score_stocks(df, "2024-07-01")
print(scores)  # {'000001.SZ': 0.85, '000002.SZ': 0.92}

# 筛选和排序
qualified = {k: v for k, v in scores.items() if v > 0.9}
top_5 = sorted(qualified.items(), key=lambda x: x[1], reverse=True)[:5]
```

## 故障排除

### 问题1：训练时内存不足
**解决**：减少`training.total_timesteps`或使用更小的数据集

### 问题2：评分全部<0.9
**解决**：降低`score_threshold`到0.85或0.8

### 问题3：模型加载失败
**解决**：检查模型路径和算法类型（PPO/SAC/TD3）

### 问题4：缺少依赖
**解决**：
```bash
pip install pytz stable-baselines3 finrl
```

## 下一步优化

1. **多股票训练**：在多只代表性股票上训练，提高泛化能力
2. **特征增强**：添加基本面、市场情绪等特征
3. **集成学习**：训练多个模型并集成预测
4. **实时交易**：接入实时数据和交易接口

## 参考文档

- 详细文档：[docs/zz500-rl-quickstart.md](docs/zz500-rl-quickstart.md)
- 实现计划：`.claude/plans/finsys-train-rl-zz500-db-5-zz500-db-rl-fluffy-octopus.md`
- 项目主文档：[README.md](README.md)
