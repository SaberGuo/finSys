# 多股票评分交易系统实施总结

## 概述

成功将训练系统从单股票交易模式改造为多股票选择和评分系统。新系统将**评分**（RL模型）和**组合管理**（规则引擎）分离，实现了更灵活和可扩展的架构。

## 实施内容

### 1. 核心组件

#### 1.1 StockScoringEnv (评分环境)
**文件**: `finquant/training/scoring_env.py`

**功能**:
- 单股票评分环境，输出连续评分值
- 观察空间: `[close, volume, indicators...]` (9维)
- 动作空间: 单个连续评分 (无界)
- 奖励: 每日收益率或未来收益率

**特性**:
- 支持观察归一化 (z-score)
- 支持两种奖励类型: `daily_return` 和 `future_return`
- 可配置的未来时间窗口

#### 1.2 PortfolioManager (组合管理器)
**文件**: `finquant/training/portfolio_manager.py`

**功能**:
- 基于RL评分的规则化组合管理
- 最多持仓10个股票
- 自动止损 (-5%) 和止盈 (+20%)
- 评分为负时卖出

**核心逻辑**:
1. 更新持仓价格
2. 检查止损条件
3. 检查止盈条件
4. 检查评分退出条件
5. 从最高评分股票中补充空缺

**关键修复**:
- 防止同一天内卖出后立即买回同一股票
- 正确初始化持仓的盈亏百分比

### 2. 配置扩展

#### 2.1 新增配置类
**文件**: `finquant/config/settings.py`

```python
class ScoringConfig(BaseModel):
    enabled: bool = False
    reward_type: str = "daily_return"
    future_horizon: int = 1
    normalize_obs: bool = True

class PortfolioConfig(BaseModel):
    max_positions: int = 10
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.20
    score_threshold: float = 0.0
    position_sizing: str = "equal"
    transaction_cost_pct: float = 0.001

class TrainingConfig(BaseModel):
    # ... 现有字段 ...
    mode: str = "trading"  # "trading" 或 "scoring"
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
```

#### 2.2 示例配置
**文件**: `config/csi500_scoring.yaml`

关键配置:
```yaml
training:
  mode: "scoring"
  scoring:
    enabled: true
    reward_type: "daily_return"
    normalize_obs: true

portfolio:
  max_positions: 10
  stop_loss_pct: -0.05
  take_profit_pct: 0.20
  score_threshold: 0.0
```

### 3. 训练器集成

#### 3.1 Trainer 类更新
**文件**: `finquant/training/trainer.py`

**新增方法**:
- `train()`: 根据 `mode` 参数路由到不同训练方法
- `_train_trading()`: 原有的多股票交易训练
- `_train_scoring()`: 新的单股票评分训练

**关键差异**:
- 评分模式要求 `stock_dim=1`
- 使用 `StockScoringEnv` 而非 `StockTradingEnv`
- 模型文件名包含 "scoring" 标识

### 4. CLI 更新

#### 4.1 train 命令
**文件**: `finquant/cli/main.py`

**新增选项**:
```bash
--mode [trading|scoring]  # 训练模式选择
```

**验证逻辑**:
- 评分模式要求单股票数据
- 自动检测并提示用户

### 5. 环境构建器

#### 5.1 build_env 函数
**文件**: `finquant/training/env.py`

**新增参数**:
- `mode`: "trading" 或 "scoring"
- `reward_type`: 评分模式的奖励类型
- `future_horizon`: 未来收益的时间窗口
- `normalize_obs`: 是否归一化观察

**逻辑**:
```python
if mode == "scoring":
    if stock_dim != 1:
        raise ValueError("Scoring mode requires stock_dim=1")
    return build_scoring_env(...)
else:
    return build_trading_env(...)  # 原有逻辑
```

### 6. 测试覆盖

#### 6.1 StockScoringEnv 测试
**文件**: `tests/unit/test_scoring_env.py`

**测试用例** (19个):
- 环境初始化验证
- 观察和动作空间验证
- 重置和步进功能
- 奖励计算 (daily_return 和 future_return)
- 观察归一化
- 多轮次测试

**结果**: ✅ 19/19 通过

#### 6.2 PortfolioManager 测试
**文件**: `tests/unit/test_portfolio_manager.py`

**测试用例** (23个):
- 初始化验证
- 买入/卖出逻辑
- 止损触发
- 止盈触发
- 负评分退出
- 交易成本计算
- 最大持仓约束
- 评分阈值
- 边界情况

**结果**: ✅ 23/23 通过

## 使用方法

### 训练评分模型

```bash
# 1. 准备单股票数据
finsys data fetch --config config/csi500_scoring.yaml

# 2. 训练评分模型
finsys train \
  --config config/csi500_scoring.yaml \
  --data-file data/processed/600519_SH.parquet \
  --mode scoring \
  --output models/scoring

# 输出: models/scoring/ppo_scoring_600519_SH_20240630.zip
```

### 使用评分模型进行回测

```python
from finquant.training.portfolio_manager import PortfolioManager
from finquant.config.settings import load_config
import pandas as pd

# 加载配置
config = load_config("config/csi500_scoring.yaml")

# 初始化组合管理器
pm = PortfolioManager(
    initial_cash=config.portfolio.initial_amount,
    max_positions=config.portfolio.max_positions,
    stop_loss_pct=config.portfolio.stop_loss_pct,
    take_profit_pct=config.portfolio.take_profit_pct,
    score_threshold=config.portfolio.score_threshold,
)

# 加载评分模型
from stable_baselines3 import PPO
model = PPO.load("models/scoring/ppo_scoring_600519_SH_20240630")

# 每日更新
for date in trading_days:
    # 1. 获取所有股票的特征
    features = get_features(date)
    
    # 2. 使用模型评分
    scores = {}
    for ticker, feat in features.items():
        score = model.predict(feat)[0][0]
        scores[ticker] = score
    
    # 3. 获取当前价格
    prices = get_prices(date)
    
    # 4. 更新组合
    result = pm.update(date, scores, prices)
    
    print(f"{date}: {result['num_positions']} positions, value={result['total_value']:.2f}")

# 获取回测报告
summary = pm.get_summary()
print(f"Total return: {summary['total_return']:.2%}")
```

## 架构优势

### 1. 可扩展性
- **单股票训练**: 在单股票上训练，可对任意数量股票评分
- **无需重训练**: 添加新股票无需重新训练模型
- **降低维度**: 观察空间从 2854 维 (317股票) 降至 9 维

### 2. 向后兼容
- **默认行为不变**: `training.mode` 默认为 "trading"
- **现有配置继续工作**: 无需修改现有配置文件
- **渐进式迁移**: 可以逐步从交易模式迁移到评分模式

### 3. 清晰分离
- **RL 负责预测**: 模型只输出评分，不直接交易
- **规则负责执行**: 组合管理器处理止损、止盈、持仓限制
- **易于调整**: 修改交易规则无需重新训练模型

### 4. 灵活配置
- **多种奖励类型**: daily_return 或 future_return
- **可调参数**: 止损、止盈、最大持仓、评分阈值
- **持仓策略**: equal 或 score_weighted

## 关键设计决策

### 1. 为什么创建新环境而不是修改 FinRL?
- FinRL 的 `StockTradingEnv` 与组合管理紧密耦合
- 评分模型应该是无状态的（不关心现金、持仓）
- 更清晰的关注点分离

### 2. 为什么动作空间是无界的?
- 允许模型表达置信度（高正值 = 强买入信号）
- 比有界 [-1, 1] 更具表现力
- 可以后处理应用 sigmoid/tanh 归一化

### 3. 为什么组合管理是规则化的?
- 止损、止盈是业务逻辑，不是学习行为
- 更容易修改规则而无需重新训练
- 支持 A/B 测试不同策略

### 4. 为什么防止同日买回?
- 避免"卖出-买入"循环
- 更符合实际交易逻辑
- 减少不必要的交易成本

## 下一步工作

### Phase 1 完成 ✅
- [x] 核心环境 (StockScoringEnv)
- [x] 组合管理器 (PortfolioManager)
- [x] 配置扩展
- [x] 训练器集成
- [x] CLI 更新
- [x] 单元测试

### Phase 2 待完成
- [ ] 集成测试 (端到端训练和回测)
- [ ] 性能测试 (训练时间、评分延迟)
- [ ] 批量训练脚本 (对所有股票训练评分模型)
- [ ] 回测命令 (CLI 支持评分模式回测)

### Phase 3 待完成
- [ ] 用户文档
- [ ] API 文档
- [ ] 迁移指南
- [ ] 示例 Jupyter Notebook

## 文件清单

### 新增文件
- `finquant/training/scoring_env.py` (92 行)
- `finquant/training/portfolio_manager.py` (470 行)
- `tests/unit/test_scoring_env.py` (280 行)
- `tests/unit/test_portfolio_manager.py` (380 行)
- `config/csi500_scoring.yaml` (60 行)

### 修改文件
- `finquant/training/env.py` (+60 行)
- `finquant/training/trainer.py` (+200 行)
- `finquant/config/settings.py` (+70 行)
- `finquant/cli/main.py` (+30 行)

### 测试结果
- StockScoringEnv: 19/19 通过 ✅
- PortfolioManager: 23/23 通过 ✅
- 总覆盖率: 96% (scoring_env.py)

## 总结

成功实现了多股票评分交易系统的核心功能，包括：

1. ✅ **评分环境**: 单股票输入 → 连续评分输出
2. ✅ **组合管理**: 基于评分的规则化交易
3. ✅ **配置系统**: 灵活的参数配置
4. ✅ **训练集成**: 支持评分模式训练
5. ✅ **CLI 支持**: 命令行工具更新
6. ✅ **测试覆盖**: 42 个单元测试全部通过

系统现在支持两种训练模式：
- **Trading Mode** (原有): 多股票组合交易
- **Scoring Mode** (新增): 单股票评分 + 规则化组合管理

新架构具有更好的可扩展性、向后兼容性和清晰的关注点分离。
