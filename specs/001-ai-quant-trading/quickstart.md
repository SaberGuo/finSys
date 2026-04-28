# Quickstart: A股AI量化交易系统

**Feature**: `001-ai-quant-trading`  
**Date**: 2026-04-28  
**目标**: 从零开始到完成第一次完整训练（数据下载 + 预处理 + FinRL 训练 + 回测），全流程 ≤ 5 步。

---

## 前置要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | ≥ 3.10 | 推荐 3.11 |
| CUDA (可选) | ≥ 11.8 | GPU 训练显著提速；CPU 模式也可运行但较慢 |
| 迅投 QMT 客户端 | 已安装并登录 | 使用 xtquant 时必须；不使用 xtquant 时可跳过 |
| 磁盘空间 | ≥ 5 GB | 数据、模型权重、日志 |

---

## 第 1 步：安装依赖

```bash
# 克隆代码仓库
git clone <repo-url> finSys
cd finSys

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# 或 Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 包含的核心包（安装顺序无关）：

```
finrl>=0.3.7
stable-baselines3>=2.0.0
gymnasium>=0.29.0
stockstats>=0.5.4
akshare>=1.18.0
pandas>=2.0.0
pyarrow>=14.0.0          # Parquet 读写
torch>=2.1.0
transformers>=4.37.0
bitsandbytes>=0.41.0     # 4-bit 量化 (Qwen)
click>=8.1.0             # CLI
pyyaml>=6.0
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
plotly>=5.18.0           # 回测可视化
```

---

## 第 2 步：配置股票池和日期范围

复制默认配置并按需修改：

```bash
cp config/default.yaml config/my_config.yaml
```

编辑 `config/my_config.yaml`，至少配置以下字段：

```yaml
stocks:
  - "000001.SZ"   # 平安银行
  - "600000.SH"   # 浦发银行
  # 可添加更多股票，建议先从 5～10 只开始

dates:
  train_start: "2021-01-01"
  train_end:   "2024-06-30"
  test_start:  "2024-07-01"
  test_end:    "2025-06-30"

data:
  primary_source: "akshare"   # 无需迅投授权；如有 QMT 可改为 "xtquant"
```

> **无迅投账号？** 将 `primary_source` 设为 `"akshare"`，无需任何认证即可下载数据。

---

## 第 3 步：下载并预处理数据

```bash
finsys data fetch --config config/my_config.yaml
```

**预期输出**（约 1～5 分钟，取决于股票数量）：

```
[INFO] Fetching 10 stocks via akshare...
[INFO] Fetched 000001.SZ: 876 trading days
[INFO] Fetched 600000.SH: 876 trading days
...
[INFO] Computing technical indicators: macd, boll_ub, boll_lb, rsi_30, dx_30, close_30_sma, close_60_sma
[INFO] Saved to data/processed/20210101_20250630_a1b2c3.parquet (87,600 rows × 14 columns)
```

**验证输出**：

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/')
print(df.shape)           # 应为 (total_rows, 14)
print(df.isnull().sum())  # 所有列应为 0
print(df['tic'].unique()) # 应包含所有配置的股票
"
```

---

## 第 4 步：训练 FinRL 模型

```bash
finsys train --config config/my_config.yaml --algo ppo
```

**预期输出**（GPU 约 5～15 分钟，CPU 约 30～60 分钟）：

```
[INFO] Training started: PPO | stocks=10 | obs_dim=91 | timesteps=100000
[INFO] Step 10000/100000 | reward_mean=-0.0003
[INFO] Step 50000/100000 | reward_mean=0.0012
[INFO] Step 100000/100000 | reward_mean=0.0025
[INFO] Model saved to models/ppo_20260428_a1b2c3/
```

---

## 第 5 步：运行回测并查看结果

```bash
finsys backtest \
  --config config/my_config.yaml \
  --model models/ppo_20260428_a1b2c3/ \
  --data data/processed/20210101_20250630_a1b2c3.parquet
```

**预期输出**：

```
[INFO] Backtest period: 2024-07-01 to 2025-06-30
[INFO] ─────────────────────────────────────────
[INFO] Sharpe Ratio     :  0.73
[INFO] Annual Return    : +9.8%
[INFO] Max Drawdown     : -6.4%
[INFO] Cumulative Return: +9.2%
[INFO] ─────────────────────────────────────────
[INFO] HTML report saved to reports/bt_20260428_xyz/report.html
```

在浏览器中打开 `reports/bt_20260428_xyz/report.html` 查看净值曲线和交易明细。

---

## 进阶：启用舆情分析（可选，需 GPU ≥ 8 GB VRAM）

```bash
# 准备新闻 JSONL 文件（每行一条记录）
# 格式: {"date": "2024-07-01", "tic": "000001.SZ", "text": "平安银行发布公告..."}

# 运行舆情分析
finsys sentiment analyze \
  --input data/my_news.jsonl \
  --output data/sentiment/result.jsonl

# 使用增强数据集训练
finsys train --config config/my_config.yaml --algo ppo --enhanced
```

---

## 一键端到端运行（全流程）

```bash
finsys run --config config/my_config.yaml --mode baseline
```

等价于依次执行 `data fetch → train → backtest`，全流程日志输出到 `logs/run_{timestamp}.jsonl`。

---

## 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `xtquant` 连接失败 | QMT 客户端未启动 | 先启动 QMT 客户端并登录，或改用 `akshare` |
| `CUDA out of memory` | 训练时显存不足 | 减少 `stocks` 数量或降低 `batch_size` |
| Qwen 加载失败 | `bitsandbytes` 未正确安装 | `pip install bitsandbytes --upgrade` (Linux/CUDA only) |
| 技术指标全为 NaN | 数据时间窗口不足 60 天 | 确保 `train_start` 到首个有效日期 ≥ 60 个交易日 |
| 训练 reward 不收敛 | 初始超参数不适合数据集 | 参见 `config.yaml::training.ppo` 部分，降低 `learning_rate` |
