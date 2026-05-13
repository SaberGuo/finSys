"""演示verbose模式下的选股详细输出"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from finquant.selection.strategies.breakout import BreakoutStrategy, BreakoutConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

print("=" * 70)
print("MA突破选股策略 - Verbose模式演示")
print("=" * 70)

# 创建合成数据：模拟一个突破场景
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
tickers = ['600030.SH', '000001.SZ', '000002.SZ']

data = []
for tic in tickers:
    for i, date in enumerate(dates):
        # 基础价格
        base_price = 100 if tic == '600030.SH' else 50

        # 模拟突破：在8月1日附近突破
        if i < 150:  # 前150天在MA下方
            close = base_price * 0.95 + np.random.randn() * 0.5
        elif i == 150:  # 8月1日突破
            close = base_price * 1.08  # 突破8%
        else:  # 突破后保持在上方
            close = base_price * 1.05 + np.random.randn() * 0.5

        # 模拟放量
        if i == 150:
            volume = 10000000  # 突破日放量
        else:
            volume = 5000000 + np.random.randint(-1000000, 1000000)

        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'tic': tic,
            'open': close * 0.99,
            'high': close * 1.01,
            'low': close * 0.98,
            'close': close,
            'volume': max(volume, 1000000)
        })

market_df = pd.DataFrame(data)

# 创建策略（verbose=True）
config = BreakoutConfig(
    ma_periods=[60, 120],
    volume_multiplier=1.2,
    breakout_threshold=1.02,
    lookback_days=30,
    confirmation_days=2,
    anti_jitter_mode='either',
    top_k=10
)

strategy = BreakoutStrategy(config, verbose=True)

# 准备索引数据（使用第一只股票）
index_df = market_df[market_df['tic'] == tickers[0]].copy()

# 测试突破日期
test_date = '2020-05-30'  # dates[150]的日期

print("\n测试日期:", test_date)
print("\n" + "=" * 70)

result = strategy.select(market_df, index_df, test_date)

print("\n" + "=" * 70)
print("选股结果总结")
print("=" * 70)
print(f"日期: {result.date}")
print(f"选中股票数: {len(result.selected_tickers)}")
print(f"市场状态: {result.market_state.value}")
print(f"活跃因子: {', '.join(result.active_factors)}")

if result.selected_tickers:
    print(f"\n选中股票:")
    for tic in result.selected_tickers:
        print(f"  {tic}: {result.scores[tic]:.4f}")
else:
    print("\n未选中任何股票")

print("\n" + "=" * 70)
print("说明:")
print("=" * 70)
print("使用 --verbose 参数运行 finquant selection run 命令时，")
print("会显示每只股票的详细筛选过程：")
print("  - 配置参数")
print("  - 候选股票数")
print("  - 通过筛选的股票数")
print("  - 每只选中股票的详细信息：")
print("    * 评分")
print("    * 收盘价")
print("    * MA突破幅度")
print("    * 成交量放大倍数")
print("\n示例命令:")
print("  finquant selection run \\")
print("    --config config/selection_breakout.yaml \\")
print("    --start 2020-08-01 \\")
print("    --end 2020-08-31 \\")
print("    --verbose")
print("=" * 70)
