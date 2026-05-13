"""测试突破策略 - 使用更宽松的参数"""

from finquant.config.settings import load_config
from finquant.data.pipeline import DataPipeline
from finquant.selection import create_strategy
import pandas as pd

print("=" * 70)
print("测试突破策略 - 宽松参数版本")
print("=" * 70)

# 加载配置
print("\n1. 加载配置...")
config = load_config("config/selection_breakout_test.yaml")
print(f"   MA周期: {config.selection.breakout.ma_periods}")
print(f"   放量倍数: {config.selection.breakout.volume_multiplier}")
print(f"   突破阈值: {config.selection.breakout.breakout_threshold}")
print(f"   防抖动模式: {config.selection.breakout.anti_jitter_mode}")

# 加载数据
print("\n2. 加载数据...")
pipeline = DataPipeline(config)
market_df = pipeline.fetch()
print(f"   数据形状: {market_df.shape}")
print(f"   日期范围: {market_df['date'].min()} 到 {market_df['date'].max()}")

# 创建策略
print("\n3. 创建策略...")
strategy = create_strategy(config)

# 准备索引数据
index_df = market_df[market_df['tic'] == config.stocks[0]].copy()

# 测试多个日期
print("\n4. 测试多个日期...")
test_dates = ["2020-08-03", "2020-09-01", "2020-10-01", "2020-11-02", "2020-12-01"]

for test_date in test_dates:
    try:
        result = strategy.select(market_df, index_df, test_date)
        print(f"\n   {test_date}:")
        print(f"     选中: {len(result.selected_tickers)} 只股票")

        if result.selected_tickers:
            print(f"     股票: {', '.join(result.selected_tickers[:5])}")
            print(f"     得分范围: {min(result.scores.values()):.4f} ~ {max(result.scores.values()):.4f}")

        if result.exclusion_reasons:
            print(f"     排除: {len(result.exclusion_reasons)} 只股票")

    except Exception as e:
        print(f"   {test_date}: 错误 - {e}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
