"""诊断为什么没有选中股票"""

from finquant.config.settings import load_config
from finquant.data.pipeline import DataPipeline
import pandas as pd
import numpy as np

print("=" * 70)
print("诊断突破策略")
print("=" * 70)

# 加载数据
config = load_config("config/selection_breakout_test.yaml")
pipeline = DataPipeline(config)
market_df = pipeline.fetch()

print(f"\n数据概况:")
print(f"  总记录数: {len(market_df)}")
print(f"  股票数: {market_df['tic'].nunique()}")
print(f"  日期范围: {market_df['date'].min()} ~ {market_df['date'].max()}")

# 选择一只股票进行详细分析
test_tic = "600030.SH"
test_date = "2020-08-03"

stock_df = market_df[market_df['tic'] == test_tic].sort_values('date').copy()
print(f"\n分析股票: {test_tic}")
print(f"  记录数: {len(stock_df)}")

# 计算MA
stock_df['ma60'] = stock_df['close'].rolling(60).mean()
stock_df['ma120'] = stock_df['close'].rolling(120).mean()
stock_df['vol_ma20'] = stock_df['volume'].rolling(20).mean()

# 查看测试日期前后的数据
test_idx = stock_df[stock_df['date'] == test_date].index
if len(test_idx) > 0:
    idx = test_idx[0]
    window = stock_df.loc[max(0, idx-5):min(len(stock_df)-1, idx+2)]

    print(f"\n{test_date} 前后数据:")
    print(window[['date', 'close', 'ma60', 'ma120', 'volume', 'vol_ma20']].to_string())

    # 检查突破条件
    today = stock_df.loc[idx]
    if idx > 0:
        yesterday = stock_df.loc[idx-1]

        print(f"\n突破条件检查:")
        print(f"  1. MA60突破:")
        print(f"     今天: close({today['close']:.2f}) > ma60({today['ma60']:.2f}) = {today['close'] > today['ma60']}")
        print(f"     昨天: close({yesterday['close']:.2f}) <= ma60({yesterday['ma60']:.2f}) = {yesterday['close'] <= yesterday['ma60']}")
        print(f"     → 突破: {today['close'] > today['ma60'] and yesterday['close'] <= yesterday['ma60']}")

        print(f"\n  2. MA120突破:")
        print(f"     今天: close({today['close']:.2f}) > ma120({today['ma120']:.2f}) = {today['close'] > today['ma120']}")
        print(f"     昨天: close({yesterday['close']:.2f}) <= ma120({yesterday['ma120']:.2f}) = {yesterday['close'] <= yesterday['ma120']}")
        print(f"     → 突破: {today['close'] > today['ma120'] and yesterday['close'] <= yesterday['ma120']}")

        print(f"\n  3. 放量:")
        print(f"     volume({today['volume']}) >= 1.2 * vol_ma20({today['vol_ma20']:.0f})")
        print(f"     → {today['volume']} >= {1.2 * today['vol_ma20']:.0f} = {today['volume'] >= 1.2 * today['vol_ma20']}")

        print(f"\n  4. 价格阈值:")
        print(f"     close({today['close']:.2f}) >= ma60({today['ma60']:.2f}) * 1.02 = {today['close'] >= today['ma60'] * 1.02}")
        print(f"     close({today['close']:.2f}) >= ma120({today['ma120']:.2f}) * 1.02 = {today['close'] >= today['ma120'] * 1.02}")

# 统计所有股票在测试日期的情况
print(f"\n\n所有股票在 {test_date} 的情况:")
test_data = market_df[market_df['date'] == test_date].copy()
print(f"  有数据的股票数: {len(test_data)}")

# 计算每只股票的MA
for tic in test_data['tic'].unique():
    stock_df = market_df[market_df['tic'] == tic].sort_values('date').copy()
    stock_df['ma60'] = stock_df['close'].rolling(60).mean()
    stock_df['ma120'] = stock_df['close'].rolling(120).mean()

    today_data = stock_df[stock_df['date'] == test_date]
    if len(today_data) > 0:
        today = today_data.iloc[0]
        has_ma60 = not pd.isna(today['ma60'])
        has_ma120 = not pd.isna(today['ma120'])
        above_ma60 = today['close'] > today['ma60'] if has_ma60 else False
        above_ma120 = today['close'] > today['ma120'] if has_ma120 else False

        print(f"  {tic}: MA60={has_ma60}, MA120={has_ma120}, "
              f"above_MA60={above_ma60}, above_MA120={above_ma120}")

print("\n" + "=" * 70)
