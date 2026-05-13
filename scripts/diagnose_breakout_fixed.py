"""诊断为什么没有选中股票 - 修复版"""

from finquant.config.settings import load_config
from finquant.data.pipeline import DataPipeline
import pandas as pd

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

stock_df = market_df[market_df['tic'] == test_tic].sort_values('date').reset_index(drop=True).copy()
print(f"\n分析股票: {test_tic}")
print(f"  记录数: {len(stock_df)}")

# 计算MA
stock_df['ma60'] = stock_df['close'].rolling(60).mean()
stock_df['ma120'] = stock_df['close'].rolling(120).mean()
stock_df['vol_ma20'] = stock_df['volume'].rolling(20).mean()

# 查看测试日期前后的数据
test_rows = stock_df[stock_df['date'] == test_date]
if len(test_rows) > 0:
    idx = test_rows.index[0]
    window = stock_df.iloc[max(0, idx-5):min(len(stock_df), idx+3)]

    print(f"\n{test_date} 前后数据:")
    print(window[['date', 'close', 'ma60', 'ma120', 'volume', 'vol_ma20']].to_string())

    # 检查突破条件
    today = stock_df.iloc[idx]
    if idx > 0:
        yesterday = stock_df.iloc[idx-1]

        print(f"\n突破条件检查 ({test_date}):")
        print(f"\n  1. MA60突破:")
        if pd.notna(today['ma60']) and pd.notna(yesterday['ma60']):
            print(f"     今天: close({today['close']:.2f}) > ma60({today['ma60']:.2f}) = {today['close'] > today['ma60']}")
            print(f"     昨天: close({yesterday['close']:.2f}) <= ma60({yesterday['ma60']:.2f}) = {yesterday['close'] <= yesterday['ma60']}")
            print(f"     → 突破: {today['close'] > today['ma60'] and yesterday['close'] <= yesterday['ma60']}")
        else:
            print(f"     MA60数据不足 (today={pd.notna(today['ma60'])}, yesterday={pd.notna(yesterday['ma60'])})")

        print(f"\n  2. MA120突破:")
        if pd.notna(today['ma120']) and pd.notna(yesterday['ma120']):
            print(f"     今天: close({today['close']:.2f}) > ma120({today['ma120']:.2f}) = {today['close'] > today['ma120']}")
            print(f"     昨天: close({yesterday['close']:.2f}) <= ma120({yesterday['ma120']:.2f}) = {yesterday['close'] <= yesterday['ma120']}")
            print(f"     → 突破: {today['close'] > today['ma120'] and yesterday['close'] <= yesterday['ma120']}")
        else:
            print(f"     MA120数据不足 (today={pd.notna(today['ma120'])}, yesterday={pd.notna(yesterday['ma120'])})")

        print(f"\n  3. 放量:")
        if pd.notna(today['vol_ma20']):
            print(f"     volume({today['volume']}) >= 1.2 * vol_ma20({today['vol_ma20']:.0f})")
            print(f"     → {today['volume']} >= {1.2 * today['vol_ma20']:.0f} = {today['volume'] >= 1.2 * today['vol_ma20']}")
        else:
            print(f"     vol_ma20数据不足")

        print(f"\n  4. 价格阈值:")
        if pd.notna(today['ma60']):
            print(f"     close({today['close']:.2f}) >= ma60({today['ma60']:.2f}) * 1.02 = {today['close'] >= today['ma60'] * 1.02}")
        if pd.notna(today['ma120']):
            print(f"     close({today['close']:.2f}) >= ma120({today['ma120']:.2f}) * 1.02 = {today['close'] >= today['ma120'] * 1.02}")

# 统计所有股票的MA数据可用性
print(f"\n\n所有股票在 {test_date} 的MA数据可用性:")
for tic in sorted(market_df['tic'].unique()):
    stock_df = market_df[market_df['tic'] == tic].sort_values('date').reset_index(drop=True).copy()
    stock_df['ma60'] = stock_df['close'].rolling(60).mean()
    stock_df['ma120'] = stock_df['close'].rolling(120).mean()

    today_data = stock_df[stock_df['date'] == test_date]
    if len(today_data) > 0:
        today = today_data.iloc[0]
        has_ma60 = pd.notna(today['ma60'])
        has_ma120 = pd.notna(today['ma120'])
        above_ma60 = today['close'] > today['ma60'] if has_ma60 else None
        above_ma120 = today['close'] > today['ma120'] if has_ma120 else None

        print(f"  {tic}: MA60={'✓' if has_ma60 else '✗'}, MA120={'✓' if has_ma120 else '✗'}, "
              f"above_MA60={above_ma60}, above_MA120={above_ma120}")

print("\n结论:")
print("  如果MA120数据不足(✗)，说明从2020-01-02到2020-08-03不足120个交易日")
print("  需要更早的历史数据，或者使用更短的MA周期（如MA30, MA60）")

print("\n" + "=" * 70)
