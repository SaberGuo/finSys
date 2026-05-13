"""测试从 zz500_data.db 加载数据并运行突破策略"""

from finquant.config.settings import load_config
from finquant.data.pipeline import DataPipeline
from finquant.selection import create_strategy

print("=" * 70)
print("测试突破策略数据加载")
print("=" * 70)

# 加载配置
print("\n1. 加载配置...")
config = load_config("config/selection_breakout.yaml")
print(f"   策略类型: {config.selection.strategy_type}")
print(f"   股票数量: {len(config.stocks)}")
print(f"   股票列表: {config.stocks[:3]}...")
print(f"   数据源: {config.data.source_priority}")
print(f"   日期范围: {config.dates.test_start} 到 {config.dates.test_end}")

# 加载市场数据
print("\n2. 加载市场数据...")
pipeline = DataPipeline(config)
market_df = pipeline.fetch()
print(f"   数据形状: {market_df.shape}")
print(f"   列名: {list(market_df.columns)}")
print(f"   日期范围: {market_df['date'].min()} 到 {market_df['date'].max()}")
print(f"   股票数量: {market_df['tic'].nunique()}")
print("\n   前3条数据:")
print(market_df.head(3))

# 创建策略
print("\n3. 创建策略...")
strategy = create_strategy(config)
print(f"   策略类型: {type(strategy).__name__}")
print(f"   MA周期: {strategy.config.ma_periods}")
print(f"   放量倍数: {strategy.config.volume_multiplier}")
print(f"   突破阈值: {strategy.config.breakout_threshold}")

# 准备索引数据（使用第一只股票作为示例）
print("\n4. 准备索引数据...")
index_df = market_df[market_df['tic'] == config.stocks[0]].copy()
print(f"   索引数据形状: {index_df.shape}")

# 运行选股（选择一个有足够历史数据的日期）
print("\n5. 运行选股...")
test_date = "2020-08-01"
print(f"   选股日期: {test_date}")

try:
    result = strategy.select(market_df, index_df, test_date)
    print(f"   ✓ 选股成功!")
    print(f"   选中股票数: {len(result.selected_tickers)}")
    print(f"   市场状态: {result.market_state.value}")

    if result.selected_tickers:
        print(f"\n   选中的股票:")
        for tic in result.selected_tickers[:5]:
            print(f"     {tic}: {result.scores[tic]:.4f}")
    else:
        print(f"   未选中任何股票（可能是条件太严格或数据不足）")

    if result.exclusion_reasons:
        print(f"\n   排除的股票:")
        for tic, reason in list(result.exclusion_reasons.items())[:3]:
            print(f"     {tic}: {reason}")

except Exception as e:
    print(f"   ✗ 选股失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
