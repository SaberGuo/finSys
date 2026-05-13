#!/usr/bin/env python3
"""基于因子选股结果进行RL训练。

从selection结果中：
1. 读取每日选股结果
2. 提取候选股票池
3. 获取5分钟数据并训练RL模型
4. 回测并输出报告

用法:
    # 使用指定日期的选股结果
    python scripts/selection_rl_train.py --selection-date 2024-01-15 --train-days 30 --test-days 10

    # 使用日期范围内的选股结果（每日更新候选池）
    python scripts/selection_rl_train.py --start 2024-01-01 --end 2024-03-31 --algo sac --timesteps 200000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finquant.config.settings import (
    AppConfig,
    DataConfig,
    DatesConfig,
    EnvironmentConfig,
    FrequencyConfig,
    TrainingConfig,
)
from finquant.data.preprocessor import preprocess_5min_data
from finquant.data.sources.db_5min import Db5MinDataSource
from finquant.features.technical import compute_indicators
from finquant.training.trainer import Trainer
from finquant.utils.logging import get_logger

logger = get_logger("selection_rl_train")

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "zz500_data.db"
DEFAULT_SELECTION_DIR = PROJECT_ROOT / "data" / "selection"


def load_selection_result(selection_dir: Path, date: str) -> dict:
    """加载指定日期的选股结果。

    Args:
        selection_dir: 选股结果目录
        date: 日期 (YYYY-MM-DD)

    Returns:
        选股结果字典
    """
    result_file = selection_dir / f"{date}_selection.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Selection result not found: {result_file}")

    with open(result_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_selected_stocks_for_period(
    selection_dir: Path,
    start_date: str,
    end_date: str,
) -> dict[str, list[str]]:
    """获取时间段内每日的选股结果。

    Args:
        selection_dir: 选股结果目录
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        {date: [selected_tickers]} 字典
    """
    result_files = sorted(selection_dir.glob("*_selection.json"))
    daily_selections = {}

    for file in result_files:
        date = file.stem.replace("_selection", "")
        if start_date <= date <= end_date:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                daily_selections[date] = data["selected_tickers"]

    return daily_selections


def fetch_5min_data_for_stocks(
    db_path: Path,
    stocks: list[str],
    start_date: str,
    end_date: str,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """从数据库获取5分钟数据。

    Args:
        db_path: 数据库路径
        stocks: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        indicators: 技术指标列表

    Returns:
        处理后的DataFrame
    """
    logger.info(f"Fetching 5min data for {len(stocks)} stocks from {start_date} to {end_date}")

    # 直接从db_5min获取数据
    source = Db5MinDataSource(db_path=str(db_path))
    raw_df = source.download(
        symbols=stocks,
        start_date=start_date,
        end_date=end_date,
    )

    if raw_df.empty:
        raise ValueError(f"No 5min data found for stocks {stocks} in date range {start_date} to {end_date}")

    logger.info(f"Raw data: {len(raw_df)} rows")

    # 预处理
    processed_df = preprocess_5min_data(raw_df)

    # 计算技术指标
    default_indicators = indicators or [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
        "adx_14",
        "atr_14",
    ]
    enriched_df = compute_indicators(processed_df, default_indicators)

    logger.info(f"Enriched data: {len(enriched_df)} rows, {enriched_df['tic'].nunique()} tickers")

    return enriched_df


def train_with_single_selection(
    selection_result: dict,
    db_path: Path,
    train_days: int,
    test_days: int,
    algo: str = "ppo",
    timesteps: int = 100_000,
    output_dir: Path | None = None,
) -> dict:
    """使用单日选股结果进行训练。

    Args:
        selection_result: 选股结果字典
        db_path: 数据库路径
        train_days: 训练天数
        test_days: 测试天数
        algo: 算法
        timesteps: 训练步数
        output_dir: 输出目录

    Returns:
        回测报告字典
    """
    selection_date = selection_result["date"]
    selected_stocks = selection_result["selected_tickers"]

    logger.info(f"Using selection from {selection_date}: {len(selected_stocks)} stocks")
    logger.info(f"Market state: {selection_result['market_state']}")
    logger.info(f"Active factors: {', '.join(selection_result['active_factors'])}")

    # 计算时间范围
    sel_dt = datetime.strptime(selection_date, "%Y-%m-%d")
    train_start = (sel_dt - timedelta(days=train_days)).strftime("%Y-%m-%d")
    train_end = selection_date
    test_start = (sel_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    test_end = (sel_dt + timedelta(days=test_days)).strftime("%Y-%m-%d")

    # 获取5分钟数据
    df = fetch_5min_data_for_stocks(
        db_path=db_path,
        stocks=selected_stocks,
        start_date=train_start,
        end_date=test_end,
    )

    # 分割训练/测试集
    train_df = df[df["date"] <= train_end].copy()
    test_df = df[df["date"] > train_end].copy()

    logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")

    if len(test_df) == 0:
        raise ValueError("测试集为空，请增加测试天数")

    # 创建配置
    config = AppConfig(
        stocks=selected_stocks,
        dates=DatesConfig(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ),
        data=DataConfig(),
        frequency=FrequencyConfig(value="5min"),
        environment=EnvironmentConfig(),
        training=TrainingConfig(algorithm=algo, total_timesteps=timesteps),
    )

    # 训练
    logger.info(f"Starting {algo.upper()} training for {timesteps} steps...")
    trainer = Trainer(config)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = trainer.train(train_df=train_df, output_dir=output_dir)
    else:
        model_path = trainer.train(train_df=train_df)

    logger.info(f"Model saved to: {model_path}")

    # 回测
    logger.info("Running backtest...")
    report = trainer.backtest(
        model_path=model_path,
        test_df=test_df,
        expected_obs_dim=trainer._last_train_obs_dim,
    )

    return report.to_dict()


def main():
    parser = argparse.ArgumentParser(description="基于因子选股结果进行RL训练")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH),
                        help="zz500_data.db 路径")
    parser.add_argument("--selection-dir", type=str, default=str(DEFAULT_SELECTION_DIR),
                        help="选股结果目录 (默认: data/selection)")

    # 单日选股模式
    parser.add_argument("--selection-date", type=str, default=None,
                        help="使用指定日期的选股结果 (YYYY-MM-DD)")
    parser.add_argument("--train-days", type=int, default=30,
                        help="训练天数 (默认: 30)")
    parser.add_argument("--test-days", type=int, default=10,
                        help="测试天数 (默认: 10)")

    # 多日选股模式（未来扩展）
    parser.add_argument("--start", type=str, default=None,
                        help="开始日期 (YYYY-MM-DD) - 用于多日选股模式")
    parser.add_argument("--end", type=str, default=None,
                        help="结束日期 (YYYY-MM-DD) - 用于多日选股模式")

    # 训练参数
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "td3"],
                        help="RL算法 (默认: ppo)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="训练步数 (默认: 100000)")
    parser.add_argument("--indicators", type=str, nargs="+", default=None,
                        help="技术指标列表")
    parser.add_argument("--output-dir", type=str, default="runs/selection",
                        help="输出目录 (默认: runs/selection)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细信息")

    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"数据库不存在: {db_path}")
        sys.exit(1)

    selection_dir = Path(args.selection_dir)
    if not selection_dir.exists():
        logger.error(f"选股结果目录不存在: {selection_dir}")
        logger.info("请先运行: finsys selection run --start YYYY-MM-DD --end YYYY-MM-DD")
        sys.exit(1)

    # 单日选股模式
    if args.selection_date:
        try:
            selection_result = load_selection_result(selection_dir, args.selection_date)
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info(f"可用的选股结果: {', '.join([f.stem.replace('_selection', '') for f in sorted(selection_dir.glob('*_selection.json'))[:5]])}...")
            sys.exit(1)

        output_dir = PROJECT_ROOT / args.output_dir / f"{args.selection_date}_{datetime.now().strftime('%H%M%S')}"

        try:
            report = train_with_single_selection(
                selection_result=selection_result,
                db_path=db_path,
                train_days=args.train_days,
                test_days=args.test_days,
                algo=args.algo,
                timesteps=args.timesteps,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.error(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 输出结果
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"\n选股日期: {args.selection_date}")
        print(f"市场状态: {selection_result['market_state']}")
        print(f"活跃因子: {', '.join(selection_result['active_factors'])}")

        print(f"\n股票池 ({len(selection_result['selected_tickers'])}只):")
        for i, (ticker, score) in enumerate(list(selection_result['scores'].items())[:10], 1):
            print(f"  {i}. {ticker} (score: {score:.4f})")

        print(f"\n模型: {args.algo.upper()}")
        print(f"训练步数: {args.timesteps:,}")

        print(f"\n回测结果:")
        print(f"  年化收益率: {report.get('annual_return', 0):.2%}")
        print(f"  最大回撤: {report.get('max_drawdown', 0):.2%}")
        print(f"  夏普比率: {report.get('sharpe_ratio', 0):.2f}")
        print(f"  胜率: {report.get('win_rate', 0):.2%}")
        print(f"  总交易次数: {report.get('total_trades', 0)}")

        print(f"\n输出目录: {output_dir}")

        # 保存元数据
        meta = {
            "selection_date": args.selection_date,
            "selection_result": selection_result,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "algorithm": args.algo,
            "timesteps": args.timesteps,
            "report": report,
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"元数据保存: {meta_path}")

    # 多日选股模式（未来扩展）
    elif args.start and args.end:
        logger.error("多日选股模式尚未实现")
        logger.info("当前仅支持单日选股模式，请使用 --selection-date 参数")
        sys.exit(1)

    else:
        logger.error("请指定 --selection-date 或 --start/--end 参数")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
