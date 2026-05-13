#!/usr/bin/env python3
"""随机选择时间和股票进行RL训练。

从zz500_data.db中：
1. 随机选择训练时间段
2. 随机选择N只股票
3. 构建5分钟数据集并训练RL模型
4. 回测并输出报告

用法:
    python scripts/random_rl_train.py --stock-count 10 --train-days 60
    python scripts/random_rl_train.py --stock-count 20 --train-days 90 --algo sac --timesteps 200000
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finquant.config.settings import AppConfig, DataConfig, DatesConfig, EnvironmentConfig, FrequencyConfig, TrainingConfig
from finquant.data.pipeline import DataPipeline
from finquant.data.preprocessor import preprocess_5min_data
from finquant.data.sources.db_5min import Db5MinDataSource
from finquant.features.technical import compute_indicators
from finquant.training.env import build_env
from finquant.training.trainer import Trainer
from finquant.utils.logging import get_logger

logger = get_logger("random_rl_train")

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "zz500_data.db"


def get_db_info(db_path: Path) -> tuple[list[str], str, str]:
    """获取数据库中的股票列表和日期范围。

    Returns:
        (股票代码列表, 最早日期, 最晚日期)
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 获取所有股票代码
        cursor.execute("SELECT DISTINCT code FROM minute_data ORDER BY code")
        codes = [row[0] for row in cursor.fetchall()]

        # 获取日期范围
        cursor.execute("SELECT MIN(date), MAX(date) FROM minute_data")
        min_date, max_date = cursor.fetchone()

    return codes, min_date, max_date


def random_select_stocks(all_codes: list[str], count: int) -> list[str]:
    """随机选择N只股票。

    Args:
        all_codes: 数据库中的代码列表 (如 '600000.SH')
        count: 选择的股票数量

    Returns:
        随机选择的股票代码列表 (如 '600000.SH')
    """
    selected = random.sample(all_codes, min(count, len(all_codes)))
    return selected


def random_select_time_range(
    min_date: str,
    max_date: str,
    train_days: int,
    test_days: int = 30,
) -> tuple[str, str, str, str]:
    """随机选择训练和测试时间段。

    Args:
        min_date: 数据库最早日期 (YYYY-MM-DD)
        max_date: 数据库最晚日期 (YYYY-MM-DD)
        train_days: 训练天数
        test_days: 测试天数

    Returns:
        (train_start, train_end, test_start, test_end)
    """
    min_dt = datetime.strptime(min_date, "%Y-%m-%d")
    max_dt = datetime.strptime(max_date, "%Y-%m-%d")

    total_days = train_days + test_days
    # 确保有足够的数据
    available_days = (max_dt - min_dt).days
    if available_days < total_days:
        raise ValueError(f"数据库只有 {available_days} 天数据，但需要 {total_days} 天")

    # 随机选择训练开始日期
    latest_start = max_dt - timedelta(days=total_days)
    start_offset = random.randint(0, (latest_start - min_dt).days)
    train_start = min_dt + timedelta(days=start_offset)
    train_end = train_start + timedelta(days=train_days)
    test_start = train_end + timedelta(days=1)
    test_end = test_start + timedelta(days=test_days)

    return (
        train_start.strftime("%Y-%m-%d"),
        train_end.strftime("%Y-%m-%d"),
        test_start.strftime("%Y-%m-%d"),
        min(test_end.strftime("%Y-%m-%d"), max_date),
    )


def fetch_all_stocks_data(
    db_path: Path,
    all_codes: list[str],
    min_date: str,
    max_date: str,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """从数据库预取所有股票数据。

    Args:
        db_path: 数据库路径
        all_codes: 所有股票代码列表 (标准格式如 '600000.SH')
        min_date: 最早日期
        max_date: 最晚日期
        indicators: 技术指标列表

    Returns:
        包含所有股票的完整DataFrame
    """
    logger.info(f"Pre-fetching all {len(all_codes)} stocks from {min_date} to {max_date}")

    # 直接查询数据库（不使用Db5MinDataSource，因为它的格式转换不匹配）
    placeholders = ",".join("?" for _ in all_codes)
    query = f"""
        SELECT code, date, time, open, high, low, close, volume, amount
        FROM minute_data
        WHERE code IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY code, date, time
    """

    with sqlite3.connect(db_path) as conn:
        params = (*all_codes, min_date, max_date)
        raw_df = pd.read_sql_query(query, conn, params=params)

    raw_df = raw_df.rename(columns={"code": "tic"})

    logger.info(f"Raw data: {len(raw_df)} rows")

    # 清理无效数据
    for col in ["open", "high", "low", "close", "volume"]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    # 移除价格<=0或volume<0的行
    raw_df = raw_df.dropna(subset=["open", "high", "low", "close", "volume"])
    raw_df = raw_df[(raw_df["close"] > 0) & (raw_df["volume"] >= 0)]

    # 移除完全没有有效数据的股票
    valid_tickers = raw_df.groupby("tic").size()
    valid_tickers = valid_tickers[valid_tickers > 0].index.tolist()
    raw_df = raw_df[raw_df["tic"].isin(valid_tickers)]

    logger.info(f"After cleaning: {len(raw_df)} rows, {len(valid_tickers)} valid tickers")

    if len(raw_df) == 0:
        raise ValueError("No valid data after cleaning")

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

    logger.info(f"Pre-fetched data: {len(enriched_df)} rows, {enriched_df['tic'].nunique()} tickers")

    return enriched_df


def filter_epoch_data(
    all_df: pd.DataFrame,
    stocks: list[str],
    train_start: str,
    test_end: str,
) -> pd.DataFrame:
    """从预取的数据中过滤出当前epoch的数据。

    Args:
        all_df: 预取的完整DataFrame
        stocks: 当前epoch的股票列表 (标准格式如 '600000.SH')
        train_start: 训练开始日期
        test_end: 测试结束日期

    Returns:
        过滤后的DataFrame
    """
    filtered_df = all_df[all_df["tic"].isin(stocks)].copy()
    filtered_df = filtered_df[(filtered_df["date"] >= train_start) & (filtered_df["date"] <= test_end)]

    logger.info(f"Filtered data: {len(filtered_df)} rows, {filtered_df['tic'].nunique()} tickers")

    return filtered_df


def fetch_random_data(
    db_path: Path,
    stocks: list[str],
    train_start: str,
    test_end: str,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """从数据库获取随机选择的股票数据。

    Args:
        db_path: 数据库路径
        stocks: 股票代码列表 (标准格式如 '600000.SH')
        train_start: 训练开始日期
        test_end: 测试结束日期
        indicators: 技术指标列表

    Returns:
        处理后的DataFrame
    """
    logger.info(f"Fetching data for {len(stocks)} stocks from {train_start} to {test_end}")

    # 直接查询数据库
    placeholders = ",".join("?" for _ in stocks)
    query = f"""
        SELECT code, date, time, open, high, low, close, volume, amount
        FROM minute_data
        WHERE code IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY code, date, time
    """

    with sqlite3.connect(db_path) as conn:
        params = (*stocks, train_start, test_end)
        raw_df = pd.read_sql_query(query, conn, params=params)

    raw_df = raw_df.rename(columns={"code": "tic"})

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


def train_and_backtest(
    df: pd.DataFrame,
    train_end: str,
    algo: str = "ppo",
    timesteps: int = 100_000,
    output_dir: Path | None = None,
    model: Any = None,
    epoch: int | None = None,
) -> tuple[dict, Any]:
    """训练模型并回测。

    Args:
        df: 完整数据集
        train_end: 训练结束日期 (YYYY-MM-DD)
        algo: 算法 (ppo/sac/td3)
        timesteps: 训练步数
        output_dir: 输出目录
        model: 已有模型（用于增量训练）
        epoch: Epoch编号（用于日志）

    Returns:
        (回测报告字典, 训练后的模型)
    """
    from finrl.agents.stablebaselines3.models import DRLAgent

    # 分割训练/测试集
    train_df = df[df["date"] <= train_end].copy()
    test_df = df[df["date"] > train_end].copy()

    logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")

    if len(test_df) == 0:
        raise ValueError("测试集为空，请增加数据时间范围")

    # 创建配置
    stock_dim = train_df["tic"].nunique()
    config = AppConfig(
        stocks=train_df["tic"].unique().tolist(),
        dates=DatesConfig(
            train_start=train_df["date"].min(),
            train_end=train_end,
            test_start=test_df["date"].min() if len(test_df) > 0 else train_end,
            test_end=test_df["date"].max() if len(test_df) > 0 else train_end,
        ),
        data=DataConfig(),
        frequency=FrequencyConfig(value="5min"),
        environment=EnvironmentConfig(),
        training=TrainingConfig(algorithm=algo, total_timesteps=timesteps),
    )

    # 训练
    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
    logger.info(f"Starting {algo.upper()} training{epoch_str} for {timesteps} steps...")
    trainer = Trainer(config)

    # 构建环境
    from finquant.training.env import build_env

    indicators = config.indicators if hasattr(config, "indicators") else []
    env = build_env(
        train_df,
        stock_dim=stock_dim,
        initial_amount=config.environment.initial_amount,
        hmax=config.environment.hmax,
        buy_cost_pct=config.environment.buy_cost_pct,
        sell_cost_pct=config.environment.sell_cost_pct,
        reward_scaling=config.environment.reward_scaling,
        indicators=indicators,
    )

    agent = DRLAgent(env=env)

    # 使用已有模型或创建新模型
    if model is None:
        # FinRL expects lowercase algorithm names
        hyperparams = getattr(config.training, algo, {}) or {}
        model = agent.get_model(algo, model_kwargs=hyperparams)
        logger.info("Created new model")
    else:
        logger.info("Continuing training from existing model")

    # 训练
    tb_log_name = f"{algo}_epoch{epoch}" if epoch is not None else algo
    trained_model = agent.train_model(
        model=model,
        tb_log_name=tb_log_name,
        total_timesteps=timesteps,
    )

    # 保存模型
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"{algo}_epoch{epoch}" if epoch is not None else algo
        model_path = output_dir / f"{model_name}.zip"
        trained_model.save(str(model_path))
        logger.info(f"Model saved to: {model_path}")
    else:
        model_path = Path(f"{algo}_temp.zip")
        trained_model.save(str(model_path))

    # 回测
    logger.info("Running backtest...")
    obs_dim = env.observation_space.shape[0]
    report = trainer.backtest(
        model_path=model_path,
        test_df=test_df,
        expected_obs_dim=obs_dim,
    )

    return report.to_dict(), trained_model


def save_epoch_results(
    epoch: int,
    stocks: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    timesteps: int,
    report: dict,
    output_dir: Path,
) -> None:
    """保存单个epoch的结果。

    Args:
        epoch: Epoch编号
        stocks: 股票列表
        train_start: 训练开始日期
        train_end: 训练结束日期
        test_start: 测试开始日期
        test_end: 测试结束日期
        timesteps: 训练步数
        report: 回测报告字典
        output_dir: 输出目录
    """
    epoch_info = {
        "epoch": epoch,
        "stocks": stocks,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "timesteps": timesteps,
        "metrics": {
            "annual_return": report.get("annual_return", 0),
            "max_drawdown": report.get("max_drawdown", 0),
            "sharpe_ratio": report.get("sharpe_ratio", 0),
            "win_rate": report.get("win_rate", 0),
            "total_trades": report.get("total_trades", 0),
        },
    }

    epoch_info_path = output_dir / f"epoch_{epoch}_info.json"
    with open(epoch_info_path, "w", encoding="utf-8") as f:
        json.dump(epoch_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Epoch {epoch} results saved to {epoch_info_path}")


def aggregate_epoch_metrics(epoch_results: list[dict]) -> dict:
    """聚合所有epoch的指标统计。

    Args:
        epoch_results: 所有epoch的结果列表

    Returns:
        聚合后的统计字典
    """
    import numpy as np

    metrics_keys = ["annual_return", "max_drawdown", "sharpe_ratio", "win_rate", "total_trades"]
    aggregated = {
        "total_epochs": len(epoch_results),
        "total_timesteps": sum(r["timesteps"] for r in epoch_results),
        "metrics": {},
    }

    for key in metrics_keys:
        values = [r["metrics"][key] for r in epoch_results if key in r["metrics"]]
        if values:
            aggregated["metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="随机选择时间和股票进行RL训练")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH),
                        help="zz500_data.db 路径")
    parser.add_argument("--stock-count", type=int, default=5,
                        help="随机选择的股票数量 (默认: 10)")
    parser.add_argument("--train-days", type=int, default=90,
                        help="训练天数 (默认: 60)")
    parser.add_argument("--test-days", type=int, default=30,
                        help="测试天数 (默认: 30)")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "td3"],
                        help="RL算法 (默认: ppo)")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="训练步数 (默认: 100000, 仅在epochs=1时使用)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 1)")
    parser.add_argument("--stocks-per-epoch", type=int, default=5,
                        help="每轮随机选择的股票数量 (默认: 10)")
    parser.add_argument("--timesteps-per-epoch", type=int, default=10_000,
                        help="每轮训练步数 (默认: 10000)")
    parser.add_argument("--indicators", type=str, nargs="+", default=None,
                        help="技术指标列表 (默认: macd, boll_ub, boll_lb, rsi_30, dx_30, close_30_sma, close_60_sma, adx_14, atr_14)")
    parser.add_argument("--output-dir", type=str, default="runs/random",
                        help="输出目录 (默认: runs/random)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (用于可复现)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细信息")

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"数据库不存在: {db_path}")
        sys.exit(1)

    # 获取数据库信息
    logger.info("Reading database info...")
    all_codes, min_date, max_date = get_db_info(db_path)
    logger.info(f"Database: {len(all_codes)} stocks, {min_date} ~ {max_date}")

    # 确定训练参数
    num_epochs = args.epochs
    stocks_per_epoch = args.stocks_per_epoch if num_epochs > 1 else args.stock_count
    timesteps_per_epoch = args.timesteps_per_epoch if num_epochs > 1 else args.timesteps

    logger.info(f"Training configuration: {num_epochs} epochs, {stocks_per_epoch} stocks/epoch, {timesteps_per_epoch} steps/epoch")

    # 创建输出目录
    output_dir = PROJECT_ROOT / args.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 多轮训练模式
    if num_epochs > 1:
        logger.info("=" * 60)
        logger.info(f"Multi-epoch training mode: {num_epochs} epochs")
        logger.info("=" * 60)

        # 预取所有股票数据
        try:
            all_stocks_df = fetch_all_stocks_data(
                db_path=db_path,
                all_codes=all_codes,
                min_date=min_date,
                max_date=max_date,
                indicators=args.indicators,
            )
        except Exception as e:
            logger.error(f"数据预取失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 初始化
        epoch_results = []
        model = None
        start_time = time.time()

        # Epoch循环
        for epoch in range(num_epochs):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"[Epoch {epoch + 1}/{num_epochs}] Starting...")
            logger.info(f"{'=' * 60}")

            # 随机选择股票
            selected_stocks = random_select_stocks(all_codes, stocks_per_epoch)
            logger.info(f"[Epoch {epoch + 1}/{num_epochs}] Stocks: {', '.join(selected_stocks[:3])}... ({len(selected_stocks)} total)")

            # 随机选择时间范围
            train_start, train_end, test_start, test_end = random_select_time_range(
                min_date, max_date, args.train_days, args.test_days
            )
            logger.info(f"[Epoch {epoch + 1}/{num_epochs}] Time: train={train_start}~{train_end}, test={test_start}~{test_end}")

            # 过滤数据
            try:
                epoch_df = filter_epoch_data(
                    all_df=all_stocks_df,
                    stocks=selected_stocks,
                    train_start=train_start,
                    test_end=test_end,
                )
            except Exception as e:
                logger.error(f"[Epoch {epoch + 1}/{num_epochs}] 数据过滤失败: {e}")
                continue

            if len(epoch_df) == 0:
                logger.warning(f"[Epoch {epoch + 1}/{num_epochs}] 数据为空，跳过")
                continue

            # 训练并回测
            epoch_output_dir = output_dir / f"epoch_{epoch}"
            try:
                report, model = train_and_backtest(
                    df=epoch_df,
                    train_end=train_end,
                    algo=args.algo,
                    timesteps=timesteps_per_epoch,
                    output_dir=epoch_output_dir,
                    model=model,
                    epoch=epoch,
                )
            except Exception as e:
                logger.error(f"[Epoch {epoch + 1}/{num_epochs}] 训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 保存epoch结果
            save_epoch_results(
                epoch=epoch,
                stocks=selected_stocks,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                timesteps=timesteps_per_epoch,
                report=report,
                output_dir=epoch_output_dir,
            )

            # 记录结果
            epoch_result = {
                "epoch": epoch,
                "stocks": selected_stocks,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "timesteps": timesteps_per_epoch,
                "metrics": {
                    "annual_return": report.get("annual_return", 0),
                    "max_drawdown": report.get("max_drawdown", 0),
                    "sharpe_ratio": report.get("sharpe_ratio", 0),
                    "win_rate": report.get("win_rate", 0),
                    "total_trades": report.get("total_trades", 0),
                },
            }
            epoch_results.append(epoch_result)

            # 输出当前结果
            logger.info(f"[Epoch {epoch + 1}/{num_epochs}] Backtest: annual_return={report.get('annual_return', 0):.2%}, sharpe={report.get('sharpe_ratio', 0):.2f}")

            # 计算ETA
            elapsed = time.time() - start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            eta = avg_time_per_epoch * (num_epochs - epoch - 1)
            logger.info(f"[Epoch {epoch + 1}/{num_epochs}] Elapsed: {elapsed:.1f}s, ETA: {eta / 60:.1f}min")

        # 保存最终模型
        if model is not None:
            final_model_path = output_dir / "final_model.zip"
            model.save(str(final_model_path))
            logger.info(f"Final model saved to: {final_model_path}")

        # 聚合结果
        if epoch_results:
            # 保存所有epoch的指标到CSV
            metrics_rows = []
            for r in epoch_results:
                row = {
                    "epoch": r["epoch"],
                    "train_start": r["train_start"],
                    "train_end": r["train_end"],
                    "test_start": r["test_start"],
                    "test_end": r["test_end"],
                    "timesteps": r["timesteps"],
                    **r["metrics"],
                }
                metrics_rows.append(row)

            metrics_df = pd.DataFrame(metrics_rows)
            metrics_csv_path = output_dir / "all_epochs_metrics.csv"
            metrics_df.to_csv(metrics_csv_path, index=False)
            logger.info(f"All epochs metrics saved to: {metrics_csv_path}")

            # 聚合统计
            aggregated = aggregate_epoch_metrics(epoch_results)
            aggregated_path = output_dir / "aggregated_summary.json"
            with open(aggregated_path, "w", encoding="utf-8") as f:
                json.dump(aggregated, f, indent=2, ensure_ascii=False)
            logger.info(f"Aggregated summary saved to: {aggregated_path}")

            # 输出汇总
            print("\n" + "=" * 60)
            print(f"多轮训练完成! ({num_epochs} epochs)")
            print("=" * 60)
            print(f"\n总训练步数: {aggregated['total_timesteps']:,}")
            print(f"\n聚合指标:")
            for key, stats in aggregated["metrics"].items():
                print(f"  {key}:")
                print(f"    mean: {stats['mean']:.4f}")
                print(f"    std:  {stats['std']:.4f}")
                print(f"    min:  {stats['min']:.4f}")
                print(f"    max:  {stats['max']:.4f}")

            print(f"\n输出目录: {output_dir}")

    else:
        # 单轮训练模式（原有逻辑）
        logger.info("Single-epoch training mode")

        # 随机选择股票
        selected_stocks = random_select_stocks(all_codes, args.stock_count)
        logger.info(f"Selected {len(selected_stocks)} stocks: {', '.join(selected_stocks[:5])}{'...' if len(selected_stocks) > 5 else ''}")

        # 随机选择时间范围
        train_start, train_end, test_start, test_end = random_select_time_range(
            min_date, max_date, args.train_days, args.test_days
        )
        logger.info(f"Time range: train={train_start}~{train_end}, test={test_start}~{test_end}")

        # 获取数据
        try:
            df = fetch_random_data(
                db_path=db_path,
                stocks=selected_stocks,
                train_start=train_start,
                test_end=test_end,
                indicators=args.indicators,
            )
        except Exception as e:
            logger.error(f"数据获取失败: {e}")
            sys.exit(1)

        if len(df) == 0:
            logger.error("获取的数据为空")
            sys.exit(1)

        # 训练并回测
        try:
            report, _ = train_and_backtest(
                df=df,
                train_end=train_end,
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
        print(f"\n股票池 ({len(selected_stocks)}只):")
        for i, s in enumerate(selected_stocks, 1):
            print(f"  {i}. {s}")

        print(f"\n时间范围:")
        print(f"  训练: {train_start} ~ {train_end} ({args.train_days}天)")
        print(f"  测试: {test_start} ~ {test_end} ({args.test_days}天)")

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
            "stocks": selected_stocks,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "algorithm": args.algo,
            "timesteps": args.timesteps,
            "report": report,
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"元数据保存: {meta_path}")


if __name__ == "__main__":
    main()
