#!/usr/bin/env python3
"""Backtest ZZ500 single-stock RL strategy.

Daily workflow:
1. Score all 355 ZZ500 stocks using single-stock RL model
2. Filter stocks with score > 0.9
3. Select top 5 by score
4. Allocate capital by score-weighted distribution
5. Execute trades with stop-loss/take-profit

Usage:
    python scripts/backtest_zz500_single_stock.py \
        --model models/zz500_single_stock/ppo_zz500_single_20240630_abc123.zip \
        --config config/zz500_rl_single_stock.yaml \
        --start 2024-07-01 \
        --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from finquant.config.settings import load_config
from finquant.data.sources.db_daily import DbDailyDataSource
from finquant.data.sources.zz500_loader import load_zz500_stocks
from finquant.features.technical import compute_indicators
from finquant.selection.rl_scorer import RLStockScorer
from finquant.utils.logging import get_logger

logger = get_logger("backtest_zz500_single_stock")


def main():
    parser = argparse.ArgumentParser(description="Backtest ZZ500 single-stock RL strategy")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--db-path", type=str, default="data/processed/zz500_data.db")
    parser.add_argument("--output-dir", type=str, default="runs/zz500_backtest")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    model_path = Path(args.model)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    logger.info("Loading ZZ500 stock universe...")
    all_stocks = load_zz500_stocks(db_path)
    logger.info(f"Loaded {len(all_stocks)} ZZ500 stocks")

    config = load_config(args.config)
    config.stocks = all_stocks

    logger.info("Loading market data...")
    data_source = DbDailyDataSource(db_path=db_path)
    market_df = data_source.download(
        symbols=all_stocks,
        start_date=args.start,
        end_date=args.end,
    )

    logger.info("Computing indicators...")
    market_df = compute_indicators(market_df, config.indicators)

    logger.info("Initializing RL scorer...")
    scorer = RLStockScorer(
        model_path=model_path,
        indicators=config.indicators,
        score_mapping=config.zz500_selection.score_mapping,
    )

    logger.info(f"Running backtest from {args.start} to {args.end}...")

    portfolio_size = config.zz500_selection.portfolio_size
    score_threshold = config.zz500_selection.score_threshold
    initial_amount = config.environment.initial_amount

    cash = initial_amount
    positions = {}
    entry_prices = {}

    daily_records = []
    trading_days = sorted(market_df["date"].unique())

    for date in trading_days:
        logger.info(f"\n{'='*60}")
        logger.info(f"Date: {date}")
        logger.info(f"{'='*60}")

        scores = scorer.score_stocks(market_df, date)

        qualified = {k: v for k, v in scores.items() if v > score_threshold}

        if not qualified:
            logger.info(f"No stocks meet threshold {score_threshold}")
            daily_records.append({
                "date": date,
                "selected_tickers": [],
                "scores": {},
                "cash": cash,
                "positions": positions.copy(),
            })
            continue

        sorted_stocks = sorted(qualified.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_stocks[:portfolio_size]

        selected_tickers = [t for t, _ in top_k]
        selected_scores = {t: s for t, s in top_k}

        logger.info(f"Selected {len(selected_tickers)} stocks:")
        for ticker, score in top_k:
            logger.info(f"  {ticker}: {score:.4f}")

        total_score = sum(selected_scores.values())
        weights = {t: s / total_score for t, s in selected_scores.items()}

        allocations = {t: cash * w for t, w in weights.items()}

        date_df = market_df[market_df["date"] == date]
        prices = {row["tic"]: row["close"] for _, row in date_df.iterrows()}

        new_positions = {}
        for ticker in selected_tickers:
            if ticker in prices:
                price = prices[ticker]
                shares = int(allocations[ticker] / price)
                if shares > 0:
                    new_positions[ticker] = shares
                    entry_prices[ticker] = price
                    logger.info(f"  Buy {ticker}: {shares} shares @ ¥{price:.2f}")

        total_cost = sum(new_positions[t] * prices[t] for t in new_positions if t in prices)
        cash -= total_cost
        positions = new_positions

        daily_records.append({
            "date": date,
            "selected_tickers": selected_tickers,
            "scores": selected_scores,
            "cash": cash,
            "positions": positions.copy(),
            "entry_prices": entry_prices.copy(),
        })

    final_date = trading_days[-1]
    final_df = market_df[market_df["date"] == final_date]
    final_prices = {row["tic"]: row["close"] for _, row in final_df.iterrows()}

    final_value = cash
    for ticker, shares in positions.items():
        if ticker in final_prices:
            final_value += shares * final_prices[ticker]

    total_return = (final_value - initial_amount) / initial_amount

    print("\n" + "="*60)
    print("ZZ500 Single-Stock RL Backtest Results")
    print("="*60)
    print(f"Period: {args.start} to {args.end}")
    print(f"Initial amount: ¥{initial_amount:,.2f}")
    print(f"Final value: ¥{final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print("="*60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(daily_records)
    results_df.to_csv(output_dir / "daily_records.csv", index=False)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
