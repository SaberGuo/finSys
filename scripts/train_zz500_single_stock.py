#!/usr/bin/env python3
"""Train single-stock RL model on ZZ500 data.

The model learns to evaluate individual stocks (stock_dim=1) and can be
applied to any stock in the universe during inference.

Usage:
    python scripts/train_zz500_single_stock.py \
        --config config/zz500_rl_single_stock.yaml \
        --db-path data/processed/zz500_data.db
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finquant.config.settings import load_config
from finquant.data.sources.db_daily import DbDailyDataSource
from finquant.data.sources.zz500_loader import load_zz500_stocks
from finquant.features.technical import compute_indicators
from finquant.training.trainer import Trainer
from finquant.utils.logging import get_logger

logger = get_logger("train_zz500_single_stock")


def main():
    parser = argparse.ArgumentParser(description="Train ZZ500 single-stock RL model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--db-path", type=str, default="data/processed/zz500_data.db")
    parser.add_argument("--output-dir", type=str, default="models/zz500_single_stock")
    args = parser.parse_args()

    db_path = Path(args.db_path)

    logger.info("Loading ZZ500 stock universe...")
    all_stocks = load_zz500_stocks(db_path)
    logger.info(f"Loaded {len(all_stocks)} ZZ500 stocks")

    config = load_config(args.config)
    config.stocks = all_stocks

    logger.info("Loading training data...")
    data_source = DbDailyDataSource(db_path=db_path)
    train_df = data_source.download(
        symbols=all_stocks,
        start_date=config.dates.train_start,
        end_date=config.dates.train_end,
    )

    logger.info("Computing technical indicators...")
    train_df = compute_indicators(train_df, config.indicators)

    logger.info(f"Training data: {len(train_df)} rows, {train_df['tic'].nunique()} stocks")

    # For single-stock training, we train on one stock at a time
    # Select a representative stock for training (e.g., most liquid stock)
    stock_counts = train_df.groupby("tic").size()
    most_complete_stock = stock_counts.idxmax()

    logger.info(f"Training on single stock: {most_complete_stock} (most complete data)")
    single_stock_df = train_df[train_df["tic"] == most_complete_stock].copy()

    logger.info(f"Single-stock training data: {len(single_stock_df)} rows")

    logger.info(f"Starting {config.training.algorithm.upper()} training (single-stock mode)...")
    trainer = Trainer(config)

    model_path = trainer.train(
        train_df=single_stock_df,
        output_dir=Path(args.output_dir),
        indicator_set_id="zz500_single",
    )

    logger.info(f"Training complete! Model saved to: {model_path}")

    import json
    meta = {
        "model_type": "single_stock",
        "stock_dim": 1,
        "total_stocks": len(all_stocks),
        "training_stock": most_complete_stock,
        "training_rows": len(single_stock_df),
        "config_path": args.config,
        "model_path": str(model_path),
    }
    meta_path = Path(args.output_dir) / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
