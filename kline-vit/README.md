# kline-vit: K-line Chart ViT Recognition & Backtesting

An independent project that fine-tunes a Vision Transformer (ViT) on dual-panel (weekly + daily) K-line chart images from Hong Kong stocks, then backtests the resulting signals with Backtrader.

See [quickstart.md](../specs/005-kline-vit-backtest/quickstart.md) for the full workflow.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Build image dataset
python -m kline_vit build-dataset --db-path ../data/processed/hk_stocks.db --output-dir data/images --config config/default.yaml

# 2. Train ViT model
python -m kline_vit train --train-csv data/dataset_train.csv --val-csv data/dataset_val.csv --test-csv data/dataset_test.csv --config config/default.yaml

# 3. Run backtest
python -m kline_vit backtest --model-path models/best_model.pth --db-path ../data/processed/hk_stocks.db --config config/default.yaml

# 4. Infer single image
python -m kline_vit infer --model-path models/best_model.pth --image-path data/images/test/hk.00001/2025-06-01.png --config config/default.yaml
```

## Project Structure

```
kline-vit/
├── config/default.yaml       # All configurable parameters
├── kline_vit/
│   ├── cli.py                # CLI entry point
│   ├── data/
│   │   ├── db_reader.py      # SQLite reader
│   │   ├── renderer.py       # mplfinance dual-panel renderer
│   │   ├── label_generator.py
│   │   └── dataset.py        # PyTorch KlineDataset
│   ├── model/
│   │   ├── vit_finetuner.py  # timm ViT fine-tuning
│   │   └── inference.py      # InferenceEngine
│   └── backtest/
│       ├── strategy.py       # Backtrader strategy
│       └── runner.py         # BacktestRunner
└── tests/
    ├── unit/
    ├── integration/
    └── contract/
```

## Running Tests

```bash
cd kline-vit
pytest --cov=kline_vit --cov-report=term-missing
```
