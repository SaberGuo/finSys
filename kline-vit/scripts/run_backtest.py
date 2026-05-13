#!/usr/bin/env python
"""Convenience script to run Backtrader backtest using default finSys paths."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # finSys root
DB_PATH = ROOT / "data" / "processed" / "hk_stocks.db"
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pth"

if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "kline_vit",
        "backtest",
        "--model-path", str(MODEL_PATH),
        "--db-path", str(DB_PATH),
        "--config", "config/default.yaml",
    ] + sys.argv[1:]
    subprocess.run(cmd, cwd=Path(__file__).parent.parent)
