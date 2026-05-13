#!/usr/bin/env python
"""Convenience script to build K-line image dataset using default finSys paths."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # finSys root
DB_PATH = ROOT / "data" / "processed" / "hk_stocks.db"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "images"

if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "kline_vit",
        "build-dataset",
        "--db-path", str(DB_PATH),
        "--output-dir", str(OUTPUT_DIR),
        "--config", "config/default.yaml",
    ] + sys.argv[1:]
    subprocess.run(cmd, cwd=Path(__file__).parent.parent)
