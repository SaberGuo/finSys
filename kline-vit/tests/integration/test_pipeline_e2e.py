import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "hk_stocks.db"


def make_synthetic_db(tmp_path: Path) -> str:
    """Create a tiny SQLite DB with synthetic data for 3 stocks."""
    import sqlite3
    db_path = tmp_path / "test_hk.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE daily_data (
            code TEXT, date TEXT, open REAL, high REAL,
            low REAL, close REAL, volume INTEGER, amount REAL
        )
    """)
    codes = ["hk.00001", "hk.00002", "hk.00003"]
    dates = pd.date_range("2020-01-02", periods=200, freq="B").strftime("%Y-%m-%d").tolist()
    rows = []
    for code in codes:
        price = 10.0
        for d in dates:
            price = max(1.0, price + np.random.randn() * 0.1)
            rows.append((code, d, price, price * 1.01, price * 0.99, price, 100000, price * 100000))
    conn.executemany("INSERT INTO daily_data VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(db_path)


def test_build_dataset_index_produces_csv(tmp_path):
    from kline_vit.data.dataset import build_dataset_index
    db_path = make_synthetic_db(tmp_path)
    config = {
        "data": {
            "db_path": db_path,
            "image_dir": str(tmp_path / "images"),
            "daily_window": 60,
            "weekly_window": 13,
            "label_horizon": 5,
            "label_threshold": 0.02,
            "train_end": "2020-06-30",
            "val_end": "2020-09-30",
        }
    }
    df = build_dataset_index(db_path, str(tmp_path / "images"), config, "train",
                              filter_codes=["hk.00001"])
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"image_path", "label", "code", "date", "future_return"}


def test_build_dataset_no_future_leakage(tmp_path):
    from kline_vit.data.dataset import build_dataset_index
    db_path = make_synthetic_db(tmp_path)
    config = {
        "data": {
            "db_path": db_path,
            "image_dir": str(tmp_path / "images"),
            "daily_window": 60,
            "weekly_window": 13,
            "label_horizon": 5,
            "label_threshold": 0.02,
            "train_end": "2020-06-30",
            "val_end": "2020-09-30",
        }
    }
    train_df = build_dataset_index(db_path, str(tmp_path / "images"), config, "train",
                                    filter_codes=["hk.00001"])
    val_df = build_dataset_index(db_path, str(tmp_path / "images"), config, "val",
                                  filter_codes=["hk.00001"])
    if len(train_df) > 0 and len(val_df) > 0:
        train_dates = set(train_df["date"])
        val_dates = set(val_df["date"])
        assert train_dates.isdisjoint(val_dates), "Train and val must not share dates"


def test_images_created_on_disk(tmp_path):
    from kline_vit.data.dataset import build_dataset_index
    db_path = make_synthetic_db(tmp_path)
    image_dir = str(tmp_path / "images")
    config = {
        "data": {
            "db_path": db_path,
            "image_dir": image_dir,
            "daily_window": 60,
            "weekly_window": 13,
            "label_horizon": 5,
            "label_threshold": 0.02,
            "train_end": "2020-06-30",
            "val_end": "2020-09-30",
        }
    }
    df = build_dataset_index(db_path, image_dir, config, "train",
                              filter_codes=["hk.00001"])
    for _, row in df.iterrows():
        assert Path(row["image_path"]).exists(), f"Image not found: {row['image_path']}"
