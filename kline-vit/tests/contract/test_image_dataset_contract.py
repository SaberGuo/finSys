import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_dummy_csv(tmp_path: Path, n: int = 10, split: str = "train", date_prefix: str = "2023") -> Path:
    img_dir = tmp_path / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        img_path = img_dir / f"hk.0000{i % 3}" / f"{date_prefix}-0{(i % 9) + 1}-01.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img.save(img_path)
        records.append({
            "image_path": str(img_path),
            "label": i % 2,
            "code": f"hk.0000{i % 3}",
            "date": f"{date_prefix}-0{(i % 9) + 1}-01",
            "future_return": 0.03 if i % 2 == 1 else -0.01,
        })
    csv_path = tmp_path / f"dataset_{split}.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


def test_dataset_item_shape_contract(tmp_path):
    from kline_vit.data.dataset import KlineDataset
    csv_path = make_dummy_csv(tmp_path, n=5)
    ds = KlineDataset(str(csv_path))
    img, label = ds[0]
    assert img.shape == (3, 224, 224), f"Expected (3,224,224), got {img.shape}"
    assert img.dtype == torch.float32
    assert label in (0, 1)


def test_no_data_leakage_contract(tmp_path):
    from kline_vit.data.dataset import KlineDataset
    train_csv = make_dummy_csv(tmp_path, n=5, split="train", date_prefix="2023")
    val_csv = make_dummy_csv(tmp_path, n=5, split="val", date_prefix="2024")
    train_ds = KlineDataset(str(train_csv))
    val_ds = KlineDataset(str(val_csv))
    train_dates = set(train_ds.index_df["date"])
    val_dates = set(val_ds.index_df["date"])
    assert train_dates.isdisjoint(val_dates), "Train and val dates must not overlap"


def test_image_normalized(tmp_path):
    """Image tensor should be normalized (not raw 0-255)."""
    from kline_vit.data.dataset import KlineDataset
    csv_path = make_dummy_csv(tmp_path, n=3)
    ds = KlineDataset(str(csv_path))
    img, _ = ds[0]
    # After ImageNet normalization, values should be in roughly [-3, 3]
    assert img.min() >= -5.0
    assert img.max() <= 5.0
