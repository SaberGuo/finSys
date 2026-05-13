import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_tiny_dataset(tmp_path: Path, n: int = 50):
    """Create tiny synthetic dataset for training integration test."""
    import pandas as pd
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        img_path = img_dir / f"img_{i:04d}.png"
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(img_path)
        records.append({
            "image_path": str(img_path),
            "label": i % 2,
            "code": "hk.00001",
            "date": f"2023-01-{(i % 28) + 1:02d}",
            "future_return": 0.03 if i % 2 == 1 else -0.01,
        })
    csv_path = tmp_path / "dataset_train.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return str(csv_path)


def test_training_pipeline_e2e(tmp_path):
    from torch.utils.data import DataLoader
    from kline_vit.data.dataset import KlineDataset
    from kline_vit.model.vit_finetuner import ViTFineTuner

    csv_path = make_tiny_dataset(tmp_path, n=20)
    ds = KlineDataset(csv_path)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    config = {
        "model": {
            "name": "vit_tiny_patch16_224",
            "pretrained": False,
            "num_classes": 2,
            "freeze_layers": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "batch_size": 4,
            "max_epochs": 2,
            "early_stop_patience": 2,
            "model_save_path": str(tmp_path / "best_model.pth"),
        }
    }

    finetuner = ViTFineTuner(config)
    history = finetuner.train(loader, loader)

    assert Path(config["model"]["model_save_path"]).exists()
    assert len(history) >= 1
    assert all(k in history[0] for k in ("train_loss", "val_loss", "val_acc"))


def test_model_save_and_load(tmp_path):
    import timm
    from kline_vit.model.inference import InferenceEngine

    model_path = tmp_path / "model.pth"
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    torch.save(model.state_dict(), model_path)

    config = {
        "model": {"name": "vit_tiny_patch16_224", "pretrained": False, "num_classes": 2},
        "backtest": {"signal_threshold": 0.6},
    }
    engine = InferenceEngine(str(model_path), config, device="cpu")
    assert engine.model is not None
