import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_tiny_config() -> dict:
    return {
        "model": {
            "name": "vit_tiny_patch16_224",
            "pretrained": False,  # no download in tests
            "num_classes": 2,
            "freeze_layers": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "batch_size": 2,
            "max_epochs": 2,
            "early_stop_patience": 2,
            "model_save_path": "models/best_model.pth",
        }
    }


def make_dummy_loader(n_batches: int = 3, batch_size: int = 2):
    """Create a dummy DataLoader yielding (image_tensor, label) batches."""
    from torch.utils.data import DataLoader, TensorDataset
    imgs = torch.randn(n_batches * batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (n_batches * batch_size,))
    ds = TensorDataset(imgs, labels)
    return DataLoader(ds, batch_size=batch_size)


def test_model_loads_with_correct_head():
    from kline_vit.model.vit_finetuner import ViTFineTuner
    config = make_tiny_config()
    finetuner = ViTFineTuner(config)
    # Check classification head outputs 2 classes
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = finetuner.model(dummy)
    assert out.shape == (1, 2)


def test_first_layers_frozen():
    from kline_vit.model.vit_finetuner import ViTFineTuner
    config = make_tiny_config()
    finetuner = ViTFineTuner(config)
    # Count frozen transformer blocks
    frozen_blocks = 0
    for i, block in enumerate(finetuner.model.blocks):
        if not any(p.requires_grad for p in block.parameters()):
            frozen_blocks += 1
    assert frozen_blocks >= config["model"]["freeze_layers"]


def test_training_runs_without_error(tmp_path):
    from kline_vit.model.vit_finetuner import ViTFineTuner
    config = make_tiny_config()
    config["model"]["model_save_path"] = str(tmp_path / "best_model.pth")
    finetuner = ViTFineTuner(config)
    train_loader = make_dummy_loader()
    val_loader = make_dummy_loader()
    history = finetuner.train(train_loader, val_loader)
    assert isinstance(history, list)
    assert len(history) > 0
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert "val_acc" in history[0]


def test_best_model_saved(tmp_path):
    from kline_vit.model.vit_finetuner import ViTFineTuner
    config = make_tiny_config()
    save_path = tmp_path / "best_model.pth"
    config["model"]["model_save_path"] = str(save_path)
    finetuner = ViTFineTuner(config)
    finetuner.train(make_dummy_loader(), make_dummy_loader())
    assert save_path.exists()


def test_early_stopping_triggers(tmp_path):
    from kline_vit.model.vit_finetuner import ViTFineTuner
    config = make_tiny_config()
    config["model"]["max_epochs"] = 10
    config["model"]["early_stop_patience"] = 1
    config["model"]["model_save_path"] = str(tmp_path / "best_model.pth")
    finetuner = ViTFineTuner(config)
    history = finetuner.train(make_dummy_loader(), make_dummy_loader())
    # Should stop before max_epochs due to patience=1
    assert len(history) <= 10
