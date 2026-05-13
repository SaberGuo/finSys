import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_dummy_model(tmp_path: Path) -> str:
    import timm
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    model_path = tmp_path / "err_model.pth"
    torch.save(model.state_dict(), model_path)
    return str(model_path)


def make_config() -> dict:
    return {
        "model": {"name": "vit_tiny_patch16_224", "pretrained": False, "num_classes": 2},
        "backtest": {"signal_threshold": 0.6},
    }


def test_model_not_found_raises_with_hint():
    from kline_vit.model.inference import InferenceEngine
    with pytest.raises(FileNotFoundError) as exc_info:
        InferenceEngine("/nonexistent/model.pth", make_config())
    assert "train" in str(exc_info.value).lower() or "model" in str(exc_info.value).lower()


def test_image_not_found_raises(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    engine = InferenceEngine(model_path, make_config(), device="cpu")
    with pytest.raises(FileNotFoundError):
        engine.predict_single("/nonexistent/image.png")


def test_empty_batch_returns_empty_list(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    engine = InferenceEngine(model_path, make_config(), device="cpu")
    results = engine.predict_batch([])
    assert results == []
