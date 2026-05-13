import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_dummy_model(tmp_path: Path) -> str:
    """Save a tiny ViT model to disk and return path."""
    import timm
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    model_path = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), model_path)
    return str(model_path)


def make_dummy_image(tmp_path: Path) -> str:
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    img_path = tmp_path / "test_image.png"
    img.save(img_path)
    return str(img_path)


def make_config() -> dict:
    return {
        "model": {
            "name": "vit_tiny_patch16_224",
            "pretrained": False,
            "num_classes": 2,
        },
        "backtest": {
            "signal_threshold": 0.6,
        }
    }


def test_inference_result_probability_range(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    img_path = make_dummy_image(tmp_path)
    engine = InferenceEngine(model_path, make_config(), device="cpu")
    result = engine.predict_single(img_path)
    assert 0.0 <= result.buy_probability <= 1.0


def test_inference_result_label_binary(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    img_path = make_dummy_image(tmp_path)
    engine = InferenceEngine(model_path, make_config(), device="cpu")
    result = engine.predict_single(img_path)
    assert result.label in (0, 1)


def test_inference_model_not_found():
    from kline_vit.model.inference import InferenceEngine
    with pytest.raises(FileNotFoundError):
        InferenceEngine("/nonexistent/model.pth", make_config())


def test_inference_image_not_found(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    engine = InferenceEngine(model_path, make_config(), device="cpu")
    with pytest.raises(FileNotFoundError):
        engine.predict_single("/nonexistent/image.png")


def test_batch_inference_order(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    images = [make_dummy_image(tmp_path / f"img{i}.png") for i in range(3)]
    for p in images:
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img.save(p)
    engine = InferenceEngine(model_path, make_config(), device="cpu")
    results = engine.predict_batch(images)
    assert len(results) == len(images)
    for r, path in zip(results, images):
        assert r.image_path == path


def test_batch_empty_input(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    engine = InferenceEngine(model_path, make_config(), device="cpu")
    results = engine.predict_batch([])
    assert results == []
