import time
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
    model_path = tmp_path / "perf_model.pth"
    torch.save(model.state_dict(), model_path)
    return str(model_path)


def make_dummy_images(tmp_path: Path, n: int) -> list[str]:
    paths = []
    for i in range(n):
        img_path = tmp_path / f"img_{i:04d}.png"
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img.save(img_path)
        paths.append(str(img_path))
    return paths


def make_config() -> dict:
    return {
        "model": {"name": "vit_tiny_patch16_224", "pretrained": False, "num_classes": 2},
        "backtest": {"signal_threshold": 0.6},
    }


def test_single_inference_under_2s(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    img_path = make_dummy_images(tmp_path, 1)[0]
    engine = InferenceEngine(model_path, make_config(), device="cpu")

    t0 = time.perf_counter()
    result = engine.predict_single(img_path)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 2000, f"Single inference took {elapsed_ms:.0f}ms, expected < 2000ms"
    assert result.inference_time_ms < 2000


def test_batch_100_images_under_30s(tmp_path):
    from kline_vit.model.inference import InferenceEngine
    model_path = make_dummy_model(tmp_path)
    images = make_dummy_images(tmp_path, 100)
    engine = InferenceEngine(model_path, make_config(), device="cpu")

    t0 = time.perf_counter()
    results = engine.predict_batch(images)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(results) == 100
    assert elapsed_ms < 30000, f"Batch 100 took {elapsed_ms:.0f}ms, expected < 30000ms"
