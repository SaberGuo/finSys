import time
from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from kline_vit.model.types import InferenceResult

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


class InferenceEngine:
    """Loads a trained ViT model and runs inference on K-line images."""

    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str = "auto",
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run 'train' first to generate a model."
            )

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        model_cfg = config.get("model", {})
        self.model = timm.create_model(
            model_cfg.get("name", "vit_tiny_patch16_224"),
            pretrained=False,
            num_classes=model_cfg.get("num_classes", 2),
        )
        self.model.load_state_dict(
            torch.load(str(model_path), map_location=self._device)
        )
        self.model.to(self._device)
        self.model.eval()

        bt_cfg = config.get("backtest", {})
        self._threshold = float(bt_cfg.get("signal_threshold", 0.6))

    def predict_single(self, image_path: str) -> InferenceResult:
        """Run inference on a single image file."""
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        t0 = time.perf_counter()
        try:
            tensor = self._load_image(str(img_path))
            result = self._infer_batch(tensor.unsqueeze(0), [image_path])[0]
        except RuntimeError:
            # CUDA OOM fallback
            self.model.to("cpu")
            self._device = torch.device("cpu")
            tensor = self._load_image(str(img_path))
            result = self._infer_batch(tensor.unsqueeze(0), [image_path])[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return InferenceResult(
            image_path=image_path,
            buy_probability=result.buy_probability,
            label=result.label,
            inference_time_ms=elapsed_ms,
        )

    def predict_batch(self, image_paths: list[str]) -> list[InferenceResult]:
        """Run batch inference on a list of image files."""
        if not image_paths:
            return []

        results = []
        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            valid_paths = []
            for p in batch_paths:
                try:
                    tensors.append(self._load_image(p))
                    valid_paths.append(p)
                except FileNotFoundError:
                    raise

            batch_tensor = torch.stack(tensors)
            batch_results = self._infer_batch(batch_tensor, valid_paths)
            results.extend(batch_results)

        return results

    def _load_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return _TRANSFORM(img)

    def _infer_batch(
        self, batch: torch.Tensor, paths: list[str]
    ) -> list[InferenceResult]:
        batch = batch.to(self._device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            buy_probs = probs[:, 1].cpu().numpy()

        results = []
        for path, prob in zip(paths, buy_probs):
            label = 1 if float(prob) >= self._threshold else 0
            results.append(InferenceResult(
                image_path=path,
                buy_probability=float(prob),
                label=label,
                inference_time_ms=0.0,
            ))
        return results
