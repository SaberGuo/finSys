from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from kline_vit.data.db_reader import DBReader
from kline_vit.data.label_generator import LabelGenerator

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Kline-specific normalization: neutral mean preserves red/green color contrast
_KLINE_MEAN = [0.5, 0.5, 0.5]
_KLINE_STD = [0.5, 0.5, 0.5]

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_KLINE_MEAN, std=_KLINE_STD),
])


class KlineDataset(Dataset):
    """PyTorch Dataset for K-line chart images."""

    def __init__(
        self,
        index_csv: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        csv_path = Path(index_csv)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {index_csv}. Run 'build-dataset' first."
            )
        self.index_df = pd.read_csv(csv_path)
        self.transform = transform or _DEFAULT_TRANSFORM

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.index_df.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        label = int(row["label"])
        return tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset."""
        counts = self.index_df["label"].value_counts().sort_index()
        total = len(self.index_df)
        n_classes = 2
        weights = torch.tensor(
            [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
            dtype=torch.float32,
        )
        return weights


def build_dataset_index(
    db_path: str,
    image_dir: str,
    config: dict,
    split: str,
    filter_codes: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
) -> pd.DataFrame:
    """
    Build dataset index CSV for the given split.

    Split boundaries (from config):
      train: date <= train_end
      val:   train_end < date <= val_end  (with 5-day gap)
      test:  date > val_end               (with 5-day gap)
    """
    data_cfg = config["data"]
    daily_window: int = data_cfg.get("daily_window", 60)
    weekly_window: int = data_cfg.get("weekly_window", 13)
    label_horizon: int = data_cfg.get("label_horizon", 5)
    label_threshold: float = data_cfg.get("label_threshold", 0.02)
    train_end: str = data_cfg.get("train_end", "2023-12-31")
    val_end: str = data_cfg.get("val_end", "2024-12-31")
    gap_days = label_horizon  # avoid leakage

    reader = DBReader(db_path)
    from kline_vit.data.renderer import KlineRenderer
    renderer = KlineRenderer()
    label_gen = LabelGenerator(horizon=label_horizon, threshold=label_threshold)

    codes = filter_codes if filter_codes else reader.get_all_codes()

    records = []
    for idx, code in enumerate(codes, 1):
        try:
            min_date, max_date = reader.get_date_range(code)
        except KeyError:
            if progress_callback:
                progress_callback(code, idx, len(codes), len(records))
            continue

        # Determine anchor date range for this split
        all_dates = reader.get_tradeable_dates(code, min_date, max_date)
        if len(all_dates) < daily_window + label_horizon + 5:
            continue

        for i, anchor_date in enumerate(all_dates):
            # Split filtering
            if split == "train" and anchor_date > train_end:
                continue
            elif split == "val" and (anchor_date <= train_end or anchor_date > val_end):
                continue
            elif split == "test" and anchor_date <= val_end:
                continue

            # Need enough history
            history_idx = i
            if history_idx < daily_window:
                continue

            # Fetch data
            daily_df = reader.get_daily_data(code, anchor_date, n_days=daily_window + weekly_window * 5)
            if len(daily_df) < daily_window:
                continue

            # Resample weekly
            weekly_df = _resample_weekly(daily_df, weekly_window)
            if len(weekly_df) < weekly_window:
                continue

            # Get future data for label (need data beyond anchor_date)
            future_df = reader.get_future_close(code, anchor_date, label_horizon)
            if len(future_df) < label_horizon:
                continue

            # Combine for label generation
            combined_df = pd.concat([daily_df, future_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

            result = label_gen.generate(combined_df, anchor_date)
            if result is None:
                continue
            label, future_return = result

            # Render image
            try:
                img_path = renderer.render(
                    code, anchor_date,
                    daily_df.tail(daily_window),
                    weekly_df.tail(weekly_window),
                    output_dir=image_dir,
                    split=split,
                )
            except Exception:
                continue

            records.append({
                "image_path": img_path,
                "label": label,
                "code": code,
                "date": anchor_date,
                "future_return": future_return,
            })

        if progress_callback:
            progress_callback(code, idx, len(codes), len(records))

    return pd.DataFrame(records)


def _resample_weekly(daily_df: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (W-FRI), return last n_weeks rows."""
    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["date"])
    df = df.set_index("Date")
    weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    weekly = weekly.reset_index()
    weekly["date"] = weekly["Date"].dt.strftime("%Y-%m-%d")
    weekly = weekly.drop(columns=["Date"])
    return weekly.tail(n_weeks).reset_index(drop=True)
