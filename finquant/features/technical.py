from __future__ import annotations

import pandas as pd


DEFAULT_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


def _rsi(series: pd.Series, period: int = 30) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean().replace(0, 1e-9)
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def compute_indicators(frame: pd.DataFrame, indicators: list[str] | None = None) -> pd.DataFrame:
    data = frame.copy()
    selected = indicators or DEFAULT_INDICATORS

    out_frames = []
    for _, group in data.groupby("tic", sort=False):
        g = group.sort_values("date").copy()
        close = g["close"].astype(float)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        g["macd"] = ema12 - ema26

        ma20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
        g["boll_ub"] = ma20 + (2 * std20)
        g["boll_lb"] = ma20 - (2 * std20)

        g["rsi_30"] = _rsi(close, 30).clip(0, 100)
        g["dx_30"] = close.pct_change().abs().rolling(30, min_periods=1).mean().mul(100).clip(0, 100)
        g["close_30_sma"] = close.rolling(30, min_periods=1).mean()
        g["close_60_sma"] = close.rolling(60, min_periods=1).mean()

        out_frames.append(g)

    combined = pd.concat(out_frames, ignore_index=True)
    for col in selected:
        if col not in combined.columns:
            raise ValueError(f"indicator not supported: {col}")

    selected_cols = [c for c in selected if c in combined.columns]
    combined[selected_cols] = combined[selected_cols].fillna(0.0)
    return combined
