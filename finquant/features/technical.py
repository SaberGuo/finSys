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


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX).

    ADX measures trend strength regardless of direction.
    Values > 25 indicate strong trend, < 20 indicate weak trend.
    """
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smoothed indicators
    atr = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / atr.replace(0, 1e-9))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / atr.replace(0, 1e-9))

    # ADX calculation
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    adx = dx.rolling(period, min_periods=1).mean()

    return adx.fillna(0.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR measures market volatility.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr.fillna(0.0)


def compute_indicators(frame: pd.DataFrame, indicators: list[str] | None = None) -> pd.DataFrame:
    data = frame.copy()
    selected = set(indicators or DEFAULT_INDICATORS)

    out_frames = []
    for _, group in data.groupby("tic", sort=False):
        g = group.sort_values("date").copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float) if "high" in g.columns else close
        low = g["low"].astype(float) if "low" in g.columns else close

        if "macd" in selected:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            g["macd"] = ema12 - ema26

        if "boll_ub" in selected or "boll_lb" in selected:
            ma20 = close.rolling(20, min_periods=1).mean()
            std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
            if "boll_ub" in selected:
                g["boll_ub"] = ma20 + (2 * std20)
            if "boll_lb" in selected:
                g["boll_lb"] = ma20 - (2 * std20)

        if "rsi_30" in selected:
            g["rsi_30"] = _rsi(close, 30).clip(0, 100)
        if "dx_30" in selected:
            g["dx_30"] = close.pct_change().abs().rolling(30, min_periods=1).mean().mul(100).clip(0, 100)
        if "close_30_sma" in selected:
            g["close_30_sma"] = close.rolling(30, min_periods=1).mean()
        if "close_60_sma" in selected:
            g["close_60_sma"] = close.rolling(60, min_periods=1).mean()

        if "adx_14" in selected:
            g["adx_14"] = _adx(high, low, close, 14)
        if "atr_14" in selected:
            g["atr_14"] = _atr(high, low, close, 14)

        if "volume_ratio" in selected:
            volume = g["volume"].astype(float)
            vol_sma = volume.rolling(20, min_periods=1).mean().replace(0, 1e-9)
            g["volume_ratio"] = volume / vol_sma

        out_frames.append(g)

    combined = pd.concat(out_frames, ignore_index=True)
    for col in selected:
        if col not in combined.columns:
            raise ValueError(f"indicator not supported: {col}")

    combined[list(selected)] = combined[list(selected)].fillna(0.0)
    return combined


def compute_indicators_for_set(
    frame: pd.DataFrame,
    indicator_set: "IndicatorSet",  # type: ignore[name-defined]
) -> pd.DataFrame:
    """Compute indicators for a specific *indicator_set*, respecting window params."""
    return compute_indicators(frame, indicators=indicator_set.indicators)
