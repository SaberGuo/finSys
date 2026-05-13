"""FinRL environment wrapper with observation-space dimension validation (T030)."""
from __future__ import annotations

import sys
import types
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Default 7 technical indicators matching MarketDataset schema
INDICATORS: list[str] = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# Per-stock observation dims: close + volume + len(INDICATORS)
OBS_DIM_PER_STOCK: int = 2 + len(INDICATORS)  # = 9

# FinRL imports optional data-source modules at package-import time.
# Provide tiny stubs so local/test environments don't need those deps.
_OPTIONAL_STUBS: dict[str, Any] = {
    "alpaca_trade_api": {"REST": object},
    "wrds": {},
    "yfinance": {},
    "ccxt": {},
    "jqdatasdk": {},
    "quantconnect": {},
}
for _mod_name, _attrs in _OPTIONAL_STUBS.items():
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_mod_name] = _stub


def compute_obs_dim(
    stock_dim: int,
    indicators: list[str] | None = None,
    fusion_indicators: list[str] | None = None,
) -> int:
    """Return total observation-space dimension for *stock_dim* stocks.

    Formula: ``1 + (2 + len(indicators) + len(fusion_indicators)) * stock_dim``
    where 1 = cash balance, 2 = close + volume per stock.
    """
    if stock_dim <= 0:
        raise ValueError(f"stock_dim must be > 0, got {stock_dim}")
    ind = indicators if indicators is not None else INDICATORS
    fusion = fusion_indicators or []
    return 1 + (2 + len(ind) + len(fusion)) * stock_dim


def _prepare_env_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort and index *df* for FinRL ``StockTradingEnv``.

    Daily data uses ``date`` factorize; 5min data uses ``(date, time)``
    combined factorize so each bar gets a unique integer index.
    """
    env_df = df.sort_values(["date", "tic"]).reset_index(drop=True).copy()
    if "time" in env_df.columns:
        env_df.index = pd.factorize(
            env_df["date"].astype(str) + "_" + env_df["time"].astype(str)
        )[0]
    else:
        env_df.index = env_df["date"].factorize()[0]
    return env_df


def build_env(
    df: pd.DataFrame,
    stock_dim: int,
    mode: str = "trading",
    initial_amount: float = 1_000_000.0,
    hmax: int = 100,
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
    reward_scaling: float = 1e-4,
    indicators: list[str] | None = None,
    fusion_indicators: list[str] | None = None,
    reward_type: str = "daily_return",
    future_horizon: int = 1,
    normalize_obs: bool = True,
) -> Any:
    """Create and return a FinRL ``StockTradingEnv`` or ``StockScoringEnv`` instance.

    Parameters
    ----------
    mode:
        Environment mode: "trading" (default) or "scoring".
        - "trading": Multi-stock portfolio trading (FinRL StockTradingEnv)
        - "scoring": Single-stock scoring for selection (StockScoringEnv)
    df:
        MarketDataset DataFrame. Must contain all required indicator columns.
    stock_dim:
        Number of unique tickers (N). Validated against ``len(df['tic'].unique())``.
        For scoring mode, must be 1.
    initial_amount:
        Starting cash (RMB). Only used in trading mode.
    hmax:
        Max shares per trade action. Only used in trading mode.
    buy_cost_pct:
        Transaction cost fraction for buys. Only used in trading mode.
    sell_cost_pct:
        Transaction cost fraction for sells. Only used in trading mode.
    reward_scaling:
        Scales portfolio value change to RL reward. Only used in trading mode.
    indicators:
        List of technical indicator column names. Defaults to :data:`INDICATORS`.
    fusion_indicators:
        Optional list of additional feature columns (e.g. sentiment,
        fundamentals) to append to the observation space. Only used in trading mode.
    reward_type:
        Reward calculation for scoring mode: "daily_return" or "future_return".
    future_horizon:
        Number of days ahead for future_return calculation in scoring mode.
    normalize_obs:
        Whether to normalize observations in scoring mode.

    Returns
    -------
    StockTradingEnv or StockScoringEnv
        Configured environment based on mode.

    Raises
    ------
    ValueError
        If *stock_dim* doesn't match the number of unique tickers in *df*,
        or if scoring mode is used with stock_dim != 1.
    """
    if mode not in ["trading", "scoring"]:
        raise ValueError(f"mode must be 'trading' or 'scoring', got {mode!r}")

    if mode == "scoring":
        if stock_dim != 1:
            raise ValueError(
                f"Scoring mode requires stock_dim=1, got {stock_dim}. "
                "Train on single stocks and score them independently."
            )
        from finquant.training.scoring_env import build_scoring_env

        ind = indicators if indicators is not None else INDICATORS
        return build_scoring_env(
            df=df,
            indicators=ind,
            reward_type=reward_type,
            future_horizon=future_horizon,
            normalize_obs=normalize_obs,
        )

    # Trading mode (existing logic)
    try:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # type: ignore[import]
    except ModuleNotFoundError:
        stocktrading_path: Path | None = None
        for root in sys.path:
            candidate = Path(root) / "finrl" / "meta" / "env_stock_trading" / "env_stocktrading.py"
            if candidate.exists():
                stocktrading_path = candidate
                break
        if stocktrading_path is None:
            raise

        spec = importlib_util.spec_from_file_location(
            "_finrl_env_stocktrading_local", stocktrading_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load FinRL env module from {stocktrading_path}")
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        StockTradingEnv = module.StockTradingEnv

    ind = indicators if indicators is not None else INDICATORS
    fusion = fusion_indicators or []
    tech_list = ind + fusion

    actual_n = df["tic"].nunique()
    if actual_n != stock_dim:
        raise ValueError(
            f"stock_dim={stock_dim} but df contains {actual_n} unique tickers"
        )

    expected_dim = compute_obs_dim(stock_dim, ind, fusion)

    env_df = _prepare_env_df(df)

    env = StockTradingEnv(
        df=env_df,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=initial_amount,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=expected_dim,
        action_space=stock_dim,
        tech_indicator_list=tech_list,
    )

    # Validate observation_space dimension matches expectation
    actual_dim = env.observation_space.shape[0]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Observation space dimension mismatch: expected {expected_dim}, "
            f"got {actual_dim}. Check that df columns match indicators list."
        )

    # Normalize to Gymnasium-style outputs expected by the test contracts.
    original_reset = env.reset
    original_step = env.step

    def _reset(*args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        result = original_reset(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return np.asarray(obs, dtype=float), info

    def _step(*args: Any, **kwargs: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = original_step(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise ValueError("Unexpected env.step return format")
        return np.asarray(obs, dtype=float), float(reward), bool(terminated), bool(truncated), info

    env.reset = _reset  # type: ignore[method-assign]
    env.step = _step  # type: ignore[method-assign]

    return env
