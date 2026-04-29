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


def compute_obs_dim(stock_dim: int, indicators: list[str] | None = None) -> int:
    """Return total observation-space dimension for *stock_dim* stocks.

    Formula: ``1 + (2 + len(indicators)) * stock_dim``
    where 1 = cash balance, 2 = close + volume per stock.
    """
    if stock_dim <= 0:
        raise ValueError(f"stock_dim must be > 0, got {stock_dim}")
    ind = indicators if indicators is not None else INDICATORS
    return 1 + (2 + len(ind)) * stock_dim


def build_env(
    df: pd.DataFrame,
    stock_dim: int,
    initial_amount: float = 1_000_000.0,
    hmax: int = 100,
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
    reward_scaling: float = 1e-4,
    indicators: list[str] | None = None,
) -> Any:
    """Create and return a FinRL ``StockTradingEnv`` instance.

    Parameters
    ----------
    df:
        MarketDataset DataFrame. Must contain all required indicator columns.
    stock_dim:
        Number of unique tickers (N). Validated against ``len(df['tic'].unique())``.
    initial_amount:
        Starting cash (RMB).
    hmax:
        Max shares per trade action.
    buy_cost_pct:
        Transaction cost fraction for buys.
    sell_cost_pct:
        Transaction cost fraction for sells.
    reward_scaling:
        Scales portfolio value change to RL reward.
    indicators:
        List of technical indicator column names. Defaults to :data:`INDICATORS`.

    Returns
    -------
    StockTradingEnv
        Configured FinRL environment.

    Raises
    ------
    ValueError
        If *stock_dim* doesn't match the number of unique tickers in *df*.
    """
    # FinRL imports optional Alpaca modules at package-import time.
    # Provide a tiny stub so local/test environments don't need alpaca deps.
    if "alpaca_trade_api" not in sys.modules:
        alpaca_stub = types.ModuleType("alpaca_trade_api")
        alpaca_stub.REST = object  # type: ignore[attr-defined]
        sys.modules["alpaca_trade_api"] = alpaca_stub

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

    actual_n = df["tic"].nunique()
    if actual_n != stock_dim:
        raise ValueError(
            f"stock_dim={stock_dim} but df contains {actual_n} unique tickers"
        )

    expected_dim = compute_obs_dim(stock_dim, ind)

    # FinRL expects each trading day to share a common integer index across all
    # tickers; otherwise ``df.loc[day, :]`` can return a scalar row instead of
    # a per-day DataFrame slice for multi-stock environments.
    env_df = df.sort_values(["date", "tic"]).reset_index(drop=True).copy()
    env_df.index = env_df["date"].factorize()[0]

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
        tech_indicator_list=ind,
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
