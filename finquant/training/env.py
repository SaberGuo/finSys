"""FinRL environment wrapper with observation-space dimension validation (T030)."""
from __future__ import annotations

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
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # type: ignore[import]

    ind = indicators if indicators is not None else INDICATORS

    actual_n = df["tic"].nunique()
    if actual_n != stock_dim:
        raise ValueError(
            f"stock_dim={stock_dim} but df contains {actual_n} unique tickers"
        )

    expected_dim = compute_obs_dim(stock_dim, ind)

    env = StockTradingEnv(
        df=df,
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

    return env
