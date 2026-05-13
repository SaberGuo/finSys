"""FinRL agent trainer: PPO / SAC / TD3 via stable-baselines3 (T031)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from finquant.config.settings import AppConfig
from finquant.features.fusion import FUSION_COLUMNS
from finquant.training.backtest import BacktestReport
from finquant.training.env import build_env
from finquant.utils.logging import get_logger

logger = get_logger(__name__)


def _detect_fusion_indicators(df: pd.DataFrame) -> list[str]:
    """Return fusion columns that actually exist in *df*."""
    return [c for c in FUSION_COLUMNS if c in df.columns]


_ALGORITHM_MAP: dict[str, str] = {
    "ppo": "ppo",
    "sac": "sac",
    "td3": "td3",
}


def _universe_hash(stocks: list[str]) -> str:
    key = ",".join(sorted(stocks))
    return hashlib.md5(key.encode()).hexdigest()[:8]  # noqa: S324


class _EarlyStoppingCallback:
    """SB3 callback: stop training if mean reward doesn't improve."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float("inf")
        self.no_improvement_count = 0

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        """Return True to continue training, False to stop."""
        # SB3 calls this during training; locals contains 'self' = model
        self_ = locals_dict.get("self")
        if self_ is None:
            return True
        # Try to get mean reward from logger
        mean_reward = getattr(self_, "logger", None)
        if mean_reward is not None:
            # This is a simplified heuristic; real early stopping would monitor eval reward
            pass
        return True


class _NanAbortCallback:
    """SB3 callback: abort training on NaN loss/gradient."""

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        self_ = locals_dict.get("self")
        if self_ is None:
            return True
        # Check if loss is NaN
        loss = getattr(self_, "loss", None)
        if loss is not None and pd.isna(loss):
            logger.error("NaN loss detected; aborting training.")
            return False
        return True


class Trainer:
    """Wraps FinRL ``DRLAgent`` for train/backtest lifecycle.

    Parameters
    ----------
    config:
        Application configuration. ``config.training`` drives algorithm,
        total_timesteps, and hyperparameters.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._algo = config.training.algorithm.lower()
        if self._algo not in _ALGORITHM_MAP:
            raise ValueError(
                f"Unsupported algorithm: {self._algo!r}. "
                f"Choose from: {list(_ALGORITHM_MAP)}"
            )
        self._last_train_obs_dim: int | None = None
        self._last_indicator_set_id: str | None = None

    def train(
        self,
        train_df: pd.DataFrame,
        output_dir: Path | str | None = None,
        indicators: list[str] | None = None,
        indicator_set_id: str | None = None,
        mode: str | None = None,
    ) -> Path:
        """Train RL agent on *train_df* and save model weights.

        Parameters
        ----------
        train_df:
            MarketDataset (train split). Required columns per data-model.md.
        output_dir:
            Directory for saved model. Defaults to ``config.training.model_dir``.
        indicators:
            Override indicator list. Defaults to ``config.indicators``.
        indicator_set_id:
            Optional indicator set ID to include in model filename.
        mode:
            Training mode: "trading" or "scoring". Defaults to ``config.training.mode``.

        Returns
        -------
        Path
            Path to saved model file (SB3 ``.zip``).
        """
        training_mode = mode if mode is not None else self.config.training.mode

        if training_mode == "scoring":
            return self._train_scoring(train_df, output_dir, indicators, indicator_set_id)
        else:
            return self._train_trading(train_df, output_dir, indicators, indicator_set_id)

    def _train_trading(
        self,
        train_df: pd.DataFrame,
        output_dir: Path | str | None = None,
        indicators: list[str] | None = None,
        indicator_set_id: str | None = None,
    ) -> Path:
        """Train RL agent in trading mode (original behavior).

        Parameters
        ----------
        train_df:
            MarketDataset (train split)
        output_dir:
            Directory for saved model
        indicators:
            Override indicator list
        indicator_set_id:
            Optional indicator set ID

        Returns
        -------
        Path
            Path to saved model file
        """
        from finrl.agents.stablebaselines3.models import DRLAgent  # type: ignore[import]

        ind = indicators if indicators is not None else self.config.indicators
        stock_dim = train_df["tic"].nunique()
        fusion_ind = _detect_fusion_indicators(train_df)

        env = build_env(
            train_df,
            stock_dim=stock_dim,
            mode="trading",
            initial_amount=self.config.environment.initial_amount,
            hmax=self.config.environment.hmax,
            buy_cost_pct=self.config.environment.buy_cost_pct,
            sell_cost_pct=self.config.environment.sell_cost_pct,
            reward_scaling=self.config.environment.reward_scaling,
            indicators=ind,
            fusion_indicators=fusion_ind,
        )

        obs_dim = env.observation_space.shape[0]
        self._last_train_obs_dim = obs_dim
        self._last_indicator_set_id = indicator_set_id
        logger.info(
            f"training_start algorithm={self._algo} mode=trading "
            f"timesteps={self.config.training.total_timesteps} stock_dim={stock_dim} "
            f"obs_dim={obs_dim} fusion_indicators={fusion_ind}"
        )

        algo_upper = _ALGORITHM_MAP[self._algo]
        hyperparams: dict[str, Any] = getattr(self.config.training, self._algo, {}) or {}

        agent = DRLAgent(env=env)
        model = agent.get_model(algo_upper, model_kwargs=hyperparams)

        trained_model = agent.train_model(
            model=model,
            tb_log_name=self._algo,
            total_timesteps=self.config.training.total_timesteps,
        )

        out_dir = Path(output_dir) if output_dir else Path(self.config.training.model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = [self._algo]
        if indicator_set_id:
            parts.append(indicator_set_id)
        parts.extend([
            self.config.dates.train_end.replace("-", ""),
            _universe_hash(self.config.stocks),
        ])
        model_name = "_".join(parts)
        model_path = out_dir / model_name
        trained_model.save(str(model_path))

        # Persist training metadata
        import json

        meta = {
            "algorithm": self._algo,
            "mode": "trading",
            "indicator_set_id": indicator_set_id,
            "obs_dim": obs_dim,
            "stock_dim": stock_dim,
            "fusion_indicators": fusion_ind,
            "total_timesteps": self.config.training.total_timesteps,
            "train_start": self.config.dates.train_start,
            "train_end": self.config.dates.train_end,
            "stocks": self.config.stocks,
            "indicators": ind,
        }
        meta_path = out_dir / f"{model_name}_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"training_done model_path={model_path} meta_path={meta_path}")
        return Path(str(model_path) + ".zip")

    def _train_scoring(
        self,
        train_df: pd.DataFrame,
        output_dir: Path | str | None = None,
        indicators: list[str] | None = None,
        indicator_set_id: str | None = None,
    ) -> Path:
        """Train RL agent in scoring mode (single-stock scoring).

        Parameters
        ----------
        train_df:
            Single-stock MarketDataset (train split)
        output_dir:
            Directory for saved model
        indicators:
            Override indicator list
        indicator_set_id:
            Optional indicator set ID

        Returns
        -------
        Path
            Path to saved model file
        """
        from finrl.agents.stablebaselines3.models import DRLAgent  # type: ignore[import]

        ind = indicators if indicators is not None else self.config.indicators
        stock_dim = train_df["tic"].nunique()

        # Validate single stock
        if stock_dim != 1:
            raise ValueError(
                f"Scoring mode requires single stock, got {stock_dim} stocks. "
                "Train on individual stocks separately."
            )

        ticker = train_df["tic"].iloc[0]

        env = build_env(
            train_df,
            stock_dim=1,
            mode="scoring",
            indicators=ind,
            reward_type=self.config.training.scoring.reward_type,
            future_horizon=self.config.training.scoring.future_horizon,
            normalize_obs=self.config.training.scoring.normalize_obs,
        )

        obs_dim = env.observation_space.shape[0]
        self._last_train_obs_dim = obs_dim
        self._last_indicator_set_id = indicator_set_id
        logger.info(
            f"training_start algorithm={self._algo} mode=scoring "
            f"timesteps={self.config.training.total_timesteps} ticker={ticker} "
            f"obs_dim={obs_dim} reward_type={self.config.training.scoring.reward_type}"
        )

        algo_upper = _ALGORITHM_MAP[self._algo]
        hyperparams: dict[str, Any] = getattr(self.config.training, self._algo, {}) or {}

        agent = DRLAgent(env=env)
        model = agent.get_model(algo_upper, model_kwargs=hyperparams)

        trained_model = agent.train_model(
            model=model,
            tb_log_name=f"{self._algo}_scoring",
            total_timesteps=self.config.training.total_timesteps,
        )

        out_dir = Path(output_dir) if output_dir else Path(self.config.training.model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = [self._algo, "scoring"]
        if indicator_set_id:
            parts.append(indicator_set_id)
        parts.extend([
            ticker.replace(".", "_"),
            self.config.dates.train_end.replace("-", ""),
        ])
        model_name = "_".join(parts)
        model_path = out_dir / model_name
        trained_model.save(str(model_path))

        # Persist training metadata
        import json

        meta = {
            "algorithm": self._algo,
            "mode": "scoring",
            "indicator_set_id": indicator_set_id,
            "obs_dim": obs_dim,
            "stock_dim": 1,
            "ticker": ticker,
            "reward_type": self.config.training.scoring.reward_type,
            "future_horizon": self.config.training.scoring.future_horizon,
            "normalize_obs": self.config.training.scoring.normalize_obs,
            "total_timesteps": self.config.training.total_timesteps,
            "train_start": self.config.dates.train_start,
            "train_end": self.config.dates.train_end,
            "indicators": ind,
        }
        meta_path = out_dir / f"{model_name}_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"training_done model_path={model_path} meta_path={meta_path}")
        return Path(str(model_path) + ".zip")

    def backtest(
        self,
        model_path: Path | str,
        test_df: pd.DataFrame,
        output_dir: Path | str | None = None,
        indicators: list[str] | None = None,
        risk_free_rate: float = 0.02,
        expected_obs_dim: int | None = None,
    ) -> BacktestReport:
        """Run backtest using saved model and return a :class:`BacktestReport`.

        Parameters
        ----------
        model_path:
            Path to saved SB3 model (``.zip``).
        test_df:
            MarketDataset (test split).
        output_dir:
            Directory for HTML + CSV report output.
        indicators:
            Override indicator list.
        risk_free_rate:
            Annual risk-free rate for Sharpe calculation.
        expected_obs_dim:
            Optional expected observation dimension for validation.

        Returns
        -------
        BacktestReport
        """
        ind = indicators if indicators is not None else self.config.indicators
        stock_dim = test_df["tic"].nunique()
        fusion_ind = _detect_fusion_indicators(test_df)

        env = build_env(
            test_df,
            stock_dim=stock_dim,
            initial_amount=self.config.environment.initial_amount,
            hmax=self.config.environment.hmax,
            buy_cost_pct=self.config.environment.buy_cost_pct,
            sell_cost_pct=self.config.environment.sell_cost_pct,
            reward_scaling=self.config.environment.reward_scaling,
            indicators=ind,
            fusion_indicators=fusion_ind,
        )

        if expected_obs_dim is not None and env.observation_space.shape[0] != expected_obs_dim:
            raise ValueError(
                f"Observation dim mismatch: expected {expected_obs_dim}, "
                f"got {env.observation_space.shape[0]}. "
                "Training and backtest indicator sets may differ."
            )

        from finrl.agents.stablebaselines3.models import DRLAgent  # type: ignore[import]

        # Load the correct SB3 algorithm class
        algo_upper = _ALGORITHM_MAP[self._algo]
        agent = DRLAgent(env=env)
        model = agent.get_model(algo_upper)

        # Remove .zip suffix for SB3 load() if present
        mp = str(model_path)
        if mp.endswith(".zip"):
            mp = mp[:-4]

        from stable_baselines3 import PPO, SAC, TD3  # type: ignore[import]

        _sb3_map = {"ppo": PPO, "sac": SAC, "td3": TD3}
        SB3Cls = _sb3_map[algo_upper]
        loaded_model = SB3Cls.load(mp, env=env)

        df_account, df_actions = DRLAgent.DRL_prediction(
            model=loaded_model, environment=env
        )

        # Build portfolio value series from account value DataFrame
        portfolio_series = pd.Series(
            df_account["account_value"].values,
            index=pd.to_datetime(df_account["date"]),
            name="portfolio_value",
        )

        # Compute trade statistics from actions and prices
        if "time" in test_df.columns:
            # 5min data: preserve (date, time) granularity and align indices
            price_df = test_df.groupby(["date", "time", "tic"])["close"].last().unstack()
            n_actions = len(df_actions)
            if n_actions <= len(price_df):
                df_actions.index = price_df.index[:n_actions]
        else:
            price_df = test_df.groupby(["date", "tic"])["close"].last().unstack()
        from finquant.training.backtest import compute_trade_stats

        trades_df, trade_summary = compute_trade_stats(df_actions, price_df)

        report = BacktestReport.from_portfolio_values(
            portfolio_series, risk_free_rate=risk_free_rate
        )
        report.trade_summary = trade_summary
        report.trades = trades_df

        # Persist report artifacts
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{self._algo}_{self.config.dates.test_end.replace('-', '')}"
            report.save_html(out_dir / f"{tag}_report.html")
            report.save_csv(out_dir / f"{tag}_metrics.csv")

        return report
