"""FinRL agent trainer: PPO / SAC / TD3 via stable-baselines3 (T031)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from finquant.config.settings import AppConfig
from finquant.training.backtest import BacktestReport
from finquant.training.env import build_env
from finquant.utils.logging import get_logger

logger = get_logger(__name__)

_ALGORITHM_MAP: dict[str, str] = {
    "ppo": "PPO",
    "sac": "SAC",
    "td3": "TD3",
}


def _universe_hash(stocks: list[str]) -> str:
    key = ",".join(sorted(stocks))
    return hashlib.md5(key.encode()).hexdigest()[:8]  # noqa: S324


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

    def train(
        self,
        train_df: pd.DataFrame,
        output_dir: Path | str | None = None,
        indicators: list[str] | None = None,
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

        Returns
        -------
        Path
            Path to saved model file (SB3 ``.zip``).
        """
        from finrl.agents.stablebaselines3.models import DRLAgent  # type: ignore[import]

        ind = indicators if indicators is not None else self.config.indicators
        stock_dim = train_df["tic"].nunique()

        env = build_env(
            train_df,
            stock_dim=stock_dim,
            initial_amount=self.config.environment.initial_amount,
            hmax=self.config.environment.hmax,
            buy_cost_pct=self.config.environment.buy_cost_pct,
            sell_cost_pct=self.config.environment.sell_cost_pct,
            reward_scaling=self.config.environment.reward_scaling,
            indicators=ind,
        )

        algo_upper = _ALGORITHM_MAP[self._algo]
        hyperparams: dict[str, Any] = getattr(self.config.training, self._algo, {}) or {}

        agent = DRLAgent(env=env)
        model = agent.get_model(algo_upper, model_kwargs=hyperparams)

        logger.info(
            "training_start",
            algorithm=self._algo,
            total_timesteps=self.config.training.total_timesteps,
            stock_dim=stock_dim,
        )

        trained_model = agent.train_model(
            model=model,
            tb_log_name=self._algo,
            total_timesteps=self.config.training.total_timesteps,
        )

        out_dir = Path(output_dir) if output_dir else Path(self.config.training.model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_name = (
            f"{self._algo}_{self.config.dates.train_end.replace('-', '')}"
            f"_{_universe_hash(self.config.stocks)}"
        )
        model_path = out_dir / model_name
        trained_model.save(str(model_path))

        logger.info("training_done", model_path=str(model_path))
        return Path(str(model_path) + ".zip")

    def backtest(
        self,
        model_path: Path | str,
        test_df: pd.DataFrame,
        output_dir: Path | str | None = None,
        indicators: list[str] | None = None,
        risk_free_rate: float = 0.02,
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

        Returns
        -------
        BacktestReport
        """
        from finrl.agents.stablebaselines3.models import DRLAgent  # type: ignore[import]

        ind = indicators if indicators is not None else self.config.indicators
        stock_dim = test_df["tic"].nunique()

        env = build_env(
            test_df,
            stock_dim=stock_dim,
            initial_amount=self.config.environment.initial_amount,
            hmax=self.config.environment.hmax,
            buy_cost_pct=self.config.environment.buy_cost_pct,
            sell_cost_pct=self.config.environment.sell_cost_pct,
            reward_scaling=self.config.environment.reward_scaling,
            indicators=ind,
        )

        # Load the correct SB3 algorithm class
        algo_upper = _ALGORITHM_MAP[self._algo]
        agent = DRLAgent(env=env)
        model = agent.get_model(algo_upper)

        # Remove .zip suffix for SB3 load() if present
        mp = str(model_path)
        if mp.endswith(".zip"):
            mp = mp[:-4]

        from stable_baselines3 import PPO, SAC, TD3  # type: ignore[import]

        _sb3_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
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

        report = BacktestReport.from_portfolio_values(
            portfolio_series, risk_free_rate=risk_free_rate
        )

        # Persist report artifacts
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{self._algo}_{self.config.dates.test_end.replace('-', '')}"
            report.save_html(out_dir / f"{tag}_report.html")
            report.save_csv(out_dir / f"{tag}_metrics.csv")

        return report
