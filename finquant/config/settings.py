from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DatesConfig(BaseModel):
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class DataConfig(BaseModel):
    source_priority: list[str] = Field(default_factory=lambda: ["xtquant", "akshare", "baostock"])
    output_dir: str = "data/processed"
    raw_dir: str = "data/raw"
    xtquant: dict[str, Any] = Field(default_factory=dict)
    baostock: dict[str, Any] = Field(default_factory=dict)


class EnvironmentConfig(BaseModel):
    initial_amount: float = 1_000_000
    hmax: int = 100
    buy_cost_pct: float = 0.001
    sell_cost_pct: float = 0.001
    reward_scaling: float = 0.0001


class ScoringConfig(BaseModel):
    """Configuration for scoring-based training."""
    enabled: bool = False
    reward_type: str = "daily_return"
    future_horizon: int = 1
    normalize_obs: bool = True

    @field_validator("reward_type")
    @classmethod
    def validate_reward_type(cls, v: str) -> str:
        if v not in ("daily_return", "future_return"):
            raise ValueError(f"reward_type must be 'daily_return' or 'future_return', got {v!r}")
        return v


class PortfolioConfig(BaseModel):
    """Configuration for portfolio management."""
    max_positions: int = 10
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.20
    score_threshold: float = 0.0
    position_sizing: str = "equal"
    transaction_cost_pct: float = 0.001

    @field_validator("position_sizing")
    @classmethod
    def validate_position_sizing(cls, v: str) -> str:
        if v not in ("equal", "score_weighted"):
            raise ValueError(f"position_sizing must be 'equal' or 'score_weighted', got {v!r}")
        return v

    @field_validator("stop_loss_pct")
    @classmethod
    def validate_stop_loss_pct(cls, v: float) -> float:
        if v >= 0:
            raise ValueError(f"stop_loss_pct must be negative, got {v}")
        return v

    @field_validator("take_profit_pct")
    @classmethod
    def validate_take_profit_pct(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"take_profit_pct must be positive, got {v}")
        return v


class TrainingConfig(BaseModel):
    algorithm: str = "ppo"
    total_timesteps: int = 100_000
    model_dir: str = "models"
    mode: str = "trading"
    epochs: int = 1
    stocks_per_epoch: int = 10
    timesteps_per_epoch: int = 10_000
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    ppo: dict[str, Any] = Field(default_factory=dict)
    sac: dict[str, Any] = Field(default_factory=dict)
    td3: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("trading", "scoring"):
            raise ValueError(f"mode must be 'trading' or 'scoring', got {v!r}")
        return v


class SentimentConfig(BaseModel):
    enabled: bool = False
    news_file: str = ""
    output_dir: str = "data/sentiment"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    quantize_4bit: bool = True
    max_new_tokens: int = 512
    batch_size: int = 8


class FundamentalConfig(BaseModel):
    enabled: bool = False
    output_dir: str = "data/fundamentals"


class FrequencyConfig(BaseModel):
    value: str = "daily"

    @field_validator("value")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        if v not in ("daily", "5min"):
            raise ValueError(f"frequency must be 'daily' or '5min', got {v!r}")
        return v


class IndicatorSetConfig(BaseModel):
    id: str
    name: str = ""
    frequency: str = "daily"
    indicators: list[str] = Field(default_factory=list)
    window_params: dict[str, Any] = Field(default_factory=dict)
    target: str = "future_return"

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        if v not in ("daily", "5min"):
            raise ValueError(f"frequency must be 'daily' or '5min', got {v!r}")
        return v


class TargetConfig(BaseModel):
    type: str = "future_return"
    horizon: int = 1
    threshold: float = 0.0

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ("future_return", "direction"):
            raise ValueError(f"target type must be 'future_return' or 'direction', got {v!r}")
        return v


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"
    structured: bool = True


class MarketStateConfig(BaseModel):
    auto_optimize: bool = False


class FactorConfig(BaseModel):
    id: str
    name: str = ""
    category: str
    required_columns: list[str] = Field(default_factory=list)
    missing_strategy: str = "fill_neutral"
    direction: int = 1

    @field_validator("missing_strategy")
    @classmethod
    def validate_missing_strategy(cls, v: str) -> str:
        if v not in ("fill_neutral", "skip", "reduce_weight"):
            raise ValueError(f"missing_strategy must be 'fill_neutral', 'skip', or 'reduce_weight', got {v!r}")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError(f"direction must be 1 or -1, got {v!r}")
        return v


class ScreenConfig(BaseModel):
    top_k: int = 10
    exclude_st: bool = True
    exclude_halt: bool = True
    exclude_limit_up: bool = True
    score_range: tuple[float, float] = (-1.0, 1.0)


class BreakoutConfig(BaseModel):
    ma_periods: list[int] = Field(default_factory=lambda: [120, 250])
    volume_multiplier: float = 1.5
    volume_ma_period: int = 20
    breakout_threshold: float = 1.05
    lookback_days: int = 60
    confirmation_days: int = 3
    anti_jitter_mode: str = "threshold"
    top_k: int = 10
    exclude_st: bool = True
    exclude_halt: bool = True

    @field_validator("anti_jitter_mode")
    @classmethod
    def validate_anti_jitter_mode(cls, v: str) -> str:
        if v not in ("threshold", "confirmation", "both", "either"):
            raise ValueError(f"anti_jitter_mode must be 'threshold', 'confirmation', 'both', or 'either', got {v!r}")
        return v


class SelectionConfig(BaseModel):
    strategy_type: str = "factor_based"
    index_ticker: str = "000905.SH"
    top_k: int = 10
    ic_window: int = 60
    ic_min_periods: int = 20
    normalizer: str = "zscore"
    market_state: MarketStateConfig = Field(default_factory=MarketStateConfig)
    exclude_st: bool = True
    exclude_halt: bool = True
    breakout: BreakoutConfig | None = None

    @field_validator("strategy_type")
    @classmethod
    def validate_strategy_type(cls, v: str) -> str:
        if v not in ("factor_based", "breakout"):
            raise ValueError(f"strategy_type must be 'factor_based' or 'breakout', got {v!r}")
        return v

    @field_validator("normalizer")
    @classmethod
    def validate_normalizer(cls, v: str) -> str:
        if v not in ("zscore", "rank", "mad"):
            raise ValueError(f"normalizer must be 'zscore', 'rank', or 'mad', got {v!r}")
        return v


class ZZ500SelectionConfig(BaseModel):
    """Configuration for ZZ500 RL-based stock selection."""
    portfolio_size: int = 5
    score_threshold: float = 0.9
    score_mapping: str = "sigmoid"
    position_sizing: str = "score_weighted"
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.20
    rl_exit_threshold: float = -0.2


class AppConfig(BaseModel):
    stocks: list[str]
    dates: DatesConfig
    data: DataConfig = Field(default_factory=DataConfig)
    indicators: list[str] = Field(
        default_factory=lambda: [
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ]
    )
    frequency: FrequencyConfig = Field(default_factory=FrequencyConfig)
    indicator_sets: list[IndicatorSetConfig] = Field(default_factory=list)
    target: TargetConfig = Field(default_factory=TargetConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    fundamentals: FundamentalConfig = Field(default_factory=FundamentalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    selection: SelectionConfig | None = None
    zz500_selection: ZZ500SelectionConfig = Field(default_factory=ZZ500SelectionConfig)
    live_trading: bool = False

    @field_validator("frequency", mode="before")
    @classmethod
    def validate_frequency(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"value": value}
        return value

    @field_validator("stocks")
    @classmethod
    def validate_stocks(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("stocks cannot be empty")
        return value


DEFAULT_CONFIG_PATH = Path("config/default.yaml.example")


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
