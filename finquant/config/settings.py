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


class TrainingConfig(BaseModel):
    algorithm: str = "ppo"
    total_timesteps: int = 100_000
    model_dir: str = "models"
    ppo: dict[str, Any] = Field(default_factory=dict)
    sac: dict[str, Any] = Field(default_factory=dict)
    td3: dict[str, Any] = Field(default_factory=dict)


class SentimentConfig(BaseModel):
    enabled: bool = False
    news_file: str = ""
    output_dir: str = "data/sentiment"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    quantize_4bit: bool = True
    max_new_tokens: int = 512
    batch_size: int = 8


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"
    structured: bool = True


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
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    live_trading: bool = False

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
