"""Indicator set configuration and registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from finquant.config.settings import IndicatorSetConfig
from finquant.features.technical import compute_indicators


@dataclass
class IndicatorSet:
    """Runtime representation of an indicator set configuration."""

    id: str
    name: str
    frequency: str
    indicators: list[str]
    window_params: dict[str, Any] = field(default_factory=dict)
    target: str = "future_return"

    @classmethod
    def from_config(cls, config: IndicatorSetConfig) -> "IndicatorSet":
        return cls(
            id=config.id,
            name=config.name or config.id,
            frequency=config.frequency,
            indicators=list(config.indicators),
            window_params=dict(config.window_params),
            target=config.target,
        )

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators in this set on *frame*."""
        return compute_indicators(frame, indicators=self.indicators)


class IndicatorSetRegistry:
    """Hold and lookup predefined indicator sets."""

    def __init__(self) -> None:
        self._sets: dict[str, IndicatorSet] = {}

    def register(self, indicator_set: IndicatorSet) -> None:
        self._sets[indicator_set.id] = indicator_set

    def get(self, indicator_set_id: str) -> IndicatorSet:
        if indicator_set_id not in self._sets:
            raise KeyError(f"indicator set '{indicator_set_id}' not found")
        return self._sets[indicator_set_id]

    def list_ids(self) -> list[str]:
        return list(self._sets.keys())

    @classmethod
    def from_configs(cls, configs: list[IndicatorSetConfig]) -> "IndicatorSetRegistry":
        registry = cls()
        for cfg in configs:
            registry.register(IndicatorSet.from_config(cfg))
        return registry
