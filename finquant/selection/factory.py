"""Strategy factory for creating selection strategies."""

from finquant.config.settings import AppConfig, SelectionConfig
from finquant.selection.pipeline import SelectionPipeline
from finquant.selection.strategy import SelectionStrategy
from finquant.selection.strategies.breakout import BreakoutStrategy, BreakoutConfig
from finquant.selection.strategies.factor_based import FactorBasedStrategy


def create_strategy(config: AppConfig, verbose: bool = False) -> SelectionStrategy:
    """Create a selection strategy based on configuration.

    Args:
        config: Application configuration with selection settings
        verbose: Enable detailed logging output

    Returns:
        SelectionStrategy instance (FactorBasedStrategy or BreakoutStrategy)

    Raises:
        ValueError: If strategy_type is invalid or required config is missing

    Example:
        config = AppConfig.from_yaml("config.yaml")
        strategy = create_strategy(config, verbose=True)
        result = strategy.select(market_df, index_df, "2023-01-01")
    """
    if config.selection is None:
        raise ValueError("selection config is required")

    selection_config = config.selection
    strategy_type = selection_config.strategy_type

    if strategy_type == "factor_based":
        return _create_factor_based_strategy(config)
    elif strategy_type == "breakout":
        return _create_breakout_strategy(selection_config, verbose)
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")


def _create_factor_based_strategy(config: AppConfig) -> FactorBasedStrategy:
    """Create factor-based strategy using existing pipeline."""
    pipeline = SelectionPipeline.from_config(config)
    return FactorBasedStrategy(pipeline)


def _create_breakout_strategy(selection_config: SelectionConfig, verbose: bool = False) -> BreakoutStrategy:
    """Create breakout strategy from config."""
    if selection_config.breakout is None:
        # Use default BreakoutConfig
        breakout_config = BreakoutConfig()
    else:
        # Convert Pydantic model to BreakoutConfig dataclass
        breakout_config = BreakoutConfig(
            ma_periods=selection_config.breakout.ma_periods,
            volume_multiplier=selection_config.breakout.volume_multiplier,
            volume_ma_period=selection_config.breakout.volume_ma_period,
            breakout_threshold=selection_config.breakout.breakout_threshold,
            lookback_days=selection_config.breakout.lookback_days,
            confirmation_days=selection_config.breakout.confirmation_days,
            anti_jitter_mode=selection_config.breakout.anti_jitter_mode,
            top_k=selection_config.breakout.top_k,
            exclude_st=selection_config.breakout.exclude_st,
            exclude_halt=selection_config.breakout.exclude_halt,
        )

    return BreakoutStrategy(breakout_config, verbose=verbose)
