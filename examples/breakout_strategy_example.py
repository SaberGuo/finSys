"""
Example: Using the MA Breakout Selection Strategy

This script demonstrates how to use the breakout strategy to select stocks
based on moving average breakouts with volume confirmation.
"""

from finquant.config.settings import AppConfig
from finquant.selection import create_strategy
import pandas as pd


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===\n")

    # Load configuration
    config = AppConfig.from_yaml("config/selection_breakout.yaml")

    # Create strategy
    strategy = create_strategy(config)
    print(f"Created strategy: {type(strategy).__name__}")
    print(f"Strategy config: {strategy.config}\n")

    # In a real scenario, you would load actual market data
    # For this example, we'll show the structure
    print("Expected data structure:")
    print("market_df columns: date, tic, open, high, low, close, volume")
    print("index_df columns: date, tic, open, high, low, close, volume\n")


def example_custom_config():
    """Example with custom configuration."""
    print("=== Custom Configuration Example ===\n")

    config_dict = {
        "stocks": ["600000.SH", "600016.SH", "600019.SH"],
        "dates": {
            "train_start": "2023-01-01",
            "train_end": "2023-06-30",
            "test_start": "2023-07-01",
            "test_end": "2023-12-31",
        },
        "selection": {
            "strategy_type": "breakout",
            "breakout": {
                "ma_periods": [60, 120],           # Shorter MAs for more signals
                "volume_multiplier": 2.0,          # Stronger volume requirement
                "breakout_threshold": 1.03,        # 3% above MA
                "lookback_days": 30,               # Shorter lookback
                "anti_jitter_mode": "confirmation", # Use confirmation mode
                "top_k": 5,
            },
        },
    }

    config = AppConfig(**config_dict)
    strategy = create_strategy(config)

    print("Custom strategy configuration:")
    print(f"  MA periods: {strategy.config.ma_periods}")
    print(f"  Volume multiplier: {strategy.config.volume_multiplier}")
    print(f"  Breakout threshold: {strategy.config.breakout_threshold}")
    print(f"  Anti-jitter mode: {strategy.config.anti_jitter_mode}")
    print(f"  Top K: {strategy.config.top_k}\n")


def example_strategy_comparison():
    """Example comparing breakout vs factor-based strategies."""
    print("=== Strategy Comparison Example ===\n")

    # Breakout strategy
    breakout_config = AppConfig.from_yaml("config/selection_breakout.yaml")
    breakout_strategy = create_strategy(breakout_config)

    # Factor-based strategy
    factor_config = AppConfig.from_yaml("config/selection_factor_based.yaml")
    factor_strategy = create_strategy(factor_config)

    print(f"Breakout strategy: {type(breakout_strategy).__name__}")
    print(f"Factor-based strategy: {type(factor_strategy).__name__}")
    print("\nBoth strategies implement the same SelectionStrategy interface")
    print("and can be used interchangeably.\n")


def example_result_structure():
    """Show the structure of selection results."""
    print("=== Selection Result Structure ===\n")

    print("SelectionResult attributes:")
    print("  - date: str                      # Selection date")
    print("  - selected_tickers: list[str]    # Top-k selected stocks")
    print("  - scores: dict[str, float]       # Stock scores")
    print("  - market_state: MarketState      # Market condition")
    print("  - active_factors: list[str]      # Factors used")
    print("  - factor_weights: dict           # Factor weights")
    print("  - index_metrics: dict            # Index metrics")
    print("  - exclusion_reasons: dict        # Why stocks excluded")
    print("\nExample usage:")
    print("  result = strategy.select(market_df, index_df, '2023-08-01')")
    print("  for tic in result.selected_tickers:")
    print("      print(f'{tic}: {result.scores[tic]:.4f}')\n")


def example_cli_usage():
    """Show CLI usage examples."""
    print("=== CLI Usage Examples ===\n")

    print("1. Run breakout selection:")
    print("   finquant selection run \\")
    print("     --config config/selection_breakout.yaml \\")
    print("     --start 2023-07-01 \\")
    print("     --end 2023-12-31 \\")
    print("     --output-dir data/selection_breakout \\")
    print("     --verbose\n")

    print("2. Run factor-based selection for comparison:")
    print("   finquant selection run \\")
    print("     --config config/selection_factor_based.yaml \\")
    print("     --start 2023-07-01 \\")
    print("     --end 2023-12-31 \\")
    print("     --output-dir data/selection_factor \\")
    print("     --verbose\n")

    print("3. Compare results:")
    print("   # Results are saved as JSON files in output directories")
    print("   # You can load and compare them programmatically\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MA Breakout Selection Strategy - Usage Examples")
    print("=" * 60)
    print()

    example_basic_usage()
    example_custom_config()
    example_strategy_comparison()
    example_result_structure()
    example_cli_usage()

    print("=" * 60)
    print("For more details, see docs/breakout-strategy-guide.md")
    print("=" * 60)
