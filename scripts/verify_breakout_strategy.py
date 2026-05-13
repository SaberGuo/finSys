"""
Verification script for MA Breakout Selection Strategy implementation.

This script verifies that all components are working correctly.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from finquant.selection import create_strategy, SelectionStrategy
        from finquant.selection.strategies import BreakoutStrategy, FactorBasedStrategy
        from finquant.config.settings import load_config, BreakoutConfig
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config_loading():
    """Test that config files can be loaded."""
    print("\nTesting config loading...")
    try:
        from finquant.config.settings import load_config

        # Test breakout config
        breakout_config = load_config("config/selection_breakout.yaml")
        assert breakout_config.selection.strategy_type == "breakout"
        assert breakout_config.selection.breakout.ma_periods == [120, 250]
        print("  ✓ Breakout config loaded successfully")

        # Test factor-based config
        factor_config = load_config("config/selection_factor_based.yaml")
        assert factor_config.selection.strategy_type == "factor_based"
        print("  ✓ Factor-based config loaded successfully")

        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_strategy_creation():
    """Test that strategies can be created."""
    print("\nTesting strategy creation...")
    try:
        from finquant.config.settings import load_config
        from finquant.selection import create_strategy
        from finquant.selection.strategies import BreakoutStrategy, FactorBasedStrategy

        # Test breakout strategy
        breakout_config = load_config("config/selection_breakout.yaml")
        breakout_strategy = create_strategy(breakout_config)
        assert isinstance(breakout_strategy, BreakoutStrategy)
        print("  ✓ Breakout strategy created successfully")

        # Test factor-based strategy
        factor_config = load_config("config/selection_factor_based.yaml")
        factor_strategy = create_strategy(factor_config)
        assert isinstance(factor_strategy, FactorBasedStrategy)
        print("  ✓ Factor-based strategy created successfully")

        return True
    except Exception as e:
        print(f"  ✗ Strategy creation failed: {e}")
        return False


def test_strategy_interface():
    """Test that strategies implement the correct interface."""
    print("\nTesting strategy interface...")
    try:
        from finquant.config.settings import load_config
        from finquant.selection import create_strategy, SelectionStrategy

        config = load_config("config/selection_breakout.yaml")
        strategy = create_strategy(config)

        # Check that strategy implements SelectionStrategy
        assert isinstance(strategy, SelectionStrategy)
        assert hasattr(strategy, "select")
        print("  ✓ Strategy implements SelectionStrategy interface")

        return True
    except Exception as e:
        print(f"  ✗ Interface test failed: {e}")
        return False


def test_config_validation():
    """Test that config validation works."""
    print("\nTesting config validation...")
    try:
        from finquant.config.settings import AppConfig
        from pydantic import ValidationError

        # Test invalid strategy type
        try:
            config_dict = {
                "stocks": ["600000.SH"],
                "dates": {
                    "train_start": "2023-01-01",
                    "train_end": "2023-06-30",
                    "test_start": "2023-07-01",
                    "test_end": "2023-12-31",
                },
                "selection": {
                    "strategy_type": "invalid_type",
                },
            }
            AppConfig(**config_dict)
            print("  ✗ Should have raised ValidationError for invalid strategy_type")
            return False
        except ValidationError:
            print("  ✓ Invalid strategy_type rejected correctly")

        # Test invalid anti_jitter_mode
        try:
            config_dict = {
                "stocks": ["600000.SH"],
                "dates": {
                    "train_start": "2023-01-01",
                    "train_end": "2023-06-30",
                    "test_start": "2023-07-01",
                    "test_end": "2023-12-31",
                },
                "selection": {
                    "strategy_type": "breakout",
                    "breakout": {
                        "anti_jitter_mode": "invalid_mode",
                    },
                },
            }
            AppConfig(**config_dict)
            print("  ✗ Should have raised ValidationError for invalid anti_jitter_mode")
            return False
        except ValidationError:
            print("  ✓ Invalid anti_jitter_mode rejected correctly")

        return True
    except Exception as e:
        print(f"  ✗ Validation test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("MA Breakout Selection Strategy - Verification")
    print("=" * 60)

    tests = [
        test_imports,
        test_config_loading,
        test_strategy_creation,
        test_strategy_interface,
        test_config_validation,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All verification tests passed!")
        print("\nThe MA Breakout Selection Strategy is ready to use.")
        print("\nNext steps:")
        print("  1. Run unit tests: pytest tests/unit/test_breakout_strategy.py")
        print("  2. Run integration tests: pytest tests/integration/test_breakout_selection.py")
        print("  3. Try the example: python examples/breakout_strategy_example.py")
        print("  4. Read the guide: docs/breakout-strategy-guide.md")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
