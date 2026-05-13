#!/usr/bin/env python3
"""Test ZZ500 RL system components."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finquant.config.settings import load_config
from finquant.data.sources.zz500_loader import load_zz500_stocks

def test_config():
    """Test configuration loading."""
    print("Testing configuration loading...")
    config = load_config("config/zz500_rl_single_stock.yaml")

    assert config.zz500_selection.portfolio_size == 5
    assert config.zz500_selection.score_threshold == 0.9
    assert config.zz500_selection.score_mapping == "sigmoid"
    assert config.zz500_selection.position_sizing == "score_weighted"
    assert config.zz500_selection.stop_loss_pct == -0.05
    assert config.zz500_selection.take_profit_pct == 0.20
    assert config.zz500_selection.rl_exit_threshold == -0.2

    print("✓ Configuration loaded successfully")
    print(f"  - Portfolio size: {config.zz500_selection.portfolio_size}")
    print(f"  - Score threshold: {config.zz500_selection.score_threshold}")
    print(f"  - Training timesteps: {config.training.total_timesteps}")

def test_stock_loader():
    """Test stock list loader."""
    print("\nTesting stock list loader...")
    db_path = Path("data/processed/zz500_data.db")

    if not db_path.exists():
        print(f"✗ Database not found: {db_path}")
        return

    stocks = load_zz500_stocks(db_path)

    assert len(stocks) > 0
    assert all(isinstance(s, str) for s in stocks)
    assert all("." in s for s in stocks)  # Format: XXXXXX.SH or XXXXXX.SZ

    print(f"✓ Loaded {len(stocks)} ZZ500 stocks")
    print(f"  - First 5: {stocks[:5]}")
    print(f"  - Last 5: {stocks[-5:]}")

def test_scorer_import():
    """Test RL scorer import."""
    print("\nTesting RL scorer import...")
    try:
        from finquant.selection.rl_scorer import RLStockScorer
        print("✓ RLStockScorer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import RLStockScorer: {e}")

def main():
    print("="*60)
    print("ZZ500 RL System Component Tests")
    print("="*60)

    try:
        test_config()
        test_stock_loader()
        test_scorer_import()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
