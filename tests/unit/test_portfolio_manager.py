"""Unit tests for PortfolioManager."""
from __future__ import annotations

import pytest

from finquant.training.portfolio_manager import PortfolioManager, Position, Trade


@pytest.fixture
def portfolio() -> PortfolioManager:
    """Create a portfolio manager with default settings."""
    return PortfolioManager(
        initial_cash=1_000_000,
        max_positions=10,
        stop_loss_pct=-0.05,
        take_profit_pct=0.20,
        score_threshold=0.0,
        position_sizing="equal",
        transaction_cost_pct=0.001,
    )


@pytest.fixture
def sample_scores() -> dict[str, float]:
    """Sample stock scores."""
    return {
        "600519.SH": 0.8,
        "000001.SZ": 0.6,
        "600036.SH": 0.4,
        "000002.SZ": 0.2,
        "600000.SH": -0.1,
    }


@pytest.fixture
def sample_prices() -> dict[str, float]:
    """Sample stock prices."""
    return {
        "600519.SH": 100.0,
        "000001.SZ": 50.0,
        "600036.SH": 80.0,
        "000002.SZ": 30.0,
        "600000.SH": 60.0,
    }


class TestPortfolioManager:
    """Test suite for PortfolioManager."""

    def test_init_valid(self) -> None:
        """Test portfolio initialization with valid parameters."""
        pm = PortfolioManager(
            initial_cash=1_000_000,
            max_positions=10,
            stop_loss_pct=-0.05,
            take_profit_pct=0.20,
        )

        assert pm.cash == 1_000_000
        assert pm.max_positions == 10
        assert pm.stop_loss_pct == -0.05
        assert pm.take_profit_pct == 0.20
        assert pm.num_positions == 0
        assert pm.total_value == 1_000_000

    def test_init_invalid_max_positions(self) -> None:
        """Test that invalid max_positions raises ValueError."""
        with pytest.raises(ValueError, match="max_positions must be > 0"):
            PortfolioManager(initial_cash=1_000_000, max_positions=0)

    def test_init_invalid_stop_loss(self) -> None:
        """Test that non-negative stop_loss_pct raises ValueError."""
        with pytest.raises(ValueError, match="stop_loss_pct must be negative"):
            PortfolioManager(initial_cash=1_000_000, stop_loss_pct=0.05)

    def test_init_invalid_take_profit(self) -> None:
        """Test that non-positive take_profit_pct raises ValueError."""
        with pytest.raises(ValueError, match="take_profit_pct must be positive"):
            PortfolioManager(initial_cash=1_000_000, take_profit_pct=-0.20)

    def test_init_invalid_position_sizing(self) -> None:
        """Test that invalid position_sizing raises ValueError."""
        with pytest.raises(ValueError, match="position_sizing must be"):
            PortfolioManager(initial_cash=1_000_000, position_sizing="invalid")

    def test_buy_position(self, portfolio: PortfolioManager, sample_prices: dict[str, float]) -> None:
        """Test buying a new position."""
        result = portfolio.update(
            date="2024-01-01",
            scores={"600519.SH": 0.8},
            prices=sample_prices,
        )

        assert portfolio.num_positions == 1
        assert "600519.SH" in portfolio.positions
        assert len(result["trades"]) == 1
        assert result["trades"][0].action == "buy"
        assert result["trades"][0].ticker == "600519.SH"

    def test_fill_vacancies(self, portfolio: PortfolioManager, sample_scores: dict[str, float], sample_prices: dict[str, float]) -> None:
        """Test filling vacancies from highest-scored stocks."""
        result = portfolio.update(
            date="2024-01-01",
            scores=sample_scores,
            prices=sample_prices,
        )

        # Should buy top 4 stocks (positive scores)
        assert portfolio.num_positions == 4
        assert "600519.SH" in portfolio.positions  # score 0.8
        assert "000001.SZ" in portfolio.positions  # score 0.6
        assert "600036.SH" in portfolio.positions  # score 0.4
        assert "000002.SZ" in portfolio.positions  # score 0.2
        assert "600000.SH" not in portfolio.positions  # score -0.1 (negative)

    def test_max_positions_constraint(self, sample_prices: dict[str, float]) -> None:
        """Test that max_positions constraint is enforced."""
        pm = PortfolioManager(initial_cash=1_000_000, max_positions=2)

        scores = {f"stock_{i}": 0.5 for i in range(10)}
        prices = {f"stock_{i}": 100.0 for i in range(10)}

        pm.update(date="2024-01-01", scores=scores, prices=prices)

        assert pm.num_positions == 2

    def test_stop_loss_trigger(self, portfolio: PortfolioManager, sample_prices: dict[str, float]) -> None:
        """Test stop-loss selling."""
        # Buy position
        portfolio.update(
            date="2024-01-01",
            scores={"600519.SH": 0.8},
            prices=sample_prices,
        )

        # Price drops 6% (below -5% stop-loss)
        new_prices = sample_prices.copy()
        new_prices["600519.SH"] = 94.0  # -6% from 100

        result = portfolio.update(
            date="2024-01-02",
            scores={"600519.SH": 0.8},  # Score still positive
            prices=new_prices,
        )

        # Should sell due to stop-loss
        assert portfolio.num_positions == 0
        assert len(result["trades"]) == 1
        assert result["trades"][0].action == "sell"
        assert result["trades"][0].reason == "stop_loss"

    def test_take_profit_trigger(self, portfolio: PortfolioManager, sample_prices: dict[str, float]) -> None:
        """Test take-profit selling."""
        # Buy position
        portfolio.update(
            date="2024-01-01",
            scores={"600519.SH": 0.8},
            prices=sample_prices,
        )

        # Price rises 21% (above 20% take-profit)
        new_prices = sample_prices.copy()
        new_prices["600519.SH"] = 121.0  # +21% from 100

        result = portfolio.update(
            date="2024-01-02",
            scores={"600519.SH": 0.8},
            prices=new_prices,
        )

        # Should sell due to take-profit
        assert portfolio.num_positions == 0
        assert len(result["trades"]) == 1
        assert result["trades"][0].action == "sell"
        assert result["trades"][0].reason == "take_profit"

    def test_negative_score_exit(self, portfolio: PortfolioManager, sample_prices: dict[str, float]) -> None:
        """Test selling when score becomes negative."""
        # Buy position
        portfolio.update(
            date="2024-01-01",
            scores={"600519.SH": 0.8},
            prices=sample_prices,
        )

        # Score becomes negative
        result = portfolio.update(
            date="2024-01-02",
            scores={"600519.SH": -0.2},
            prices=sample_prices,
        )

        # Should sell due to negative score
        assert portfolio.num_positions == 0
        assert len(result["trades"]) == 1
        assert result["trades"][0].action == "sell"
        assert result["trades"][0].reason == "negative_score"

    def test_transaction_costs(self, portfolio: PortfolioManager, sample_prices: dict[str, float]) -> None:
        """Test that transaction costs are applied."""
        initial_cash = portfolio.cash

        # Buy position
        portfolio.update(
            date="2024-01-01",
            scores={"600519.SH": 0.8},
            prices=sample_prices,
        )

        # Cash should decrease by more than just share cost
        pos = portfolio.positions["600519.SH"]
        share_cost = pos.shares * pos.entry_price
        transaction_cost = share_cost * 0.001
        expected_cash = initial_cash - share_cost - transaction_cost

        assert abs(portfolio.cash - expected_cash) < 1.0  # Allow small rounding error

    def test_position_update_price(self) -> None:
        """Test Position.update_price method."""
        pos = Position(
            ticker="600519.SH",
            shares=100,
            entry_price=100.0,
            entry_date="2024-01-01",
        )

        pos.update_price(110.0)

        assert pos.current_price == 110.0
        assert pos.unrealized_pnl == 1000.0  # (110 - 100) * 100
        assert pos.unrealized_pnl_pct == 0.10  # 10%

    def test_position_market_value(self) -> None:
        """Test Position.market_value property."""
        pos = Position(
            ticker="600519.SH",
            shares=100,
            entry_price=100.0,
            entry_date="2024-01-01",
            current_price=110.0,
        )

        assert pos.market_value == 11000.0  # 100 * 110

    def test_get_portfolio_series(self, portfolio: PortfolioManager, sample_scores: dict[str, float], sample_prices: dict[str, float]) -> None:
        """Test get_portfolio_series method."""
        # Run multiple updates
        for i in range(5):
            portfolio.update(
                date=f"2024-01-0{i+1}",
                scores=sample_scores,
                prices=sample_prices,
            )

        series = portfolio.get_portfolio_series()

        assert len(series) == 5
        assert series.name == "portfolio_value"
        assert all(series > 0)

    def test_get_trades_df(self, portfolio: PortfolioManager, sample_scores: dict[str, float], sample_prices: dict[str, float]) -> None:
        """Test get_trades_df method."""
        portfolio.update(
            date="2024-01-01",
            scores=sample_scores,
            prices=sample_prices,
        )

        trades_df = portfolio.get_trades_df()

        assert len(trades_df) > 0
        assert "date" in trades_df.columns
        assert "ticker" in trades_df.columns
        assert "action" in trades_df.columns
        assert "shares" in trades_df.columns
        assert "price" in trades_df.columns
        assert "amount" in trades_df.columns
        assert "reason" in trades_df.columns
        assert "score" in trades_df.columns

    def test_get_summary(self, portfolio: PortfolioManager, sample_scores: dict[str, float], sample_prices: dict[str, float]) -> None:
        """Test get_summary method."""
        # Run updates
        for i in range(3):
            portfolio.update(
                date=f"2024-01-0{i+1}",
                scores=sample_scores,
                prices=sample_prices,
            )

        summary = portfolio.get_summary()

        assert "initial_cash" in summary
        assert "final_value" in summary
        assert "total_return" in summary
        assert "num_trades" in summary
        assert "num_buys" in summary
        assert "num_sells" in summary
        assert summary["initial_cash"] == 1_000_000

    def test_daily_return_calculation(self, portfolio: PortfolioManager, sample_prices: dict[str, float]) -> None:
        """Test daily return calculation."""
        # Day 1: Buy
        result1 = portfolio.update(
            date="2024-01-01",
            scores={"600519.SH": 0.8},
            prices=sample_prices,
        )

        # Day 2: Price increases
        new_prices = sample_prices.copy()
        new_prices["600519.SH"] = 105.0  # +5%

        result2 = portfolio.update(
            date="2024-01-02",
            scores={"600519.SH": 0.8},
            prices=new_prices,
        )

        # Daily return should be positive
        assert result2["daily_return"] > 0

    def test_insufficient_cash(self) -> None:
        """Test behavior when cash is insufficient."""
        pm = PortfolioManager(initial_cash=1000, max_positions=10)

        scores = {f"stock_{i}": 0.5 for i in range(10)}
        prices = {f"stock_{i}": 1000.0 for i in range(10)}  # Each stock costs more than total cash

        result = pm.update(date="2024-01-01", scores=scores, prices=prices)

        # Should buy at most 1 stock (or 0 if transaction costs prevent it)
        assert pm.num_positions <= 1

    def test_empty_scores(self, portfolio: PortfolioManager, sample_prices: dict[str, float]) -> None:
        """Test update with empty scores."""
        result = portfolio.update(
            date="2024-01-01",
            scores={},
            prices=sample_prices,
        )

        assert portfolio.num_positions == 0
        assert len(result["trades"]) == 0

    def test_all_negative_scores(self, portfolio: PortfolioManager, sample_prices: dict[str, float]) -> None:
        """Test update when all scores are negative."""
        negative_scores = {ticker: -0.5 for ticker in sample_prices}

        result = portfolio.update(
            date="2024-01-01",
            scores=negative_scores,
            prices=sample_prices,
        )

        # Should not buy any stocks
        assert portfolio.num_positions == 0
        assert len(result["trades"]) == 0

    def test_score_threshold(self) -> None:
        """Test score_threshold parameter."""
        pm = PortfolioManager(
            initial_cash=1_000_000,
            max_positions=10,
            score_threshold=0.5,  # Only buy if score >= 0.5
        )

        scores = {
            "stock_1": 0.8,  # Above threshold
            "stock_2": 0.4,  # Below threshold
            "stock_3": 0.6,  # Above threshold
        }
        prices = {ticker: 100.0 for ticker in scores}

        pm.update(date="2024-01-01", scores=scores, prices=prices)

        # Should only buy stocks with score >= 0.5
        assert pm.num_positions == 2
        assert "stock_1" in pm.positions
        assert "stock_3" in pm.positions
        assert "stock_2" not in pm.positions


class TestTrade:
    """Test suite for Trade dataclass."""

    def test_trade_creation(self) -> None:
        """Test Trade creation."""
        trade = Trade(
            date="2024-01-01",
            ticker="600519.SH",
            action="buy",
            shares=100,
            price=100.0,
            amount=10000.0,
            reason="fill_vacancy",
            score=0.8,
        )

        assert trade.date == "2024-01-01"
        assert trade.ticker == "600519.SH"
        assert trade.action == "buy"
        assert trade.shares == 100
        assert trade.price == 100.0
        assert trade.amount == 10000.0
        assert trade.reason == "fill_vacancy"
        assert trade.score == 0.8
