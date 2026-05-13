"""Rule-based portfolio management using RL stock scores.

This module provides a portfolio manager that uses stock scores from an RL model
to make trading decisions. The manager handles:
- Position sizing and allocation
- Stop-loss and take-profit rules
- Score-based entry and exit signals
- Max position constraints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Position:
    """Represents a single stock position in the portfolio.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol
    shares : int
        Number of shares held
    entry_price : float
        Price at which position was opened
    entry_date : str
        Date when position was opened (YYYY-MM-DD)
    current_price : float
        Current market price
    unrealized_pnl : float
        Unrealized profit/loss in currency
    unrealized_pnl_pct : float
        Unrealized profit/loss as percentage
    """

    ticker: str
    shares: int
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_price(self, price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.shares
        self.unrealized_pnl_pct = (price - self.entry_price) / self.entry_price

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.current_price * self.shares


@dataclass
class Trade:
    """Represents a single trade execution.

    Attributes
    ----------
    date : str
        Trade date (YYYY-MM-DD)
    ticker : str
        Stock ticker symbol
    action : str
        Trade action: "buy" or "sell"
    shares : int
        Number of shares traded
    price : float
        Execution price
    amount : float
        Total trade amount (shares × price)
    reason : str
        Reason for trade (e.g., "fill_vacancy", "stop_loss", "take_profit")
    score : float, optional
        Stock score at time of trade
    """

    date: str
    ticker: str
    action: str
    shares: int
    price: float
    amount: float
    reason: str
    score: float | None = None


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time.

    Attributes
    ----------
    date : str
        State date (YYYY-MM-DD)
    cash : float
        Available cash
    positions : dict[str, Position]
        Current positions by ticker
    total_value : float
        Total portfolio value (cash + positions)
    daily_return : float
        Daily return percentage
    """

    date: str
    cash: float
    positions: dict[str, Position]
    total_value: float
    daily_return: float = 0.0

    @property
    def position_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)


class PortfolioManager:
    """Rule-based portfolio manager using RL stock scores.

    The manager executes a daily update cycle:
    1. Update existing positions with current prices
    2. Check stop-loss and take-profit conditions
    3. Check score-based exit signals
    4. Fill vacancies from highest-scored stocks

    Parameters
    ----------
    initial_cash : float
        Starting cash amount
    max_positions : int, default=10
        Maximum number of concurrent positions
    stop_loss_pct : float, default=-0.05
        Stop-loss threshold (e.g., -0.05 = -5%)
    take_profit_pct : float, default=0.20
        Take-profit threshold (e.g., 0.20 = +20%)
    score_threshold : float, default=0.0
        Minimum score to hold position (sell if score < threshold)
    position_sizing : str, default="equal"
        Position sizing method: "equal" or "score_weighted"
    transaction_cost_pct : float, default=0.001
        Transaction cost as percentage (0.001 = 0.1%)
    """

    def __init__(
        self,
        initial_cash: float,
        max_positions: int = 10,
        stop_loss_pct: float = -0.05,
        take_profit_pct: float = 0.20,
        score_threshold: float = 0.0,
        position_sizing: str = "equal",
        transaction_cost_pct: float = 0.001,
    ) -> None:
        if max_positions <= 0:
            raise ValueError(f"max_positions must be > 0, got {max_positions}")
        if stop_loss_pct >= 0:
            raise ValueError(f"stop_loss_pct must be negative, got {stop_loss_pct}")
        if take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be positive, got {take_profit_pct}")
        if position_sizing not in ["equal", "score_weighted"]:
            raise ValueError(
                f"position_sizing must be 'equal' or 'score_weighted', got {position_sizing!r}"
            )

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.score_threshold = score_threshold
        self.position_sizing = position_sizing
        self.transaction_cost_pct = transaction_cost_pct

        self.positions: dict[str, Position] = {}
        self.trade_history: list[Trade] = []
        self.state_history: list[PortfolioState] = []
        self._prev_total_value = initial_cash

    @property
    def total_value(self) -> float:
        """Current total portfolio value."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    def update(
        self,
        date: str,
        scores: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, Any]:
        """Execute daily portfolio update.

        Parameters
        ----------
        date : str
            Current date (YYYY-MM-DD)
        scores : dict[str, float]
            Stock scores from RL model {ticker: score}
        prices : dict[str, float]
            Current prices {ticker: price}

        Returns
        -------
        dict
            Update summary with trades and portfolio state
        """
        trades: list[Trade] = []
        sold_today: set[str] = set()  # Track stocks sold today to avoid re-buying

        # Step 1: Update existing positions with current prices
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.update_price(prices[ticker])

        # Step 2: Check stop-loss conditions
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            if pos.unrealized_pnl_pct <= self.stop_loss_pct:
                trade = self._sell_position(date, ticker, "stop_loss", scores.get(ticker))
                trades.append(trade)
                sold_today.add(ticker)

        # Step 3: Check take-profit conditions
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            if pos.unrealized_pnl_pct >= self.take_profit_pct:
                trade = self._sell_position(date, ticker, "take_profit", scores.get(ticker))
                trades.append(trade)
                sold_today.add(ticker)

        # Step 4: Check score-based exits
        for ticker in list(self.positions.keys()):
            score = scores.get(ticker, 0.0)
            if score < self.score_threshold:
                trade = self._sell_position(date, ticker, "negative_score", score)
                trades.append(trade)
                sold_today.add(ticker)

        # Step 5: Fill vacancies from highest-scored stocks
        vacancies = self.max_positions - self.num_positions
        if vacancies > 0:
            # Filter out stocks already held, sold today, and with negative scores
            candidates = {
                ticker: score
                for ticker, score in scores.items()
                if ticker not in self.positions
                and ticker not in sold_today
                and score >= self.score_threshold
                and ticker in prices
            }

            # Sort by score descending
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

            # Buy top N candidates
            for ticker, score in sorted_candidates[:vacancies]:
                if self.cash <= 0:
                    break
                trade = self._buy_position(date, ticker, prices[ticker], score)
                if trade:
                    trades.append(trade)

        # Step 6: Record portfolio state
        daily_return = (self.total_value - self._prev_total_value) / self._prev_total_value
        state = PortfolioState(
            date=date,
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=self.total_value,
            daily_return=daily_return,
        )
        self.state_history.append(state)
        self._prev_total_value = self.total_value

        return {
            "date": date,
            "trades": trades,
            "num_positions": self.num_positions,
            "cash": self.cash,
            "total_value": self.total_value,
            "daily_return": daily_return,
        }

    def _buy_position(
        self, date: str, ticker: str, price: float, score: float
    ) -> Trade | None:
        """Buy a new position.

        Parameters
        ----------
        date : str
            Trade date
        ticker : str
            Stock ticker
        price : float
            Current price
        score : float
            Stock score

        Returns
        -------
        Trade or None
            Trade object if executed, None if insufficient cash
        """
        # Calculate position size
        if self.position_sizing == "equal":
            # Equal allocation among max positions
            target_amount = self.cash / (self.max_positions - self.num_positions)
        else:  # score_weighted
            # Allocate proportional to score (not implemented yet, use equal for now)
            target_amount = self.cash / (self.max_positions - self.num_positions)

        # Calculate shares (integer)
        shares = int(target_amount / price)
        if shares <= 0:
            return None

        # Calculate actual cost including transaction costs
        cost = shares * price
        transaction_cost = cost * self.transaction_cost_pct
        total_cost = cost + transaction_cost

        if total_cost > self.cash:
            # Adjust shares to fit available cash
            shares = int(self.cash / (price * (1 + self.transaction_cost_pct)))
            if shares <= 0:
                return None
            cost = shares * price
            transaction_cost = cost * self.transaction_cost_pct
            total_cost = cost + transaction_cost

        # Execute trade
        self.cash -= total_cost
        pos = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            entry_date=date,
            current_price=price,
        )
        pos.update_price(price)  # Initialize P&L values
        self.positions[ticker] = pos

        trade = Trade(
            date=date,
            ticker=ticker,
            action="buy",
            shares=shares,
            price=price,
            amount=cost,
            reason="fill_vacancy",
            score=score,
        )
        self.trade_history.append(trade)
        return trade

    def _sell_position(
        self, date: str, ticker: str, reason: str, score: float | None = None
    ) -> Trade:
        """Sell an existing position.

        Parameters
        ----------
        date : str
            Trade date
        ticker : str
            Stock ticker
        reason : str
            Reason for selling
        score : float, optional
            Current stock score

        Returns
        -------
        Trade
            Trade object
        """
        pos = self.positions.pop(ticker)
        proceeds = pos.shares * pos.current_price
        transaction_cost = proceeds * self.transaction_cost_pct
        net_proceeds = proceeds - transaction_cost

        self.cash += net_proceeds

        trade = Trade(
            date=date,
            ticker=ticker,
            action="sell",
            shares=pos.shares,
            price=pos.current_price,
            amount=proceeds,
            reason=reason,
            score=score,
        )
        self.trade_history.append(trade)
        return trade

    def get_portfolio_series(self) -> pd.Series:
        """Get portfolio value time series.

        Returns
        -------
        pd.Series
            Portfolio value indexed by date
        """
        if not self.state_history:
            return pd.Series(dtype=float)

        return pd.Series(
            [state.total_value for state in self.state_history],
            index=pd.to_datetime([state.date for state in self.state_history]),
            name="portfolio_value",
        )

    def get_trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame.

        Returns
        -------
        pd.DataFrame
            Trade history with columns: date, ticker, action, shares, price, amount, reason, score
        """
        if not self.trade_history:
            return pd.DataFrame(columns=["date", "ticker", "action", "shares", "price", "amount", "reason", "score"])

        return pd.DataFrame([
            {
                "date": trade.date,
                "ticker": trade.ticker,
                "action": trade.action,
                "shares": trade.shares,
                "price": trade.price,
                "amount": trade.amount,
                "reason": trade.reason,
                "score": trade.score,
            }
            for trade in self.trade_history
        ])

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio performance summary.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self.state_history:
            return {
                "initial_cash": self.initial_cash,
                "final_value": self.cash,
                "total_return": 0.0,
                "num_trades": 0,
            }

        final_value = self.state_history[-1].total_value
        total_return = (final_value - self.initial_cash) / self.initial_cash

        trades_df = self.get_trades_df()
        num_buys = len(trades_df[trades_df["action"] == "buy"])
        num_sells = len(trades_df[trades_df["action"] == "sell"])

        return {
            "initial_cash": self.initial_cash,
            "final_value": final_value,
            "total_return": total_return,
            "num_trades": len(self.trade_history),
            "num_buys": num_buys,
            "num_sells": num_sells,
            "max_positions": self.max_positions,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
        }
