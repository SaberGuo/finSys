"""Backtest metrics computation and report generation (T032)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252


def compute_trade_stats(
    df_actions: pd.DataFrame, price_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute individual trade records and summary statistics.

    Parameters
    ----------
    df_actions:
        DataFrame indexed by date with tickers as columns.
        Values are share holdings (not deltas).
    price_df:
        DataFrame indexed by date with tickers as columns.
        Values are closing prices.

    Returns
    -------
    trades: pd.DataFrame
        One row per trade with columns [date, tic, action, shares, price, amount].
    summary: dict[str, Any]
        Aggregated trade statistics.
    """
    # Position changes = actual trades (first day is treated as opening from 0)
    changes = df_actions.diff().fillna(df_actions.iloc[0])

    # Align price_df to changes so we can use positional .values access
    # (avoids all index-ambiguity issues with duplicate or MultiIndex)
    price_aligned = price_df.reindex(index=changes.index, columns=changes.columns)
    changes_vals = changes.values
    price_vals = price_aligned.values

    trades: list[dict[str, Any]] = []
    # Running cost basis per ticker for win-rate calculation
    cost_basis: dict[str, float] = {}  # total cost
    position: dict[str, float] = {}    # total shares
    wins = 0
    losses = 0

    for i in range(len(changes)):
        idx = changes.index[i]
        # Handle both simple index (date) and MultiIndex (date, time)
        if isinstance(idx, tuple):
            date, time = idx[0], idx[1]
        else:
            date, time = idx, None

        for j, tic in enumerate(changes.columns):
            delta = changes_vals[i, j]
            if abs(delta) < 0.5:
                continue
            price = price_vals[i, j]
            if pd.isna(price):
                continue
            action = "BUY" if delta > 0 else "SELL"
            shares = abs(delta)
            amount = shares * price
            trades.append(
                {
                    "date": str(date),
                    "tic": tic,
                    "action": action,
                    "shares": int(round(shares)),
                    "price": round(float(price), 3),
                    "amount": round(float(amount), 2),
                }
            )
            if action == "BUY":
                position[tic] = position.get(tic, 0.0) + shares
                cost_basis[tic] = cost_basis.get(tic, 0.0) + amount
            else:
                pos = position.get(tic, 0.0)
                cb = cost_basis.get(tic, 0.0)
                if pos > 0:
                    avg_cost = cb / pos
                    if price > avg_cost:
                        wins += 1
                    else:
                        losses += 1
                    # Reduce proportionally
                    ratio = shares / pos
                    position[tic] = max(0.0, pos - shares)
                    cost_basis[tic] = max(0.0, cb - cb * ratio)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        n_buy = n_sell = 0
        total_amount = 0.0
        avg_amount = 0.0
    else:
        n_buy = len(trades_df[trades_df["action"] == "BUY"])
        n_sell = len(trades_df[trades_df["action"] == "SELL"])
        total_amount = trades_df["amount"].sum()
        avg_amount = trades_df["amount"].mean()
    total_trades = n_buy + n_sell
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    summary: dict[str, Any] = {
        "total_trades": int(total_trades),
        "buy_count": int(n_buy),
        "sell_count": int(n_sell),
        "total_trade_amount": round(float(total_amount), 2),
        "avg_trade_amount": round(float(avg_amount), 2),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": round(float(win_rate), 4),
    }
    return trades_df, summary


def _trade_rows_html(trades_df: pd.DataFrame) -> str:
    if trades_df.empty:
        return "<p>No trades recorded.</p>"
    rows = ""
    for _, row in trades_df.iterrows():
        rows += (
            f"<tr><td>{row['date']}</td><td>{row['tic']}</td>"
            f"<td>{row['action']}</td><td>{row['shares']}</td>"
            f"<td>{row['price']}</td><td>{row['amount']}</td></tr>"
        )
    return (
        '<table border="1" cellpadding="4">'
        '<thead><tr><th>Date</th><th>Ticker</th><th>Action</th>'
        '<th>Shares</th><th>Price</th><th>Amount</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def compute_sharpe(
    daily_returns: pd.Series, risk_free_rate: float = 0.02
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    daily_returns:
        Daily portfolio return series.
    risk_free_rate:
        Annual risk-free rate (converted to daily internally).
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_cagr(daily_returns: pd.Series) -> float:
    """Compound Annual Growth Rate.

    Parameters
    ----------
    daily_returns:
        Daily portfolio return series.
    """
    n = len(daily_returns)
    if n == 0:
        return 0.0
    total = (1.0 + daily_returns).prod()
    years = n / TRADING_DAYS_PER_YEAR
    if years == 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def compute_max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum drawdown (negative value, e.g. -0.15 = −15 %).

    Parameters
    ----------
    daily_returns:
        Daily portfolio return series.
    """
    cum = (1.0 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min()) if len(drawdown) > 0 else 0.0


@dataclass
class BacktestReport:
    """Summary of backtest performance metrics and portfolio value series."""

    sharpe: float
    cagr: float
    max_drawdown: float
    total_return: float
    portfolio_values: pd.Series
    risk_free_rate: float = 0.02
    extra: dict[str, Any] = field(default_factory=dict)
    trade_summary: dict[str, Any] = field(default_factory=dict)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_portfolio_values(
        cls,
        portfolio_values: pd.Series,
        benchmark: pd.Series | None = None,
        risk_free_rate: float = 0.02,
    ) -> "BacktestReport":
        """Build report from a portfolio-value time series.

        Parameters
        ----------
        portfolio_values:
            Time-indexed Series of portfolio value (absolute RMB amount).
        benchmark:
            Optional benchmark value series for comparison (unused for now).
        risk_free_rate:
            Annual risk-free rate.
        """
        daily_returns = portfolio_values.pct_change().dropna()
        sharpe = compute_sharpe(daily_returns, risk_free_rate=risk_free_rate)
        cagr = compute_cagr(daily_returns)
        mdd = compute_max_drawdown(daily_returns)
        start_val = portfolio_values.iloc[0]
        end_val = portfolio_values.iloc[-1]
        total_return = float((end_val - start_val) / start_val) if start_val != 0 else 0.0

        extra: dict[str, Any] = {}
        if benchmark is not None:
            bench_returns = benchmark.pct_change().dropna()
            extra["benchmark_cagr"] = compute_cagr(bench_returns)
            extra["benchmark_sharpe"] = compute_sharpe(bench_returns, risk_free_rate)

        return cls(
            sharpe=sharpe,
            cagr=cagr,
            max_drawdown=mdd,
            total_return=total_return,
            portfolio_values=portfolio_values,
            risk_free_rate=risk_free_rate,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return scalar metrics as a flat dict."""
        result: dict[str, Any] = {
            "sharpe": self.sharpe,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
        }
        result.update(self.extra)
        result.update(self.trade_summary)
        return result

    def save_csv(self, path: Path) -> None:
        """Save scalar metrics to a CSV with columns [metric, value]."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"metric": k, "value": v} for k, v in self.to_dict().items()]
        pd.DataFrame(rows).to_csv(path, index=False)

    def save_html(self, path: Path) -> None:
        """Save an interactive HTML report using pyecharts.

        Falls back to plain HTML if pyecharts is not installed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._save_html_pyecharts(path)
        except ImportError:
            self._save_html_plain(path)

    def _save_html_pyecharts(self, path: Path) -> None:
        from pyecharts import options as opts  # type: ignore[import]
        from pyecharts.charts import Line, Page  # type: ignore[import]

        dates = (
            self.portfolio_values.index.strftime("%Y-%m-%d").tolist()
            if hasattr(self.portfolio_values.index, "strftime")
            else [str(d) for d in self.portfolio_values.index]
        )
        values = [round(float(v), 2) for v in self.portfolio_values.values]

        metrics = self.to_dict()
        trade_sub = (
            f" | Trades: {metrics.get('total_trades', 0)} "
            f"(Win {metrics.get('win_rate', 0):.1%})"
            if "total_trades" in metrics
            else ""
        )
        subtitle = (
            f"Sharpe: {metrics['sharpe']:.3f} | "
            f"CAGR: {metrics['cagr']:.2%} | "
            f"MDD: {metrics['max_drawdown']:.2%} | "
            f"Total Return: {metrics['total_return']:.2%}"
            f"{trade_sub}"
        )

        line = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "Portfolio Value",
                values,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Backtest Report", subtitle=subtitle
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(type_="category", name="Date"),
                yaxis_opts=opts.AxisOpts(type_="value", name="Value (¥)"),
                datazoom_opts=[opts.DataZoomOpts()],
            )
        )

        page = Page()
        page.add(line)
        page.render(str(path))

    def _save_html_plain(self, path: Path) -> None:
        """Minimal HTML fallback when pyecharts is unavailable."""
        metrics = self.to_dict()
        rows_html = "".join(
            f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" if isinstance(v, float)
            else f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in metrics.items()
        )
        trade_table = _trade_rows_html(self.trades)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest Report</title></head>
<body>
<h1>Backtest Report</h1>
<h2>Performance Metrics</h2>
<table border="1" cellpadding="6">
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<h2>Trade History</h2>
{trade_table}
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
