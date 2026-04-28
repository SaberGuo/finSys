"""Backtest metrics computation and report generation (T032)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252


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
        subtitle = (
            f"Sharpe: {metrics['sharpe']:.3f} | "
            f"CAGR: {metrics['cagr']:.2%} | "
            f"MDD: {metrics['max_drawdown']:.2%} | "
            f"Total Return: {metrics['total_return']:.2%}"
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
            f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in metrics.items()
        )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest Report</title></head>
<body>
<h1>Backtest Report</h1>
<table border="1" cellpadding="6">
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
