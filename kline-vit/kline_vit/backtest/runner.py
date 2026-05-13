import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from kline_vit.data.db_reader import DBReader

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    benchmark_return: float
    excess_return: float


class BacktestRunner:
    """Runs Backtrader backtest using ViT model signals."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.bt_cfg = config.get("backtest", {})

    def run(self, inference_engine, db_path: str) -> BacktestReport:
        import backtrader as bt
        from kline_vit.backtest.strategy import _make_bt_strategy
        KlineSignalStrategy = _make_bt_strategy()
        start_date = self.bt_cfg.get("start_date", "2025-01-01")
        end_date = self.bt_cfg.get("end_date", "2026-05-07")
        initial_cash = float(self.bt_cfg.get("initial_cash", 1_000_000))
        commission = float(self.bt_cfg.get("commission", 0.001))
        signal_threshold = float(self.bt_cfg.get("signal_threshold", 0.6))
        stop_loss_pct = float(self.bt_cfg.get("stop_loss_pct", 0.08))
        max_pos_pct = float(self.bt_cfg.get("max_position_pct", 0.20))
        image_dir = self.config.get("data", {}).get("image_dir", "data/images")

        reader = DBReader(db_path)
        codes = reader.get_all_codes()

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=int(max_pos_pct * 100))

        # Add data feeds
        feeds_added = 0
        for code in codes[:10]:  # limit to first 10 stocks for backtest
            df = reader.get_daily_data(code, end_date, n_days=10000)
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            if len(df) < 5:
                continue
            df["Date"] = pd.to_datetime(df["date"])
            df = df.set_index("Date")
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            df["OpenInterest"] = 0
            feed = bt.feeds.PandasData(dataname=df)
            feed._name = code
            cerebro.adddata(feed, name=code)
            feeds_added += 1

        if feeds_added == 0:
            return BacktestReport(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

        cerebro.addstrategy(
            KlineSignalStrategy,
            inference_engine=inference_engine,
            image_dir=str(Path(image_dir) / "test"),
            signal_threshold=signal_threshold,
            stop_loss_pct=stop_loss_pct,
        )

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        results = cerebro.run()
        strat = results[0]

        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()

        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash

        # Annualize return
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            years = max((end_dt - start_dt).days / 365.25, 0.01)
            annual_return = (1 + total_return) ** (1 / years) - 1
        except Exception:
            annual_return = total_return

        sharpe = sharpe_analysis.get("sharperatio") or 0.0
        if sharpe is None:
            sharpe = 0.0

        max_dd = drawdown_analysis.get("max", {}).get("drawdown", 0.0) / 100.0

        # Trade stats
        total_trades = int(trades_analysis.get("total", {}).get("closed", 0))
        won = int(trades_analysis.get("won", {}).get("total", 0))
        lost = int(trades_analysis.get("lost", {}).get("total", 0))
        win_rate = won / max(total_trades, 1)

        gross_won = float(trades_analysis.get("won", {}).get("pnl", {}).get("total", 0.0))
        gross_lost = abs(float(trades_analysis.get("lost", {}).get("pnl", {}).get("total", 0.0)))
        profit_factor = gross_won / max(gross_lost, 1e-9)

        # Benchmark: buy-and-hold first stock
        benchmark_return = self._compute_benchmark(reader, codes[0], start_date, end_date)
        excess_return = annual_return - benchmark_return

        report = BacktestReport(
            annual_return=float(annual_return),
            max_drawdown=float(max_dd),
            sharpe_ratio=float(sharpe),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            total_trades=total_trades,
            benchmark_return=float(benchmark_return),
            excess_return=float(excess_return),
        )

        self._save_report(report)
        return report

    def _compute_benchmark(
        self, reader: DBReader, code: str, start_date: str, end_date: str
    ) -> float:
        df = reader.get_daily_data(code, end_date, n_days=10000)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if len(df) < 2:
            return 0.0
        total_return = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"]
        try:
            years = max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25, 0.01)
            return float((1 + total_return) ** (1 / years) - 1)
        except Exception:
            return float(total_return)

    def _save_report(self, report: BacktestReport) -> None:
        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        report_dict = {
            "annual_return": report.annual_return,
            "max_drawdown": report.max_drawdown,
            "sharpe_ratio": report.sharpe_ratio,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "total_trades": report.total_trades,
            "benchmark_return": report.benchmark_return,
            "excess_return": report.excess_return,
        }
        with open(out_dir / "backtest_report.json", "w") as f:
            json.dump(report_dict, f, indent=2)
