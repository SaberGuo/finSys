"""动态选股+RL交易回测模块。

每日流程：
1. 开盘前（9:00）执行选股，获取当日候选股票池
2. 盘中（9:30-15:00）使用训练好的RL模型在候选池上进行交易
3. 收盘后记录当日持仓、收益等信息
4. 次日重复上述流程

这个模块实现了"选股+交易"双阶段的动态回测。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finquant.config.settings import AppConfig
from finquant.data.preprocessor import preprocess_5min_data
from finquant.data.sources.db_5min import Db5MinDataSource
from finquant.features.technical import compute_indicators
from finquant.selection import SelectionPipeline
from finquant.training.env import build_env
from finquant.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DailyTradingRecord:
    """单日交易记录。"""

    date: str
    market_state: str
    selected_tickers: list[str]
    selection_scores: dict[str, float]

    # 交易统计
    start_value: float
    end_value: float
    daily_return: float

    # 持仓信息
    positions: dict[str, int]  # {ticker: shares}
    cash: float

    # 交易动作
    trades: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "date": self.date,
            "market_state": self.market_state,
            "selected_tickers": self.selected_tickers,
            "selection_scores": self.selection_scores,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "daily_return": self.daily_return,
            "positions": self.positions,
            "cash": self.cash,
            "trades": self.trades,
        }


@dataclass
class DynamicBacktestReport:
    """动态回测报告。"""

    start_date: str
    end_date: str
    initial_amount: float
    final_value: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float

    # 每日记录
    daily_records: list[DailyTradingRecord]

    # 汇总统计
    total_trades: int
    win_rate: float
    avg_daily_return: float
    volatility: float

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_amount": self.initial_amount,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_daily_return": self.avg_daily_return,
            "volatility": self.volatility,
            "daily_records": [r.to_dict() for r in self.daily_records],
        }

    def save(self, output_path: Path) -> None:
        """保存报告到JSON文件。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {output_path}")


class DynamicBacktester:
    """动态选股+RL交易回测器。

    每日流程：
    1. 开盘前执行选股
    2. 使用RL模型在候选池上交易
    3. 记录当日结果

    Parameters
    ----------
    config : AppConfig
        配置对象
    model_path : Path
        训练好的RL模型路径
    selection_pipeline : SelectionPipeline
        选股流水线
    db_path : Path
        5分钟数据库路径
    """

    def __init__(
        self,
        config: AppConfig,
        model_path: Path,
        selection_pipeline: SelectionPipeline,
        db_path: Path | None = None,
    ):
        self.config = config
        self.model_path = model_path
        self.selection_pipeline = selection_pipeline
        self.db_path = db_path or Path("data/processed/zz500_data.db")

        # 加载RL模型
        self.model = self._load_model()

        # 初始化账户
        self.initial_amount = config.environment.initial_amount
        self.cash = self.initial_amount
        self.positions: dict[str, int] = {}  # {ticker: shares}
        self.account_value_history: list[float] = []

    def _load_model(self):
        """加载训练好的RL模型。"""
        from stable_baselines3 import PPO, SAC, TD3

        algo = self.config.training.algorithm.lower()
        model_class = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo]

        logger.info(f"Loading {algo.upper()} model from {self.model_path}")
        return model_class.load(str(self.model_path))

    def run(
        self,
        market_df: pd.DataFrame,
        index_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> DynamicBacktestReport:
        """运行动态回测。

        Parameters
        ----------
        market_df : pd.DataFrame
            市场数据（日频），用于选股
        index_df : pd.DataFrame
            指数数据（日频），用于市场状态分类
        start_date : str
            回测开始日期 (YYYY-MM-DD)
        end_date : str
            回测结束日期 (YYYY-MM-DD)

        Returns
        -------
        DynamicBacktestReport
            回测报告
        """
        # 获取交易日列表
        trading_days = sorted(
            [d for d in market_df["date"].unique() if start_date <= d <= end_date]
        )

        logger.info(f"Running dynamic backtest from {start_date} to {end_date}")
        logger.info(f"Total trading days: {len(trading_days)}")

        daily_records = []

        for date in trading_days:
            try:
                record = self._trade_single_day(market_df, index_df, date)
                daily_records.append(record)

                logger.info(
                    f"{date}: {record.market_state}, "
                    f"{len(record.selected_tickers)} stocks, "
                    f"value={record.end_value:.2f}, "
                    f"return={record.daily_return:.2%}, "
                    f"trades={len(record.trades)}"
                )
            except Exception as e:
                logger.error(f"Error on {date}: {e}")
                continue

        # 生成回测报告
        report = self._generate_report(
            start_date=start_date,
            end_date=end_date,
            daily_records=daily_records,
        )

        return report

    def _trade_single_day(
        self,
        market_df: pd.DataFrame,
        index_df: pd.DataFrame,
        date: str,
    ) -> DailyTradingRecord:
        """执行单日交易。

        流程：
        1. 开盘前选股
        2. 获取当日5分钟数据
        3. 使用RL模型执行交易
        4. 记录结果
        """
        # 1. 开盘前选股（9:00）
        logger.info(f"\n{'='*80}")
        logger.info(f"[{date} 09:00] 开盘前选股开始")
        logger.info(f"{'='*80}")

        selection_result = self.selection_pipeline.run(market_df, index_df, date)

        # 打印详细选股信息
        logger.info(f"[{date} 09:00] 市场状态: {selection_result.market_state.value}")
        logger.info(f"[{date} 09:00] 活跃因子: {', '.join(selection_result.active_factors)}")

        # 打印因子权重
        logger.info(f"[{date} 09:00] 因子权重:")
        for factor_id, weight in selection_result.factor_weights.items():
            logger.info(f"  - {factor_id}: {weight:.4f}")

        # 打印指数指标
        logger.info(f"[{date} 09:00] 指数指标:")
        for metric, value in selection_result.index_metrics.items():
            logger.info(f"  - {metric}: {value:.4f}")

        selected_tickers = selection_result.selected_tickers
        if not selected_tickers:
            logger.warning(f"[{date} 09:00] 未选出任何股票，跳过交易")
            if selection_result.exclusion_reasons:
                logger.info(f"[{date} 09:00] 排除原因统计:")
                exclusion_stats = {}
                for reason in selection_result.exclusion_reasons.values():
                    exclusion_stats[reason] = exclusion_stats.get(reason, 0) + 1
                for reason, count in exclusion_stats.items():
                    logger.info(f"  - {reason}: {count}只")
            # 返回空记录
            return DailyTradingRecord(
                date=date,
                market_state=selection_result.market_state.value,
                selected_tickers=[],
                selection_scores={},
                start_value=self._calculate_account_value(date, market_df),
                end_value=self._calculate_account_value(date, market_df),
                daily_return=0.0,
                positions=self.positions.copy(),
                cash=self.cash,
                trades=[],
            )

        # 打印选中的股票及其得分
        logger.info(f"[{date} 09:00] 选中股票 ({len(selected_tickers)}只):")
        sorted_tickers = sorted(
            selected_tickers,
            key=lambda t: selection_result.scores.get(t, 0),
            reverse=True
        )
        for i, ticker in enumerate(sorted_tickers, 1):
            score = selection_result.scores.get(ticker, 0)
            logger.info(f"  {i}. {ticker}: {score:.4f}")

        # 打印排除信息
        if selection_result.exclusion_reasons:
            exclusion_stats = {}
            for reason in selection_result.exclusion_reasons.values():
                exclusion_stats[reason] = exclusion_stats.get(reason, 0) + 1
            logger.info(f"[{date} 09:00] 排除股票统计:")
            for reason, count in exclusion_stats.items():
                logger.info(f"  - {reason}: {count}只")

        # 2. 获取当日5分钟数据（9:30-15:00）
        logger.info(f"[{date} 09:30] 开盘，开始获取盘中数据")
        intraday_df = self._fetch_intraday_data(selected_tickers, date)

        if intraday_df.empty:
            logger.warning(f"[{date}] 无盘中数据，跳过交易")
            return DailyTradingRecord(
                date=date,
                market_state=selection_result.market_state.value,
                selected_tickers=selected_tickers,
                selection_scores=selection_result.scores,
                start_value=self._calculate_account_value(date, market_df),
                end_value=self._calculate_account_value(date, market_df),
                daily_return=0.0,
                positions=self.positions.copy(),
                cash=self.cash,
                trades=[],
            )

        # 3. 记录开盘时账户价值
        start_value = self._calculate_account_value(date, market_df)
        logger.info(f"[{date} 09:30] 账户初始价值: ¥{start_value:,.2f} (现金: ¥{self.cash:,.2f})")

        # 4. 使用RL模型执行交易（含止损止盈）
        logger.info(f"[{date} 09:30] 开始RL交易决策")
        trades = self._execute_rl_trading(intraday_df, selected_tickers, date)

        # 5. 记录收盘时账户价值
        end_value = self._calculate_account_value(date, market_df)
        daily_return = (end_value - start_value) / start_value if start_value > 0 else 0.0

        logger.info(f"[{date} 15:00] 收盘，账户最终价值: ¥{end_value:,.2f}")
        logger.info(f"[{date}] 当日收益率: {daily_return:.2%}")
        logger.info(f"{'='*80}\n")

        return DailyTradingRecord(
            date=date,
            market_state=selection_result.market_state.value,
            selected_tickers=selected_tickers,
            selection_scores=selection_result.scores,
            start_value=start_value,
            end_value=end_value,
            daily_return=daily_return,
            positions=self.positions.copy(),
            cash=self.cash,
            trades=trades,
        )

    def _fetch_intraday_data(
        self,
        tickers: list[str],
        date: str,
    ) -> pd.DataFrame:
        """获取当日5分钟数据。"""
        source = Db5MinDataSource(db_path=str(self.db_path))

        try:
            raw_df = source.download(
                symbols=tickers,
                start_date=date,
                end_date=date,
            )

            if raw_df.empty:
                return pd.DataFrame()

            # 预处理
            processed_df = preprocess_5min_data(raw_df)

            # 计算技术指标
            enriched_df = compute_indicators(processed_df, self.config.indicators)

            return enriched_df
        except Exception as e:
            logger.error(f"Failed to fetch intraday data for {date}: {e}")
            return pd.DataFrame()

    def _execute_rl_trading(
        self,
        intraday_df: pd.DataFrame,
        selected_tickers: list[str],
        date: str,
    ) -> list[dict[str, Any]]:
        """使用RL模型执行交易，包含止损止盈监控。

        止损止盈策略：
        - 止损点：入场价格 * 0.95 (-5%)
        - 止盈点：入场价格 * 1.10 (+10%)
        - RL模型决策是否触发止损止盈

        Note: 模型训练时使用固定的stock_dim，回测时需要匹配该维度。
        如果实际股票数量少于模型期望，会跳过RL交易。
        """
        trades = []

        # 获取实际有数据的股票列表
        actual_tickers = intraday_df["tic"].unique().tolist()

        # 如果某些选中的股票没有5分钟数据，记录警告
        missing_tickers = set(selected_tickers) - set(actual_tickers)
        if missing_tickers:
            logger.warning(
                f"[{date}] 以下股票无5分钟数据，将被排除: {', '.join(missing_tickers)}"
            )

        if not actual_tickers:
            logger.warning(f"[{date}] 无可交易股票")
            return trades

        # 检查模型期望的观测空间维度
        expected_obs_dim = self.model.observation_space.shape[0]
        # 反推模型期望的stock_dim（假设num_indicators已知）
        # obs_dim = 1 + stock_dim * (2 + num_indicators)
        num_indicators = len(self.config.indicators)
        expected_stock_dim = (expected_obs_dim - 1) // (2 + num_indicators)

        if len(actual_tickers) != expected_stock_dim:
            logger.warning(
                f"[{date}] 股票数量不匹配: 实际{len(actual_tickers)}只, "
                f"模型期望{expected_stock_dim}只, 跳过RL交易"
            )
            logger.info(
                f"[{date}] 提示: 请使用 top_k={expected_stock_dim} 的选股配置，"
                f"或使用匹配的stock_dim重新训练模型"
            )
            return trades

        # 跟踪每只股票的入场价格和止损止盈点
        entry_prices: dict[str, float] = {}
        stop_loss_prices: dict[str, float] = {}
        take_profit_prices: dict[str, float] = {}

        try:
            # 构建交易环境（使用实际股票数量）
            env = build_env(
                intraday_df,
                stock_dim=len(actual_tickers),
                initial_amount=self.cash,
                hmax=self.config.environment.hmax,
                buy_cost_pct=self.config.environment.buy_cost_pct,
                sell_cost_pct=self.config.environment.sell_cost_pct,
                reward_scaling=self.config.environment.reward_scaling,
                indicators=self.config.indicators,
                fusion_indicators=[],
            )

            # 运行一个episode
            obs, _ = env.reset()
            done = False
            step = 0

            while not done and step < 1000:  # 最多1000步
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                # 获取当前状态（价格和持仓）
                if hasattr(env, "state_memory") and len(env.state_memory) > 0:
                    current_state = env.state_memory[-1]
                    # state格式: [cash, price1, holdings1, price2, holdings2, ...]

                    # 检查每只股票的交易动作和止损止盈
                    for i, ticker in enumerate(actual_tickers):
                        price_idx = 1 + i * 2
                        holdings_idx = 2 + i * 2

                        if price_idx < len(current_state) and holdings_idx < len(current_state):
                            current_price = float(current_state[price_idx])
                            current_holdings = int(current_state[holdings_idx])

                            # 检测买入动作（持仓从0变为正）
                            if ticker not in entry_prices and current_holdings > 0:
                                entry_prices[ticker] = current_price
                                stop_loss_prices[ticker] = current_price * 0.95  # -5%
                                take_profit_prices[ticker] = current_price * 1.10  # +10%

                                logger.info(
                                    f"  [{ticker}] 买入 {current_holdings}股 @ ¥{current_price:.2f}, "
                                    f"止损: ¥{stop_loss_prices[ticker]:.2f}, "
                                    f"止盈: ¥{take_profit_prices[ticker]:.2f}"
                                )

                            # 检测卖出动作（持仓变为0）
                            elif ticker in entry_prices and current_holdings == 0:
                                entry_price = entry_prices[ticker]
                                pnl_pct = (current_price - entry_price) / entry_price

                                # 判断是否触发止损或止盈
                                if current_price <= stop_loss_prices[ticker]:
                                    reason = "止损"
                                elif current_price >= take_profit_prices[ticker]:
                                    reason = "止盈"
                                else:
                                    reason = "RL决策"

                                logger.info(
                                    f"  [{ticker}] 卖出 @ ¥{current_price:.2f} ({reason}), "
                                    f"盈亏: {pnl_pct:.2%}"
                                )

                                # 清除跟踪
                                del entry_prices[ticker]
                                del stop_loss_prices[ticker]
                                del take_profit_prices[ticker]

                            # 监控持仓的止损止盈状态
                            elif ticker in entry_prices and current_holdings > 0:
                                entry_price = entry_prices[ticker]
                                pnl_pct = (current_price - entry_price) / entry_price

                                # 检查是否接近止损或止盈点
                                if current_price <= stop_loss_prices[ticker] * 1.01:  # 接近止损（1%容差）
                                    logger.debug(
                                        f"  [{ticker}] 接近止损点: 当前¥{current_price:.2f}, "
                                        f"止损¥{stop_loss_prices[ticker]:.2f}, 浮亏{pnl_pct:.2%}"
                                    )
                                elif current_price >= take_profit_prices[ticker] * 0.99:  # 接近止盈（1%容差）
                                    logger.debug(
                                        f"  [{ticker}] 接近止盈点: 当前¥{current_price:.2f}, "
                                        f"止盈¥{take_profit_prices[ticker]:.2f}, 浮盈{pnl_pct:.2%}"
                                    )

                # 记录交易动作
                if hasattr(env, "actions_memory") and len(env.actions_memory) > 0:
                    last_action = env.actions_memory[-1]
                    trades.append({
                        "step": step,
                        "action": last_action.tolist() if isinstance(last_action, np.ndarray) else last_action,
                        "reward": float(reward),
                    })

            # 更新账户状态（从环境中获取最终状态）
            if hasattr(env, "state_memory") and len(env.state_memory) > 0:
                final_state = env.state_memory[-1]
                # final_state格式: [cash, price1, holdings1, price2, holdings2, ...]
                self.cash = float(final_state[0])

                # 更新持仓
                self.positions = {}
                for i, ticker in enumerate(actual_tickers):
                    holdings_idx = 2 + i * 2
                    if holdings_idx < len(final_state):
                        shares = int(final_state[holdings_idx])
                        if shares > 0:
                            self.positions[ticker] = shares

                # 打印最终持仓
                if self.positions:
                    logger.info(f"[{date} 15:00] 收盘持仓:")
                    for ticker, shares in self.positions.items():
                        if ticker in entry_prices:
                            entry_price = entry_prices[ticker]
                            price_idx = 1 + actual_tickers.index(ticker) * 2
                            current_price = float(final_state[price_idx])
                            pnl_pct = (current_price - entry_price) / entry_price
                            logger.info(
                                f"  - {ticker}: {shares}股 @ ¥{current_price:.2f}, "
                                f"浮动盈亏: {pnl_pct:.2%}"
                            )
                else:
                    logger.info(f"[{date} 15:00] 收盘无持仓")

        except Exception as e:
            logger.error(f"RL trading execution failed: {e}")

        return trades

    def _calculate_account_value(
        self,
        date: str,
        market_df: pd.DataFrame,
    ) -> float:
        """计算账户总价值。"""
        total_value = self.cash

        # 获取当日收盘价
        date_df = market_df[market_df["date"] == date]

        for ticker, shares in self.positions.items():
            ticker_df = date_df[date_df["tic"] == ticker]
            if not ticker_df.empty:
                close_price = ticker_df.iloc[0]["close"]
                total_value += shares * close_price

        return total_value

    def _generate_report(
        self,
        start_date: str,
        end_date: str,
        daily_records: list[DailyTradingRecord],
    ) -> DynamicBacktestReport:
        """生成回测报告。"""
        if not daily_records:
            raise ValueError("No daily records to generate report")

        # 计算总收益
        final_value = daily_records[-1].end_value
        total_return = (final_value - self.initial_amount) / self.initial_amount

        # 计算年化收益
        days = len(daily_records)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0.0

        # 计算最大回撤
        values = [r.end_value for r in daily_records]
        max_drawdown = self._calculate_max_drawdown(values)

        # 计算夏普比率
        daily_returns = [r.daily_return for r in daily_records]
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)

        # 计算交易统计
        total_trades = sum(len(r.trades) for r in daily_records)

        # 计算胜率
        positive_days = sum(1 for r in daily_records if r.daily_return > 0)
        win_rate = positive_days / len(daily_records) if daily_records else 0.0

        # 计算平均日收益和波动率
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0.0
        volatility = np.std(daily_returns) if daily_returns else 0.0

        return DynamicBacktestReport(
            start_date=start_date,
            end_date=end_date,
            initial_amount=self.initial_amount,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            daily_records=daily_records,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_daily_return=avg_daily_return,
            volatility=volatility,
        )

    @staticmethod
    def _calculate_max_drawdown(values: list[float]) -> float:
        """计算最大回撤。"""
        if not values:
            return 0.0

        peak = values[0]
        max_dd = 0.0

        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def _calculate_sharpe_ratio(
        daily_returns: list[float],
        risk_free_rate: float = 0.02,
    ) -> float:
        """计算夏普比率。"""
        if not daily_returns or len(daily_returns) < 2:
            return 0.0

        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)

        if std_return == 0:
            return 0.0

        # 年化
        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252)

        return float(sharpe)


@dataclass
class Trade:
    """单笔交易记录（适用于简单回测）。"""

    ticker: str
    date: str
    entry_time: str
    entry_price: float
    shares: int
    exit_time: str | None = None
    exit_price: float | None = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_reason: str = ""  # stop_loss / take_profit / close
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "ticker": self.ticker,
            "date": self.date,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


class SimpleBacktester:
    """简单回测器：选股 + 止损止盈策略。

    每日流程：
    1. 开盘前（9:00）执行选股
    2. 开盘（9:35）等权重买入所有选中股票
    3. 逐根5分钟K线监控止损(-5%)和止盈(+10%)
    4. 触发止损/止盈时立即卖出
    5. 收盘前(15:00)未平仓的股票按收盘价强制平仓
    6. 记录每笔交易的详细盈亏情况

    Parameters
    ----------
    config : AppConfig
        配置对象
    selection_pipeline : SelectionPipeline
        选股流水线
    stop_loss_pct : float
        止损百分比（默认-0.05 = -5%）
    take_profit_pct : float
        止盈百分比（默认0.10 = +10%）
    db_path : Path | None
        5分钟数据库路径
    """

    def __init__(
        self,
        config: AppConfig,
        selection_pipeline: SelectionPipeline,
        stop_loss_pct: float = -0.05,
        take_profit_pct: float = 0.10,
        db_path: Path | None = None,
    ):
        self.config = config
        self.selection_pipeline = selection_pipeline
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.db_path = db_path or Path("data/processed/zz500_data.db")

        # 初始化账户
        self.initial_amount = config.environment.initial_amount
        self.cash = self.initial_amount
        self.positions: dict[str, int] = {}  # {ticker: shares}
        self.trade_history: list[Trade] = []  # 所有交易记录

    def run(
        self,
        market_df: pd.DataFrame,
        index_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> DynamicBacktestReport:
        """运行简单回测。

        Parameters
        ----------
        market_df : pd.DataFrame
            市场数据（日频），用于选股
        index_df : pd.DataFrame
            指数数据（日频），用于市场状态分类
        start_date : str
            回测开始日期 (YYYY-MM-DD)
        end_date : str
            回测结束日期 (YYYY-MM-DD)

        Returns
        -------
        DynamicBacktestReport
            回测报告
        """
        # 获取交易日列表
        trading_days = sorted(
            [d for d in market_df["date"].unique() if start_date <= d <= end_date]
        )

        logger.info(f"Running simple backtest from {start_date} to {end_date}")
        logger.info(f"Total trading days: {len(trading_days)}")
        logger.info(
            f"Stop loss: {self.stop_loss_pct:.1%}, Take profit: {self.take_profit_pct:.1%}"
        )

        daily_records = []

        for date in trading_days:
            try:
                record = self._trade_single_day(market_df, index_df, date)
                daily_records.append(record)

                logger.info(
                    f"{date}: {record.market_state}, "
                    f"{len(record.selected_tickers)} stocks, "
                    f"value={record.end_value:.2f}, "
                    f"return={record.daily_return:.2%}, "
                    f"trades={len(record.trades)}"
                )
            except Exception as e:
                logger.error(f"Error on {date}: {e}")
                continue

        # 生成回测报告
        report = self._generate_report(
            start_date=start_date,
            end_date=end_date,
            daily_records=daily_records,
        )

        return report

    def _trade_single_day(
        self,
        market_df: pd.DataFrame,
        index_df: pd.DataFrame,
        date: str,
    ) -> DailyTradingRecord:
        """执行单日简单交易。"""
        # 1. 开盘前选股（9:00）
        logger.info(f"\n{'='*80}")
        logger.info(f"[{date} 09:00] 开盘前选股开始 (简单策略)")
        logger.info(f"{'='*80}")

        selection_result = self.selection_pipeline.run(market_df, index_df, date)

        # 打印详细选股信息
        logger.info(f"[{date} 09:00] 市场状态: {selection_result.market_state.value}")
        logger.info(
            f"[{date} 09:00] 活跃因子: {', '.join(selection_result.active_factors)}"
        )
        logger.info(f"[{date} 09:00] 因子权重:")
        for factor_id, weight in selection_result.factor_weights.items():
            logger.info(f"  - {factor_id}: {weight:.4f}")

        selected_tickers = selection_result.selected_tickers
        if not selected_tickers:
            logger.warning(f"[{date} 09:00] 未选出任何股票，跳过交易")
            return DailyTradingRecord(
                date=date,
                market_state=selection_result.market_state.value,
                selected_tickers=[],
                selection_scores={},
                start_value=self._calculate_account_value(date, market_df),
                end_value=self._calculate_account_value(date, market_df),
                daily_return=0.0,
                positions=self.positions.copy(),
                cash=self.cash,
                trades=[],
            )

        # 打印选中的股票及其得分
        logger.info(f"[{date} 09:00] 选中股票 ({len(selected_tickers)}只):")
        sorted_tickers = sorted(
            selected_tickers,
            key=lambda t: selection_result.scores.get(t, 0),
            reverse=True,
        )
        for i, ticker in enumerate(sorted_tickers, 1):
            score = selection_result.scores.get(ticker, 0)
            logger.info(f"  {i}. {ticker}: {score:.4f}")

        # 2. 获取当日5分钟数据（9:30-15:00）
        logger.info(f"[{date} 09:30] 开盘，开始获取盘中数据")
        intraday_df = self._fetch_intraday_data(selected_tickers, date)

        if intraday_df.empty:
            logger.warning(f"[{date}] 无盘中数据，跳过交易")
            return DailyTradingRecord(
                date=date,
                market_state=selection_result.market_state.value,
                selected_tickers=selected_tickers,
                selection_scores=selection_result.scores,
                start_value=self._calculate_account_value(date, market_df),
                end_value=self._calculate_account_value(date, market_df),
                daily_return=0.0,
                positions=self.positions.copy(),
                cash=self.cash,
                trades=[],
            )

        # 3. 记录开盘时账户价值
        start_value = self._calculate_account_value(date, market_df)
        logger.info(
            f"[{date} 09:30] 账户初始价值: ¥{start_value:,.2f} (现金: ¥{self.cash:,.2f})"
        )

        # 4. 执行简单交易（买入 + 止损止盈监控）
        logger.info(f"[{date} 09:30] 开始简单交易决策")
        trades = self._execute_simple_trading(
            intraday_df, selected_tickers, date
        )

        # 5. 记录收盘时账户价值
        end_value = self._calculate_account_value(date, market_df)
        daily_return = (end_value - start_value) / start_value if start_value > 0 else 0.0

        logger.info(f"[{date} 15:00] 收盘，账户最终价值: ¥{end_value:,.2f}")
        logger.info(f"[{date}] 当日收益率: {daily_return:.2%}")
        logger.info(f"{'='*80}\n")

        # 将 Trade 对象转换为 dict 列表
        trade_dicts = [t.to_dict() for t in trades]

        return DailyTradingRecord(
            date=date,
            market_state=selection_result.market_state.value,
            selected_tickers=selected_tickers,
            selection_scores=selection_result.scores,
            start_value=start_value,
            end_value=end_value,
            daily_return=daily_return,
            positions=self.positions.copy(),
            cash=self.cash,
            trades=trade_dicts,
        )

    def _fetch_intraday_data(
        self,
        tickers: list[str],
        date: str,
    ) -> pd.DataFrame:
        """获取当日5分钟数据。"""
        source = Db5MinDataSource(db_path=str(self.db_path))

        try:
            raw_df = source.download(
                symbols=tickers,
                start_date=date,
                end_date=date,
            )

            if raw_df.empty:
                return pd.DataFrame()

            # 预处理
            processed_df = preprocess_5min_data(raw_df)

            # 计算技术指标
            enriched_df = compute_indicators(processed_df, self.config.indicators)

            return enriched_df
        except Exception as e:
            logger.error(f"Failed to fetch intraday data for {date}: {e}")
            return pd.DataFrame()

    def _execute_simple_trading(
        self,
        intraday_df: pd.DataFrame,
        selected_tickers: list[str],
        date: str,
    ) -> list[Trade]:
        """执行简单交易策略：买入 + 止损止盈。

        策略：
        1. 等权重买入所有选中股票（第一根K线收盘价）
        2. 逐根K线监控，触及止损/止盈立即卖出
        3. 收盘前未平仓的，按收盘价强制平仓
        """
        trades: list[Trade] = []

        # 实际有数据的股票
        actual_tickers = intraday_df["tic"].unique().tolist()

        # 检测缺少5分钟数据的股票
        missing_tickers = set(selected_tickers) - set(actual_tickers)
        if missing_tickers:
            logger.warning(
                f"[{date}] 以下股票无5分钟数据，将被排除: {', '.join(missing_tickers)}"
            )

        if not actual_tickers:
            logger.warning(f"[{date}] 无可交易股票")
            return trades

        buy_cost_pct = self.config.environment.buy_cost_pct
        sell_cost_pct = self.config.environment.sell_cost_pct

        # 每只股票的买入资金（等权重分配）
        allocation_per_stock = self.cash / len(actual_tickers)

        # ---- 阶段1：买入 ----
        logger.info(f"[{date}] 买入 {len(actual_tickers)}只股票，每只约¥{allocation_per_stock:,.2f}")

        # 用于跟踪开仓信息的字典
        open_trades: dict[str, Trade] = {}

        for ticker in actual_tickers:
            ticker_df = intraday_df[intraday_df["tic"] == ticker].sort_values("time")
            if len(ticker_df) < 2:
                logger.warning(f"[{date}] {ticker} K线不足2根，跳过")
                continue

            # 第一根K线收盘价作为买入价
            entry_price = float(ticker_df.iloc[0]["close"])
            entry_time = str(ticker_df.iloc[0]["time"])

            if entry_price <= 0:
                logger.warning(f"[{date}] {ticker} 买入价无效({entry_price})，跳过")
                continue

            # 计算可买入数量
            max_shares = int(allocation_per_stock / (entry_price * (1 + buy_cost_pct)))

            if max_shares <= 0:
                logger.warning(
                    f"[{date}] {ticker} 资金不足，跳过 (价格{entry_price:.2f}，可用{self.cash:.2f})"
                )
                continue

            # 计算买入成本（含手续费）
            buy_value = max_shares * entry_price
            buy_cost = buy_value * buy_cost_pct
            total_cost = buy_value + buy_cost

            self.cash -= total_cost
            self.positions[ticker] = max_shares

            # 计算止损止盈价格
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)

            logger.info(
                f"  [{ticker}] 买入 {max_shares}股 @ ¥{entry_price:.2f} "
                f"(成本¥{total_cost:,.2f}), "
                f"止损: ¥{stop_loss:.2f}({self.stop_loss_pct:.1%}), "
                f"止盈: ¥{take_profit:.2f}({self.take_profit_pct:.1%})"
            )

            # 创建开仓交易记录
            trade = Trade(
                ticker=ticker,
                date=date,
                entry_time=entry_time,
                entry_price=entry_price,
                shares=max_shares,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            open_trades[ticker] = trade

        # ---- 阶段2：逐根K线监控止损止盈 ----
        logger.info(f"[{date}] 开始逐K线监控止损止盈...")

        for ticker in list(open_trades.keys()):
            if ticker not in self.positions:
                continue

            ticker_df = intraday_df[intraday_df["tic"] == ticker].sort_values("time")
            trade = open_trades[ticker]
            entry_price = trade.entry_price
            shares = trade.shares
            stop_loss = trade.stop_loss
            take_profit = trade.take_profit

            # 从第2根K线开始监控
            for idx in range(1, len(ticker_df)):
                k = ticker_df.iloc[idx]
                current_price = float(k["close"])
                current_time = str(k["time"])

                # 检查是否触发止损
                if current_price <= stop_loss:
                    # 执行卖出
                    sell_value = shares * current_price
                    sell_cost = sell_value * sell_cost_pct
                    net_sell_value = sell_value - sell_cost

                    self.cash += net_sell_value
                    del self.positions[ticker]

                    # 计算盈亏
                    buy_total_cost = shares * entry_price * (1 + buy_cost_pct)
                    pnl = net_sell_value - buy_total_cost
                    pnl_pct = pnl / buy_total_cost

                    trade.exit_time = current_time
                    trade.exit_price = current_price
                    trade.exit_reason = "stop_loss"
                    trade.pnl = pnl
                    trade.pnl_pct = pnl_pct

                    logger.info(
                        f"  [{ticker}] 卖出 {shares}股 @ ¥{current_price:.2f} "
                        f"(止损触发), 盈亏: ¥{pnl:,.2f}({pnl_pct:.2%})"
                    )
                    trades.append(trade)
                    break

                # 检查是否触发止盈
                if current_price >= take_profit:
                    # 执行卖出
                    sell_value = shares * current_price
                    sell_cost = sell_value * sell_cost_pct
                    net_sell_value = sell_value - sell_cost

                    self.cash += net_sell_value
                    del self.positions[ticker]

                    # 计算盈亏
                    buy_total_cost = shares * entry_price * (1 + buy_cost_pct)
                    pnl = net_sell_value - buy_total_cost
                    pnl_pct = pnl / buy_total_cost

                    trade.exit_time = current_time
                    trade.exit_price = current_price
                    trade.exit_reason = "take_profit"
                    trade.pnl = pnl
                    trade.pnl_pct = pnl_pct

                    logger.info(
                        f"  [{ticker}] 卖出 {shares}股 @ ¥{current_price:.2f} "
                        f"(止盈触发), 盈亏: ¥{pnl:,.2f}({pnl_pct:.2%})"
                    )
                    trades.append(trade)
                    break
            else:
                # 未触发止损止盈，收盘前平仓
                if ticker in self.positions:
                    last_k = ticker_df.iloc[-1]
                    current_price = float(last_k["close"])
                    current_time = str(last_k["time"])

                    # 执行卖出
                    sell_value = shares * current_price
                    sell_cost = sell_value * sell_cost_pct
                    net_sell_value = sell_value - sell_cost

                    self.cash += net_sell_value
                    del self.positions[ticker]

                    # 计算盈亏
                    buy_total_cost = shares * entry_price * (1 + buy_cost_pct)
                    pnl = net_sell_value - buy_total_cost
                    pnl_pct = pnl / buy_total_cost

                    # 判断收盘盈亏方向
                    if pnl >= 0:
                        exit_reason = "close_profit"
                    else:
                        exit_reason = "close_loss"

                    trade.exit_time = current_time
                    trade.exit_price = current_price
                    trade.exit_reason = exit_reason
                    trade.pnl = pnl
                    trade.pnl_pct = pnl_pct

                    logger.info(
                        f"  [{ticker}] 收盘平仓 {shares}股 @ ¥{current_price:.2f}, "
                        f"盈亏: ¥{pnl:,.2f}({pnl_pct:.2%})"
                    )
                    trades.append(trade)

        # ---- 阶段3：打印当日交易汇总 ----
        if trades:
            total_pnl = sum(t.pnl for t in trades)
            win_count = sum(1 for t in trades if t.pnl > 0)
            loss_count = sum(1 for t in trades if t.pnl <= 0)
            logger.info(
                f"[{date}] 当日交易汇总: "
                f"总盈亏¥{total_pnl:,.2f}, "
                f"盈利{win_count}笔, 亏损{loss_count}笔, "
                f"胜率{win_count/len(trades):.0%}"
            )
        else:
            logger.info(f"[{date}] 当日无交易")

        # 统计平仓原因
        reason_counts = {}
        for t in trades:
            reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1
        if reason_counts:
            reason_str = ", ".join(f"{k}:{v}" for k, v in reason_counts.items())
            logger.info(f"[{date}] 平仓原因: {reason_str}")

        return trades

    def _calculate_account_value(
        self,
        date: str,
        market_df: pd.DataFrame,
    ) -> float:
        """计算账户总价值。"""
        total_value = self.cash

        # 获取当日收盘价
        date_df = market_df[market_df["date"] == date]

        for ticker, shares in self.positions.items():
            ticker_df = date_df[date_df["tic"] == ticker]
            if not ticker_df.empty:
                close_price = ticker_df.iloc[0]["close"]
                total_value += shares * close_price

        return total_value

    def _generate_report(
        self,
        start_date: str,
        end_date: str,
        daily_records: list[DailyTradingRecord],
    ) -> DynamicBacktestReport:
        """生成回测报告。"""
        if not daily_records:
            raise ValueError("No daily records to generate report")

        # 计算总收益
        final_value = daily_records[-1].end_value
        total_return = (final_value - self.initial_amount) / self.initial_amount

        # 计算年化收益
        days = len(daily_records)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0.0

        # 计算最大回撤
        values = [r.end_value for r in daily_records]
        max_drawdown = DynamicBacktester._calculate_max_drawdown(values)

        # 计算夏普比率
        daily_returns = [r.daily_return for r in daily_records]
        sharpe_ratio = DynamicBacktester._calculate_sharpe_ratio(daily_returns)

        # 计算交易统计
        total_trades = sum(len(r.trades) for r in daily_records)

        # 计算胜率
        positive_days = sum(1 for r in daily_records if r.daily_return > 0)
        win_rate = positive_days / len(daily_records) if daily_records else 0.0

        # 计算平均日收益和波动率
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0.0
        volatility = np.std(daily_returns) if daily_returns else 0.0

        # 统计交易盈亏
        all_trades: list[dict[str, Any]] = []
        for r in daily_records:
            all_trades.extend(r.trades)

        if all_trades:
            trade_pnls = [t.get("pnl", 0) for t in all_trades]
            winning_trades = sum(1 for p in trade_pnls if p > 0)
            trade_win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0.0
            avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0.0
            total_trade_pnl = sum(trade_pnls)

            logger.info(f"\n{'='*80}")
            logger.info("回测交易统计")
            logger.info(f"{'='*80}")
            logger.info(f"总交易笔数: {len(all_trades)}")
            logger.info(f"盈利笔数: {winning_trades}")
            logger.info(f"亏损笔数: {len(all_trades) - winning_trades}")
            logger.info(f"交易胜率: {trade_win_rate:.2%}")
            logger.info(f"平均单笔盈亏: ¥{avg_trade_pnl:,.2f}")
            logger.info(f"总交易盈亏: ¥{total_trade_pnl:,.2f}")
            logger.info(f"{'='*80}\n")

        return DynamicBacktestReport(
            start_date=start_date,
            end_date=end_date,
            initial_amount=self.initial_amount,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            daily_records=daily_records,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_daily_return=avg_daily_return,
            volatility=volatility,
        )
