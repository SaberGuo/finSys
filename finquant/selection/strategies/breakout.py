"""MA breakout selection strategy with volume confirmation."""

from dataclasses import dataclass
from typing import Literal
import logging

import pandas as pd
import numpy as np

from finquant.selection import SelectionResult, MarketState
from finquant.selection.strategy import SelectionStrategy
from finquant.selection.market_state import MarketStateClassifier

logger = logging.getLogger(__name__)


@dataclass
class BreakoutConfig:
    """Configuration for breakout selection strategy.

    Attributes:
        ma_periods: Moving average periods to check (e.g., [120, 250])
        volume_multiplier: Volume surge threshold (e.g., 1.5 = 150% of MA)
        volume_ma_period: Rolling window for volume MA (default: 20)
        breakout_threshold: Price must be this multiple above MA (e.g., 1.05 = 5%)
        lookback_days: Days to check for first breakout (default: 60)
        confirmation_days: Days to confirm breakout (default: 3)
        anti_jitter_mode: How to apply anti-jitter logic
        top_k: Number of stocks to select
        exclude_st: Exclude ST stocks
        exclude_halt: Exclude halted stocks (volume=0)
    """
    ma_periods: list[int] = None
    volume_multiplier: float = 1.5
    volume_ma_period: int = 20
    breakout_threshold: float = 1.05
    lookback_days: int = 60
    confirmation_days: int = 3
    anti_jitter_mode: Literal["threshold", "confirmation", "both", "either"] = "threshold"
    top_k: int = 10
    exclude_st: bool = True
    exclude_halt: bool = True

    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [120, 250]


class BreakoutStrategy(SelectionStrategy):
    """Stock selection based on MA breakout with volume confirmation.

    This strategy identifies stocks that:
    1. Break through key moving averages (MA120/MA250)
    2. Show volume surge (volume >= K * vol_ma20)
    3. Are breaking out for the first time in N days
    4. Pass anti-jitter filters (price threshold or confirmation days)

    Args:
        config: BreakoutConfig with strategy parameters

    Example:
        config = BreakoutConfig(
            ma_periods=[120, 250],
            volume_multiplier=1.5,
            breakout_threshold=1.05,
            lookback_days=60,
            top_k=10
        )
        strategy = BreakoutStrategy(config)
        result = strategy.select(market_df, index_df, "2023-01-01")
    """

    def __init__(self, config: BreakoutConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.market_state_classifier = MarketStateClassifier()

    def select(
        self,
        market_df: pd.DataFrame,
        index_df: pd.DataFrame,
        as_of_date: str,
    ) -> SelectionResult:
        """Select stocks based on MA breakout signals.

        Args:
            market_df: Market data with columns [date, tic, open, high, low, close, volume]
            index_df: Index data (not used in this strategy)
            as_of_date: Selection date in YYYY-MM-DD format

        Returns:
            SelectionResult with selected tickers and breakout scores
        """
        if self.verbose:
            logger.info(f"\n{'='*70}")
            logger.info(f"MA突破选股 - {as_of_date}")
            logger.info(f"{'='*70}")
            logger.info(f"配置参数:")
            logger.info(f"  MA周期: {self.config.ma_periods}")
            logger.info(f"  放量倍数: {self.config.volume_multiplier}x")
            logger.info(f"  突破阈值: {self.config.breakout_threshold} ({(self.config.breakout_threshold-1)*100:.0f}%)")
            logger.info(f"  回溯天数: {self.config.lookback_days}")
            logger.info(f"  确认天数: {self.config.confirmation_days}")
            logger.info(f"  防抖模式: {self.config.anti_jitter_mode}")
            logger.info(f"  选股数量: {self.config.top_k}")

        # Filter data up to as_of_date
        df = market_df[market_df["date"] <= as_of_date].copy()

        if df.empty:
            return self._empty_result(as_of_date)

        # Compute required indicators for each stock
        df = self._compute_indicators(df)

        # Get data for as_of_date
        today_df = df[df["date"] == as_of_date].copy()

        if today_df.empty:
            return self._empty_result(as_of_date)

        if self.verbose:
            logger.info(f"\n候选股票数: {len(today_df)}")

        # Apply breakout filters
        candidates = self._filter_breakouts(df, today_df, as_of_date)

        if candidates.empty:
            if self.verbose:
                logger.info(f"✗ 无符合突破条件的股票")
            return self._empty_result(as_of_date)

        if self.verbose:
            logger.info(f"✓ 通过突破筛选: {len(candidates)} 只股票")

        # Score and rank candidates
        scores = self._score_candidates(candidates)

        # Apply exclusion rules
        selected_tickers, all_scores, exclusion_reasons = self._apply_exclusions(
            candidates, scores
        )

        if self.verbose and exclusion_reasons:
            logger.info(f"\n排除股票:")
            for tic, reason in exclusion_reasons.items():
                logger.info(f"  {tic}: {reason}")

        # Select top_k
        selected_tickers = selected_tickers[: self.config.top_k]

        # Classify market state based on index data
        try:
            # Prepare index data with required indicators
            index_df_prepared = self._prepare_index_for_classification(index_df, as_of_date)
            market_state = self.market_state_classifier.classify(index_df_prepared, as_of_date)
            index_metrics = self.market_state_classifier.get_index_metrics(index_df_prepared, as_of_date)
        except (ValueError, KeyError) as e:
            # Fallback if classification fails
            if self.verbose:
                logger.warning(f"Market state classification failed: {e}, using OSCILLATION")
            market_state = MarketState.OSCILLATION
            index_metrics = {}

        if self.verbose:
            logger.info(f"\n市场状态: {market_state.value}")
            if index_metrics:
                logger.info(f"指数指标:")
                logger.info(f"  收盘价: {index_metrics.get('close', 0):.2f}")
                logger.info(f"  ADX: {index_metrics.get('adx', 0):.2f}")
                logger.info(f"  MA50: {index_metrics.get('ma50', 0):.2f}")
                logger.info(f"  相对MA50: {index_metrics.get('ma50_ratio', 1.0):.2%}")
            logger.info(f"\n最终选中: {len(selected_tickers)} 只股票")
            if selected_tickers:
                logger.info(f"股票详情:")
                for tic in selected_tickers:
                    row = candidates[candidates["tic"] == tic].iloc[0]
                    logger.info(f"\n  {tic}:")
                    logger.info(f"    评分: {all_scores[tic]:.4f}")
                    logger.info(f"    收盘价: {row['close']:.2f}")

                    # Show which MA was broken
                    for ma_period in self.config.ma_periods:
                        ma_col = f"ma{ma_period}"
                        if ma_col in row and not pd.isna(row[ma_col]):
                            ma_val = row[ma_col]
                            pct_above = (row['close'] - ma_val) / ma_val * 100
                            if row['close'] > ma_val:
                                logger.info(f"    MA{ma_period}: {ma_val:.2f} (突破 +{pct_above:.2f}%)")

                    # Show volume surge
                    if not pd.isna(row['vol_ma']) and row['vol_ma'] > 0:
                        vol_ratio = row['volume'] / row['vol_ma']
                        logger.info(f"    成交量: {row['volume']:,.0f} ({vol_ratio:.2f}x 均量)")

        return SelectionResult(
            date=as_of_date,
            selected_tickers=selected_tickers,
            scores={tic: all_scores[tic] for tic in selected_tickers},
            market_state=market_state,
            active_factors=["ma_breakout", "volume_surge"],
            factor_weights={"ma_breakout": 0.6, "volume_surge": 0.4},
            index_metrics=index_metrics or {},
            exclusion_reasons=exclusion_reasons,
        )

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MA and volume indicators for each stock."""
        result = []

        for tic, group in df.groupby("tic"):
            group = group.sort_values("date").copy()

            # Compute MAs
            for period in self.config.ma_periods:
                group[f"ma{period}"] = group["close"].rolling(period).mean()

            # Compute volume MA
            group["vol_ma"] = group["volume"].rolling(self.config.volume_ma_period).mean()

            result.append(group)

        return pd.concat(result, ignore_index=True)

    def _filter_breakouts(
        self, df: pd.DataFrame, today_df: pd.DataFrame, as_of_date: str
    ) -> pd.DataFrame:
        """Filter stocks that meet all breakout criteria."""
        candidates = []

        for _, row in today_df.iterrows():
            tic = row["tic"]
            stock_df = df[df["tic"] == tic].sort_values("date")

            # Check each MA period
            for ma_period in self.config.ma_periods:
                ma_col = f"ma{ma_period}"

                if ma_col not in row or pd.isna(row[ma_col]):
                    continue

                # Check breakout conditions
                if self._check_breakout(stock_df, row, ma_col, as_of_date):
                    candidates.append(row)
                    break  # Only need one MA to break out

        if not candidates:
            return pd.DataFrame()

        return pd.DataFrame(candidates)

    def _check_breakout(
        self, stock_df: pd.DataFrame, today_row: pd.Series, ma_col: str, as_of_date: str
    ) -> bool:
        """Check if stock meets all breakout conditions."""
        # 1. MA Breakout: close_t > MA_t AND close_{t-1} <= MA_{t-1}
        if not self._compute_ma_breakout(stock_df, today_row, ma_col, as_of_date):
            return False

        # 2. Volume Surge: volume_t >= K * vol_ma20_t
        if not self._compute_volume_surge(today_row):
            return False

        # 3. First Breakout: No breakout in past N days
        if not self._check_first_breakout(stock_df, ma_col, as_of_date):
            return False

        # 4. Anti-Jitter: Apply threshold or confirmation logic
        if not self._apply_anti_jitter(stock_df, today_row, ma_col, as_of_date):
            return False

        return True

    def _compute_ma_breakout(
        self, stock_df: pd.DataFrame, today_row: pd.Series, ma_col: str, as_of_date: str
    ) -> bool:
        """Check if close crosses above MA today."""
        today_close = today_row["close"]
        today_ma = today_row[ma_col]

        if pd.isna(today_close) or pd.isna(today_ma):
            return False

        # Today: close > MA
        if today_close <= today_ma:
            return False

        # Yesterday: close <= MA
        yesterday_data = stock_df[stock_df["date"] < as_of_date].tail(1)
        if yesterday_data.empty:
            return False

        yesterday_row = yesterday_data.iloc[0]
        yesterday_close = yesterday_row["close"]
        yesterday_ma = yesterday_row[ma_col]

        if pd.isna(yesterday_close) or pd.isna(yesterday_ma):
            return False

        return bool(yesterday_close <= yesterday_ma)

    def _compute_volume_surge(self, today_row: pd.Series) -> bool:
        """Check if volume surges above threshold."""
        volume = today_row["volume"]
        vol_ma = today_row["vol_ma"]

        if pd.isna(volume) or pd.isna(vol_ma) or vol_ma == 0:
            return False

        return bool(volume >= self.config.volume_multiplier * vol_ma)

    def _check_first_breakout(
        self, stock_df: pd.DataFrame, ma_col: str, as_of_date: str
    ) -> bool:
        """Check if this is the first breakout in lookback period."""
        lookback_df = stock_df[
            (stock_df["date"] < as_of_date)
        ].tail(self.config.lookback_days)

        if lookback_df.empty:
            return True  # No history, consider it first breakout

        # Check if any day in lookback had close > MA
        for _, row in lookback_df.iterrows():
            if not pd.isna(row["close"]) and not pd.isna(row[ma_col]):
                if row["close"] > row[ma_col]:
                    return False  # Found a previous breakout

        return True

    def _apply_anti_jitter(
        self, stock_df: pd.DataFrame, today_row: pd.Series, ma_col: str, as_of_date: str
    ) -> bool:
        """Apply anti-jitter mechanism based on mode."""
        mode = self.config.anti_jitter_mode

        threshold_pass = self._check_threshold(today_row, ma_col)
        confirmation_pass = self._check_confirmation(stock_df, ma_col, as_of_date)

        if mode == "threshold":
            return threshold_pass
        elif mode == "confirmation":
            return confirmation_pass
        elif mode == "both":
            return threshold_pass and confirmation_pass
        elif mode == "either":
            return threshold_pass or confirmation_pass
        else:
            return True

    def _check_threshold(self, today_row: pd.Series, ma_col: str) -> bool:
        """Check if close is at least threshold% above MA."""
        close = today_row["close"]
        ma = today_row[ma_col]

        if pd.isna(close) or pd.isna(ma) or ma == 0:
            return False

        return bool(close >= ma * self.config.breakout_threshold)

    def _check_confirmation(
        self, stock_df: pd.DataFrame, ma_col: str, as_of_date: str
    ) -> bool:
        """Check if close stays above MA for confirmation_days."""
        # Get last N days including today
        recent_df = stock_df[stock_df["date"] <= as_of_date].tail(
            self.config.confirmation_days
        )

        if len(recent_df) < self.config.confirmation_days:
            return False  # Not enough data

        # All days must have close > MA
        for _, row in recent_df.iterrows():
            if pd.isna(row["close"]) or pd.isna(row[ma_col]):
                return False
            if row["close"] <= row[ma_col]:
                return False

        return True

    def _score_candidates(self, candidates: pd.DataFrame) -> dict[str, float]:
        """Score candidates based on breakout strength.

        Scores are normalized to [-1.0, 1.0] range using tanh transformation.
        """
        scores = {}

        for _, row in candidates.iterrows():
            tic = row["tic"]

            # Compute breakout strength for each MA
            breakout_strengths = []
            for ma_period in self.config.ma_periods:
                ma_col = f"ma{ma_period}"
                if ma_col in row and not pd.isna(row[ma_col]) and row[ma_col] > 0:
                    strength = (row["close"] - row[ma_col]) / row[ma_col]
                    breakout_strengths.append(strength)

            # Compute volume surge strength
            vol_strength = 0.0
            if not pd.isna(row["vol_ma"]) and row["vol_ma"] > 0:
                vol_strength = (row["volume"] - row["vol_ma"]) / row["vol_ma"]

            # Combined score: 60% breakout + 40% volume
            if breakout_strengths:
                breakout_score = max(breakout_strengths)
                raw_score = 0.6 * breakout_score + 0.4 * vol_strength

                # Normalize to [-1.0, 1.0] using tanh
                # tanh(x) maps (-inf, inf) -> (-1, 1)
                # Scale by 5 to make typical values (0.05-0.5) map to reasonable range
                normalized_score = float(np.tanh(raw_score * 5))
                scores[tic] = normalized_score

        return scores

    def _apply_exclusions(
        self, candidates: pd.DataFrame, scores: dict[str, float]
    ) -> tuple[list[str], dict[str, float], dict[str, str]]:
        """Apply exclusion rules and return selected tickers."""
        exclusion_reasons = {}
        valid_tickers = []

        for tic, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            row = candidates[candidates["tic"] == tic].iloc[0]

            # Check ST stocks
            if self.config.exclude_st and "ST" in tic:
                exclusion_reasons[tic] = "ST stock"
                continue

            # Check halted stocks
            if self.config.exclude_halt and row["volume"] == 0:
                exclusion_reasons[tic] = "halted (volume=0)"
                continue

            valid_tickers.append(tic)

        return valid_tickers, scores, exclusion_reasons

    def _empty_result(self, as_of_date: str) -> SelectionResult:
        """Return empty result when no candidates found."""
        return SelectionResult(
            date=as_of_date,
            selected_tickers=[],
            scores={},
            market_state=MarketState.OSCILLATION,
            active_factors=[],
            factor_weights={},
            index_metrics={},
            exclusion_reasons={},
        )

    def _prepare_index_for_classification(self, index_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        """Prepare index data with required indicators for market state classification.

        Args:
            index_df: Raw index data
            as_of_date: Current date

        Returns:
            Index DataFrame with adx_14, ma50, and volume_ma20_ratio
        """
        df = index_df.copy()

        # Sort by date
        df = df.sort_values('date')

        # Add adx_14 if not present (use dx_30 as proxy or compute from scratch)
        if 'adx_14' not in df.columns:
            if 'dx_30' in df.columns:
                # Use dx_30 as a proxy for adx_14
                df['adx_14'] = df['dx_30']
            else:
                # Fallback: set to 15 (neutral value)
                df['adx_14'] = 15.0

        # Add ma50 if not present
        if 'ma50' not in df.columns:
            df['ma50'] = df['close'].rolling(50, min_periods=1).mean()

        # Add volume_ma20_ratio if not present
        if 'volume_ma20_ratio' not in df.columns and 'volume' in df.columns:
            volume_ma20 = df['volume'].rolling(20, min_periods=1).mean().replace(0, 1e-9)
            df['volume_ma20_ratio'] = df['volume'] / volume_ma20

        return df
