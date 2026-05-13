from typing import Optional

import pandas as pd


class LabelGenerator:
    """Generates binary buy/hold labels based on future N-day return."""

    def __init__(self, horizon: int = 5, threshold: float = 0.02) -> None:
        self.horizon = horizon
        self.threshold = threshold

    def generate(
        self,
        df: pd.DataFrame,
        anchor_date: str,
    ) -> Optional[tuple[int, float]]:
        """
        Compute label for anchor_date using future horizon-day return.

        Returns (label, future_return) or None if insufficient future data.
        label=1 (buy) if future_return > threshold, else label=0.
        """
        future_rows = df[df["date"] > anchor_date].head(self.horizon)
        if len(future_rows) < self.horizon:
            return None

        anchor_rows = df[df["date"] <= anchor_date]
        if anchor_rows.empty:
            return None

        anchor_close = anchor_rows.iloc[-1]["close"]
        future_close = future_rows.iloc[-1]["close"]

        if anchor_close <= 0:
            return None

        future_return = (future_close - anchor_close) / anchor_close
        # strictly greater than threshold for buy signal
        label = 1 if future_return > self.threshold else 0
        return label, float(future_return)
