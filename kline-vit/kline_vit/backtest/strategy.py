import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _compute_signal(
    engine,
    image_path: str,
    threshold: float,
    in_position: bool,
) -> int:
    """
    Compute trading signal.
    Returns: 1 (BUY), -1 (SELL), 0 (HOLD)
    """
    try:
        result = engine.predict_single(image_path)
        prob = result.buy_probability
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Inference failed for {image_path}: {e}")
        return 0  # HOLD on error

    if prob >= threshold and not in_position:
        return 1   # BUY
    elif prob < threshold and in_position:
        return -1  # SELL
    return 0       # HOLD


class KlineSignalStrategy:
    """Backtrader strategy driven by ViT model signals.

    Import backtrader lazily to allow unit testing without bt installed.
    Actual bt.Strategy subclass is created at runtime.
    """
    pass


def _make_bt_strategy():
    """Create the actual Backtrader strategy class (requires backtrader installed)."""
    import backtrader as bt

    class _KlineSignalStrategy(bt.Strategy):
        params = (
            ("inference_engine", None),
            ("image_dir", "data/images/test"),
            ("signal_threshold", 0.6),
            ("stop_loss_pct", 0.08),
        )

        def __init__(self) -> None:
            self.order = None
            self.stop_order = None
            self.buy_price = None

        def next(self) -> None:
            if self.order:
                return
            current_date = self.data.datetime.date(0).strftime("%Y-%m-%d")
            code = getattr(self.data, "_name", "unknown")
            image_path = str(Path(self.p.image_dir) / code / f"{current_date}.png")
            in_position = self.position.size > 0
            signal = _compute_signal(
                self.p.inference_engine, image_path,
                self.p.signal_threshold, in_position,
            )
            if signal == 1:
                self.order = self.buy()
                self.buy_price = self.data.close[0]
            elif signal == -1:
                self.order = self.sell()
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None

        def notify_order(self, order) -> None:
            if order.status in (order.Completed,):
                if order.isbuy():
                    stop_price = order.executed.price * (1 - self.p.stop_loss_pct)
                    self.stop_order = self.sell(exectype=bt.Order.Stop, price=stop_price)
                    logger.info(f"BUY @ {order.executed.price:.2f}, stop @ {stop_price:.2f}")
                else:
                    logger.info(f"SELL @ {order.executed.price:.2f}")
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self.order = None

        def notify_trade(self, trade) -> None:
            if trade.isclosed:
                logger.info(f"Trade P&L: gross={trade.pnl:.2f}, net={trade.pnlcomm:.2f}")

    return _KlineSignalStrategy
