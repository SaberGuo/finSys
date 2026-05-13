from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


class KlineRenderer:
    """Renders dual-panel (weekly + daily) K-line charts using mplfinance."""

    def render(
        self,
        code: str,
        anchor_date: str,
        daily_df: pd.DataFrame,
        weekly_df: pd.DataFrame,
        output_dir: str = "data/images",
        split: str = "",
    ) -> str:
        """
        Render dual-panel K-line image and save as PNG.

        Returns the absolute path to the saved image.
        """
        if split:
            out_dir = Path(output_dir) / split / code
        else:
            out_dir = Path(output_dir) / code
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{anchor_date}.png"

        daily_ohlcv = self._to_mpf_df(daily_df)
        weekly_ohlcv = self._to_mpf_df(weekly_df)

        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            rc={"axes.labelsize": 0, "xtick.labelsize": 0, "ytick.labelsize": 0},
        )

        fig, axes = plt.subplots(1, 2, figsize=(8.96, 4.48), dpi=50)

        # Left panel: weekly
        mpf.plot(
            weekly_ohlcv,
            type="candle",
            style=style,
            ax=axes[0],
            volume=False,
            show_nontrading=False,
        )
        axes[0].set_title("")
        axes[0].axis("off")

        # Right panel: daily
        mpf.plot(
            daily_ohlcv,
            type="candle",
            style=style,
            ax=axes[1],
            volume=False,
            show_nontrading=False,
        )
        axes[1].set_title("")
        axes[1].axis("off")

        plt.tight_layout(pad=0)
        fig.savefig(str(out_path), dpi=50, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return str(out_path)

    @staticmethod
    def _to_mpf_df(df: pd.DataFrame) -> pd.DataFrame:
        """Convert a date-string OHLCV DataFrame to mplfinance-compatible format."""
        mpf_df = df.copy()
        mpf_df["Date"] = pd.to_datetime(mpf_df["date"])
        mpf_df = mpf_df.set_index("Date")
        mpf_df = mpf_df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        return mpf_df[["Open", "High", "Low", "Close"]]
