from __future__ import annotations

from pathlib import Path

import pandas as pd

from finquant.config.settings import AppConfig
from finquant.data.preprocessor import preprocess_market_data
from finquant.data.sources.akshare import AkshareDataSource
from finquant.data.sources.baostock import BaostockDataSource
from finquant.data.sources.base import DataSource
from finquant.data.sources.xtquant import XtquantDataSource
from finquant.features.technical import compute_indicators
from finquant.utils.logging import get_logger


class DataSourceExhaustedError(RuntimeError):
    pass


class DataPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("finquant.data")
        self.adapters: dict[str, DataSource] = {
            "xtquant": XtquantDataSource(),
            "akshare": AkshareDataSource(),
            "baostock": BaostockDataSource(),
        }

    def fetch(self) -> pd.DataFrame:
        last_error: Exception | None = None
        for source_name in self.config.data.source_priority:
            adapter = self.adapters.get(source_name)
            if adapter is None:
                continue
            try:
                raw = adapter.download(
                    symbols=self.config.stocks,
                    start_date=self.config.dates.train_start,
                    end_date=self.config.dates.test_end,
                )
                self.logger.info(f"downloaded from {source_name}")
                processed = preprocess_market_data(raw)
                enriched = compute_indicators(processed, self.config.indicators)
                return enriched.sort_values(["date", "tic"]).reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.logger.warning(f"source {source_name} failed: {exc}")

        raise DataSourceExhaustedError("all data sources failed") from last_error

    def fetch_and_save(self, output_dir: str | Path | None = None) -> Path:
        df = self.fetch()
        out_dir = Path(output_dir or self.config.data.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        start = self.config.dates.train_start.replace("-", "")
        end = self.config.dates.test_end.replace("-", "")
        file_path = out_dir / f"{start}_{end}_dataset.parquet"
        df.to_parquet(file_path, index=False)
        self.logger.info(f"saved dataset to {file_path}")
        return file_path
