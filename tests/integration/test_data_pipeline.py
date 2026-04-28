from __future__ import annotations

import pytest

from finquant.config.settings import load_config
from finquant.data.pipeline import DataPipeline
from finquant.data.sources.base import DataSource


@pytest.mark.integration
@pytest.mark.skipif(True, reason="opt-in integration test; run with external setup")
def test_data_pipeline_end_to_end_opt_in() -> None:
    config = load_config("config/default.yaml.example")
    pipeline = DataPipeline(config)
    df = pipeline.fetch()
    assert not df.empty
    assert df.duplicated(subset=["date", "tic"]).sum() == 0


def test_failover_logic_uses_next_source(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSource(DataSource):
        def download(self, symbols: list[str], start_date: str, end_date: str):
            raise RuntimeError("primary source down")

    config = load_config("config/default.yaml.example")
    config.data.source_priority = ["xtquant", "akshare"]
    pipeline = DataPipeline(config)
    pipeline.adapters["xtquant"] = FailingSource()

    df = pipeline.fetch()
    assert not df.empty
    assert "macd" in df.columns
