"""Unit tests for finquant.features.sentiment (T042)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finquant.features.sentiment import SentimentRecord, SentimentProcessor


@pytest.fixture()
def mock_sentiment_config():
    from finquant.config.settings import SentimentConfig

    return SentimentConfig(
        enabled=True,
        model_id="mock-model",
        quantize_4bit=False,
        batch_size=2,
    )


@pytest.fixture()
def sample_news_jsonl(tmp_path: Path) -> Path:
    import json

    records = [
        {"date": "2024-01-02", "tic": "000001.SZ", "text": "平安银行发布强劲年报，营收增长15%。"},
        {"date": "2024-01-02", "tic": "600519.SH", "text": "茅台酒价格下跌，市场担忧需求减少。"},
        {"date": "2024-01-03", "tic": "000001.SZ", "text": "监管机构对银行业展开新一轮检查。"},
        # duplicate to test dedup
        {"date": "2024-01-03", "tic": "000001.SZ", "text": "监管机构对银行业展开新一轮检查。"},
    ]
    f = tmp_path / "news.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return f


class TestSentimentRecord:
    def test_valid_record(self) -> None:
        r = SentimentRecord(date="2024-01-02", tic="000001.SZ", score=0.8, label="positive")
        assert r.score == pytest.approx(0.8)

    def test_score_range_validation(self) -> None:
        with pytest.raises(ValueError):
            SentimentRecord(date="2024-01-02", tic="000001.SZ", score=1.5, label="positive")
        with pytest.raises(ValueError):
            SentimentRecord(date="2024-01-02", tic="000001.SZ", score=-1.5, label="negative")

    def test_to_dict(self) -> None:
        r = SentimentRecord(date="2024-01-02", tic="000001.SZ", score=0.5, label="neutral")
        d = r.to_dict()
        assert set(d.keys()) >= {"date", "tic", "score", "label"}

    def test_neutral_default(self) -> None:
        r = SentimentRecord.neutral("2024-01-02", "000001.SZ")
        assert r.score == pytest.approx(0.0)
        assert r.label == "neutral"


class TestSentimentProcessor:
    def test_init(self, mock_sentiment_config) -> None:
        proc = SentimentProcessor(mock_sentiment_config)
        assert proc is not None

    @patch("finquant.features.sentiment.QwenModel")
    def test_process_texts_returns_records(
        self, mock_model_cls, mock_sentiment_config
    ) -> None:
        mock_model = MagicMock()
        mock_model.generate.return_value = '{"score": 0.8, "label": "positive"}'
        mock_model_cls.return_value = mock_model

        proc = SentimentProcessor(mock_sentiment_config)
        texts = [
            {"date": "2024-01-02", "tic": "000001.SZ", "text": "good news"},
            {"date": "2024-01-03", "tic": "000001.SZ", "text": "bad news"},
        ]
        records = proc.process_texts(texts)
        assert len(records) == 2
        assert all(isinstance(r, SentimentRecord) for r in records)

    @patch("finquant.features.sentiment.QwenModel")
    def test_process_deduplicates_texts(
        self, mock_model_cls, mock_sentiment_config
    ) -> None:
        mock_model = MagicMock()
        mock_model.generate.return_value = '{"score": 0.5, "label": "neutral"}'
        mock_model_cls.return_value = mock_model

        proc = SentimentProcessor(mock_sentiment_config)
        texts = [
            {"date": "2024-01-02", "tic": "000001.SZ", "text": "same text"},
            {"date": "2024-01-02", "tic": "000001.SZ", "text": "same text"},  # dup
        ]
        records = proc.process_texts(texts)
        assert len(records) == 1  # deduped

    @patch("finquant.features.sentiment.QwenModel")
    def test_graceful_degradation_on_parse_error(
        self, mock_model_cls, mock_sentiment_config
    ) -> None:
        """On invalid JSON from Qwen → return neutral record."""
        mock_model = MagicMock()
        mock_model.generate.return_value = "invalid json"
        mock_model_cls.return_value = mock_model

        proc = SentimentProcessor(mock_sentiment_config)
        texts = [{"date": "2024-01-02", "tic": "000001.SZ", "text": "news"}]
        records = proc.process_texts(texts)
        assert len(records) == 1
        assert records[0].label == "neutral"

    @patch("finquant.features.sentiment.QwenModel")
    def test_graceful_degradation_on_model_failure(
        self, mock_model_cls, mock_sentiment_config
    ) -> None:
        """On Qwen generate() exception → return neutral record."""
        mock_model = MagicMock()
        mock_model.generate.side_effect = RuntimeError("GPU OOM")
        mock_model_cls.return_value = mock_model

        proc = SentimentProcessor(mock_sentiment_config)
        texts = [{"date": "2024-01-02", "tic": "000001.SZ", "text": "news"}]
        records = proc.process_texts(texts)
        assert len(records) == 1
        assert records[0].label == "neutral"

    @patch("finquant.features.sentiment.QwenModel")
    def test_process_file_saves_jsonl(
        self,
        mock_model_cls,
        mock_sentiment_config,
        sample_news_jsonl: Path,
        tmp_path: Path,
    ) -> None:
        import json

        mock_model = MagicMock()
        mock_model.generate.return_value = '{"score": 0.6, "label": "positive"}'
        mock_model_cls.return_value = mock_model

        proc = SentimentProcessor(mock_sentiment_config)
        out = proc.process_file(input_path=sample_news_jsonl, output_dir=tmp_path)
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(l) for l in lines]
        # 4 input records, but 1 duplicate → 3 unique
        assert len(records) == 3
