"""Integration tests for NLP pipeline (T044)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def news_jsonl(tmp_path: Path) -> Path:
    import json

    records = [
        {"date": "2024-01-02", "tic": "000001.SZ", "text": "平安银行超预期盈利，股价上涨5%。"},
        {"date": "2024-01-03", "tic": "600519.SH", "text": "茅台因渠道乱象被监管警告，股价承压。"},
        {"date": "2024-01-04", "tic": "000001.SZ", "text": "市场整体平稳，成交量缩减。"},
    ]
    f = tmp_path / "news.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return f


class TestNLPPipelineIntegration:
    @patch("finquant.nlp.model.AutoModelForCausalLM")
    @patch("finquant.nlp.model.AutoTokenizer")
    def test_full_pipeline_produces_sentiment_jsonl(
        self,
        mock_tokenizer_cls,
        mock_model_cls,
        news_jsonl: Path,
        tmp_path: Path,
    ) -> None:
        import json

        from finquant.config.settings import SentimentConfig
        from finquant.features.sentiment import SentimentProcessor

        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_hf_model = MagicMock()
        generated_ids = MagicMock()
        mock_hf_model.generate.return_value = generated_ids
        mock_tokenizer.batch_decode.return_value = ['{"score": 0.7, "label": "positive"}']
        mock_model_cls.from_pretrained.return_value = mock_hf_model

        cfg = SentimentConfig(
            enabled=True, model_id="mock-model", quantize_4bit=False, batch_size=2
        )
        proc = SentimentProcessor(cfg)
        out = proc.process_file(input_path=news_jsonl, output_dir=tmp_path)

        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(l) for l in lines]
        assert len(records) == 3
        for r in records:
            assert "date" in r
            assert "tic" in r
            assert "score" in r
            assert "label" in r
            assert -1.0 <= r["score"] <= 1.0

    @patch("finquant.nlp.model.AutoModelForCausalLM")
    @patch("finquant.nlp.model.AutoTokenizer")
    def test_sc003_inference_returns_score(
        self,
        mock_tokenizer_cls,
        mock_model_cls,
        tmp_path: Path,
    ) -> None:
        """SC-003: Each text produces a sentiment score in [-1, 1]."""
        from finquant.config.settings import SentimentConfig
        from finquant.features.sentiment import SentimentProcessor

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.batch_decode.return_value = ['{"score": -0.3, "label": "negative"}']

        mock_hf_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_hf_model

        cfg = SentimentConfig(
            enabled=True, model_id="mock-model", quantize_4bit=False
        )
        proc = SentimentProcessor(cfg)
        records = proc.process_texts(
            [{"date": "2024-01-02", "tic": "000001.SZ", "text": "利空消息"}]
        )
        assert len(records) == 1
        assert -1.0 <= records[0].score <= 1.0
