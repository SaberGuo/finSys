"""Unit tests for finquant.features.fundamental."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finquant.features.fundamental import FundamentalRecord, FundamentalProcessor


class TestFundamentalRecord:
    def test_valid_record(self) -> None:
        rec = FundamentalRecord(
            date="2024-01-02",
            tic="000009.SZ",
            score=0.5,
            label="positive",
            revenue_growth_hint="positive",
            profitability_hint="neutral",
            debt_hint="negative",
        )
        assert rec.score == pytest.approx(0.5)
        assert rec.revenue_growth_score == pytest.approx(1.0)
        assert rec.profitability_score == pytest.approx(0.0)
        assert rec.debt_score == pytest.approx(-1.0)

    def test_invalid_score_raises(self) -> None:
        with pytest.raises(ValueError):
            FundamentalRecord(
                date="2024-01-02",
                tic="000009.SZ",
                score=2.0,
                label="positive",
                revenue_growth_hint="positive",
                profitability_hint="neutral",
                debt_hint="negative",
            )

    def test_invalid_hint_raises(self) -> None:
        with pytest.raises(ValueError):
            FundamentalRecord(
                date="2024-01-02",
                tic="000009.SZ",
                score=0.5,
                label="positive",
                revenue_growth_hint="bullish",
                profitability_hint="neutral",
                debt_hint="negative",
            )

    def test_neutral_factory(self) -> None:
        rec = FundamentalRecord.neutral("2024-01-02", "000009.SZ")
        assert rec.score == pytest.approx(0.0)
        assert rec.label == "neutral"
        assert rec.revenue_growth_hint == "neutral"

    def test_to_dict(self) -> None:
        rec = FundamentalRecord.neutral("2024-01-02", "000009.SZ")
        d = rec.to_dict()
        assert d["date"] == "2024-01-02"
        assert d["score"] == pytest.approx(0.0)


class TestFundamentalProcessorProcessFile:
    def test_process_file(self, tmp_path: Path) -> None:
        # Create a mock news JSONL
        news_file = tmp_path / "news.jsonl"
        records = [
            {"date": "2024-01-02", "tic": "000009.SZ", "title": "Good news", "content": "Revenue up"},
            {"date": "2024-01-03", "tic": "600004.SH", "title": "Bad news", "content": "Loss"},
        ]
        with news_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Mock QwenModel to avoid loading real model
        class MockConfig:
            model_id = "dummy"
            quantize_4bit = False
            max_new_tokens = 128
            batch_size = 2

        class FakeModel:
            def __init__(self, config) -> None:
                self._config = config
                self._tokenizer = None
                self._model = None

            @property
            def is_loaded(self):
                return True

            def load(self):
                pass

            def generate(self, text: str):
                return json.dumps({
                    "score": 0.5,
                    "label": "positive",
                    "revenue_growth_hint": "positive",
                    "profitability_hint": "neutral",
                    "debt_hint": "neutral",
                })

        from unittest.mock import patch

        processor = FundamentalProcessor(MockConfig())
        processor._qwen = FakeModel(MockConfig())
        processor._loaded = True

        def fake_generate(text: str) -> str:
            return json.dumps({
                "score": 0.5,
                "label": "positive",
                "revenue_growth_hint": "positive",
                "profitability_hint": "neutral",
                "debt_hint": "neutral",
            })

        with patch.object(
            type(processor), "_ensure_loaded", lambda self: None
        ):
            with patch(
                "finquant.nlp.fundamental_analyzer.DualAnalyzer._generate_dual",
                staticmethod(fake_generate),
            ):
                out_path = processor.process_file(
                    news_file, tmp_path / "out", verbose=False
                )

        assert out_path.exists()
        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["tic"] == "000009.SZ"
        assert rec["revenue_growth_score"] == pytest.approx(1.0)
