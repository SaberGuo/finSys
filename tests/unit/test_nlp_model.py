"""Unit tests for finquant.nlp.model (T041)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from finquant.nlp.model import QwenModel, QwenLoadError


class TestQwenModel:
    def test_init_stores_config(self) -> None:
        from finquant.config.settings import SentimentConfig

        cfg = SentimentConfig(
            enabled=True,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            quantize_4bit=True,
        )
        model = QwenModel(cfg)
        assert model.model_id == cfg.model_id
        assert model.quantize_4bit is True

    def test_not_loaded_before_load(self) -> None:
        from finquant.config.settings import SentimentConfig

        cfg = SentimentConfig(enabled=True)
        model = QwenModel(cfg)
        assert not model.is_loaded

    @patch("finquant.nlp.model.AutoModelForCausalLM")
    @patch("finquant.nlp.model.AutoTokenizer")
    def test_load_calls_from_pretrained(
        self, mock_tokenizer_cls, mock_model_cls
    ) -> None:
        from finquant.config.settings import SentimentConfig

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        cfg = SentimentConfig(enabled=True, model_id="mock-model", quantize_4bit=False)
        model = QwenModel(cfg)
        model.load()

        mock_tokenizer_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()
        assert model.is_loaded

    @patch("finquant.nlp.model.AutoModelForCausalLM")
    @patch("finquant.nlp.model.AutoTokenizer")
    def test_load_idempotent(self, mock_tokenizer_cls, mock_model_cls) -> None:
        from finquant.config.settings import SentimentConfig

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        cfg = SentimentConfig(enabled=True, model_id="mock-model", quantize_4bit=False)
        model = QwenModel(cfg)
        model.load()
        model.load()  # second call should be no-op

        assert mock_tokenizer_cls.from_pretrained.call_count == 1

    @patch("finquant.nlp.model.AutoModelForCausalLM")
    @patch("finquant.nlp.model.AutoTokenizer")
    def test_generate_raises_if_not_loaded(
        self, mock_tokenizer_cls, mock_model_cls
    ) -> None:
        from finquant.config.settings import SentimentConfig

        cfg = SentimentConfig(enabled=True, model_id="mock-model", quantize_4bit=False)
        model = QwenModel(cfg)

        with pytest.raises(RuntimeError, match="not loaded"):
            model.generate("some text")

    def test_load_failure_raises_qwen_load_error(self) -> None:
        from finquant.config.settings import SentimentConfig

        cfg = SentimentConfig(
            enabled=True, model_id="nonexistent-model", quantize_4bit=False
        )
        model = QwenModel(cfg)

        with patch("finquant.nlp.model.AutoTokenizer") as mock_tok:
            mock_tok.from_pretrained.side_effect = OSError("model not found")
            with pytest.raises(QwenLoadError):
                model.load()

    @patch("finquant.nlp.model.AutoModelForCausalLM")
    @patch("finquant.nlp.model.AutoTokenizer")
    def test_quantize_4bit_passes_bnb_config(
        self, mock_tokenizer_cls, mock_model_cls
    ) -> None:
        from finquant.config.settings import SentimentConfig

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        cfg = SentimentConfig(enabled=True, model_id="mock-model", quantize_4bit=True)
        model = QwenModel(cfg)

        with patch("finquant.nlp.model.BitsAndBytesConfig") as mock_bnb:
            mock_bnb.return_value = MagicMock()
            model.load()

        mock_bnb.assert_called_once()
