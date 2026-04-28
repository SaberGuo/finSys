"""Qwen2.5-7B-Instruct model loader with 4-bit NF4 quantization (T045).

Module-level names (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
are assigned at import time so unit tests can patch them via
``@patch("finquant.nlp.model.AutoTokenizer")``.
"""
from __future__ import annotations

# These are set at module level so they can be patched in tests.
# They may be None if transformers is not installed.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    BitsAndBytesConfig = None  # type: ignore[assignment]


class QwenLoadError(Exception):
    """Raised when the Qwen model cannot be loaded."""


class QwenModel:
    """Lazy loader for Qwen2.5-7B-Instruct.

    Parameters
    ----------
    config:
        ``SentimentConfig`` from app configuration.
    """

    def __init__(self, config) -> None:
        self._config = config
        self.model_id: str = config.model_id
        self.quantize_4bit: bool = config.quantize_4bit
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    def load(self) -> None:
        """Load tokenizer and model (idempotent).

        Raises
        ------
        QwenLoadError
            If the model cannot be loaded (e.g., missing files, OOM).
        """
        if self.is_loaded:
            return
        try:
            self._do_load()
        except QwenLoadError:
            raise
        except Exception as exc:
            raise QwenLoadError(f"Failed to load Qwen model {self.model_id!r}: {exc}") from exc

    def _do_load(self) -> None:
        import finquant.nlp.model as _m  # import self to allow patching

        quantization_config = None
        if self.quantize_4bit:
            bnb_cfg = _m.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            quantization_config = bnb_cfg

        self._tokenizer = _m.AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        load_kwargs: dict = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self._model = _m.AutoModelForCausalLM.from_pretrained(
            self.model_id, **load_kwargs
        )

    def generate(self, prompt: str) -> str:
        """Run inference and return raw model output text.

        Parameters
        ----------
        prompt:
            Formatted prompt string (user message content).

        Returns
        -------
        str
            Decoded model output (expected JSON: ``{"score": ..., "label": ...}``).

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called first.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "QwenModel is not loaded. Call .load() before .generate()."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是专业的A股市场分析师。"
                    "请分析以下财经文本的情感倾向，"
                    '仅输出JSON格式：{"score": <float in [-1,1]>, '
                    '"label": <"positive"|"neutral"|"negative">}'
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt")
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self._config.max_new_tokens,
            do_sample=False,
        )
        decoded = self._tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        return decoded[0].strip()
