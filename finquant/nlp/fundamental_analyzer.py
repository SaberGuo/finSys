"""Qwen dual-analysis: sentiment + fundamental hints from news text."""
from __future__ import annotations

import json
import re
from typing import Any

from finquant.nlp.model import QwenModel

_JSON_RE = re.compile(r'\{[^{}]*"score"[^{}]*\}', re.DOTALL)


class FundamentalAnalyzer:
    """Wraps :class:`QwenModel` with a dual-analysis prompt.

    Extracts both sentiment polarity and qualitative fundamental
    signals (revenue, profitability, debt trends) from a single
    Chinese financial news text.
    """

    def __init__(self, model: QwenModel) -> None:
        self._model = model

    def analyze(self, text: str) -> dict[str, Any]:
        """Analyze *text* and return parsed dict.

        Returns
        -------
        dict
            On success::

                {
                    "score": float,
                    "label": str,
                    "revenue_growth_hint": str,
                    "profitability_hint": str,
                    "debt_hint": str,
                }

            On failure falls back to neutral defaults.
        """
        try:
            raw = self._model.generate(text)
            return self._parse_response(raw)
        except Exception:
            return _neutral_defaults()

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Extract JSON from raw model output."""
        raw = raw.strip()
        # Try direct parse first
        try:
            data = json.loads(raw)
            if "score" in data:
                return _validate(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Scan for embedded JSON block
        match = _JSON_RE.search(raw)
        if match:
            try:
                data = json.loads(match.group())
                if "score" in data:
                    return _validate(data)
            except (json.JSONDecodeError, ValueError):
                pass

        raise ValueError(f"No valid analysis JSON found in: {raw[:200]!r}")


_DUAL_SYSTEM_PROMPT: str = (
    "你是专业的A股市场分析师。请分析以下财经文本，同时给出：\n"
    "1. 整体情感倾向（看涨/看跌/中性）\n"
    "2. 基本面定性线索：营收趋势、盈利能力、偿债能力\n"
    "仅输出JSON格式：\n"
    '{\n'
    '  "score": <float in [-1,1]>,\n'
    '  "label": <"positive"|"neutral"|"negative">,\n'
    '  "revenue_growth_hint": <"positive"|"neutral"|"negative">,\n'
    '  "profitability_hint": <"positive"|"neutral"|"negative">,\n'
    '  "debt_hint": <"positive"|"neutral"|"negative">\n'
    '}'
)


class DualAnalyzer(FundamentalAnalyzer):
    """Dual analyzer that overrides the base prompt with fundamental hints."""

    def analyze(self, text: str) -> dict[str, Any]:
        """Run dual-analysis prompt and parse result."""
        try:
            raw = self._generate_dual(text)
            return self._parse_response(raw)
        except Exception:
            return _neutral_defaults()

    def _generate_dual(self, text: str) -> str:
        if not self._model.is_loaded:
            raise RuntimeError("QwenModel is not loaded. Call .load() first.")

        messages = [
            {"role": "system", "content": _DUAL_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        tokenizer = self._model._tokenizer
        model = self._model._model
        cfg = self._model._config

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([prompt_text], return_tensors="pt")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
        )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded[0].strip()


def _validate(data: dict) -> dict[str, Any]:
    """Validate and clamp dual-analysis fields."""
    score = float(data.get("score", 0.0))
    score = max(-1.0, min(1.0, score))

    label = str(data.get("label", "neutral")).lower()
    if label not in {"positive", "neutral", "negative"}:
        label = "neutral"

    def _hint(key: str) -> str:
        v = str(data.get(key, "neutral")).lower()
        return v if v in {"positive", "neutral", "negative"} else "neutral"

    return {
        "score": score,
        "label": label,
        "revenue_growth_hint": _hint("revenue_growth_hint"),
        "profitability_hint": _hint("profitability_hint"),
        "debt_hint": _hint("debt_hint"),
    }


def _neutral_defaults() -> dict[str, Any]:
    return {
        "score": 0.0,
        "label": "neutral",
        "revenue_growth_hint": "neutral",
        "profitability_hint": "neutral",
        "debt_hint": "neutral",
    }
