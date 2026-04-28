"""Qwen sentiment analyzer: prompt templates and JSON response parsing (T046)."""
from __future__ import annotations

import json
import re

from finquant.nlp.model import QwenModel

_JSON_RE = re.compile(r'\{[^{}]*"score"[^{}]*\}', re.DOTALL)


class SentimentAnalyzer:
    """Wraps :class:`QwenModel` with prompt templates and JSON parsing.

    Parameters
    ----------
    model:
        Loaded (or unloaded) :class:`QwenModel` instance.
    """

    def __init__(self, model: QwenModel) -> None:
        self._model = model

    def analyze(self, text: str) -> dict:
        """Analyze *text* and return parsed sentiment dict.

        Returns
        -------
        dict
            ``{"score": float, "label": str}`` on success,
            or ``{"score": 0.0, "label": "neutral"}`` on any failure.
        """
        try:
            raw = self._model.generate(text)
            return self._parse_response(raw)
        except Exception:
            return {"score": 0.0, "label": "neutral"}

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Extract JSON from *raw* model output.

        Tolerates surrounding text by scanning for the first ``{...}`` block
        that contains a ``"score"`` key.

        Raises
        ------
        ValueError
            If no valid JSON with required keys is found.
        """
        # Try direct parse first
        raw = raw.strip()
        try:
            data = json.loads(raw)
            if "score" in data and "label" in data:
                return _validate_sentiment_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Scan for embedded JSON block
        match = _JSON_RE.search(raw)
        if match:
            try:
                data = json.loads(match.group())
                if "score" in data and "label" in data:
                    return _validate_sentiment_dict(data)
            except (json.JSONDecodeError, ValueError):
                pass

        raise ValueError(f"No valid sentiment JSON found in: {raw[:200]!r}")


def _validate_sentiment_dict(data: dict) -> dict:
    score = float(data["score"])
    if not (-1.0 <= score <= 1.0):
        score = max(-1.0, min(1.0, score))  # clamp instead of raising
    label = str(data["label"]).lower()
    if label not in {"positive", "neutral", "negative"}:
        label = "neutral"
    return {"score": score, "label": label}
