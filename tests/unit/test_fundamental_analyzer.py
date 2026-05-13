"""Unit tests for finquant.nlp.fundamental_analyzer."""
from __future__ import annotations

import pytest

from finquant.nlp.fundamental_analyzer import (
    _neutral_defaults,
    _validate,
    FundamentalAnalyzer,
)


class TestValidate:
    def test_valid_positive(self) -> None:
        result = _validate({
            "score": 0.8,
            "label": "positive",
            "revenue_growth_hint": "positive",
            "profitability_hint": "neutral",
            "debt_hint": "negative",
        })
        assert result["score"] == pytest.approx(0.8)
        assert result["label"] == "positive"
        assert result["revenue_growth_hint"] == "positive"

    def test_clamp_score(self) -> None:
        result = _validate({"score": 2.0, "label": "positive"})
        assert result["score"] == pytest.approx(1.0)

    def test_invalid_label_fallback(self) -> None:
        result = _validate({"score": 0.5, "label": "bullish"})
        assert result["label"] == "neutral"

    def test_invalid_hint_fallback(self) -> None:
        result = _validate({"score": 0.5, "revenue_growth_hint": "up"})
        assert result["revenue_growth_hint"] == "neutral"


class TestNeutralDefaults:
    def test_all_neutral(self) -> None:
        d = _neutral_defaults()
        assert d["score"] == pytest.approx(0.0)
        assert d["label"] == "neutral"
        for k in ["revenue_growth_hint", "profitability_hint", "debt_hint"]:
            assert d[k] == "neutral"


class TestFundamentalAnalyzerParse:
    def test_direct_json(self) -> None:
        raw = '{"score": 0.7, "label": "positive", "revenue_growth_hint": "positive"}'
        result = FundamentalAnalyzer._parse_response(raw)
        assert result["score"] == pytest.approx(0.7)
        assert result["label"] == "positive"

    def test_embedded_json(self) -> None:
        raw = 'Some text before {"score": -0.3, "label": "negative"} some after'
        result = FundamentalAnalyzer._parse_response(raw)
        assert result["score"] == pytest.approx(-0.3)
        assert result["label"] == "negative"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            FundamentalAnalyzer._parse_response("No JSON here")
