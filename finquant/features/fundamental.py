"""Fundamental feature extraction: FundamentalRecord and processor."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from finquant.nlp.model import QwenModel


@dataclass
class FundamentalRecord:
    """Single fundamental analysis result (from Qwen dual-analysis).

    Attributes
    ----------
    date:
        Publication date (``YYYY-MM-DD``).
    tic:
        Stock ticker in dot-notation.
    score:
        Sentiment score in ``[-1.0, 1.0]``.
    label:
        ``"positive"``, ``"neutral"``, or ``"negative"``.
    revenue_growth_hint:
        Qualitative revenue trend signal.
    profitability_hint:
        Qualitative profitability signal.
    debt_hint:
        Qualitative debt/solvency signal.
    """

    date: str
    tic: str
    score: float
    label: str
    revenue_growth_hint: str
    profitability_hint: str
    debt_hint: str

    def __post_init__(self) -> None:
        if not (-1.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [-1, 1], got {self.score}")
        for field in ["label", "revenue_growth_hint", "profitability_hint", "debt_hint"]:
            val = getattr(self, field)
            if val not in {"positive", "neutral", "negative"}:
                raise ValueError(f"invalid {field}: {val!r}")

    @classmethod
    def neutral(cls, date: str, tic: str) -> "FundamentalRecord":
        """Return a neutral record for graceful degradation."""
        return cls(
            date=date,
            tic=tic,
            score=0.0,
            label="neutral",
            revenue_growth_hint="neutral",
            profitability_hint="neutral",
            debt_hint="neutral",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "tic": self.tic,
            "score": self.score,
            "label": self.label,
            "revenue_growth_hint": self.revenue_growth_hint,
            "profitability_hint": self.profitability_hint,
            "debt_hint": self.debt_hint,
        }

    @property
    def revenue_growth_score(self) -> float:
        """Map hint to numeric score for observation space."""
        return _HINT_MAP.get(self.revenue_growth_hint, 0.0)

    @property
    def profitability_score(self) -> float:
        return _HINT_MAP.get(self.profitability_hint, 0.0)

    @property
    def debt_score(self) -> float:
        return _HINT_MAP.get(self.debt_hint, 0.0)


_HINT_MAP: dict[str, float] = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


class FundamentalProcessor:
    """Orchestrates Qwen dual-analysis over a batch of news texts.

    Parameters
    ----------
    config:
        ``SentimentConfig`` from app configuration (re-used for Qwen
        model parameters).
    """

    def __init__(self, config) -> None:
        self._config = config
        self._qwen = QwenModel(config)
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._qwen.load()
            self._loaded = True

    def process_texts(
        self, texts: list[dict[str, str]]
    ) -> list[FundamentalRecord]:
        """Analyze a list of ``{date, tic, text}`` dicts with dual-analysis.

        Returns
        -------
        list[FundamentalRecord]
        """
        from finquant.nlp.fundamental_analyzer import DualAnalyzer

        self._ensure_loaded()
        analyzer = DualAnalyzer(self._qwen)

        # Deduplicate by (date, tic, text)
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, str]] = []
        for item in texts:
            key = (item["date"], item["tic"], item["text"])
            if key not in seen:
                seen.add(key)
                unique.append(item)

        records: list[FundamentalRecord] = []
        for item in unique:
            result = analyzer.analyze(item["text"])
            try:
                rec = FundamentalRecord(
                    date=item["date"],
                    tic=item["tic"],
                    score=result["score"],
                    label=result["label"],
                    revenue_growth_hint=result["revenue_growth_hint"],
                    profitability_hint=result["profitability_hint"],
                    debt_hint=result["debt_hint"],
                )
            except (ValueError, KeyError):
                rec = FundamentalRecord.neutral(item["date"], item["tic"])
            records.append(rec)

        return records

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        verbose: bool = False,
    ) -> Path:
        """Read JSONL news, run dual-analysis, save expanded records.

        Output JSONL contains the same fields as :class:`FundamentalRecord`
        plus numeric ``revenue_growth_score``, ``profitability_score``,
        ``debt_score`` for downstream fusion.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        texts: list[dict[str, str]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    # News fetcher may include title; concatenate for analysis
                    text_parts = []
                    if obj.get("title"):
                        text_parts.append(obj["title"])
                    if obj.get("content"):
                        text_parts.append(obj["content"])
                    texts.append(
                        {
                            "date": obj["date"],
                            "tic": obj["tic"],
                            "text": "\n".join(text_parts),
                        }
                    )

        if verbose:
            print(f"Processing {len(texts)} records from {input_path} ...")

        records = self.process_texts(texts)

        out_path = output_dir / "fundamental_records.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                d = rec.to_dict()
                # Append numeric scores for fusion
                d["revenue_growth_score"] = rec.revenue_growth_score
                d["profitability_score"] = rec.profitability_score
                d["debt_score"] = rec.debt_score
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        if verbose:
            print(f"Saved {len(records)} records to {out_path}")

        return out_path
