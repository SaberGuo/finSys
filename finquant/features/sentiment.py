"""Sentiment feature extraction: SentimentRecord and SentimentProcessor (T047)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SentimentRecord:
    """Single financial text analysis result.

    Attributes
    ----------
    date:
        Publication date (``YYYY-MM-DD``).
    tic:
        Stock ticker in dot-notation (e.g. ``000001.SZ``).
    score:
        Sentiment score in ``[-1.0, 1.0]``. Negative = bearish, positive = bullish.
    label:
        ``"positive"``, ``"neutral"``, or ``"negative"``.
    """

    date: str
    tic: str
    score: float
    label: str

    def __post_init__(self) -> None:
        if not (-1.0 <= self.score <= 1.0):
            raise ValueError(
                f"score must be in [-1, 1], got {self.score}"
            )
        if self.label not in {"positive", "neutral", "negative"}:
            raise ValueError(f"invalid label: {self.label!r}")

    @classmethod
    def neutral(cls, date: str, tic: str) -> "SentimentRecord":
        """Return a neutral record (score=0.0) for graceful degradation."""
        return cls(date=date, tic=tic, score=0.0, label="neutral")

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "tic": self.tic,
            "score": self.score,
            "label": self.label,
        }


class SentimentProcessor:
    """Orchestrates Qwen inference over a batch of news texts.

    Parameters
    ----------
    config:
        ``SentimentConfig`` from app configuration.
    """

    def __init__(self, config) -> None:
        self._config = config
        from finquant.nlp.model import QwenModel

        self._qwen = QwenModel(config)
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._qwen.load()
            self._loaded = True

    def process_texts(
        self, texts: list[dict[str, str]]
    ) -> list[SentimentRecord]:
        """Analyze a list of ``{date, tic, text}`` dicts.

        Duplicates (same ``date + tic + text``) are deduplicated before
        inference to avoid redundant model calls.

        Parameters
        ----------
        texts:
            List of dicts with keys ``date``, ``tic``, ``text``.

        Returns
        -------
        list[SentimentRecord]
        """
        from finquant.nlp.analyzer import SentimentAnalyzer

        self._ensure_loaded()
        analyzer = SentimentAnalyzer(self._qwen)

        # Deduplicate by (date, tic, text)
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, str]] = []
        for item in texts:
            key = (item["date"], item["tic"], item["text"])
            if key not in seen:
                seen.add(key)
                unique.append(item)

        records: list[SentimentRecord] = []
        for item in unique:
            result = analyzer.analyze(item["text"])
            try:
                rec = SentimentRecord(
                    date=item["date"],
                    tic=item["tic"],
                    score=result["score"],
                    label=result["label"],
                )
            except (ValueError, KeyError):
                rec = SentimentRecord.neutral(item["date"], item["tic"])
            records.append(rec)

        return records

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        verbose: bool = False,
    ) -> Path:
        """Read a JSONL file of news texts, analyze each, and save results.

        Parameters
        ----------
        input_path:
            Path to JSONL file with ``{date, tic, text}`` records per line.
        output_dir:
            Output directory. Saves ``sentiment_records.jsonl`` there.
        verbose:
            Print progress if True.

        Returns
        -------
        Path
            Path to the saved JSONL file.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        texts: list[dict[str, str]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(json.loads(line))

        if verbose:
            print(f"Processing {len(texts)} records from {input_path} ...")

        records = self.process_texts(texts)

        out_path = output_dir / "sentiment_records.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

        if verbose:
            print(f"Saved {len(records)} records to {out_path}")

        return out_path
