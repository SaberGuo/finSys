"""Multi-indicator-set comparison analyzer and report generation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ComparisonReport:
    """Aggregated metrics across multiple training runs."""

    runs: list[dict] = field(default_factory=list)

    def add_run(self, run_id: str, metrics: dict[str, float]) -> None:
        self.runs.append({"run_id": run_id, **metrics})

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.runs)

    def to_csv(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(p, index=False)
        return p

    def to_json(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.runs, f, indent=2, ensure_ascii=False)
        return p


class ComparisonAnalyzer:
    """Compare models trained with different indicator sets."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)

    def _find_run_dirs(self) -> list[Path]:
        """Find all subdirectories containing a metrics CSV and model zip."""
        runs: list[Path] = []
        if self.run_dir.exists():
            for p in self.run_dir.iterdir():
                if p.is_dir() and any(p.rglob("*.csv")):
                    runs.append(p)
        return runs

    def run(self) -> ComparisonReport:
        """Scan run_dir, load metrics, and build a ComparisonReport."""
        report = ComparisonReport()
        for run_path in self._find_run_dirs():
            run_id = run_path.name
            metrics: dict[str, float] = {}

            # Load metrics CSV if present
            for csv_file in run_path.rglob("*_metrics.csv"):
                df = pd.read_csv(csv_file)
                if "metric" in df.columns and "value" in df.columns:
                    for _, row in df.iterrows():
                        try:
                            metrics[row["metric"]] = float(row["value"])
                        except (ValueError, TypeError):
                            pass

            # Load metadata JSON if present
            for meta_file in run_path.rglob("*_metadata.json"):
                with meta_file.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                    metrics.update({k: v for k, v in meta.items() if isinstance(v, (int, float))})

            if metrics:
                report.add_run(run_id, metrics)

        # Sort by sharpe ratio descending
        report.runs.sort(
            key=lambda r: r.get("sharpe", r.get("sharpe_ratio", -float("inf"))),
            reverse=True,
        )
        return report
