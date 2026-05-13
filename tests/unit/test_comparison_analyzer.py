"""Unit tests for ComparisonAnalyzer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from finquant.training.comparison import ComparisonAnalyzer, ComparisonReport


class TestComparisonReport:
    """T044: validate ComparisonReport data structures."""

    def test_add_run(self) -> None:
        report = ComparisonReport()
        report.add_run("run_a", {"sharpe": 1.5, "max_drawdown": -0.1})
        assert len(report.runs) == 1
        assert report.runs[0]["run_id"] == "run_a"

    def test_to_dataframe(self) -> None:
        report = ComparisonReport()
        report.add_run("run_a", {"sharpe": 1.5})
        report.add_run("run_b", {"sharpe": 2.0})
        df = report.to_dataframe()
        assert len(df) == 2
        assert set(df["run_id"]) == {"run_a", "run_b"}

    def test_to_csv(self, tmp_path: Path) -> None:
        report = ComparisonReport()
        report.add_run("run_a", {"sharpe": 1.5})
        path = report.to_csv(tmp_path / "report.csv")
        assert path.exists()


class TestComparisonAnalyzer:
    """T044: validate ComparisonAnalyzer scanning."""

    def test_scan_and_rank(self, tmp_path: Path) -> None:
        # Create fake run directories
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        pd.DataFrame({"metric": ["sharpe"], "value": [1.5]}).to_csv(run_a / "run_a_metrics.csv", index=False)
        with (run_a / "metadata.json").open("w") as f:
            json.dump({"sharpe": 1.5, "total_return": 0.1}, f)

        run_b = tmp_path / "run_b"
        run_b.mkdir()
        pd.DataFrame({"metric": ["sharpe"], "value": [2.0]}).to_csv(run_b / "run_b_metrics.csv", index=False)
        with (run_b / "metadata.json").open("w") as f:
            json.dump({"sharpe": 2.0, "total_return": 0.2}, f)

        analyzer = ComparisonAnalyzer(tmp_path)
        report = analyzer.run()
        assert len(report.runs) == 2
        assert report.runs[0]["run_id"] == "run_b"  # higher sharpe first
