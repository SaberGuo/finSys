from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from finquant.utils.logging import JsonFormatter, get_logger


class TestJsonFormatter:
    def test_format_returns_valid_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello"
        assert parsed["name"] == "test"
        assert "timestamp" in parsed

    def test_format_with_non_ascii(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0, msg="中文", args=(), exc_info=None
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["message"] == "中文"


class TestGetLogger:
    def test_returns_logger_with_handlers(self, tmp_path: pytest.TempPathFactory) -> None:
        log_dir = str(tmp_path / "logs")
        logger = get_logger("finquant.test", level="DEBUG", log_dir=log_dir)
        assert logger.name == "finquant.test"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2  # stream + file

    def test_does_not_duplicate_handlers(self, tmp_path: pytest.TempPathFactory) -> None:
        log_dir = str(tmp_path / "logs")
        logger1 = get_logger("finquant.dedup", level="INFO", log_dir=log_dir)
        initial_count = len(logger1.handlers)
        logger2 = get_logger("finquant.dedup", level="INFO", log_dir=log_dir)
        assert len(logger2.handlers) == initial_count

    def test_json_output_to_file(self, tmp_path: pytest.TempPathFactory) -> None:
        log_dir = str(tmp_path / "logs")
        logger = get_logger("finquant.filetest", level="INFO", log_dir=log_dir)
        logger.info("file entry")

        log_file = tmp_path / "logs" / "finsys.log"
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        last_line = json.loads(lines[-1])
        assert last_line["message"] == "file entry"
