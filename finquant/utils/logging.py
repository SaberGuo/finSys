from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level.upper())
    logger.propagate = False

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonFormatter())
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(log_dir) / "finsys.log", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger
