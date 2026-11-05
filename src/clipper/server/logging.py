"""Rotating operational logs, separate from worker stdout/stderr."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure(level: str, log_dir: Path) -> list[logging.Handler]:
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        RotatingFileHandler(
            log_dir / "server.log", maxBytes=2 * 1024**2, backupCount=5, encoding="utf-8"
        ),
        logging.StreamHandler(),
    ]
    logger = logging.getLogger("clipper")
    logger.setLevel(level)
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return handlers
