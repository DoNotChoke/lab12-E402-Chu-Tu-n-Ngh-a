from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


_DEFAULT_ATTRS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _DEFAULT_ATTRS and not key.startswith("_")
        }
        if extras:
            payload["extra"] = extras

        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level_name: str) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root.handlers.clear()
    root.addHandler(handler)
