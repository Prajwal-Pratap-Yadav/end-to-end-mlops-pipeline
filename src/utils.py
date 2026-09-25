"""Shared helpers: logging setup, filesystem and JSON utilities, timestamps."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

LOG_LEVEL_ENV = "LOG_LEVEL"
LOG_FORMAT_ENV = "LOG_FORMAT"
_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON, the format log shippers expect."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record (and any exception info) to JSON."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure root logging once for CLIs and services.

    Args:
        level: Log level name. Defaults to ``LOG_LEVEL`` env var or ``INFO``.
        fmt: ``"text"`` or ``"json"``. Defaults to ``LOG_FORMAT`` env var or ``text``.
    """
    level_name = (level or os.environ.get(LOG_LEVEL_ENV, "INFO")).upper()
    format_name = (fmt or os.environ.get(LOG_FORMAT_ENV, "text")).lower()

    handler = logging.StreamHandler(sys.stdout)
    if format_name == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level_name)
    # Third-party libraries are noisy at INFO; keep them at WARNING.
    for noisy in ("mlflow", "alembic", "urllib3", "botocore", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory of ``path`` if needed and return it as a Path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars, paths and NaN into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json(data: Any, path: str | Path) -> Path:
    """Write ``data`` as pretty JSON, creating parent directories."""
    target = ensure_parent_dir(path)
    target.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")
    return target


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=UTC)


def utc_timestamp_slug() -> str:
    """Return a filesystem-safe UTC timestamp such as ``20260101T120000Z``."""
    return utc_now().strftime("%Y%m%dT%H%M%SZ")
