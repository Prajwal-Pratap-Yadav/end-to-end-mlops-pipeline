import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.utils import (
    JsonFormatter,
    configure_logging,
    read_json,
    to_jsonable,
    utc_timestamp_slug,
    write_json,
)


def test_to_jsonable_handles_numpy_nan_paths_and_dates() -> None:
    value = {
        "int": np.int64(3),
        "float": np.float32(0.5),
        "nan": float("nan"),
        "inf": math.inf,
        "path": Path("a/b"),
        "when": datetime(2026, 1, 1, tzinfo=UTC),
        "nested": [np.float64(1.5), (1, 2)],
    }
    assert to_jsonable(value) == {
        "int": 3,
        "float": 0.5,
        "nan": None,
        "inf": None,
        "path": "a/b",
        "when": "2026-01-01T00:00:00+00:00",
        "nested": [1.5, [1, 2]],
    }


def test_write_and_read_json(tmp_path: Path) -> None:
    target = write_json({"b": np.float64(2.0), "a": 1}, tmp_path / "deep" / "x.json")
    assert read_json(target) == {"a": 1, "b": 2.0}


def test_json_log_formatter_includes_exceptions() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "svc", logging.ERROR, __file__, 1, "failed %s", ("x",), sys.exc_info()
        )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "failed x"
    assert payload["level"] == "ERROR"
    assert "ValueError: boom" in payload["exception"]


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_configure_logging(fmt: str) -> None:
    root = logging.getLogger()
    previous = root.handlers[:], root.level
    try:
        configure_logging("debug", fmt)
        assert root.level == logging.DEBUG
        assert isinstance(root.handlers[0].formatter, JsonFormatter) == (fmt == "json")
        assert logging.getLogger("mlflow").level == logging.WARNING
    finally:
        root.handlers, root.level = previous[0], previous[1]


def test_timestamp_slug_format() -> None:
    slug = utc_timestamp_slug()
    assert len(slug) == 16
    assert slug.endswith("Z")
    assert "T" in slug
