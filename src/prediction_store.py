"""Durable log of served predictions and the ground truth that arrives later.

The API writes one row per prediction (inputs, probability, model version); the
``/feedback`` endpoint attaches the real outcome once it is known. The monitoring
job reads recent rows to detect drift and measure live performance, and the
retraining job uses labeled rows as fresh training data - closing the loop.

SQLite (WAL mode) keeps the local stack dependency-free and is safe for one
writer process plus concurrent readers on the same host. In a larger deployment
this interface would sit on Postgres or a warehouse table fed by a stream.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.schema import feature_names
from src.utils import ensure_parent_dir, utc_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    created_ts        REAL NOT NULL,
    model_name        TEXT,
    model_version     TEXT,
    features          TEXT NOT NULL,
    churn_probability REAL NOT NULL,
    churn_prediction  INTEGER NOT NULL,
    actual            INTEGER,
    feedback_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_ts ON predictions (created_ts);
CREATE INDEX IF NOT EXISTS idx_predictions_labeled ON predictions (created_ts)
    WHERE actual IS NOT NULL;
"""

METADATA_COLUMNS = [
    "prediction_id",
    "created_at",
    "model_name",
    "model_version",
    "churn_probability",
    "churn_prediction",
    "actual",
]


@dataclass(frozen=True)
class PredictionRecord:
    """One served prediction."""

    prediction_id: str
    features: dict[str, Any]
    churn_probability: float
    churn_prediction: int
    model_name: str | None
    model_version: str | None
    created_at: datetime


class PredictionStore:
    """SQLite-backed prediction and feedback log."""

    def __init__(self, path: str | Path) -> None:
        self.path = ensure_parent_dir(path)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A short-lived connection per operation is cheap for SQLite and avoids
        # sharing connections across the API's worker threads.
        conn = sqlite3.connect(self.path, timeout=30)
        # WAL + synchronous=NORMAL is the recommended durability/throughput trade-off:
        # the database cannot corrupt, only the last commits may be lost on power loss.
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    def log_predictions(self, records: Sequence[PredictionRecord]) -> None:
        """Persist a batch of predictions atomically."""
        rows = [
            (
                r.prediction_id,
                r.created_at.isoformat(),
                r.created_at.timestamp(),
                r.model_name,
                r.model_version,
                json.dumps(r.features),
                float(r.churn_probability),
                int(r.churn_prediction),
            )
            for r in records
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO predictions (prediction_id, created_at, created_ts, model_name, "
                "model_version, features, churn_probability, churn_prediction) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def add_feedback(self, prediction_id: str, actual: int) -> bool:
        """Attach the observed outcome to a prediction.

        Returns:
            ``False`` if the prediction id is unknown.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE predictions SET actual = ?, feedback_at = ? WHERE prediction_id = ?",
                (int(actual), utc_now().isoformat(), prediction_id),
            )
            return cursor.rowcount > 0

    def count(self, labeled_only: bool = False) -> int:
        """Number of logged predictions (optionally only those with feedback)."""
        query = "SELECT COUNT(*) FROM predictions"
        if labeled_only:
            query += " WHERE actual IS NOT NULL"
        with self._connect() as conn:
            (total,) = conn.execute(query).fetchone()
        return int(total)

    def recent(self, limit: int) -> pd.DataFrame:
        """The ``limit`` most recent predictions, oldest first, features expanded."""
        return self._query(
            "SELECT * FROM predictions ORDER BY created_ts DESC LIMIT ?", (int(limit),)
        )

    def labeled(self, limit: int) -> pd.DataFrame:
        """The ``limit`` most recent predictions that have ground truth, oldest first."""
        return self._query(
            "SELECT * FROM predictions WHERE actual IS NOT NULL ORDER BY created_ts DESC LIMIT ?",
            (int(limit),),
        )

    def _query(self, sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        records = []
        for row in reversed(rows):  # queries fetch newest first; return chronological order
            record = {column: row[column] for column in METADATA_COLUMNS}
            record.update(json.loads(row["features"]))
            records.append(record)
        columns = METADATA_COLUMNS + feature_names()
        frame = pd.DataFrame.from_records(records, columns=columns)
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, format="ISO8601")
        return frame
