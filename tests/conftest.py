"""Shared fixtures.

Integration tests train real (small) models against a throwaway SQLite MLflow
store in a temporary directory, so they exercise the same code paths as
production without touching the developer's ``mlflow.db``.
"""

from __future__ import annotations

import itertools
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig, load_config
from src.ingest import DriftProfile, generate_customers
from src.predict import LoadedModel, load_model
from src.prediction_store import PredictionRecord, PredictionStore
from src.schema import TARGET_COLUMN, feature_names
from src.train import TrainingResult, run_training
from src.utils import utc_now

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

# Strictly increasing timestamps for every record the helpers log in this session,
# so "most recent" queries see batches in the order the test wrote them.
_CLOCK_START = utc_now()
_CLOCK = itertools.count()


def make_test_config(root: Path, **extra: Any) -> AppConfig:
    """Production config with small data, fast models and every path under ``root``."""
    overrides: dict[str, Any] = {
        "data": {"n_samples": 1500, "min_rows": 200, "raw_path": str(root / "data/raw/churn.csv")},
        "training": {
            "cv_folds": 3,
            "reference_max_rows": 1000,
            "candidates": [
                {
                    "name": "logistic_regression",
                    "estimator": "logistic_regression",
                    "param_grid": {"C": [1.0]},
                },
                {
                    "name": "hist_gradient_boosting",
                    "estimator": "hist_gradient_boosting",
                    "param_grid": {"max_iter": [60], "max_depth": [3], "learning_rate": [0.1]},
                },
            ],
        },
        "registry": {"promotion": {"min_metrics": {"roc_auc": 0.7, "f1": 0.3}}},
        "mlflow": {"tracking_uri": f"sqlite:///{(root / 'mlflow.db').as_posix()}"},
        "serving": {
            "prediction_log_path": str(root / "data/predictions/predictions.db"),
            "reload_interval_seconds": 0,
        },
        "monitoring": {
            "window_size": 1500,
            "min_window_size": 100,
            "min_labeled_for_performance": 50,
            "report_dir": str(root / "reports"),
            "retrain": {
                "min_labeled_samples": 200,
                "max_fresh_rows": 1500,
                "min_training_rows": 200,
                "cooldown_minutes": 0,
            },
        },
    }
    for section, values in extra.items():
        overrides.setdefault(section, {}).update(values)
    # environ={} keeps developer/CI MLOPS__* variables from leaking into tests.
    return load_config(CONFIG_PATH, overrides=overrides, environ={})


@dataclass
class TrainedEnv:
    """A temporary MLflow registry with a trained champion."""

    config: AppConfig
    result: TrainingResult
    root: Path

    def load_champion(self) -> LoadedModel:
        return load_model(self.config)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    """Test config whose MLflow store, data and reports live in ``tmp_path``."""
    return make_test_config(tmp_path)


@pytest.fixture(scope="session")
def trained_env(tmp_path_factory: pytest.TempPathFactory) -> TrainedEnv:
    """Session-wide registry with champion v1 - for tests that only read it."""
    root = tmp_path_factory.mktemp("trained")
    cfg = make_test_config(root)
    result = run_training(cfg)
    return TrainedEnv(cfg, result, root)


@pytest.fixture
def fresh_env(tmp_path: Path) -> TrainedEnv:
    """Per-test registry with champion v1 - for tests that modify the registry."""
    cfg = make_test_config(tmp_path)
    return TrainedEnv(cfg, run_training(cfg), tmp_path)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """CLIs and the app reconfigure root logging; undo it so handlers never outlive a test."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level


@pytest.fixture(autouse=True)
def _no_active_mlflow_run() -> Iterator[None]:
    """Guarantee a failing test never leaks an active MLflow run into the next one."""
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()


def log_scored_customers(
    store: PredictionStore,
    model: LoadedModel,
    customers: pd.DataFrame,
    threshold: float = 0.35,
    with_feedback: bool = True,
) -> list[str]:
    """Score customers with ``model`` and write them to the prediction log in time order."""
    probabilities = model.predict_proba(customers)
    records = []
    for row, probability in zip(
        customers[feature_names()].to_dict("records"), probabilities, strict=True
    ):
        features = {
            k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()
        }
        records.append(
            PredictionRecord(
                prediction_id=uuid.uuid4().hex,
                features=features,
                churn_probability=float(probability),
                churn_prediction=int(probability >= threshold),
                model_name=model.name,
                model_version=model.version,
                created_at=_CLOCK_START + timedelta(milliseconds=next(_CLOCK)),
            )
        )
    store.log_predictions(records)
    if with_feedback:
        for record, actual in zip(records, customers[TARGET_COLUMN], strict=True):
            store.add_feedback(record.prediction_id, int(actual))
    return [r.prediction_id for r in records]


def drifted_customers(n: int, seed: int, strength: float = 1.0) -> pd.DataFrame:
    """Customers from a shifted market (covariate + concept drift)."""
    return generate_customers(n, seed=seed, drift=DriftProfile.from_strength(strength))
