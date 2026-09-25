import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluate import compute_metrics, evaluate_model
from src.ingest import generate_customers
from src.preprocess import build_pipeline, split_features_target
from src.utils import read_json


def test_perfect_predictions() -> None:
    y = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["n_samples"] == 4


def test_threshold_changes_class_metrics_not_ranking() -> None:
    y = np.array([0, 1, 1, 0, 1])
    proba = np.array([0.2, 0.4, 0.7, 0.3, 0.9])
    low = compute_metrics(y, proba, threshold=0.35)
    high = compute_metrics(y, proba, threshold=0.5)
    assert low["recall"] == 1.0
    assert high["recall"] == pytest.approx(2 / 3)
    assert low["roc_auc"] == high["roc_auc"]


def test_single_class_gives_nan_ranking_metrics() -> None:
    metrics = compute_metrics(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.8]))
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["average_precision"])
    assert metrics["recall"] == 1.0


def test_evaluate_model_writes_all_artifacts(tmp_path: Path) -> None:
    X, y = split_features_target(generate_customers(800, seed=6))
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    metrics = evaluate_model(model, X, y, 0.35, output_dir=tmp_path, with_importance=True)

    assert 0.75 < metrics["roc_auc"] <= 1.0
    for name in [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "classification_report.json",
        "metrics.json",
        "feature_importance.csv",
        "feature_importance.png",
    ]:
        assert (tmp_path / name).stat().st_size > 0, name
    assert read_json(tmp_path / "metrics.json")["roc_auc"] == pytest.approx(metrics["roc_auc"])
