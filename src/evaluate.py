"""Model evaluation: metrics, diagnostic plots and feature importance.

Used by training (test-set evaluation logged to MLflow), by the promotion gate
(champion vs challenger on identical data) and by the monitoring job (live
performance on labeled feedback). It can also evaluate any registered model
against any labeled CSV:

    python -m src.evaluate --data data/raw/churn.csv
    python -m src.evaluate --model-uri models:/churn-classifier/2 --data drifted.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering in containers and CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.utils import write_json

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute threshold-dependent and ranking metrics for a binary classifier.

    Args:
        y_true: Ground-truth labels (0/1).
        y_proba: Predicted probability of the positive class.
        threshold: Decision threshold for class predictions.

    Returns:
        Metric name to value. Ranking metrics are ``nan`` when only one class is present.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(y_proba, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y_true_arr)) == 2

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_arr, proba)) if both_classes else float("nan"),
        "average_precision": (
            float(average_precision_score(y_true_arr, proba)) if both_classes else float("nan")
        ),
        "log_loss": float(log_loss(y_true_arr, proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true_arr, proba)),
        "positive_rate": float(y_pred.mean()),
        "n_samples": float(len(y_true_arr)),
    }


def save_evaluation_plots(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    output_dir: str | Path,
) -> list[Path]:
    """Render confusion matrix, ROC and precision-recall curves as PNG files.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted positive-class probabilities.
        threshold: Decision threshold used for the confusion matrix.
        output_dir: Directory to write the images to.

    Returns:
        Paths of the written images.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true_arr, y_pred, display_labels=["retained", "churned"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    paths.append(_save(fig, out / "confusion_matrix.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true_arr, y_proba, ax=ax, name="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC curve")
    paths.append(_save(fig, out / "roc_curve.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true_arr, y_proba, ax=ax, name="model")
    ax.set_title("Precision-recall curve")
    paths.append(_save(fig, out / "precision_recall_curve.png"))
    return paths


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def compute_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    max_rows: int = 1000,
) -> pd.DataFrame:
    """Model-agnostic permutation importance on raw input features (ROC-AUC drop).

    Args:
        model: Fitted pipeline accepting raw features.
        X: Evaluation features.
        y: Evaluation labels.
        random_state: Seed for row sampling and permutations.
        max_rows: Row cap to keep evaluation fast.

    Returns:
        DataFrame with ``feature``, ``importance_mean`` and ``importance_std``, sorted.
    """
    if len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=random_state)
        y = y.loc[X.index]
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=5, random_state=random_state, n_jobs=1
    )
    importance = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(importance: pd.DataFrame, path: str | Path) -> Path:
    """Render permutation importance as a horizontal bar chart."""
    ordered = importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"])
    ax.set_xlabel("Mean ROC-AUC decrease when permuted")
    ax.set_title("Permutation feature importance")
    return _save(fig, Path(path))


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    output_dir: str | Path | None = None,
    with_importance: bool = False,
) -> dict[str, float]:
    """Score a fitted pipeline and optionally write every evaluation artifact.

    Args:
        model: Fitted pipeline.
        X: Evaluation features.
        y: Evaluation labels.
        threshold: Decision threshold.
        output_dir: If given, plots, reports and importance are written here.
        with_importance: Also compute permutation importance (slower).

    Returns:
        Metrics dictionary.
    """
    proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, proba, threshold)
    if output_dir is not None:
        out = Path(output_dir)
        save_evaluation_plots(y, proba, threshold, out)
        report = classification_report(
            np.asarray(y).astype(int),
            (proba >= threshold).astype(int),
            target_names=["retained", "churned"],
            output_dict=True,
            zero_division=0,
        )
        write_json(report, out / "classification_report.json")
        write_json(metrics, out / "metrics.json")
        if with_importance:
            importance = compute_feature_importance(model, X, y)
            importance.to_csv(out / "feature_importance.csv", index=False)
            save_feature_importance_plot(importance, out / "feature_importance.png")
    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI: evaluate a registered model against a labeled CSV."""
    from src.config import load_config
    from src.predict import load_model
    from src.preprocess import split_features_target
    from src.utils import configure_logging
    from src.validation import assert_valid

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--data", required=True, help="Labeled CSV to evaluate on")
    parser.add_argument("--model-uri", default=None, help="Defaults to the champion alias")
    parser.add_argument("--output-dir", default=None, help="Write plots and reports here")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    loaded = load_model(config, model_uri=args.model_uri)
    frame = pd.read_csv(args.data)
    assert_valid(frame, require_target=True)
    X, y = split_features_target(frame)
    metrics = evaluate_model(
        loaded.model, X, y, config.training.decision_threshold, args.output_dir
    )
    print(json.dumps({"model": loaded.describe(), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
