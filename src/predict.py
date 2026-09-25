"""Model loading from the MLflow registry and batch inference.

The same loader backs the online API, the batch scorer below, the monitoring job
(prediction drift) and the retraining job (champion evaluation).

Usage:
    python -m src.predict --input data/raw/churn.csv --output predictions.csv
    python -m src.predict --input new_customers.csv --model-uri models:/churn-classifier/1
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.config import AppConfig, load_config
from src.registry import configure_mlflow, get_version_by_alias
from src.schema import ID_COLUMN, feature_names
from src.utils import configure_logging, ensure_parent_dir, utc_now
from src.validation import assert_valid

logger = logging.getLogger(__name__)

_ALIAS_URI = re.compile(r"^models:/(?P<name>[^/@]+)@(?P<alias>[^/]+)$")
_VERSION_URI = re.compile(r"^models:/(?P<name>[^/@]+)/(?P<version>\d+)$")


class ModelNotFoundError(RuntimeError):
    """Raised when the requested model (or alias) does not exist in the registry."""


@dataclass
class LoadedModel:
    """A fitted pipeline plus the registry metadata needed for lineage."""

    model: Pipeline
    model_uri: str
    name: str | None = None
    version: str | None = None
    run_id: str | None = None
    loaded_at: datetime = field(default_factory=utc_now)
    training_metrics: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return churn probabilities for raw feature rows.

        Type coercion happens inside the pipeline's first step, so only the contract
        columns are selected here.
        """
        return np.asarray(self.model.predict_proba(frame[feature_names()])[:, 1], dtype=float)

    def describe(self) -> dict[str, Any]:
        """Metadata suitable for API responses and logs."""
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "loaded_at": self.loaded_at.isoformat(),
        }


def resolve_model_uri(
    client: MlflowClient, config: AppConfig, model_uri: str | None = None
) -> tuple[str, str | None, str | None, str | None]:
    """Resolve a model URI to an immutable ``models:/<name>/<version>`` URI.

    Resolving aliases up front means the loaded artifact and the reported version
    can never disagree, even if the alias moves while the model is loading.

    Args:
        client: MLflow client.
        config: Application config (supplies the default champion alias).
        model_uri: ``models:/name@alias``, ``models:/name/version`` or any other
            MLflow model URI. Defaults to the configured champion alias.

    Returns:
        ``(uri, name, version, run_id)``; name/version/run_id are ``None`` for
        non-registry URIs.

    Raises:
        ModelNotFoundError: If an alias does not resolve.
    """
    uri = model_uri or f"models:/{config.registry.model_name}@{config.registry.champion_alias}"
    if match := _ALIAS_URI.match(uri):
        version = get_version_by_alias(client, match["name"], match["alias"])
        if version is None:
            raise ModelNotFoundError(f"No model version for {uri}")
        return (
            f"models:/{match['name']}/{version.version}",
            match["name"],
            str(version.version),
            version.run_id,
        )
    if match := _VERSION_URI.match(uri):
        model_version = client.get_model_version(match["name"], match["version"])
        return uri, match["name"], match["version"], model_version.run_id
    return uri, None, None, None


def load_model(
    config: AppConfig,
    model_uri: str | None = None,
    client: MlflowClient | None = None,
) -> LoadedModel:
    """Load a model (the champion by default) from the MLflow registry.

    Args:
        config: Application configuration.
        model_uri: Optional explicit model URI.
        client: Optional MLflow client (one is created from the config otherwise).

    Returns:
        The loaded model with lineage metadata.
    """
    client = client or configure_mlflow(config)
    uri, name, version, run_id = resolve_model_uri(client, config, model_uri)
    logger.info("Loading model %s", uri)
    model = mlflow.sklearn.load_model(uri)
    metrics: dict[str, float] = {}
    if run_id:
        run_metrics = client.get_run(run_id).data.metrics
        metrics = {
            k.removeprefix("test_"): v for k, v in run_metrics.items() if k.startswith("test_")
        }
    return LoadedModel(
        model=model,
        model_uri=uri,
        name=name,
        version=version,
        run_id=run_id,
        training_metrics=metrics,
    )


def predict_frame(loaded: LoadedModel, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Score a table of customers.

    Args:
        loaded: Model to use.
        frame: Raw customer rows (extra columns such as ``customer_id`` are kept).
        threshold: Decision threshold for ``churn_prediction``.

    Returns:
        Copy of ``frame`` with ``churn_probability`` and ``churn_prediction`` columns.
    """
    result = frame.copy()
    proba = loaded.predict_proba(frame)
    result["churn_probability"] = np.round(proba, 6)
    result["churn_prediction"] = (proba >= threshold).astype(int)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI: batch-score a CSV with the champion (or a given) model."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--input", required=True, help="CSV of customers to score")
    parser.add_argument("--output", required=True, help="Where to write predictions")
    parser.add_argument("--model-uri", default=None, help="Defaults to the champion alias")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    loaded = load_model(config, model_uri=args.model_uri)
    frame = pd.read_csv(args.input)
    assert_valid(frame, require_target=False)
    scored = predict_frame(loaded, frame, config.training.decision_threshold)
    columns = [c for c in (ID_COLUMN, "churn_probability", "churn_prediction") if c in scored]
    scored[columns].to_csv(ensure_parent_dir(args.output), index=False)
    logger.info(
        "Scored %d rows with %s v%s -> %s (predicted churn rate %.1f%%)",
        len(scored),
        loaded.name,
        loaded.version,
        args.output,
        100 * scored["churn_prediction"].mean(),
    )


if __name__ == "__main__":
    main()
