"""Typed, validated configuration loaded from YAML with environment overrides.

The YAML file (``configs/config.yaml`` by default, or the path in ``MLOPS_CONFIG``)
is the single source of truth for tunables. Deployment-specific values can be
overridden without editing the file:

* ``MLOPS__<SECTION>__<KEY>=value`` for any nested key, e.g.
  ``MLOPS__MONITORING__DRIFT__PSI_THRESHOLD=0.25``. Values are parsed as YAML
  scalars, so ``true``, ``3`` and ``0.25`` get the right types.
* ``MLFLOW_TRACKING_URI`` (the standard MLflow variable) overrides
  ``mlflow.tracking_uri``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
CONFIG_PATH_ENV = "MLOPS_CONFIG"
ENV_PREFIX = "MLOPS__"
ENV_DELIMITER = "__"

EstimatorName = Literal["logistic_regression", "random_forest", "hist_gradient_boosting"]
MetricName = Literal["roc_auc", "f1", "average_precision", "accuracy", "precision", "recall"]


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_StrictModel):
    """Project-wide settings."""

    name: str = "churn-mlops"
    random_seed: int = 42


class DataConfig(_StrictModel):
    """Where training data comes from and the contract it must satisfy."""

    source: Literal["synthetic", "csv"] = "synthetic"
    csv_path: Path | None = None
    n_samples: int = Field(default=6000, ge=100)
    raw_path: Path = Path("data/raw/churn.csv")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    min_rows: int = Field(default=500, ge=1)
    max_missing_fraction: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _csv_source_needs_path(self) -> DataConfig:
        if self.source == "csv" and self.csv_path is None:
            raise ValueError("data.csv_path is required when data.source == 'csv'")
        return self


class ModelCandidate(_StrictModel):
    """One model family and the hyper-parameter grid to search for it."""

    name: str
    estimator: EstimatorName
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)


class TrainingConfig(_StrictModel):
    """Model selection and evaluation settings."""

    experiment_name: str = "churn-prediction"
    cv_folds: int = Field(default=5, ge=2)
    selection_metric: MetricName = "roc_auc"
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    reference_max_rows: int = Field(default=5000, ge=100)
    candidates: list[ModelCandidate] = Field(min_length=1)


class PromotionConfig(_StrictModel):
    """Champion/challenger promotion policy."""

    metric: MetricName = "roc_auc"
    min_improvement: float = 0.0
    min_metrics: dict[MetricName, float] = Field(default_factory=dict)


class RegistryConfig(_StrictModel):
    """MLflow Model Registry naming."""

    model_name: str = "churn-classifier"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)


class MlflowConfig(_StrictModel):
    """MLflow tracking server / store location."""

    tracking_uri: str = "sqlite:///mlflow.db"


class ServingConfig(_StrictModel):
    """Online inference service settings."""

    prediction_log_path: Path = Path("data/predictions/predictions.db")
    reload_interval_seconds: float = Field(default=30.0, ge=0.0)
    max_batch_size: int = Field(default=1000, ge=1)
    admin_token: str | None = None


class DriftConfig(_StrictModel):
    """Statistical drift detection thresholds."""

    n_bins: int = Field(default=10, ge=2)
    psi_threshold: float = Field(default=0.2, gt=0.0)
    p_value_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)
    drift_share_threshold: float = Field(default=0.15, gt=0.0, le=1.0)


class RetrainConfig(_StrictModel):
    """Automated retraining policy."""

    enabled: bool = True
    min_labeled_samples: int = Field(default=500, ge=10)
    max_fresh_rows: int = Field(default=5000, ge=10)
    holdout_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    min_training_rows: int = Field(default=1000, ge=10)
    cooldown_minutes: float = Field(default=30.0, ge=0.0)


class MonitoringConfig(_StrictModel):
    """Production monitoring settings."""

    window_size: int = Field(default=1000, ge=10)
    min_window_size: int = Field(default=200, ge=10)
    min_labeled_for_performance: int = Field(default=100, ge=10)
    interval_seconds: float = Field(default=60.0, gt=0.0)
    metrics_port: int = Field(default=8001, ge=1, le=65535)
    report_dir: Path = Path("reports/monitoring")
    drift: DriftConfig = Field(default_factory=DriftConfig)
    performance_thresholds: dict[MetricName, float] = Field(default_factory=dict)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)


class S3Config(_StrictModel):
    """Optional S3 (or S3-compatible) publishing of data snapshots and reports."""

    enabled: bool = False
    bucket: str | None = None
    prefix: str = "churn-mlops"
    endpoint_url: str | None = None
    region: str | None = None

    @model_validator(mode="after")
    def _enabled_needs_bucket(self) -> S3Config:
        if self.enabled and not self.bucket:
            raise ValueError("storage.s3.bucket is required when storage.s3.enabled is true")
        return self


class StorageConfig(_StrictModel):
    """External storage integrations."""

    s3: S3Config = Field(default_factory=S3Config)


class AppConfig(_StrictModel):
    """Root configuration object shared by every component."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Translate ``MLOPS__A__B=value`` environment variables into a nested dict.

    Args:
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        Nested dictionary of overrides, e.g. ``{"a": {"b": value}}``.
    """
    environ = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}
    for name, raw_value in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        keys = [part.lower() for part in name[len(ENV_PREFIX) :].split(ENV_DELIMITER) if part]
        if not keys:
            continue
        cursor = overrides
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = yaml.safe_load(raw_value) if raw_value != "" else None

    tracking_uri = environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        overrides.setdefault("mlflow", {})["tracking_uri"] = tracking_uri
    return overrides


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Return the config path from the argument, ``MLOPS_CONFIG`` or the default."""
    if path is not None:
        return Path(path)
    return Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load and validate the application configuration.

    Precedence (highest first): ``overrides`` argument, environment variables, YAML file.

    Args:
        path: YAML file to read. Defaults to ``MLOPS_CONFIG`` or ``configs/config.yaml``.
        overrides: Programmatic overrides (useful in tests and CLIs).
        environ: Environment mapping used for overrides; defaults to ``os.environ``.

    Returns:
        A fully validated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the merged configuration is invalid.
    """
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    merged = _deep_merge(raw, env_overrides(environ))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return AppConfig.model_validate(merged)
