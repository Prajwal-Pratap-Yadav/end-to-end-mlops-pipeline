"""Production model monitoring: drift, live performance and retraining triggers.

Each cycle:

1. Resolves the ``@champion`` model and downloads the reference data logged with it.
2. Reads the most recent ``window_size`` predictions from the prediction log.
3. Detects data drift (every input) and prediction drift (output distribution).
4. Measures live performance on predictions that received ground-truth feedback.
5. Decides whether to retrain (dataset drift or a metric below its threshold) and,
   if allowed, runs :func:`monitoring.retrain.run_retraining`.
6. Writes JSON + HTML reports and updates Prometheus gauges.

Usage:
    python -m monitoring.monitor --once               # single cycle, print summary
    python -m monitoring.monitor --once --no-retrain  # observe only
    python -m monitoring.monitor --serve              # loop; serve /metrics and /report
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from mlflow.tracking import MlflowClient
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from monitoring.drift import DriftReport, detect_drift
from monitoring.report import render_html
from monitoring.retrain import RetrainingOutcome, load_reference_data, run_retraining
from src.config import AppConfig, load_config
from src.evaluate import compute_metrics
from src.prediction_store import PredictionStore
from src.registry import configure_mlflow, get_version_by_alias
from src.utils import configure_logging, utc_now, utc_timestamp_slug, write_json

logger = logging.getLogger(__name__)

Status = Literal["ok", "drift", "degraded", "insufficient_data", "no_model"]
LIVE_METRICS = ("roc_auc", "f1", "precision", "recall", "accuracy")


@dataclass
class PerformanceResult:
    """Live metrics on labeled predictions served by the champion."""

    n_labeled: int
    metrics: dict[str, float] | None = None
    below_threshold: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class MonitoringResult:
    """Everything one monitoring cycle observed and decided."""

    timestamp: datetime
    status: Status
    model_name: str
    model_version: str | None
    window_size: int
    drift: DriftReport | None = None
    performance: PerformanceResult = field(default_factory=lambda: PerformanceResult(0))
    retrain_reasons: list[str] = field(default_factory=list)
    retraining: RetrainingOutcome | None = None
    report_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/HTML reports."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "window_size": self.window_size,
            "drift": self.drift.to_dict() if self.drift else None,
            "performance": {
                "n_labeled": self.performance.n_labeled,
                "metrics": self.performance.metrics,
                "below_threshold": self.performance.below_threshold,
            },
            "retrain_reasons": self.retrain_reasons,
            "retraining": self.retraining.to_dict() if self.retraining else None,
        }

    def summary(self) -> dict[str, Any]:
        """Compact summary for logs and the CLI."""
        return {
            "status": self.status,
            "model_version": self.model_version,
            "window_size": self.window_size,
            "drift_share": round(self.drift.drift_share, 3) if self.drift else None,
            "drifted_features": self.drift.drifted_features if self.drift else [],
            "prediction_psi": (
                round(self.drift.prediction_drift.psi, 3)
                if self.drift and self.drift.prediction_drift
                else None
            ),
            "live_metrics": (
                {k: round(v, 4) for k, v in self.performance.metrics.items() if k in LIVE_METRICS}
                if self.performance.metrics
                else None
            ),
            "n_labeled": self.performance.n_labeled,
            "retrain_reasons": self.retrain_reasons,
            "retraining": self.retraining.to_dict() if self.retraining else None,
            "report": self.report_paths.get("html"),
        }


class MonitorMetrics:
    """Prometheus gauges published by the monitor (own registry, testable in isolation)."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.last_run = Gauge(
            "churn_monitor_last_run_timestamp_seconds", "Unix time of the last cycle", registry=r
        )
        self.window_size = Gauge(
            "churn_monitor_window_size", "Predictions in the analysed window", registry=r
        )
        self.labeled = Gauge(
            "churn_monitor_labeled_samples",
            "Labeled predictions from the champion in the window",
            registry=r,
        )
        self.drift_share = Gauge(
            "churn_monitor_drift_share", "Share of input features with drift", registry=r
        )
        self.dataset_drift = Gauge(
            "churn_monitor_dataset_drift", "1 if dataset drift is detected", registry=r
        )
        self.feature_psi = Gauge(
            "churn_monitor_feature_psi", "PSI per input feature", ["feature"], registry=r
        )
        self.feature_drift = Gauge(
            "churn_monitor_feature_drift", "1 if the feature drifted", ["feature"], registry=r
        )
        self.prediction_psi = Gauge(
            "churn_monitor_prediction_psi", "PSI of predicted churn probability", registry=r
        )
        self.live_metric = Gauge(
            "churn_monitor_live_metric",
            "Live model metric on labeled feedback",
            ["metric"],
            registry=r,
        )
        self.retrain_recommended = Gauge(
            "churn_monitor_retrain_recommended", "1 if a retraining trigger fired", registry=r
        )
        self.champion_version = Gauge(
            "churn_monitor_champion_version", "Registry version of the champion", registry=r
        )
        self.retraining_runs = Counter(
            "churn_monitor_retraining_runs_total",
            "Retraining attempts by outcome",
            ["outcome"],
            registry=r,
        )
        self.cycle_errors = Counter(
            "churn_monitor_cycle_errors_total", "Monitoring cycles that raised", registry=r
        )

    def publish(self, result: MonitoringResult) -> None:
        """Update every gauge from a monitoring result."""
        self.last_run.set(result.timestamp.timestamp())
        self.window_size.set(result.window_size)
        self.labeled.set(result.performance.n_labeled)
        self.retrain_recommended.set(1 if result.retrain_reasons else 0)
        if result.model_version and result.model_version.isdigit():
            self.champion_version.set(int(result.model_version))
        if result.drift is not None:
            self.drift_share.set(result.drift.drift_share)
            self.dataset_drift.set(1 if result.drift.dataset_drift else 0)
            for feature in result.drift.features:
                self.feature_psi.labels(feature.feature).set(feature.psi)
                self.feature_drift.labels(feature.feature).set(1 if feature.drift_detected else 0)
            if result.drift.prediction_drift is not None:
                self.prediction_psi.set(result.drift.prediction_drift.psi)
        if result.performance.metrics:
            for name in LIVE_METRICS:
                value = result.performance.metrics.get(name)
                if value is not None:
                    self.live_metric.labels(name).set(value)
        if result.retraining is not None:
            self.retraining_runs.labels(result.retraining.status).inc()


def evaluate_live_performance(
    window: pd.DataFrame, model_version: str | None, config: AppConfig
) -> PerformanceResult:
    """Compute live metrics on labeled predictions made by the current champion."""
    labeled = window[window["actual"].notna()]
    if model_version is not None:
        labeled = labeled[labeled["model_version"].astype(str) == str(model_version)]
    result = PerformanceResult(
        n_labeled=len(labeled),
        thresholds={str(k): v for k, v in config.monitoring.performance_thresholds.items()},
    )
    if len(labeled) < config.monitoring.min_labeled_for_performance:
        return result
    if labeled["actual"].nunique() < 2:
        return result
    result.metrics = compute_metrics(
        labeled["actual"].astype(int),
        labeled["churn_probability"].to_numpy(dtype=float),
        config.training.decision_threshold,
    )
    for name, minimum in result.thresholds.items():
        value = result.metrics.get(name)
        if value is not None and value < minimum:
            result.below_threshold[name] = value
    return result


def _write_reports(result: MonitoringResult, config: AppConfig) -> dict[str, str]:
    report_dir = Path(config.monitoring.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    page = render_html(payload)
    slug = utc_timestamp_slug()
    paths = {
        "json": report_dir / f"monitoring_{slug}.json",
        "html": report_dir / f"monitoring_{slug}.html",
    }
    write_json(payload, paths["json"])
    paths["html"].write_text(page, encoding="utf-8")
    write_json(payload, report_dir / "latest.json")
    (report_dir / "latest.html").write_text(page, encoding="utf-8")

    if config.storage.s3.enabled:
        from src.s3 import S3Storage

        storage = S3Storage.from_config(config.storage.s3)
        for kind, path in paths.items():
            uri = storage.upload_file(path, f"monitoring/{path.name}")
            logger.info("Published %s report to %s", kind, uri)
    return {kind: str(path) for kind, path in paths.items()}


def run_monitoring_cycle(
    config: AppConfig,
    *,
    store: PredictionStore | None = None,
    client: MlflowClient | None = None,
    allow_retrain: bool = True,
    metrics: MonitorMetrics | None = None,
) -> MonitoringResult:
    """Run one monitoring cycle.

    Args:
        config: Application configuration.
        store: Prediction log (opened from the config when omitted).
        client: MLflow client (created from the config when omitted).
        allow_retrain: Run retraining when a trigger fires.
        metrics: Prometheus gauges to update.

    Returns:
        The monitoring result (reports are written to ``monitoring.report_dir``).
    """
    client = client or configure_mlflow(config)
    store = store or PredictionStore(config.serving.prediction_log_path)
    name = config.registry.model_name
    now = utc_now()

    champion = get_version_by_alias(client, name, config.registry.champion_alias)
    if champion is None or not champion.run_id:
        # Without a training run there is no reference data to compare against.
        result = MonitoringResult(
            now, "no_model", name, str(champion.version) if champion else None, 0
        )
    else:
        version = str(champion.version)
        window = store.recent(config.monitoring.window_size)
        if len(window) < config.monitoring.min_window_size:
            result = MonitoringResult(now, "insufficient_data", name, version, len(window))
        else:
            reference, reference_predictions = load_reference_data(client, champion.run_id)
            drift = detect_drift(
                reference,
                window,
                config.monitoring.drift,
                reference_predictions=reference_predictions["churn_probability"],
                current_predictions=window["churn_probability"],
            )
            performance = evaluate_live_performance(window, version, config)
            reasons: list[str] = []
            if drift.dataset_drift:
                reasons.append(
                    f"data drift in {len(drift.drifted_features)} feature(s): "
                    + ", ".join(drift.drifted_features)
                )
            for metric_name, value in performance.below_threshold.items():
                threshold = performance.thresholds[metric_name]
                reasons.append(f"live {metric_name} {value:.4f} < {threshold}")
            status: Status = "ok"
            if performance.below_threshold:
                status = "degraded"
            elif drift.dataset_drift:
                status = "drift"
            result = MonitoringResult(
                now, status, name, version, len(window), drift, performance, reasons
            )
            if reasons and allow_retrain:
                result.retraining = run_retraining(
                    config, store=store, client=client, reasons=reasons
                )
                logger.info("Retraining outcome: %s", result.retraining.to_dict())

    result.report_paths = _write_reports(result, config)
    if metrics is not None:
        metrics.publish(result)
    logger.info("Monitoring summary: %s", json.dumps(result.summary(), default=str))
    return result


class _MonitorHandler(BaseHTTPRequestHandler):
    """Serves /metrics, /health and the latest report."""

    metrics: MonitorMetrics
    report_dir: Path

    def do_GET(self) -> None:
        """Route GET requests."""
        if self.path == "/metrics":
            self._send(200, generate_latest(self.metrics.registry), CONTENT_TYPE_LATEST)
        elif self.path == "/health":
            self._send(200, b'{"status": "ok"}', "application/json")
        elif self.path in ("/", "/report"):
            self._send_file(self.report_dir / "latest.html", "text/html; charset=utf-8")
        elif self.path == "/report.json":
            self._send_file(self.report_dir / "latest.json", "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _send_file(self, path: Path, content_type: str) -> None:
        if path.is_file():
            self._send(200, path.read_bytes(), content_type)
        else:
            self._send(
                404, b"no report yet - the first monitoring cycle has not completed", "text/plain"
            )

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request access logs (Prometheus scrapes every few seconds)."""


def start_http_server(port: int, metrics: MonitorMetrics, report_dir: Path) -> ThreadingHTTPServer:
    """Start the monitor's HTTP endpoint in a daemon thread."""
    handler = type(
        "MonitorHandler", (_MonitorHandler,), {"metrics": metrics, "report_dir": report_dir}
    )
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)  # noqa: S104 - container service
    threading.Thread(target=server.serve_forever, daemon=True, name="monitor-http").start()
    logger.info("Monitor serving /metrics and /report on port %d", port)
    return server


def serve(config: AppConfig, allow_retrain: bool = True) -> None:
    """Run monitoring cycles forever at ``monitoring.interval_seconds``."""
    metrics = MonitorMetrics()
    start_http_server(config.monitoring.metrics_port, metrics, Path(config.monitoring.report_dir))
    while True:
        started = time.monotonic()
        try:
            run_monitoring_cycle(config, allow_retrain=allow_retrain, metrics=metrics)
        except Exception:
            metrics.cycle_errors.inc()
            logger.exception("Monitoring cycle failed")
        time.sleep(max(0.0, config.monitoring.interval_seconds - (time.monotonic() - started)))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run a single cycle (default)")
    mode.add_argument("--serve", action="store_true", help="Run continuously and serve metrics")
    parser.add_argument("--no-retrain", action="store_true", help="Never trigger retraining")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    if args.serve:
        serve(config, allow_retrain=not args.no_retrain)
        return
    result = run_monitoring_cycle(config, allow_retrain=not args.no_retrain)
    print(json.dumps(result.summary(), indent=2, default=str))


if __name__ == "__main__":
    main()
