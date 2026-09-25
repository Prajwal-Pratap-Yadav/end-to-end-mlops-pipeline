import json
import urllib.request
from pathlib import Path

import pandas as pd
import pytest
from prometheus_client import generate_latest

from monitoring.monitor import (
    MonitorMetrics,
    TriggerGate,
    evaluate_live_performance,
    run_monitoring_cycle,
    start_http_server,
)
from monitoring.report import render_html
from src.config import AppConfig
from src.ingest import generate_customers
from src.prediction_store import PredictionStore
from tests.conftest import TrainedEnv, drifted_customers, log_scored_customers, make_test_config


def _env_with_own_store(env: TrainedEnv, tmp_path: Path) -> AppConfig:
    """Reuse the trained registry but isolate the prediction log and reports."""
    return env.config.model_copy(
        update={
            "serving": env.config.serving.model_copy(
                update={"prediction_log_path": tmp_path / "predictions.db"}
            ),
            "monitoring": env.config.monitoring.model_copy(
                update={"report_dir": tmp_path / "reports"}
            ),
        }
    )


def test_trigger_gate_requires_consecutive_cycles() -> None:
    gate = TriggerGate(2)
    assert not gate.observe(True)
    assert not gate.observe(False)  # streak broken
    assert not gate.observe(True)
    assert gate.observe(True)  # second consecutive trigger
    assert gate.streak == 0  # reset after firing
    assert TriggerGate(1).observe(True)


def test_no_model_and_insufficient_data(tmp_path: Path, trained_env: TrainedEnv) -> None:
    empty = make_test_config(tmp_path / "empty")
    assert run_monitoring_cycle(empty).status == "no_model"

    config = _env_with_own_store(trained_env, tmp_path)
    store = PredictionStore(config.serving.prediction_log_path)
    log_scored_customers(store, trained_env.load_champion(), generate_customers(20, seed=1))
    result = run_monitoring_cycle(config, store=store)
    assert result.status == "insufficient_data"
    assert result.window_size == 20


def test_stable_traffic_is_ok_and_writes_reports(trained_env: TrainedEnv, tmp_path: Path) -> None:
    config = _env_with_own_store(trained_env, tmp_path)
    store = PredictionStore(config.serving.prediction_log_path)
    log_scored_customers(store, trained_env.load_champion(), generate_customers(600, seed=77))
    metrics = MonitorMetrics()

    result = run_monitoring_cycle(config, store=store, metrics=metrics)

    assert result.status == "ok"
    assert result.retrain_reasons == []
    assert result.retraining is None
    assert result.performance.metrics is not None
    assert result.performance.metrics["roc_auc"] > 0.75
    report_dir = Path(config.monitoring.report_dir)
    assert json.loads((report_dir / "latest.json").read_text())["status"] == "ok"
    assert "Model monitoring report" in (report_dir / "latest.html").read_text()

    exposition = generate_latest(metrics.registry).decode()
    assert "churn_monitor_dataset_drift 0.0" in exposition
    assert 'churn_monitor_feature_psi{feature="monthly_charges"}' in exposition
    assert 'churn_monitor_live_metric{metric="roc_auc"}' in exposition


def test_drift_is_reported_without_retraining_when_disabled(
    trained_env: TrainedEnv, tmp_path: Path
) -> None:
    config = _env_with_own_store(trained_env, tmp_path)
    store = PredictionStore(config.serving.prediction_log_path)
    log_scored_customers(store, trained_env.load_champion(), drifted_customers(600, seed=8))

    result = run_monitoring_cycle(config, store=store, allow_retrain=False)

    assert result.status in {"drift", "degraded"}
    assert result.drift is not None
    assert result.drift.dataset_drift
    assert "monthly_charges" in result.drift.drifted_features
    assert result.retrain_reasons
    assert result.retraining is None
    assert result.drift.prediction_drift is not None


def test_gate_defers_retraining_until_trigger_persists(
    trained_env: TrainedEnv, tmp_path: Path
) -> None:
    config = _env_with_own_store(trained_env, tmp_path)
    store = PredictionStore(config.serving.prediction_log_path)
    log_scored_customers(store, trained_env.load_champion(), drifted_customers(300, seed=9))
    result = run_monitoring_cycle(config, store=store, trigger_gate=TriggerGate(2))
    assert result.retraining is not None
    assert result.retraining.status == "pending"


def test_live_performance_only_counts_champion_predictions(trained_env: TrainedEnv) -> None:
    window = pd.DataFrame(
        {
            "actual": [1, 0] * 60 + [None] * 10,
            "churn_probability": [0.9, 0.1] * 60 + [0.5] * 10,
            "model_version": ["2"] * 120 + ["2"] * 10,
        }
    )
    assert evaluate_live_performance(window, "1", trained_env.config).n_labeled == 0
    result = evaluate_live_performance(window, "2", trained_env.config)
    assert result.n_labeled == 120
    assert result.metrics is not None
    assert result.metrics["roc_auc"] == 1.0
    assert result.below_threshold == {}

    inverted = window.assign(churn_probability=[0.1, 0.9] * 60 + [0.5] * 10)
    degraded = evaluate_live_performance(inverted, "2", trained_env.config)
    assert "roc_auc" in degraded.below_threshold


def test_http_endpoint_serves_metrics_health_and_report(tmp_path: Path) -> None:
    metrics = MonitorMetrics()
    metrics.window_size.set(42)
    server = start_http_server(0, metrics, tmp_path)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = urllib.request.urlopen(f"{base}/metrics", timeout=5).read().decode()
        assert "churn_monitor_window_size 42.0" in body
        assert (
            json.loads(urllib.request.urlopen(f"{base}/health", timeout=5).read())["status"] == "ok"
        )
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/report", timeout=5)
        assert missing.value.code == 404
        (tmp_path / "latest.html").write_text("<html>report</html>")
        assert b"report" in urllib.request.urlopen(f"{base}/report", timeout=5).read()
    finally:
        server.shutdown()


def test_html_report_escapes_untrusted_values() -> None:
    page = render_html(
        {
            "status": "drift",
            "model_name": "<script>alert(1)</script>",
            "model_version": "1",
            "window_size": 10,
            "retrain_reasons": ["data drift in 2 feature(s)"],
        }
    )
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "data drift in 2 feature(s)" in page
