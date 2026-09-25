"""Smoke test for a running stack (docker compose or local).

Checks that every service answers and that the API serves real predictions,
accepts feedback and exports metrics. Standard library only, so it runs on any
machine with Python - no project dependencies required.

Usage:
    python scripts/smoke_test.py                      # API only (local run)
    python scripts/smoke_test.py --stack              # all docker compose services
    python scripts/smoke_test.py --api-url http://host:8000 --timeout 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

CUSTOMER = {
    "tenure_months": 4,
    "monthly_charges": 89.5,
    "total_charges": 358.0,
    "num_support_tickets": 3,
    "avg_monthly_gb": 120.4,
    "senior_citizen": False,
    "paperless_billing": True,
    "has_tech_support": False,
    "contract_type": "month_to_month",
    "internet_service": "fiber_optic",
    "payment_method": "electronic_check",
}


def request(url: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    """GET (or POST JSON when ``payload`` is given) and decode the response."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - URLs come from the CLI, http(s) only
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        body = response.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def wait_for(url: str, timeout: float) -> None:
    """Poll ``url`` until it returns 2xx or ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            request(url, timeout=5)
            return
        except (urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
            time.sleep(3)
    raise TimeoutError(f"{url} not ready after {timeout:.0f}s ({last_error})")


def check(name: str, func: Callable[[], str], attempts: int = 1, delay: float = 5.0) -> bool:
    """Run one check (retrying ``attempts`` times) and print a PASS/FAIL line."""
    for attempt in range(1, attempts + 1):
        try:
            detail = func()
        except Exception as exc:
            if attempt == attempts:
                print(f"FAIL  {name}: {exc!r}")
                return False
            time.sleep(delay)
        else:
            print(f"PASS  {name}: {detail}")
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Run the smoke test; returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--mlflow-url", default="http://localhost:5000")
    parser.add_argument("--monitor-url", default="http://localhost:8001")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--grafana-url", default="http://localhost:3000")
    parser.add_argument(
        "--stack", action="store_true", help="Also check MLflow, monitor, Prometheus, Grafana"
    )
    parser.add_argument("--timeout", type=float, default=240, help="Seconds to wait for the API")
    args = parser.parse_args(argv)

    print(f"Waiting for {args.api_url}/ready ...")
    wait_for(f"{args.api_url}/ready", args.timeout)
    state: dict[str, Any] = {}

    def predict() -> str:
        body = request(f"{args.api_url}/predict", CUSTOMER)
        assert 0.0 <= body["churn_probability"] <= 1.0, body
        assert body["model_version"], body
        state["prediction_id"] = body["prediction_id"]
        return f"p(churn)={body['churn_probability']:.3f} from model v{body['model_version']}"

    def batch() -> str:
        body = request(f"{args.api_url}/predict/batch", {"instances": [CUSTOMER] * 3})
        assert len(body["predictions"]) == 3, body
        return "3 predictions"

    def feedback() -> str:
        body = request(
            f"{args.api_url}/feedback", {"prediction_id": state["prediction_id"], "churned": True}
        )
        assert body["status"] == "recorded", body
        return "ground truth recorded"

    def model() -> str:
        body = request(f"{args.api_url}/model")
        return f"{body['name']} v{body['version']} (offline ROC-AUC {body['training_metrics']['roc_auc']:.3f})"

    def metrics() -> str:
        text = request(f"{args.api_url}/metrics")
        assert "churn_api_predictions_total" in text, "prediction counter missing"
        return "Prometheus exposition includes prediction counters"

    checks: list[tuple[str, Callable[[], str], int]] = [
        ("POST /predict", predict, 1),
        ("POST /predict/batch", batch, 1),
        ("POST /feedback", feedback, 1),
        ("GET /model", model, 1),
        ("GET /metrics", metrics, 1),
    ]

    if args.stack:

        def mlflow() -> str:
            body = request(
                f"{args.mlflow_url}/api/2.0/mlflow/registered-models/alias"
                "?name=churn-classifier&alias=champion"
            )
            return f"registry: churn-classifier@champion -> v{body['model_version']['version']}"

        def monitor() -> str:
            assert request(f"{args.monitor_url}/health")["status"] == "ok"
            assert "churn_monitor_last_run_timestamp_seconds" in request(
                f"{args.monitor_url}/metrics"
            )
            return "healthy, exporting metrics"

        def prometheus() -> str:
            targets = request(f"{args.prometheus_url}/api/v1/targets")["data"]["activeTargets"]
            health = {t["labels"]["job"]: t["health"] for t in targets}
            assert health.get("churn-api") == "up", health
            assert health.get("churn-monitor") == "up", health
            return f"targets {health}"

        def grafana() -> str:
            assert request(f"{args.grafana_url}/api/health")["database"] == "ok"
            dashboards = request(f"{args.grafana_url}/api/search?query=churn")
            assert dashboards, "dashboard not provisioned"
            return f"dashboard '{dashboards[0]['title']}' provisioned"

        # Services start in parallel; give scrapes and provisioning up to a minute.
        checks += [
            ("MLflow registry", mlflow, 12),
            ("Monitor", monitor, 12),
            ("Prometheus scrape targets", prometheus, 12),
            ("Grafana", grafana, 12),
        ]

    results = [check(name, func, attempts) for name, func, attempts in checks]
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
