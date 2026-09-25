"""Production traffic simulator for the inference API.

Generates customers from the synthetic source (optionally drifted), sends them to
the API like real clients would, then reports the true outcomes to ``/feedback``
- mimicking ground truth that arrives after the fact. Use it to exercise
monitoring, dashboards and automated retraining end to end.

Usage:
    python -m monitoring.simulate --n 1000                     # normal traffic
    python -m monitoring.simulate --n 1500 --drift-strength 0.8 # shifted market
    python -m monitoring.simulate --api-url http://api:8000 --rps 20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.evaluate import compute_metrics
from src.ingest import DriftProfile, generate_customers
from src.schema import FEATURES, TARGET_COLUMN
from src.utils import configure_logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationSummary:
    """What the simulator sent and observed."""

    requests_sent: int
    predictions_ok: int
    errors: int
    feedback_sent: int
    drift_strength: float
    true_churn_rate: float
    mean_predicted_probability: float
    client_side_roc_auc: float | None
    model_versions: list[str]
    elapsed_seconds: float


def to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert a generated row into an API request body (booleans, nulls)."""
    payload: dict[str, Any] = {}
    for spec in FEATURES:
        value = row[spec.name]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            payload[spec.name] = None
        elif spec.kind == "binary":
            payload[spec.name] = bool(int(value))
        elif spec.kind == "categorical":
            payload[spec.name] = str(value)
        elif spec.name in ("tenure_months", "num_support_tickets"):
            payload[spec.name] = int(value)
        else:
            payload[spec.name] = float(value)
    return payload


def wait_until_ready(session: requests.Session, api_url: str, timeout: float = 120.0) -> None:
    """Block until ``/ready`` returns 200 or raise ``TimeoutError``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if session.get(f"{api_url}/ready", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"API at {api_url} not ready after {timeout:.0f}s")


def simulate(
    api_url: str,
    n: int,
    drift_strength: float = 0.0,
    feedback_rate: float = 1.0,
    seed: int = 7,
    batch_size: int = 1,
    rps: float = 0.0,
    session: requests.Session | None = None,
) -> SimulationSummary:
    """Send ``n`` generated customers to the API and report their true outcomes.

    Args:
        api_url: Base URL of the inference API.
        n: Number of customers.
        drift_strength: Covariate + concept drift in [0, 1].
        feedback_rate: Fraction of predictions that receive ground truth.
        seed: Random seed (use a new seed per run for fresh customers).
        batch_size: 1 sends ``/predict`` calls; >1 uses ``/predict/batch``.
        rps: Target requests per second (0 = as fast as possible).
        session: Optional HTTP session (injectable for tests).

    Returns:
        A :class:`SimulationSummary`.
    """
    session = session or requests.Session()
    api_url = api_url.rstrip("/")
    rng = np.random.default_rng(seed)
    customers = generate_customers(
        n, seed=seed, drift=DriftProfile.from_strength(drift_strength), id_offset=seed * 1_000_000
    )
    started = time.monotonic()
    sent = ok = errors = feedback_sent = 0
    probabilities: list[float] = []
    labels: list[int] = []
    versions: set[str] = set()

    for start in range(0, n, batch_size):
        chunk = customers.iloc[start : start + batch_size]
        payloads = [to_payload(row) for _, row in chunk.iterrows()]
        try:
            if batch_size == 1:
                response = session.post(f"{api_url}/predict", json=payloads[0], timeout=10)
                results = [response.json()] if response.ok else []
            else:
                response = session.post(
                    f"{api_url}/predict/batch", json={"instances": payloads}, timeout=30
                )
                results = response.json()["predictions"] if response.ok else []
            sent += 1
            if not response.ok:
                errors += len(payloads)
                logger.warning(
                    "Prediction failed: %s %s", response.status_code, response.text[:200]
                )
                continue
        except requests.RequestException as exc:
            errors += len(payloads)
            logger.warning("Request error: %s", exc)
            continue

        for result, (_, row) in zip(results, chunk.iterrows(), strict=True):
            ok += 1
            probabilities.append(float(result["churn_probability"]))
            labels.append(int(row[TARGET_COLUMN]))
            versions.add(str(result["model_version"]))
            if rng.random() < feedback_rate:
                fb = session.post(
                    f"{api_url}/feedback",
                    json={
                        "prediction_id": result["prediction_id"],
                        "churned": bool(row[TARGET_COLUMN]),
                    },
                    timeout=10,
                )
                feedback_sent += int(fb.ok)
        if rps > 0:
            time.sleep(1.0 / rps)

    roc_auc = None
    if labels and len(set(labels)) == 2:
        roc_auc = compute_metrics(np.array(labels), np.array(probabilities))["roc_auc"]
    return SimulationSummary(
        requests_sent=sent,
        predictions_ok=ok,
        errors=errors,
        feedback_sent=feedback_sent,
        drift_strength=drift_strength,
        true_churn_rate=float(np.mean(labels)) if labels else float("nan"),
        mean_predicted_probability=float(np.mean(probabilities)) if probabilities else float("nan"),
        client_side_roc_auc=roc_auc,
        model_versions=sorted(versions),
        elapsed_seconds=round(time.monotonic() - started, 2),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=1000, help="Customers to send")
    parser.add_argument(
        "--drift-strength", type=float, default=0.0, help="0 = baseline, 1 = severe"
    )
    parser.add_argument(
        "--feedback-rate",
        type=float,
        default=0.8,
        help="Share of predictions that later receive ground truth",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--batch-size", type=int, default=1, help="1 = /predict, >1 = /predict/batch"
    )
    parser.add_argument("--rps", type=float, default=0.0, help="Throttle to N requests/second")
    parser.add_argument(
        "--wait", type=float, default=120.0, help="Seconds to wait for API readiness"
    )
    args = parser.parse_args(argv)
    configure_logging()

    session = requests.Session()
    wait_until_ready(session, args.api_url, args.wait)
    summary = simulate(
        args.api_url,
        args.n,
        drift_strength=args.drift_strength,
        feedback_rate=args.feedback_rate,
        seed=args.seed,
        batch_size=args.batch_size,
        rps=args.rps,
        session=session,
    )
    print(json.dumps(asdict(summary), indent=2))
    if summary.predictions_ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
