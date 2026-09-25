"""Prometheus instrumentation for the inference API.

Exposed at ``GET /metrics`` and scraped by Prometheus. Route labels use the route
*template* (``/predict``), never the raw path, to keep label cardinality bounded.
The service runs one Uvicorn worker per container and scales horizontally, so the
default single-process registry is correct.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "churn_api_http_requests_total",
    "HTTP requests handled",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "churn_api_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "churn_api_predictions_total",
    "Predictions served",
    ["model_version", "predicted_class"],
)
PREDICTION_PROBABILITY = Histogram(
    "churn_api_prediction_probability",
    "Distribution of predicted churn probabilities",
    buckets=tuple(round(0.1 * i, 1) for i in range(1, 11)),
)
FEEDBACK = Counter(
    "churn_api_feedback_total",
    "Ground-truth labels received",
    ["actual_class"],
)
MODEL_INFO = Gauge(
    "churn_api_model_info",
    "Model currently serving traffic (value is always 1)",
    ["model_name", "model_version", "run_id"],
)
MODEL_LOADED = Gauge("churn_api_model_loaded", "1 when a model is loaded and serving")
MODEL_LOADED_TIMESTAMP = Gauge(
    "churn_api_model_loaded_timestamp_seconds", "Unix time the current model was loaded"
)
MODEL_RELOADS = Counter(
    "churn_api_model_reloads_total",
    "Champion reload attempts",
    ["outcome"],
)
PREDICTION_LOG_ERRORS = Counter(
    "churn_api_prediction_log_errors_total",
    "Predictions that were served but could not be written to the prediction log",
)


async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Record request count and latency for every HTTP request."""
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        REQUESTS.labels(request.method, route_path, status).inc()
        REQUEST_LATENCY.labels(request.method, route_path).observe(time.perf_counter() - start)
