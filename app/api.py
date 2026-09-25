"""HTTP endpoints of the churn inference service."""

from __future__ import annotations

import logging
import secrets
import uuid
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics
from app.model_manager import ModelManager
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
    ReloadResponse,
)
from src.config import AppConfig
from src.predict import LoadedModel
from src.prediction_store import PredictionRecord, PredictionStore
from src.utils import utc_now

logger = logging.getLogger(__name__)
router = APIRouter()


def get_config(request: Request) -> AppConfig:
    """Dependency: application config."""
    config: AppConfig = request.app.state.config
    return config


def get_manager(request: Request) -> ModelManager:
    """Dependency: model manager."""
    manager: ModelManager = request.app.state.model_manager
    return manager


def get_store(request: Request) -> PredictionStore:
    """Dependency: prediction log."""
    store: PredictionStore = request.app.state.store
    return store


ConfigDep = Annotated[AppConfig, Depends(get_config)]
ManagerDep = Annotated[ModelManager, Depends(get_manager)]
StoreDep = Annotated[PredictionStore, Depends(get_store)]


def _score(
    loaded: LoadedModel,
    customers: list[CustomerFeatures],
    store: PredictionStore,
    threshold: float,
) -> list[PredictionResponse]:
    """Score customers, log every prediction and update metrics."""
    features = [customer.to_feature_dict() for customer in customers]
    probabilities = loaded.predict_proba(pd.DataFrame(features))
    created_at = utc_now()
    records = [
        PredictionRecord(
            prediction_id=uuid.uuid4().hex,
            features=row,
            churn_probability=float(probability),
            churn_prediction=int(probability >= threshold),
            model_name=loaded.name,
            model_version=loaded.version,
            created_at=created_at,
        )
        for row, probability in zip(features, probabilities, strict=True)
    ]
    try:
        store.log_predictions(records)
    except Exception:  # serving must not fail because the audit log is unavailable
        metrics.PREDICTION_LOG_ERRORS.inc(len(records))
        logger.exception("Failed to write %d prediction(s) to the prediction log", len(records))

    version_label = loaded.version or "unknown"
    for record in records:
        metrics.PREDICTIONS.labels(version_label, str(record.churn_prediction)).inc()
        metrics.PREDICTION_PROBABILITY.observe(record.churn_probability)

    return [
        PredictionResponse(
            prediction_id=record.prediction_id,
            churn_probability=round(record.churn_probability, 6),
            churn_prediction=bool(record.churn_prediction),
            threshold=threshold,
            model_name=loaded.name,
            model_version=loaded.version,
        )
        for record in records
    ]


@router.get("/", tags=["service"], summary="Service information")
def root(config: ConfigDep) -> dict[str, str]:
    """Describe the service and point at the interactive docs."""
    return {
        "service": config.project.name,
        "model": config.registry.model_name,
        "docs": "/docs",
        "metrics": "/metrics",
    }


@router.get("/health", response_model=HealthResponse, tags=["service"], summary="Liveness probe")
def health() -> HealthResponse:
    """Return 200 while the process is alive (independent of model state)."""
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["service"],
    summary="Readiness probe",
    responses={503: {"model": ReadinessResponse}},
)
def ready(manager: ManagerDep, response: Response) -> ReadinessResponse:
    """Return 200 once a model is loaded, 503 otherwise (keeps traffic away until ready)."""
    current = manager.current
    if current is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", detail=manager.last_error)
    return ReadinessResponse(status="ready", model_version=current.version)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Predict churn for one customer",
)
def predict(
    customer: CustomerFeatures, manager: ManagerDep, store: StoreDep, config: ConfigDep
) -> PredictionResponse:
    """Score a single customer with the champion model."""
    loaded = manager.get()
    return _score(loaded, [customer], store, config.training.decision_threshold)[0]


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
    summary="Predict churn for a batch of customers",
)
def predict_batch(
    batch: BatchPredictionRequest, manager: ManagerDep, store: StoreDep, config: ConfigDep
) -> BatchPredictionResponse:
    """Score up to ``serving.max_batch_size`` customers in one request."""
    if len(batch.instances) > config.serving.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch size {len(batch.instances)} exceeds limit {config.serving.max_batch_size}",
        )
    loaded = manager.get()
    predictions = _score(loaded, batch.instances, store, config.training.decision_threshold)
    return BatchPredictionResponse(predictions=predictions)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["monitoring"],
    summary="Submit the observed outcome for a prediction",
)
def feedback(payload: FeedbackRequest, store: StoreDep) -> FeedbackResponse:
    """Attach ground truth to a logged prediction (used for live performance and retraining)."""
    if not store.add_feedback(payload.prediction_id, int(payload.churned)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown prediction_id")
    metrics.FEEDBACK.labels(str(int(payload.churned))).inc()
    return FeedbackResponse(prediction_id=payload.prediction_id)


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    tags=["model"],
    summary="Currently served model",
)
def model_info(manager: ManagerDep, config: ConfigDep) -> ModelInfoResponse:
    """Return registry lineage and offline evaluation metrics of the serving model."""
    loaded = manager.get()
    return ModelInfoResponse(
        name=loaded.name,
        version=loaded.version,
        run_id=loaded.run_id,
        model_uri=loaded.model_uri,
        loaded_at=loaded.loaded_at,
        decision_threshold=config.training.decision_threshold,
        training_metrics=loaded.training_metrics,
    )


@router.post(
    "/model/reload",
    response_model=ReloadResponse,
    tags=["model"],
    summary="Force a reload of the champion model",
)
def reload_model(
    manager: ManagerDep,
    config: ConfigDep,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> ReloadResponse:
    """Reload ``@champion`` immediately instead of waiting for the next poll."""
    expected = config.serving.admin_token
    if expected and not secrets.compare_digest(x_admin_token or "", expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")
    previous = manager.current.version if manager.current else None
    try:
        reloaded = manager.refresh(force=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Reload failed: {exc}"
        ) from exc
    current = manager.current.version if manager.current else None
    return ReloadResponse(reloaded=reloaded, previous_version=previous, current_version=current)


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus exposition endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
