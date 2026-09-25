"""FastAPI application factory for the churn inference service.

Run locally with::

    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app import metrics
from app.api import router
from app.model_manager import ModelManager, ModelUnavailableError
from src.config import AppConfig, load_config
from src.prediction_store import PredictionStore
from src.utils import configure_logging

logger = logging.getLogger(__name__)

_NOT_READY_POLL_SECONDS = 5.0


async def _reload_loop(manager: ModelManager, interval: float) -> None:
    """Poll the registry and hot-swap the model when ``@champion`` moves."""
    while True:
        await asyncio.sleep(
            interval if manager.is_ready else min(interval, _NOT_READY_POLL_SECONDS)
        )
        try:
            await asyncio.to_thread(manager.refresh)
        except Exception:
            logger.exception("Champion refresh failed; keeping the current model")


def create_app(config: AppConfig | None = None, manager: ModelManager | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config: Application config; loaded from ``MLOPS_CONFIG``/defaults when omitted.
        manager: Model manager; one backed by the MLflow registry is created when omitted.

    Returns:
        The configured application.
    """
    config = config or load_config()
    manager = manager or ModelManager(config)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.config = config
        app.state.model_manager = manager
        app.state.store = PredictionStore(config.serving.prediction_log_path)
        metrics.MODEL_LOADED.set(0)
        try:
            await asyncio.to_thread(manager.refresh)
        except Exception:
            logger.exception("Initial model load failed; the service stays not-ready and retries")

        interval = config.serving.reload_interval_seconds
        task = asyncio.create_task(_reload_loop(manager, interval)) if interval > 0 else None
        yield
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="Churn Prediction Service",
        version="1.0.0",
        description=(
            "Serves the MLflow `@champion` churn model, logs every prediction for monitoring, "
            "accepts delayed ground truth via `/feedback` and exposes Prometheus metrics."
        ),
        lifespan=lifespan,
    )
    app.middleware("http")(metrics.metrics_middleware)
    app.include_router(router)

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(request: Request, exc: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"Model not available: {exc}"},
        )

    return app


app = create_app()
