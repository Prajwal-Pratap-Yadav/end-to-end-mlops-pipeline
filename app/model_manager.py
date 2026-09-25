"""Holds the champion model and hot-swaps it when the registry alias moves.

Promotion and rollback are registry operations (moving ``@champion``); the API
notices on its next poll and swaps models atomically, without a restart. Requests
already in flight finish on the model they started with.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from mlflow.tracking import MlflowClient

from app import metrics
from src.config import AppConfig
from src.predict import LoadedModel, load_model
from src.registry import configure_mlflow, get_version_by_alias

logger = logging.getLogger(__name__)

ModelLoader = Callable[[AppConfig, str, MlflowClient], LoadedModel]


def _default_loader(config: AppConfig, model_uri: str, client: MlflowClient) -> LoadedModel:
    return load_model(config, model_uri=model_uri, client=client)


class ModelUnavailableError(RuntimeError):
    """Raised when a request arrives before any model has been loaded."""


class ModelManager:
    """Thread-safe owner of the currently served model."""

    def __init__(self, config: AppConfig, loader: ModelLoader = _default_loader) -> None:
        self.config = config
        self._loader = loader
        self._current: LoadedModel | None = None
        self._swap_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self.last_error: str | None = "model not loaded yet"

    @property
    def current(self) -> LoadedModel | None:
        """The model serving traffic, if any."""
        return self._current

    @property
    def is_ready(self) -> bool:
        """True once a model is loaded."""
        return self._current is not None

    def get(self) -> LoadedModel:
        """Return the serving model or raise :class:`ModelUnavailableError`."""
        model = self._current
        if model is None:
            raise ModelUnavailableError(self.last_error or "model not loaded")
        return model

    def refresh(self, force: bool = False) -> bool:
        """Load the champion if it differs from the serving model.

        Args:
            force: Reload even if the champion version is unchanged.

        Returns:
            True if a new model was swapped in.
        """
        with self._refresh_lock:  # one load at a time (poller vs. admin endpoint)
            client = configure_mlflow(self.config)
            name = self.config.registry.model_name
            alias = self.config.registry.champion_alias
            try:
                champion = get_version_by_alias(client, name, alias)
            except Exception as exc:
                self.last_error = f"registry unavailable: {exc}"
                metrics.MODEL_RELOADS.labels("error").inc()
                raise
            if champion is None:
                self.last_error = f"no model registered as {name}@{alias}"
                logger.warning("No %s@%s in the registry yet", name, alias)
                return False

            version = str(champion.version)
            current = self._current
            if not force and current is not None and current.version == version:
                return False

            try:
                loaded = self._loader(self.config, f"models:/{name}/{version}", client)
            except Exception as exc:
                self.last_error = f"failed to load {name} v{version}: {exc}"
                metrics.MODEL_RELOADS.labels("error").inc()
                raise
            with self._swap_lock:
                previous = self._current
                self._current = loaded
            self.last_error = None
            self._publish_metrics(loaded)
            metrics.MODEL_RELOADS.labels("success").inc()
            logger.info(
                "Serving %s v%s (previous: %s)",
                name,
                version,
                previous.version if previous else "none",
            )
            return True

    @staticmethod
    def _publish_metrics(loaded: LoadedModel) -> None:
        metrics.MODEL_INFO.clear()
        metrics.MODEL_INFO.labels(
            loaded.name or "unknown", loaded.version or "unknown", loaded.run_id or "unknown"
        ).set(1)
        metrics.MODEL_LOADED.set(1)
        metrics.MODEL_LOADED_TIMESTAMP.set(loaded.loaded_at.timestamp())
