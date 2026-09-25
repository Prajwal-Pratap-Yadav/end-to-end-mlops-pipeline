"""Automated retraining on fresh, labeled production data.

Triggered by the monitor (data drift or live performance below threshold) or run
manually. The procedure:

1. Guard rails: retraining enabled, cooldown elapsed since the last registered
   version, a champion exists, and enough ground truth has arrived.
2. Build datasets from the prediction log joined with feedback, using only the
   most recent ``max_fresh_rows`` labeled predictions (by default the same size as
   the drift window) so the new model learns the *current* regime rather than a
   blend dominated by pre-drift history. The **newest** ``holdout_fraction`` of
   those rows is held out - a time-based split, because a random split would leak
   future behaviour into training. Only if the remaining fresh rows fall below
   ``min_training_rows`` are they topped up with the champion's reference data.
3. Train through the standard pipeline (:func:`src.train.train_and_register`),
   which evaluates challenger *and* champion on the same holdout and only moves
   ``@champion`` if the challenger wins. The API picks the change up on its next poll.

Usage:
    python -m monitoring.retrain            # respects cooldown and data guards
    python -m monitoring.retrain --force    # skip cooldown / enabled checks
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import AppConfig, load_config
from src.prediction_store import PredictionStore
from src.registry import configure_mlflow, get_version_by_alias, list_versions
from src.schema import TARGET_COLUMN, feature_names
from src.train import train_and_register
from src.utils import configure_logging
from src.validation import assert_valid

logger = logging.getLogger(__name__)

RetrainStatus = Literal["promoted", "rejected", "skipped", "failed", "pending"]


@dataclass
class RetrainingOutcome:
    """Result of a retraining attempt."""

    status: RetrainStatus
    reason: str
    model_version: str | None = None
    champion_version: str | None = None
    run_id: str | None = None
    challenger_score: float | None = None
    champion_score: float | None = None
    n_fresh_train: int = 0
    n_reference_topup: int = 0
    n_holdout: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize for reports and logs."""
        return asdict(self)


def load_reference_data(client: MlflowClient, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the reference data logged with a model's training run.

    Returns:
        ``(reference_data, evaluation_predictions)`` - training rows with labels,
        and champion probabilities on held-out rows (the prediction-drift baseline).
    """
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="reference", tracking_uri=client.tracking_uri
    )
    reference = pd.read_csv(Path(local_dir) / "reference_data.csv")
    predictions = pd.read_csv(Path(local_dir) / "evaluation_predictions.csv")
    return reference, predictions


def minutes_since_last_version(client: MlflowClient, model_name: str) -> float | None:
    """Minutes since the newest registered version was created (``None`` if none)."""
    versions = list_versions(client, model_name)
    if not versions:
        return None
    created_ms = max(int(v.creation_timestamp) for v in versions)
    return (time.time() * 1000 - created_ms) / 60_000


def build_retraining_sets(
    labeled: pd.DataFrame,
    reference: pd.DataFrame,
    holdout_fraction: float,
    min_training_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split labeled production rows in time and top up training data if needed.

    Args:
        labeled: Labeled prediction-log rows in chronological order.
        reference: Champion reference data (features + target).
        holdout_fraction: Newest fraction reserved for champion/challenger evaluation.
        min_training_rows: Minimum training size; reference rows fill any gap.
        seed: Seed for sampling reference rows.

    Returns:
        ``(train_df, holdout_df, n_reference_rows_added)``.
    """
    fresh = labeled.sort_values("created_at")[feature_names()].copy()
    fresh[TARGET_COLUMN] = labeled.sort_values("created_at")["actual"].astype(int).to_numpy()
    n_holdout = max(1, round(len(fresh) * holdout_fraction))
    holdout = fresh.iloc[-n_holdout:].reset_index(drop=True)
    train = fresh.iloc[:-n_holdout].reset_index(drop=True)

    n_topup = max(0, min_training_rows - len(train))
    if n_topup:
        topup = reference[[*feature_names(), TARGET_COLUMN]].sample(
            n=min(n_topup, len(reference)), random_state=seed
        )
        train = pd.concat([train, topup], ignore_index=True)
    return train, holdout, min(n_topup, len(reference))


def run_retraining(
    config: AppConfig,
    *,
    store: PredictionStore | None = None,
    client: MlflowClient | None = None,
    reasons: list[str] | None = None,
    force: bool = False,
) -> RetrainingOutcome:
    """Retrain on fresh labeled data and let the promotion gate decide.

    Args:
        config: Application configuration.
        store: Prediction log (opened from the config when omitted).
        client: MLflow client (created from the config when omitted).
        reasons: Why retraining was triggered (recorded on the MLflow run).
        force: Ignore the ``enabled`` flag and the cooldown.

    Returns:
        The outcome; ``skipped`` when a guard rail prevented retraining.
    """
    policy = config.monitoring.retrain
    client = client or configure_mlflow(config)
    store = store or PredictionStore(config.serving.prediction_log_path)
    name = config.registry.model_name

    if not force and not policy.enabled:
        return RetrainingOutcome("skipped", "automated retraining is disabled")
    if not force:
        elapsed = minutes_since_last_version(client, name)
        if elapsed is not None and elapsed < policy.cooldown_minutes:
            return RetrainingOutcome(
                "skipped",
                f"cooldown: last version registered {elapsed:.1f} min ago (< {policy.cooldown_minutes} min)",
            )
    champion = get_version_by_alias(client, name, config.registry.champion_alias)
    if champion is None:
        return RetrainingOutcome("skipped", "no champion model to retrain from")
    if not champion.run_id:
        return RetrainingOutcome(
            "skipped", f"champion v{champion.version} has no training run (no reference data)"
        )

    labeled = store.labeled(limit=policy.max_fresh_rows)
    if len(labeled) < policy.min_labeled_samples:
        return RetrainingOutcome(
            "skipped",
            f"only {len(labeled)} labeled predictions (< {policy.min_labeled_samples} required)",
        )

    reference, _ = load_reference_data(client, champion.run_id)
    train_df, holdout_df, n_topup = build_retraining_sets(
        labeled,
        reference,
        policy.holdout_fraction,
        policy.min_training_rows,
        config.project.random_seed,
    )
    assert_valid(train_df, require_target=True, min_rows=10)
    assert_valid(holdout_df, require_target=True, min_rows=1)
    logger.info(
        "Retraining on %d fresh + %d reference rows; holdout of %d newest labeled rows",
        len(train_df) - n_topup,
        n_topup,
        len(holdout_df),
    )
    try:
        result = train_and_register(
            config,
            train_df,
            holdout_df,
            run_name="automated-retraining",
            tags={
                "trigger": "monitoring" if reasons else "manual",
                "trigger_reasons": "; ".join(reasons or ["manual"])[:500],
                "replaces_champion": str(champion.version),
            },
            data_source=str(store.path),
        )
    except Exception as exc:
        logger.exception("Retraining failed")
        return RetrainingOutcome("failed", f"{type(exc).__name__}: {exc}")

    return RetrainingOutcome(
        status="promoted" if result.decision.promote else "rejected",
        reason=result.decision.reason,
        model_version=result.model_version,
        champion_version=result.champion_version,
        run_id=result.run_id,
        challenger_score=result.decision.challenger_score,
        champion_score=result.decision.champion_score,
        n_fresh_train=len(train_df) - n_topup,
        n_reference_topup=n_topup,
        n_holdout=len(holdout_df),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Ignore cooldown and enabled flag")
    args = parser.parse_args(argv)
    configure_logging()
    outcome = run_retraining(load_config(args.config), force=args.force, reasons=["manual"])
    print(json.dumps(outcome.to_dict(), indent=2))
    if outcome.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
