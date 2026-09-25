"""MLflow Model Registry: registration, champion/challenger promotion and rollback.

Deployment is alias-based: the API serves ``models:/<name>@champion`` and hot
reloads when the alias moves, so promoting or rolling back a model is a registry
operation - no rebuild or redeploy.

Every registered version is tagged with its promotion decision and the evidence
behind it, which gives an auditable history of what served production and why.

Usage:
    python -m src.registry list
    python -m src.registry promote --version 3
    python -m src.registry rollback
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import AppConfig, PromotionConfig, load_config
from src.utils import configure_logging

logger = logging.getLogger(__name__)

TAG_DECISION = "promotion.decision"
TAG_REASON = "promotion.reason"
TAG_METRIC = "promotion.metric"
TAG_CHALLENGER_SCORE = "promotion.challenger_score"
TAG_CHAMPION_SCORE = "promotion.champion_score"
TAG_CHAMPION_VERSION = "promotion.compared_to_version"
DECISION_PROMOTED = "promoted"
DECISION_REJECTED = "rejected"


def configure_mlflow(config: AppConfig) -> MlflowClient:
    """Point the MLflow fluent API at the configured store and return a client.

    The registry URI is passed explicitly: a client built from a tracking URI alone
    silently falls back to the process-wide registry URI, which may belong to a
    different store.
    """
    uri = config.mlflow.tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def get_version_by_alias(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    """Return the model version an alias points to, or ``None`` if unset/unknown."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE", "NOT_FOUND"}:
            return None
        raise


def list_versions(client: MlflowClient, name: str) -> list[ModelVersion]:
    """Return all versions of a registered model, newest first."""
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return []
        raise
    return sorted(versions, key=lambda v: int(v.version), reverse=True)


def register_model(model_uri: str, name: str, tags: dict[str, str] | None = None) -> ModelVersion:
    """Register a logged model as a new version of ``name``.

    Args:
        model_uri: URI returned when the model was logged (e.g. ``models:/m-...``).
        name: Registered model name.
        tags: Tags to attach to the new version.

    Returns:
        The created model version.
    """
    version = mlflow.register_model(model_uri, name, tags=tags or {})
    logger.info("Registered %s version %s", name, version.version)
    return version


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the champion/challenger promotion gate."""

    promote: bool
    reason: str
    metric: str
    challenger_score: float
    champion_score: float | None = None
    failed_gates: dict[str, float] = field(default_factory=dict)

    def as_tags(self) -> dict[str, str]:
        """Render the decision as model-version tags."""
        tags = {
            TAG_DECISION: DECISION_PROMOTED if self.promote else DECISION_REJECTED,
            TAG_REASON: self.reason,
            TAG_METRIC: self.metric,
            TAG_CHALLENGER_SCORE: f"{self.challenger_score:.6f}",
        }
        if self.champion_score is not None:
            tags[TAG_CHAMPION_SCORE] = f"{self.champion_score:.6f}"
        return tags


def decide_promotion(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    policy: PromotionConfig,
) -> PromotionDecision:
    """Decide whether a challenger should replace the champion.

    The challenger must (1) pass every absolute quality gate in
    ``policy.min_metrics`` and (2) score at least ``champion + min_improvement`` on
    ``policy.metric`` when both are evaluated on the same data. With no champion,
    passing the gates is sufficient.

    Args:
        challenger_metrics: Challenger metrics on the evaluation data.
        champion_metrics: Champion metrics on the *same* data, or ``None``.
        policy: Promotion policy from the config.

    Returns:
        The decision with a human-readable reason.
    """
    metric = policy.metric
    challenger_score = float(challenger_metrics.get(metric, float("nan")))
    failed: dict[str, float] = {}
    violations: list[str] = []
    for name, minimum in policy.min_metrics.items():
        value = float(challenger_metrics.get(name, float("nan")))
        if not value >= minimum:  # also catches NaN
            failed[name] = value
            violations.append(f"{name}={value:.4f} < {minimum:.4f}")
    if failed:
        return PromotionDecision(
            False,
            f"failed quality gate: {', '.join(violations)}",
            metric,
            challenger_score,
            None,
            failed,
        )

    if champion_metrics is None:
        return PromotionDecision(
            True, "no current champion; quality gates passed", metric, challenger_score
        )

    champion_score = float(champion_metrics.get(metric, float("nan")))
    if math.isnan(champion_score):
        return PromotionDecision(
            True, "champion score unavailable; quality gates passed", metric, challenger_score
        )
    required = champion_score + policy.min_improvement
    if challenger_score >= required:
        return PromotionDecision(
            True,
            f"{metric} {challenger_score:.4f} >= champion {champion_score:.4f} + {policy.min_improvement}",
            metric,
            challenger_score,
            champion_score,
        )
    return PromotionDecision(
        False,
        f"{metric} {challenger_score:.4f} < champion {champion_score:.4f} + {policy.min_improvement}",
        metric,
        challenger_score,
        champion_score,
    )


def record_decision(
    client: MlflowClient,
    name: str,
    version: str,
    decision: PromotionDecision,
    champion_version: str | None,
) -> None:
    """Persist the promotion decision as tags on the challenger version."""
    tags = decision.as_tags()
    if champion_version is not None:
        tags[TAG_CHAMPION_VERSION] = str(champion_version)
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)


def set_alias(client: MlflowClient, name: str, alias: str, version: str) -> None:
    """Point ``alias`` at ``version`` (moving it if it already exists)."""
    client.set_registered_model_alias(name, alias, version)
    logger.info("Alias %s@%s -> version %s", name, alias, version)


def rollback(client: MlflowClient, name: str, alias: str) -> str:
    """Move the champion alias back to the previously promoted version.

    Returns:
        The version number the alias now points to.

    Raises:
        RuntimeError: If there is no champion or no earlier promoted version.
    """
    current = get_version_by_alias(client, name, alias)
    if current is None:
        raise RuntimeError(f"{name}@{alias} is not set; nothing to roll back")
    candidates = [
        v
        for v in list_versions(client, name)
        if int(v.version) < int(current.version) and v.tags.get(TAG_DECISION) == DECISION_PROMOTED
    ]
    if not candidates:
        raise RuntimeError(f"No previously promoted version older than v{current.version}")
    target = str(candidates[0].version)
    set_alias(client, name, alias, target)
    client.set_model_version_tag(name, str(current.version), "rollback.replaced_by", target)
    return target


def aliases_by_version(client: MlflowClient, name: str) -> dict[str, list[str]]:
    """Map each version number to the aliases pointing at it.

    Version search results do not carry aliases in every store, so they are read
    from the registered model itself.
    """
    try:
        aliases = client.get_registered_model(name).aliases
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return {}
        raise
    mapping: dict[str, list[str]] = {}
    for alias, version in aliases.items():
        mapping.setdefault(str(version), []).append(alias)
    return mapping


def describe_versions(client: MlflowClient, name: str) -> list[dict[str, Any]]:
    """Summarize registered versions with their aliases and promotion tags."""
    aliases = aliases_by_version(client, name)
    return [
        {
            "version": str(v.version),
            "aliases": sorted(aliases.get(str(v.version), [])),
            "run_id": v.run_id,
            "status": v.status,
            "decision": v.tags.get(TAG_DECISION, ""),
            "reason": v.tags.get(TAG_REASON, ""),
        }
        for v in list_versions(client, name)
    ]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for registry operations."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List versions, aliases and promotion decisions")
    promote_parser = sub.add_parser(
        "promote", help="Manually point the champion alias at a version"
    )
    promote_parser.add_argument("--version", required=True)
    sub.add_parser("rollback", help="Revert the champion to the previously promoted version")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    client = configure_mlflow(config)
    name, alias = config.registry.model_name, config.registry.champion_alias

    if args.command == "list":
        for row in describe_versions(client, name):
            aliases = ",".join(row["aliases"]) or "-"
            print(
                f"v{row['version']:<4} aliases={aliases:<22} {row['decision']:<9} {row['reason']}"
            )
    elif args.command == "promote":
        set_alias(client, name, alias, args.version)
        client.set_model_version_tag(name, args.version, TAG_DECISION, DECISION_PROMOTED)
        client.set_model_version_tag(name, args.version, TAG_REASON, "manual promotion")
        print(f"{name}@{alias} -> v{args.version}")
    elif args.command == "rollback":
        target = rollback(client, name, alias)
        print(f"{name}@{alias} -> v{target}")


if __name__ == "__main__":
    main()
