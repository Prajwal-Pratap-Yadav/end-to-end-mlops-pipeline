import math

import pytest

from src.config import PromotionConfig
from src.registry import (
    DECISION_PROMOTED,
    TAG_DECISION,
    configure_mlflow,
    decide_promotion,
    describe_versions,
    get_version_by_alias,
    main,
    rollback,
    set_alias,
)
from src.train import train_and_register
from tests.conftest import TrainedEnv, drifted_customers

POLICY = PromotionConfig(metric="roc_auc", min_improvement=0.01, min_metrics={"roc_auc": 0.75})


def test_first_model_is_promoted_when_it_passes_the_gate() -> None:
    decision = decide_promotion({"roc_auc": 0.8}, None, POLICY)
    assert decision.promote
    assert decision.champion_score is None


def test_quality_gate_blocks_weak_models_even_without_champion() -> None:
    decision = decide_promotion({"roc_auc": 0.7}, None, POLICY)
    assert not decision.promote
    assert decision.failed_gates == {"roc_auc": 0.7}
    assert "quality gate" in decision.reason


def test_nan_metric_fails_the_gate() -> None:
    assert not decide_promotion({"roc_auc": float("nan")}, None, POLICY).promote


def test_challenger_must_beat_champion_by_margin() -> None:
    assert decide_promotion({"roc_auc": 0.82}, {"roc_auc": 0.80}, POLICY).promote
    assert not decide_promotion({"roc_auc": 0.805}, {"roc_auc": 0.80}, POLICY).promote
    assert not decide_promotion({"roc_auc": 0.78}, {"roc_auc": 0.80}, POLICY).promote


def test_missing_champion_score_does_not_block() -> None:
    assert decide_promotion({"roc_auc": 0.8}, {"roc_auc": math.nan}, POLICY).promote


def test_decision_serialises_to_tags() -> None:
    tags = decide_promotion({"roc_auc": 0.82}, {"roc_auc": 0.80}, POLICY).as_tags()
    assert tags[TAG_DECISION] == DECISION_PROMOTED
    assert tags["promotion.champion_score"] == "0.800000"


def test_unknown_alias_or_model_returns_none(trained_env: TrainedEnv) -> None:
    client = configure_mlflow(trained_env.config)
    assert get_version_by_alias(client, "churn-classifier", "nope") is None
    assert get_version_by_alias(client, "no-such-model", "champion") is None


def test_rollback_restores_previous_champion(fresh_env: TrainedEnv) -> None:
    config = fresh_env.config
    client = configure_mlflow(config)
    name, alias = config.registry.model_name, config.registry.champion_alias

    with pytest.raises(RuntimeError, match="No previously promoted version"):
        rollback(client, name, alias)

    # Register v2 without promotion, then promote it manually and roll back.
    data = drifted_customers(1200, seed=31)
    train_and_register(config, data.iloc[:900], data.iloc[900:], promote=False)
    set_alias(client, name, alias, "2")
    client.set_model_version_tag(name, "2", TAG_DECISION, DECISION_PROMOTED)

    assert rollback(client, name, alias) == "1"
    assert str(get_version_by_alias(client, name, alias).version) == "1"  # type: ignore[union-attr]
    rows = {row["version"]: row for row in describe_versions(client, name)}
    assert rows["1"]["aliases"] == ["champion"]
    assert rows["2"]["aliases"] == ["challenger"]


def test_registry_cli(fresh_env: TrainedEnv, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = fresh_env.root / "config.yaml"
    config_path.write_text(
        "training:\n  candidates:\n    - {name: lr, estimator: logistic_regression}\n"
        f"mlflow:\n  tracking_uri: {fresh_env.config.mlflow.tracking_uri}\n"
    )
    main(["--config", str(config_path), "list"])
    assert "v1" in capsys.readouterr().out
    main(["--config", str(config_path), "promote", "--version", "1"])
    assert "champion -> v1" in capsys.readouterr().out
