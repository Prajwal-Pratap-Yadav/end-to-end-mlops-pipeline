import mlflow
import pytest

from src.config import AppConfig
from src.registry import TAG_DECISION, configure_mlflow, get_version_by_alias
from src.train import build_estimator, expand_grid, main, run_training
from tests.conftest import TrainedEnv


def test_expand_grid() -> None:
    assert expand_grid({}) == [{}]
    assert expand_grid({"b": [1, 2], "a": ["x"]}) == [{"a": "x", "b": 1}, {"a": "x", "b": 2}]


def test_unknown_estimator_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported estimator"):
        build_estimator("svm", {}, 0)


def test_training_registers_and_promotes_first_champion(trained_env: TrainedEnv) -> None:
    result = trained_env.result
    assert result.model_version == "1"
    assert result.decision.promote
    assert result.champion_version == "1"
    assert result.test_metrics["roc_auc"] > 0.75
    assert {c.name for c in result.candidates} == {"logistic_regression", "hist_gradient_boosting"}

    client = configure_mlflow(trained_env.config)
    champion = get_version_by_alias(client, "churn-classifier", "champion")
    assert champion is not None
    assert str(champion.version) == "1"  # the SQL store returns ints despite str hints
    assert champion.tags[TAG_DECISION] == "promoted"


def test_run_logs_params_metrics_lineage_and_artifacts(trained_env: TrainedEnv) -> None:
    client = configure_mlflow(trained_env.config)
    run = client.get_run(trained_env.result.run_id)
    assert run.data.params["best_candidate"] == trained_env.result.best_candidate
    assert run.data.params["decision_threshold"] == "0.35"
    assert "test_roc_auc" in run.data.metrics
    assert "cv_roc_auc_mean" in run.data.metrics
    assert run.data.tags["promotion.decision"] == "promoted"
    assert {d.dataset.name for d in run.inputs.dataset_inputs} == {"train", "evaluation"}

    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"evaluation", "reference", "promotion_decision.json"} <= artifacts
    evaluation = {a.path for a in client.list_artifacts(run.info.run_id, "evaluation")}
    assert "evaluation/roc_curve.png" in evaluation
    assert "evaluation/feature_importance.csv" in evaluation

    # Every candidate configuration is a nested child run.
    children = client.search_runs(
        [run.info.experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'"
    )
    assert len(children) == len(trained_env.result.candidates)


def test_model_is_skops_serialized_with_signature(trained_env: TrainedEnv) -> None:
    configure_mlflow(trained_env.config)  # resolve models:/ against this test registry
    model_uri = "models:/churn-classifier/1"
    info = mlflow.models.get_model_info(model_uri)
    assert info.flavors["sklearn"]["serialization_format"] == "skops"
    assert "src.preprocess.ChurnFeatureEngineer" in info.flavors["sklearn"]["skops_trusted_types"]
    assert {c.name for c in info.signature.inputs} >= {"tenure_months", "contract_type"}


def test_second_run_without_promotion_keeps_champion(fresh_env: TrainedEnv) -> None:
    result = run_training(fresh_env.config, promote=False)
    assert result.model_version == "2"
    assert not result.decision.promote
    client = configure_mlflow(fresh_env.config)
    assert str(get_version_by_alias(client, "churn-classifier", "champion").version) == "1"  # type: ignore[union-attr]
    assert str(get_version_by_alias(client, "churn-classifier", "challenger").version) == "2"  # type: ignore[union-attr]


def test_cli_skip_if_champion_exists(
    fresh_env: TrainedEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = fresh_env.root / "config.yaml"
    config_path.write_text(
        "training:\n  candidates:\n    - {name: lr, estimator: logistic_regression}\n"
        f"mlflow:\n  tracking_uri: {fresh_env.config.mlflow.tracking_uri}\n"
    )
    main(["--config", str(config_path), "--skip-if-champion-exists"])
    client = configure_mlflow(fresh_env.config)
    assert [str(v.version) for v in client.search_model_versions("name='churn-classifier'")] == [
        "1"
    ]
    assert "skipping training" in capsys.readouterr().out


def test_invalid_data_blocks_training(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.train as train_module
    from src.ingest import generate_customers
    from src.validation import DataValidationError

    broken = generate_customers(500, seed=1)
    broken.loc[0, "contract_type"] = "weekly"
    monkeypatch.setattr(train_module, "ingest", lambda cfg: (broken, cfg.data.raw_path))
    with pytest.raises(DataValidationError):
        run_training(config)
    assert not configure_mlflow(config).search_registered_models()
