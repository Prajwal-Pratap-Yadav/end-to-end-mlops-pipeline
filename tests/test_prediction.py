from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingest import generate_customers
from src.predict import ModelNotFoundError, main, predict_frame, resolve_model_uri
from src.registry import configure_mlflow
from tests.conftest import TrainedEnv


def test_champion_loads_with_lineage(trained_env: TrainedEnv) -> None:
    loaded = trained_env.load_champion()
    assert loaded.name == "churn-classifier"
    assert loaded.version == "1"
    assert loaded.run_id == trained_env.result.run_id
    assert loaded.model_uri == "models:/churn-classifier/1"
    assert loaded.training_metrics["roc_auc"] == pytest.approx(
        trained_env.result.test_metrics["roc_auc"]
    )
    assert loaded.describe()["version"] == "1"


def test_alias_resolution(trained_env: TrainedEnv) -> None:
    client = configure_mlflow(trained_env.config)
    uri, name, version, run_id = resolve_model_uri(
        client, trained_env.config, "models:/churn-classifier@champion"
    )
    assert (uri, name, version) == ("models:/churn-classifier/1", "churn-classifier", "1")
    assert run_id == trained_env.result.run_id
    with pytest.raises(ModelNotFoundError):
        resolve_model_uri(client, trained_env.config, "models:/churn-classifier@missing")


def test_predictions_are_probabilities_and_respect_threshold(trained_env: TrainedEnv) -> None:
    loaded = trained_env.load_champion()
    customers = generate_customers(200, seed=12)
    scored = predict_frame(loaded, customers, threshold=0.35)
    assert scored["churn_probability"].between(0, 1).all()
    assert (scored["churn_prediction"] == (scored["churn_probability"] >= 0.35)).all()
    assert scored["customer_id"].equals(customers["customer_id"])


def test_model_is_useful_on_unseen_customers(trained_env: TrainedEnv) -> None:
    from src.evaluate import compute_metrics

    loaded = trained_env.load_champion()
    customers = generate_customers(1500, seed=99)
    metrics = compute_metrics(customers["churned"], loaded.predict_proba(customers))
    assert metrics["roc_auc"] > 0.75


def test_prediction_is_deterministic(trained_env: TrainedEnv) -> None:
    loaded = trained_env.load_champion()
    customers = generate_customers(50, seed=5)
    np.testing.assert_array_equal(loaded.predict_proba(customers), loaded.predict_proba(customers))


def test_batch_cli(trained_env: TrainedEnv, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "training:\n  candidates:\n    - {name: lr, estimator: logistic_regression}\n"
        f"mlflow:\n  tracking_uri: {trained_env.config.mlflow.tracking_uri}\n"
    )
    source = tmp_path / "customers.csv"
    generate_customers(120, seed=3).drop(columns="churned").to_csv(source, index=False)
    output = tmp_path / "out" / "scored.csv"
    main(["--config", str(config_path), "--input", str(source), "--output", str(output)])
    scored = pd.read_csv(output)
    assert list(scored.columns) == ["customer_id", "churn_probability", "churn_prediction"]
    assert len(scored) == 120
