from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.model_manager import ModelManager
from app.schemas import EXAMPLE_CUSTOMER, CustomerFeatures
from src.config import AppConfig
from src.prediction_store import PredictionStore
from src.registry import configure_mlflow, set_alias
from src.schema import feature_names
from src.train import train_and_register
from tests.conftest import TrainedEnv, drifted_customers, make_test_config


@pytest.fixture
def client(trained_env: TrainedEnv, tmp_path: Any) -> Iterator[TestClient]:
    # Share the trained registry but give every test its own prediction log.
    config = trained_env.config.model_copy(
        update={
            "serving": trained_env.config.serving.model_copy(
                update={"prediction_log_path": tmp_path / "predictions.db", "max_batch_size": 5}
            )
        }
    )
    with TestClient(create_app(config)) as test_client:
        yield test_client


def _customer(**changes: Any) -> dict[str, Any]:
    return {**EXAMPLE_CUSTOMER, **changes}


def test_request_schema_matches_feature_contract() -> None:
    assert list(CustomerFeatures.model_fields) == feature_names()


def test_health_ready_and_root(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["model_version"] == "1"
    assert client.get("/").json()["model"] == "churn-classifier"


def test_predict_returns_probability_and_logs_prediction(client: TestClient) -> None:
    response = client.post("/predict", json=_customer())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])
    assert body["model_version"] == "1"

    store: PredictionStore = client.app.state.store  # type: ignore[attr-defined]
    logged = store.recent(1).iloc[0]
    assert logged["prediction_id"] == body["prediction_id"]
    assert logged["contract_type"] == "month_to_month"
    assert logged["senior_citizen"] == 0  # booleans stored as model-ready 0/1


def test_high_risk_customer_scores_higher_than_loyal_customer(client: TestClient) -> None:
    risky = client.post("/predict", json=_customer()).json()["churn_probability"]
    loyal = client.post(
        "/predict",
        json=_customer(
            tenure_months=60,
            total_charges=3300.0,
            num_support_tickets=0,
            contract_type="two_year",
            payment_method="credit_card",
            monthly_charges=55.0,
            has_tech_support=True,
        ),
    ).json()["churn_probability"]
    assert risky > loyal


def test_nullable_features_may_be_omitted(client: TestClient) -> None:
    payload = _customer()
    payload.pop("total_charges")
    payload.pop("avg_monthly_gb")
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        _customer(tenure_months=-1),
        _customer(contract_type="weekly"),
        _customer(monthly_charges=10_000),
        {**_customer(), "unexpected": 1},
        {k: v for k, v in _customer().items() if k != "payment_method"},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_prediction_and_size_limit(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"instances": [_customer()] * 3})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert len({p["prediction_id"] for p in predictions}) == 3

    too_big = client.post("/predict/batch", json={"instances": [_customer()] * 6})
    assert too_big.status_code == 413
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_feedback(client: TestClient) -> None:
    prediction_id = client.post("/predict", json=_customer()).json()["prediction_id"]
    ok = client.post("/feedback", json={"prediction_id": prediction_id, "churned": True})
    assert ok.status_code == 200
    assert ok.json() == {"prediction_id": prediction_id, "status": "recorded"}
    missing = client.post("/feedback", json={"prediction_id": "nope", "churned": False})
    assert missing.status_code == 404


def test_model_info(client: TestClient, trained_env: TrainedEnv) -> None:
    info = client.get("/model").json()
    assert info["version"] == "1"
    assert info["run_id"] == trained_env.result.run_id
    assert info["decision_threshold"] == 0.35
    assert info["training_metrics"]["roc_auc"] > 0.75


def test_prometheus_metrics_use_route_templates(client: TestClient) -> None:
    client.post("/predict", json=_customer())
    client.get("/definitely-not-a-route")
    text = client.get("/metrics").text
    assert 'churn_api_http_requests_total{method="POST",route="/predict",status="200"}' in text
    assert 'route="unmatched"' in text
    assert 'churn_api_model_info{model_name="churn-classifier",model_version="1"' in text
    assert "churn_api_prediction_probability_bucket" in text


def test_not_ready_without_champion(tmp_path: Any) -> None:
    config = make_test_config(tmp_path)  # empty registry
    with TestClient(create_app(config)) as empty_client:
        ready = empty_client.get("/ready")
        assert ready.status_code == 503
        assert "no model registered" in ready.json()["detail"]
        assert empty_client.post("/predict", json=_customer()).status_code == 503
        assert empty_client.get("/health").status_code == 200


def test_reload_requires_admin_token_when_configured(
    trained_env: TrainedEnv, tmp_path: Any
) -> None:
    config = trained_env.config.model_copy(
        update={
            "serving": trained_env.config.serving.model_copy(
                update={"prediction_log_path": tmp_path / "p.db", "admin_token": "s3cret"}
            )
        }
    )
    with TestClient(create_app(config)) as secured:
        assert secured.post("/model/reload").status_code == 401
        assert secured.post("/model/reload", headers={"X-Admin-Token": "wrong"}).status_code == 401
        ok = secured.post("/model/reload", headers={"X-Admin-Token": "s3cret"})
        assert ok.status_code == 200
        assert ok.json() == {"reloaded": True, "previous_version": "1", "current_version": "1"}


def test_hot_reload_when_champion_alias_moves(fresh_env: TrainedEnv) -> None:
    config: AppConfig = fresh_env.config
    manager = ModelManager(config)
    with TestClient(create_app(config, manager=manager)) as api:
        assert api.get("/model").json()["version"] == "1"
        assert not manager.refresh()  # nothing changed

        data = drifted_customers(1000, seed=41)
        train_and_register(config, data.iloc[:750], data.iloc[750:], promote=False)
        set_alias(configure_mlflow(config), "churn-classifier", "champion", "2")

        assert manager.refresh()
        assert api.get("/model").json()["version"] == "2"
        assert api.post("/predict", json=_customer()).json()["model_version"] == "2"


def test_failed_load_keeps_serving_previous_model(trained_env: TrainedEnv) -> None:
    def broken_loader(*args: Any) -> Any:
        raise OSError("artifact store unavailable")

    manager = ModelManager(trained_env.config, loader=broken_loader)
    with pytest.raises(OSError, match="artifact store"):
        manager.refresh()
    assert not manager.is_ready
    assert "artifact store unavailable" in (manager.last_error or "")
