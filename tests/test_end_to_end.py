"""The whole production loop, in-process.

simulator -> API (/predict, /feedback) -> prediction log -> monitor (drift) ->
retraining -> promotion gate -> registry alias -> API hot reload.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.model_manager import ModelManager
from monitoring.monitor import run_monitoring_cycle
from monitoring.simulate import simulate, to_payload
from src.ingest import generate_customers
from tests.conftest import TrainedEnv


class _Response:
    """Adapts an httpx response to the subset of ``requests.Response`` the simulator uses."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.status_code = response.status_code
        self.ok = response.is_success
        self.text = response.text

    def json(self) -> Any:
        return self._response.json()


class _Session:
    """Routes the simulator's HTTP calls into the FastAPI TestClient."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def post(self, url: str, json: Any, timeout: float) -> _Response:
        return _Response(self.client.post(url.removeprefix("http://api"), json=json))

    def get(self, url: str, timeout: float) -> _Response:
        return _Response(self.client.get(url.removeprefix("http://api")))


def test_payload_conversion_matches_api_contract() -> None:
    row = generate_customers(40, seed=1).iloc[0]
    payload = to_payload(row)
    assert isinstance(payload["senior_citizen"], bool)
    assert isinstance(payload["tenure_months"], int)
    assert payload["contract_type"] in {"month_to_month", "one_year", "two_year"}


class _Stub:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, body: Any) -> None:
        self._body = body

    def json(self) -> Any:
        return self._body


class _RecordingSession:
    """Fake API that remembers which customers received ground truth."""

    def __init__(self) -> None:
        self.contracts: dict[str, str] = {}
        self.labeled: set[str] = set()

    def post(self, url: str, json: Any, timeout: float) -> _Stub:
        if url.endswith("/feedback"):
            self.labeled.add(json["prediction_id"])
            return _Stub({"status": "recorded"})
        instances = json["instances"] if url.endswith("/batch") else [json]
        results = []
        for instance in instances:
            prediction_id = f"p{len(self.contracts)}"
            self.contracts[prediction_id] = instance["contract_type"]
            results.append(
                {"prediction_id": prediction_id, "churn_probability": 0.5, "model_version": "1"}
            )
        return _Stub({"predictions": results} if url.endswith("/batch") else results[0])


def test_feedback_sampling_is_independent_of_customer_attributes() -> None:
    # Regression test: feedback sampling once reused the generator's seed, so the
    # same uniforms that assigned each contract type decided who got labeled and no
    # two-year customer was ever labeled - biasing monitoring and retraining.
    session = _RecordingSession()
    simulate(
        "http://api",
        3000,
        drift_strength=0.8,
        feedback_rate=0.8,
        seed=8,
        batch_size=100,
        session=session,  # type: ignore[arg-type]
    )
    contracts = session.contracts
    labeled_share = len(session.labeled) / len(contracts)
    assert abs(labeled_share - 0.8) < 0.03
    for contract in ("month_to_month", "one_year", "two_year"):
        overall = sum(c == contract for c in contracts.values()) / len(contracts)
        labeled = sum(contracts[p] == contract for p in session.labeled) / len(session.labeled)
        assert abs(labeled - overall) < 0.03, (contract, labeled, overall)


@pytest.mark.integration
def test_drift_triggers_retraining_and_api_hot_reloads(fresh_env: TrainedEnv) -> None:
    config = fresh_env.config
    manager = ModelManager(config)
    with TestClient(create_app(config, manager=manager)) as client:
        session: Any = _Session(client)

        # 1. Normal traffic: the monitor sees a healthy model.
        normal = simulate(
            "http://api", 400, seed=101, feedback_rate=1.0, batch_size=100, session=session
        )
        assert normal.errors == 0
        assert normal.predictions_ok == 400
        healthy = run_monitoring_cycle(config, store=client.app.state.store)  # type: ignore[attr-defined]
        assert healthy.status == "ok"
        assert healthy.retraining is None

        # 2. The market shifts: drift is detected and retraining promotes a challenger.
        shifted = simulate(
            "http://api",
            1500,
            drift_strength=1.0,
            seed=102,
            feedback_rate=1.0,
            batch_size=100,
            session=session,
        )
        assert shifted.model_versions == ["1"]
        result = run_monitoring_cycle(config, store=client.app.state.store)  # type: ignore[attr-defined]
        assert result.drift is not None
        assert result.drift.dataset_drift
        assert result.retraining is not None
        assert result.retraining.status == "promoted", result.retraining.reason

        # 3. The API picks up the new champion without a restart.
        assert manager.refresh()
        assert client.get("/model").json()["version"] == "2"
        after = simulate(
            "http://api", 300, drift_strength=1.0, seed=103, batch_size=50, session=session
        )
        assert after.model_versions == ["2"]
        assert after.client_side_roc_auc is not None
        assert shifted.client_side_roc_auc is not None
        assert after.client_side_roc_auc > shifted.client_side_roc_auc
