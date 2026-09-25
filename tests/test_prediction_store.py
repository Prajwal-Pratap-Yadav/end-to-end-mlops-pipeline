from datetime import timedelta
from pathlib import Path

import pytest

from src.prediction_store import PredictionRecord, PredictionStore
from src.schema import feature_names
from src.utils import utc_now


def _record(i: int, version: str = "1") -> PredictionRecord:
    features = dict.fromkeys(feature_names(), 1)
    features.update(
        contract_type="one_year",
        internet_service="dsl",
        payment_method="credit_card",
        avg_monthly_gb=None,
    )
    return PredictionRecord(
        prediction_id=f"id-{i}",
        features=features,
        churn_probability=i / 10,
        churn_prediction=int(i >= 5),
        model_name="churn-classifier",
        model_version=version,
        created_at=utc_now() + timedelta(seconds=i),
    )


@pytest.fixture
def store(tmp_path: Path) -> PredictionStore:
    return PredictionStore(tmp_path / "nested" / "predictions.db")


def test_log_and_read_back_in_chronological_order(store: PredictionStore) -> None:
    store.log_predictions([_record(i) for i in range(8)])
    recent = store.recent(limit=3)
    assert list(recent["prediction_id"]) == ["id-5", "id-6", "id-7"]
    assert recent["contract_type"].tolist() == ["one_year"] * 3
    assert recent["avg_monthly_gb"].isna().all()
    assert str(recent["created_at"].dt.tz) == "UTC"
    assert store.count() == 8


def test_feedback_attaches_ground_truth(store: PredictionStore) -> None:
    store.log_predictions([_record(i) for i in range(4)])
    assert store.add_feedback("id-1", 1)
    assert store.add_feedback("id-3", 0)
    assert not store.add_feedback("does-not-exist", 1)

    labeled = store.labeled(limit=10)
    assert list(labeled["prediction_id"]) == ["id-1", "id-3"]
    assert labeled["actual"].tolist() == [1, 0]
    assert store.count(labeled_only=True) == 2


def test_duplicate_ids_are_rejected_atomically(store: PredictionStore) -> None:
    store.log_predictions([_record(1)])
    with pytest.raises(Exception, match="UNIQUE"):
        store.log_predictions([_record(2), _record(1)])
    assert store.count() == 1  # the failed batch left nothing behind


def test_empty_store(store: PredictionStore) -> None:
    empty = store.recent(10)
    assert empty.empty
    assert set(feature_names()) <= set(empty.columns)
