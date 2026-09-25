from pathlib import Path

import pandas as pd

from monitoring.retrain import build_retraining_sets, minutes_since_last_version, run_retraining
from src.prediction_store import PredictionStore
from src.registry import TAG_DECISION, configure_mlflow, get_version_by_alias
from src.schema import TARGET_COLUMN, feature_names
from tests.conftest import TrainedEnv, drifted_customers, log_scored_customers, make_test_config


def _labeled_window(n: int) -> pd.DataFrame:
    frame = drifted_customers(n, seed=3)
    return frame.assign(
        created_at=pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        actual=frame[TARGET_COLUMN],
    ).drop(columns=TARGET_COLUMN)


def test_holdout_is_the_newest_slice() -> None:
    labeled = _labeled_window(400).sample(frac=1, random_state=0)  # shuffled input
    reference = drifted_customers(100, seed=4)
    train, holdout, topup = build_retraining_sets(labeled, reference, 0.25, 100, seed=0)

    ordered = labeled.sort_values("created_at")
    newest = ordered.tail(100)
    pd.testing.assert_frame_equal(
        holdout[feature_names()].reset_index(drop=True),
        newest[feature_names()].reset_index(drop=True),
    )
    assert len(train) == 300
    assert topup == 0
    assert holdout[TARGET_COLUMN].tolist() == newest["actual"].astype(int).tolist()


def test_scarce_fresh_data_is_topped_up_with_reference() -> None:
    reference = drifted_customers(500, seed=5)
    train, holdout, topup = build_retraining_sets(_labeled_window(100), reference, 0.25, 300, 0)
    assert len(holdout) == 25
    assert topup == 225
    assert len(train) == 300


def test_guard_rails(fresh_env: TrainedEnv, tmp_path: Path) -> None:
    config = fresh_env.config
    store = PredictionStore(config.serving.prediction_log_path)

    disabled = config.model_copy(
        update={
            "monitoring": config.monitoring.model_copy(
                update={"retrain": config.monitoring.retrain.model_copy(update={"enabled": False})}
            )
        }
    )
    assert run_retraining(disabled, store=store).status == "skipped"

    cooling = config.model_copy(
        update={
            "monitoring": config.monitoring.model_copy(
                update={
                    "retrain": config.monitoring.retrain.model_copy(update={"cooldown_minutes": 60})
                }
            )
        }
    )
    outcome = run_retraining(cooling, store=store)
    assert outcome.status == "skipped"
    assert "cooldown" in outcome.reason
    minutes = minutes_since_last_version(configure_mlflow(config), "churn-classifier")
    assert minutes is not None
    assert minutes < 5

    not_enough = run_retraining(config, store=store)
    assert not_enough.status == "skipped"
    assert "labeled predictions" in not_enough.reason

    empty = make_test_config(tmp_path / "empty")
    assert run_retraining(empty).reason == "no champion model to retrain from"


def test_retraining_on_shifted_market_promotes_a_better_model(fresh_env: TrainedEnv) -> None:
    config = fresh_env.config
    store = PredictionStore(config.serving.prediction_log_path)
    champion = fresh_env.load_champion()
    log_scored_customers(store, champion, drifted_customers(1500, seed=21))

    outcome = run_retraining(config, store=store, reasons=["data drift"])

    assert outcome.status == "promoted", outcome.reason
    assert outcome.model_version == "2"
    assert outcome.n_holdout == 375
    assert outcome.n_fresh_train == 1125
    assert outcome.n_reference_topup == 0
    assert outcome.challenger_score is not None
    assert outcome.champion_score is not None
    assert outcome.challenger_score >= outcome.champion_score + 0.01

    client = configure_mlflow(config)
    new_champion = get_version_by_alias(client, "churn-classifier", "champion")
    assert new_champion is not None
    assert str(new_champion.version) == "2"
    assert new_champion.tags[TAG_DECISION] == "promoted"
    run = client.get_run(outcome.run_id)  # type: ignore[arg-type]
    assert run.data.tags["trigger"] == "monitoring"
    assert run.data.tags["replaces_champion"] == "1"


def test_retraining_on_unchanged_market_is_rejected(fresh_env: TrainedEnv) -> None:
    from src.ingest import generate_customers

    config = fresh_env.config
    store = PredictionStore(config.serving.prediction_log_path)
    log_scored_customers(store, fresh_env.load_champion(), generate_customers(800, seed=22))

    outcome = run_retraining(config, store=store, reasons=["manual"], force=True)

    # Same distribution: the challenger cannot beat the champion by the margin.
    assert outcome.status == "rejected", outcome.reason
    client = configure_mlflow(config)
    assert str(get_version_by_alias(client, "churn-classifier", "champion").version) == "1"  # type: ignore[union-attr]
