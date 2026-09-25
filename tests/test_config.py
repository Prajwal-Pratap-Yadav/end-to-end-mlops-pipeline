from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import AppConfig, env_overrides, load_config
from tests.conftest import CONFIG_PATH


def test_repository_config_is_valid() -> None:
    config = load_config(CONFIG_PATH, environ={})
    assert isinstance(config, AppConfig)
    assert config.registry.model_name == "churn-classifier"
    assert config.registry.champion_alias == "champion"
    assert len(config.training.candidates) == 3
    # Paths use forward slashes so the same config works on Windows and Linux.
    assert "\\" not in str(config.data.raw_path.as_posix())


def test_env_overrides_are_nested_and_typed() -> None:
    overrides = env_overrides(
        {
            "MLOPS__TRAINING__CV_FOLDS": "3",
            "MLOPS__MONITORING__DRIFT__PSI_THRESHOLD": "0.25",
            "MLOPS__MONITORING__RETRAIN__ENABLED": "false",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "UNRELATED": "ignored",
        }
    )
    assert overrides == {
        "training": {"cv_folds": 3},
        "monitoring": {"drift": {"psi_threshold": 0.25}, "retrain": {"enabled": False}},
        "mlflow": {"tracking_uri": "http://mlflow:5000"},
    }


def test_load_config_applies_environment_then_explicit_overrides() -> None:
    config = load_config(
        CONFIG_PATH,
        overrides={"training": {"cv_folds": 4}},
        environ={"MLOPS__TRAINING__CV_FOLDS": "3", "MLFLOW_TRACKING_URI": "sqlite:///x.db"},
    )
    assert config.training.cv_folds == 4  # explicit override wins over the environment
    assert config.mlflow.tracking_uri == "sqlite:///x.db"


def test_config_path_can_come_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(CONFIG_PATH.read_text().replace("cv_folds: 5", "cv_folds: 2"))
    monkeypatch.setenv("MLOPS_CONFIG", str(custom))
    assert load_config(environ={}).training.cv_folds == 2


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_config(CONFIG_PATH, overrides={"training": {"cv_fold": 3}}, environ={})


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(CONFIG_PATH, overrides={"data": {"test_size": 1.5}}, environ={})


def test_csv_source_requires_a_path() -> None:
    with pytest.raises(ValidationError, match="csv_path"):
        load_config(CONFIG_PATH, overrides={"data": {"source": "csv"}}, environ={})


def test_enabled_s3_requires_a_bucket() -> None:
    with pytest.raises(ValidationError, match="bucket"):
        load_config(CONFIG_PATH, overrides={"storage": {"s3": {"enabled": True}}}, environ={})


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", environ={})
