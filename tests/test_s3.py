"""S3 integration, exercised against moto's in-process S3 implementation."""

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from src.config import AppConfig, S3Config
from src.ingest import ingest
from src.s3 import S3Storage


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        yield


def test_upload_list_download_round_trip(aws: None, tmp_path: Path) -> None:
    storage = S3Storage.from_config(S3Config(enabled=True, bucket="mlops", prefix="churn/"))
    storage.ensure_bucket()
    storage.ensure_bucket()  # idempotent

    (tmp_path / "reports" / "sub").mkdir(parents=True)
    (tmp_path / "reports" / "a.json").write_text("{}")
    (tmp_path / "reports" / "sub" / "b.html").write_text("<html/>")
    uris = storage.upload_directory(tmp_path / "reports", "monitoring")

    assert uris == ["s3://mlops/churn/monitoring/a.json", "s3://mlops/churn/monitoring/sub/b.html"]
    assert storage.list_keys("monitoring") == ["monitoring/a.json", "monitoring/sub/b.html"]
    downloaded = storage.download_file("monitoring/sub/b.html", tmp_path / "out" / "b.html")
    assert downloaded.read_text() == "<html/>"


def test_from_config_requires_bucket() -> None:
    with pytest.raises(ValueError, match="bucket"):
        S3Storage.from_config(S3Config())


def test_ingestion_publishes_snapshot_when_enabled(aws: None, config: AppConfig) -> None:
    boto3.client("s3").create_bucket(Bucket="lake")
    enabled = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"s3": S3Config(enabled=True, bucket="lake")}
            )
        }
    )
    ingest(enabled)
    keys = S3Storage("lake", "churn-mlops").list_keys("data")
    assert keys == ["data/raw/churn.csv"]
