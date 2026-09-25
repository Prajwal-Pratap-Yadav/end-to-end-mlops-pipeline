"""Thin S3 client for publishing data snapshots and monitoring reports.

Works against AWS S3 or any S3-compatible store (MinIO, LocalStack, Ceph) via
``storage.s3.endpoint_url``. Credentials come from the standard AWS chain
(environment variables, shared config, instance/role credentials) - never from
the repository.

Disabled by default (``storage.s3.enabled: false``); when enabled, ingestion
uploads each raw snapshot and the monitor uploads every drift report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError

from src.config import S3Config

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)


class S3Storage:
    """Upload/download helpers scoped to a bucket and key prefix."""

    def __init__(self, bucket: str, prefix: str = "", client: S3Client | None = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client: S3Client = client or boto3.client("s3")

    @classmethod
    def from_config(cls, config: S3Config) -> S3Storage:
        """Create a storage client from the ``storage.s3`` config section."""
        if not config.bucket:
            raise ValueError("storage.s3.bucket must be set to use S3 storage")
        kwargs: dict[str, Any] = {}
        if config.endpoint_url:
            kwargs["endpoint_url"] = config.endpoint_url
        if config.region:
            kwargs["region_name"] = config.region
        return cls(config.bucket, config.prefix, boto3.client("s3", **kwargs))

    def key(self, relative_key: str) -> str:
        """Prefix a relative key with the configured prefix."""
        relative_key = relative_key.lstrip("/")
        return f"{self.prefix}/{relative_key}" if self.prefix else relative_key

    def uri(self, relative_key: str) -> str:
        """Return the ``s3://`` URI for a relative key."""
        return f"s3://{self.bucket}/{self.key(relative_key)}"

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist (useful for local S3 emulators)."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            logger.info("Creating bucket %s", self.bucket)
            self.client.create_bucket(Bucket=self.bucket)

    def upload_file(self, local_path: str | Path, relative_key: str) -> str:
        """Upload a single file and return its ``s3://`` URI."""
        self.client.upload_file(str(local_path), self.bucket, self.key(relative_key))
        return self.uri(relative_key)

    def upload_directory(self, local_dir: str | Path, relative_prefix: str) -> list[str]:
        """Upload every file under ``local_dir`` preserving relative paths."""
        root = Path(local_dir)
        uris = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            uris.append(self.upload_file(path, f"{relative_prefix.rstrip('/')}/{relative}"))
        return uris

    def download_file(self, relative_key: str, local_path: str | Path) -> Path:
        """Download an object to ``local_path`` (parent directories are created)."""
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self.key(relative_key), str(target))
        return target

    def list_keys(self, relative_prefix: str = "") -> list[str]:
        """List object keys (relative to the storage prefix) under a prefix."""
        full_prefix = (
            self.key(relative_prefix)
            if relative_prefix
            else (f"{self.prefix}/" if self.prefix else "")
        )
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                keys.append(key[len(self.prefix) + 1 :] if self.prefix else key)
        return keys
