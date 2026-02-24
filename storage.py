# -*- coding: utf-8 -*-
"""
Storage abstraction layer for rag service (Pipeline_RAG).
Supports both local filesystem and cloud storage (S3/MinIO).
"""

import os
import tempfile
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, BinaryIO
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def download_file(self, source: str, destination: Path) -> Path:
        """Download a file from storage to local path."""
        pass

    @abstractmethod
    def upload_file(self, source: Path, destination: str) -> str:
        """Upload a file from local path to storage. Returns the storage URL."""
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if a file exists in storage."""
        pass

    @abstractmethod
    def get_file_hash(self, path: str) -> str:
        """Get MD5 hash of a file."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage (original behavior)."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, source: str, destination: Path) -> Path:
        """For local storage, just return the source path if it exists."""
        source_path = Path(source)
        if source_path.exists():
            return source_path
        # If source is relative, resolve from base_dir
        full_path = self.base_dir / source
        if full_path.exists():
            return full_path
        raise FileNotFoundError(f"File not found: {source}")

    def upload_file(self, source: Path, destination: str) -> str:
        """For local storage, copy file to destination."""
        import shutil
        dest_path = self.base_dir / destination
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_path)
        return str(dest_path)

    def file_exists(self, path: str) -> bool:
        full_path = self.base_dir / path if not Path(path).is_absolute() else Path(path)
        return full_path.exists()

    def get_file_hash(self, path: str) -> str:
        full_path = self.base_dir / path if not Path(path).is_absolute() else Path(path)
        with open(full_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()


class S3StorageBackend(StorageBackend):
    """
    S3/MinIO cloud storage backend.

    Environment variables:
    - AWS_ACCESS_KEY_ID: Access key
    - AWS_SECRET_ACCESS_KEY: Secret key
    - S3_BUCKET or AWS_S3_BUCKET: Bucket name
    - AWS_REGION or S3_REGION: AWS region (default: ap-southeast-1)
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.bucket = bucket or os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET", "rag-papers")
        self.access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_KEY")
        self.region = region or os.getenv("AWS_REGION") or os.getenv("S3_REGION", "ap-southeast-1")

        # Initialize S3 client
        client_kwargs = {
            "service_name": "s3",
            "region_name": self.region,
        }

        if self.access_key and self.secret_key:
            client_kwargs["aws_access_key_id"] = self.access_key
            client_kwargs["aws_secret_access_key"] = self.secret_key

        self.client = boto3.client(**client_kwargs)

        # Ensure bucket exists
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                if self.region == "us-east-1":
                    self.client.create_bucket(Bucket=self.bucket)
                else:
                    self.client.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.region}
                    )
            except ClientError as e:
                print(f"Warning: Could not create bucket: {e}")

    def _get_s3_key(self, url_or_key: str) -> str:
        """Extract S3 key from URL or return as-is if already a key."""
        if url_or_key.startswith("s3://"):
            parsed = urlparse(url_or_key)
            return parsed.path.lstrip("/")
        elif url_or_key.startswith("http"):
            parsed = urlparse(url_or_key)
            # Handle S3 URL format: bucket.s3.region.amazonaws.com/key
            # or s3.region.amazonaws.com/bucket/key
            path = parsed.path.lstrip("/")

            # Check if hostname contains bucket name (virtual-hosted style)
            # e.g., mybucket.s3.ap-southeast-1.amazonaws.com
            hostname = parsed.netloc
            if hostname.endswith(".amazonaws.com"):
                # Extract bucket from hostname if virtual-hosted style
                parts = hostname.split(".")
                if len(parts) >= 4 and parts[1] == "s3":
                    # mybucket.s3.region.amazonaws.com/key -> key is the path
                    return path
                elif parts[0] == "s3":
                    # s3.region.amazonaws.com/bucket/key -> remove bucket from path
                    if path.startswith(self.bucket + "/"):
                        return path[len(self.bucket) + 1:]
            return path
        return url_or_key

    def download_file(self, source: str, destination: Path) -> Path:
        """Download file from S3 to local path."""
        s3_key = self._get_s3_key(source)
        destination.parent.mkdir(parents=True, exist_ok=True)

        print(f"📥 [S3] Downloading s3://{self.bucket}/{s3_key} -> {destination}")
        self.client.download_file(self.bucket, s3_key, str(destination))

        return destination

    def upload_file(self, source: Path, destination: str) -> str:
        """Upload file from local path to S3. Returns S3 URL."""
        s3_key = destination.lstrip("/")

        print(f"📤 [S3] Uploading {source} -> s3://{self.bucket}/{s3_key}")
        self.client.upload_file(str(source), self.bucket, s3_key)

        # Return direct S3 URL
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"

    def file_exists(self, path: str) -> bool:
        """Check if file exists in S3."""
        s3_key = self._get_s3_key(path)
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def get_file_hash(self, path: str) -> str:
        """Get MD5 hash (ETag) from S3 object."""
        s3_key = self._get_s3_key(path)
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=s3_key)
            # ETag is MD5 for non-multipart uploads
            etag = response.get("ETag", "").strip('"')
            return etag
        except ClientError:
            return ""


def get_storage_backend() -> StorageBackend:
    """
    Factory function to get the appropriate storage backend.

    Uses S3 if AWS_S3_BUCKET or S3_BUCKET is set, otherwise uses local storage.
    NOTE: Local storage uses TEMP_DIR now (cloud-first architecture)
    """
    from config import TEMP_DIR

    if os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET"):
        print("🌩️ [STORAGE] Using S3 storage backend")
        return S3StorageBackend()
    else:
        print("📁 [STORAGE] Using local storage backend (temp directory)")
        return LocalStorageBackend(TEMP_DIR)
