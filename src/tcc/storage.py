"""Storage configuration for the TCC pipeline.

Reads a YAML config file and provides resolved paths for raw data,
processed data, and S3 storage options.  Both local and S3 backends
are supported; the active backend is selected by ``storage_backend``
in the YAML file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StorageConfig:
    """Resolved storage configuration.

    Attributes
    ----------
    storage_backend:
        ``"local"`` or ``"s3"``.
    dataset_name:
        Dataset name (e.g. ``"pouring"``).
    raw_dir:
        Path to raw data.  Local path or ``s3://`` URI.
    processed_dir:
        Path to processed dataset root (includes dataset_name subdir).
        Local path or ``s3://`` URI.
    cache_dir:
        Local cache directory (S3 mode only, ``None`` for local).
    s3_storage_options:
        kwargs for ``fsspec.open`` when using S3 (``None`` for local).
    """

    storage_backend: str
    dataset_name: str
    raw_dir: str
    processed_dir: str
    cache_dir: str | None = None
    s3_storage_options: dict[str, Any] | None = None


def _resolve_env(value: str) -> str:
    """Expand ``${VAR:-default}`` patterns in a string."""

    def _sub(m: re.Match[str]) -> str:
        var, default = m.group(1), m.group(2) or ""
        return os.environ.get(var, default)

    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", _sub, value)


def load_storage_config(yaml_path: str) -> StorageConfig:
    """Load a storage config YAML and return a resolved :class:`StorageConfig`.

    Parameters
    ----------
    yaml_path:
        Path to the YAML config file (e.g. ``configs/pouring.yaml``).

    Raises
    ------
    ValueError
        If ``storage_backend`` is not ``"local"`` or ``"s3"``.
    """
    with open(yaml_path) as fh:  # noqa: PTH123
        raw: dict[str, Any] = yaml.safe_load(fh)

    backend: str = raw["storage_backend"]
    dataset_name: str = raw["dataset_name"]

    if backend not in ("local", "s3"):
        msg = f"storage_backend must be 'local' or 's3', got '{backend}'"
        raise ValueError(msg)

    if backend == "local":
        local: dict[str, str] = raw["local"]
        return StorageConfig(
            storage_backend="local",
            dataset_name=dataset_name,
            raw_dir=local["raw_dir"],
            processed_dir=str(Path(local["processed_dir"]) / dataset_name),
            cache_dir=None,
            s3_storage_options=None,
        )

    # S3 backend
    s3: dict[str, Any] = raw["s3"]
    endpoint = _resolve_env(str(s3["endpoint"]))
    bucket: str = s3["bucket"]
    prefix: str = s3["prefix"]
    use_ssl: bool = s3.get("use_ssl", endpoint.startswith("https"))

    return StorageConfig(
        storage_backend="s3",
        dataset_name=dataset_name,
        raw_dir=f"s3://{bucket}/{prefix}/raw",
        processed_dir=f"s3://{bucket}/{prefix}/processed",
        cache_dir=s3.get("cache_dir", "data/cache"),
        s3_storage_options={
            "endpoint_url": endpoint,
            "key": os.environ.get("MINIO_ACCESS_KEY", ""),
            "secret": os.environ.get("MINIO_SECRET_KEY", ""),
            "use_ssl": use_ssl,
        },
    )
