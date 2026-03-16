"""Tests for tcc.storage — config loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tcc.storage import StorageConfig, load_storage_config  # noqa: F401


@pytest.fixture()
def local_config_path(tmp_path: Path) -> Path:
    """Write a minimal local-mode config and return its path."""
    cfg = {
        "storage_backend": "local",
        "dataset_name": "pouring",
        "local": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
        },
        "s3": {
            "endpoint": "http://localhost:9000",
            "bucket": "landing",
            "prefix": "tcc/pouring",
            "use_ssl": False,
            "cache_dir": str(tmp_path / "cache"),
        },
    }
    p = tmp_path / "pouring.yaml"
    p.write_text(yaml.dump(cfg))
    return p


@pytest.fixture()
def s3_config_path(tmp_path: Path) -> Path:
    """Write a minimal S3-mode config and return its path."""
    cfg = {
        "storage_backend": "s3",
        "dataset_name": "pouring",
        "local": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
        },
        "s3": {
            "endpoint": "http://localhost:9000",
            "bucket": "landing",
            "prefix": "tcc/pouring",
            "use_ssl": False,
            "cache_dir": str(tmp_path / "cache"),
        },
    }
    p = tmp_path / "pouring.yaml"
    p.write_text(yaml.dump(cfg))
    return p


class TestLoadStorageConfig:
    def test_loads_local_backend(self, local_config_path: Path) -> None:
        cfg = load_storage_config(str(local_config_path))
        assert cfg.storage_backend == "local"
        assert cfg.dataset_name == "pouring"

    def test_loads_s3_backend(self, s3_config_path: Path) -> None:
        cfg = load_storage_config(str(s3_config_path))
        assert cfg.storage_backend == "s3"

    def test_invalid_backend_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            yaml.dump(
                {
                    "storage_backend": "gcs",
                    "dataset_name": "x",
                    "local": {"raw_dir": "a", "processed_dir": "b"},
                    "s3": {
                        "endpoint": "x",
                        "bucket": "x",
                        "prefix": "x",
                        "use_ssl": False,
                        "cache_dir": "c",
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="storage_backend"):
            load_storage_config(str(p))


class TestStorageConfigPaths:
    def test_raw_dir_local(self, local_config_path: Path, tmp_path: Path) -> None:
        cfg = load_storage_config(str(local_config_path))
        assert cfg.raw_dir == str(tmp_path / "raw")

    def test_processed_dir_local(self, local_config_path: Path, tmp_path: Path) -> None:
        cfg = load_storage_config(str(local_config_path))
        assert cfg.processed_dir == str(tmp_path / "processed" / "pouring")

    def test_raw_dir_s3(self, s3_config_path: Path) -> None:
        cfg = load_storage_config(str(s3_config_path))
        assert cfg.raw_dir == "s3://landing/tcc/pouring/raw"

    def test_processed_dir_s3(self, s3_config_path: Path) -> None:
        cfg = load_storage_config(str(s3_config_path))
        assert cfg.processed_dir == "s3://landing/tcc/pouring/processed"

    def test_cache_dir_s3(self, s3_config_path: Path) -> None:
        cfg = load_storage_config(str(s3_config_path))
        assert cfg.cache_dir is not None

    def test_s3_storage_options(self, s3_config_path: Path) -> None:
        cfg = load_storage_config(str(s3_config_path))
        opts = cfg.s3_storage_options
        assert opts is not None
        assert opts["endpoint_url"] == "http://localhost:9000"

    def test_local_has_no_s3_options(self, local_config_path: Path) -> None:
        cfg = load_storage_config(str(local_config_path))
        assert cfg.s3_storage_options is None
