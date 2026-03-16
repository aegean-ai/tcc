# Notebook Split with Configurable Storage Backends — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic TCC pouring tutorial notebook into two stages (data-prep and training) with a YAML config that switches between local and S3 storage backends.

**Architecture:** A `configs/pouring.yaml` defines storage profiles (local vs S3). A thin `src/tcc/storage.py` module reads the config and provides fsspec-based path resolution. Two notebooks share the config: `tcc_data_prep.ipynb` (download + process + store) and `tcc_training.ipynb` (train + evaluate + visualize). The existing `scripts/execute_notebook.py` + Makefile chain them via papermill.

**Tech Stack:** fsspec + s3fs (storage abstraction), papermill (notebook execution), PyYAML (config), existing tcc package.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/tcc/storage.py` | Create | Read `pouring.yaml`, return resolved paths via fsspec |
| `tests/test_storage.py` | Create | Unit tests for storage config loading + path resolution |
| `configs/pouring.yaml` | Create | Storage backend config (local and S3 profiles) |
| `notebooks/self-supervised/tcc_data_prep.ipynb` | Create | Stage 1: download, process, store |
| `notebooks/self-supervised/tcc_training.ipynb` | Create | Stage 2: train, evaluate, visualize |
| `pyproject.toml` | Modify | Add `fsspec` + `s3fs` to `[notebooks]` extras |
| `Makefile` | Modify | Add `data-prep` and `train` targets |
| `notebooks/self-supervised/tcc_pouring_tutorial_aegean_main.ipynb` | Keep | Original monolithic notebook stays (not deleted) |

---

## Chunk 1: Storage Module + Config

### Task 1: Add fsspec + s3fs dependencies

**Files:**
- Modify: `pyproject.toml:43-50`

- [ ] **Step 1: Add dependencies to pyproject.toml**

In `[project.optional-dependencies] notebooks`, add `fsspec` and `s3fs`:

```toml
notebooks = [
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
    "wandb>=0.18.0",
    "huggingface-hub>=0.20.0",
    "boto3>=1.34.0",
    "fsspec>=2024.6.0",
    "s3fs>=2024.6.0",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add fsspec and s3fs to notebook extras"
```

---

### Task 2: Create the storage config YAML

**Files:**
- Create: `configs/pouring.yaml`

- [ ] **Step 1: Create configs/pouring.yaml**

```yaml
# Storage configuration for the TCC pouring pipeline.
# Set `storage_backend` to "local" or "s3" to switch profiles.

storage_backend: local

dataset_name: pouring

# ── Local profile ─────────────────────────────────────────────────
# Used when storage_backend=local (Colab, dev container, no S3).
local:
  raw_dir: data/pouring
  processed_dir: data/pouring_processed

# ── S3 profile ────────────────────────────────────────────────────
# Used when storage_backend=s3 (MinIO lakehouse).
# Credentials come from environment: MINIO_ACCESS_KEY, MINIO_SECRET_KEY.
s3:
  endpoint: ${MINIO_ENDPOINT:-http://192.168.1.26:9000}
  bucket: landing
  prefix: tcc/pouring
  use_ssl: false
  # Local cache directory for S3 mode (frames are cached locally for training)
  cache_dir: data/pouring_cache
```

- [ ] **Step 2: Commit**

```bash
git add configs/pouring.yaml
git commit -m "feat: add pouring storage config YAML"
```

---

### Task 3: Create src/tcc/storage.py

**Files:**
- Create: `src/tcc/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage.py`:

```python
"""Tests for tcc.storage — config loading and path resolution."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from tcc.storage import StorageConfig, load_storage_config


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
        p.write_text(yaml.dump({"storage_backend": "gcs", "dataset_name": "x",
                                "local": {"raw_dir": "a", "processed_dir": "b"},
                                "s3": {"endpoint": "x", "bucket": "x", "prefix": "x",
                                       "use_ssl": False, "cache_dir": "c"}}))
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tcc.storage'`

- [ ] **Step 3: Implement src/tcc/storage.py**

```python
"""Storage configuration for the TCC pipeline.

Reads a YAML config file and provides resolved paths for raw data,
processed data, and S3 storage options.  Both local and S3 backends
are supported; the active backend is selected by ``storage_backend``
in the YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    import re

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
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)

    backend = raw["storage_backend"]
    dataset_name = raw["dataset_name"]

    if backend not in ("local", "s3"):
        msg = f"storage_backend must be 'local' or 's3', got '{backend}'"
        raise ValueError(msg)

    if backend == "local":
        local = raw["local"]
        return StorageConfig(
            storage_backend="local",
            dataset_name=dataset_name,
            raw_dir=local["raw_dir"],
            processed_dir=os.path.join(local["processed_dir"], dataset_name),
            cache_dir=None,
            s3_storage_options=None,
        )

    # S3 backend
    s3 = raw["s3"]
    endpoint = _resolve_env(str(s3["endpoint"]))
    bucket = s3["bucket"]
    prefix = s3["prefix"]
    use_ssl = s3.get("use_ssl", endpoint.startswith("https"))

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Run linter and type checker**

Run: `make lint-check && make type-check`
Expected: No errors in `src/tcc/storage.py` or `tests/test_storage.py`

- [ ] **Step 6: Commit**

```bash
git add src/tcc/storage.py tests/test_storage.py
git commit -m "feat: add storage config module with local/S3 backends"
```

---

## Chunk 2: Data Prep Notebook

### Task 4: Create tcc_data_prep.ipynb

**Files:**
- Create: `notebooks/self-supervised/tcc_data_prep.ipynb`

This notebook is extracted from cells 0-20 of the monolithic notebook, adapted to use `StorageConfig` for all path resolution. It covers:

1. Environment setup (detect Colab vs devcontainer)
2. Load storage config from YAML
3. Download raw data from HuggingFace (always — HF is the canonical source)
4. Convert videos to image-folder layout
5. If S3 backend: upload processed data to S3

**Papermill parameters cell** (tagged `parameters`):

```python
# ── Papermill parameters ───────────────────────────────────────────
config_path = "configs/pouring.yaml"
fps = 15
image_width = 224
image_height = 224
```

- [ ] **Step 1: Create the notebook**

Create `notebooks/self-supervised/tcc_data_prep.ipynb` with the following cells:

**Cell 0** (markdown): Title and overview
```markdown
# TCC Data Preparation
## Stage 1: Download, process, and store the pouring dataset

This notebook downloads the multiview pouring dataset from HuggingFace,
extracts frames from videos, and stores the processed data to the
configured storage backend (local disk or S3).

**Config:** Edit `configs/pouring.yaml` to switch between local and S3 storage.

**Usage:**
```bash
# Local execution
make data-prep

# With papermill parameters
make data-prep NB_PARAMS="-p config_path configs/pouring.yaml"
```
```

**Cell 1** (code): Environment detection + imports
```python
import sys
import os
import pathlib
import subprocess

IN_COLAB = "google.colab" in sys.modules

print("Python:", sys.version)
print("Running in Colab:", IN_COLAB)
```

**Cell 2** (code): Colab setup (clone + install)
```python
if IN_COLAB:
    REPO_DIR = pathlib.Path("tcc")
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "https://github.com/aegean-ai/tcc", str(REPO_DIR)], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./tcc[notebooks]"], check=True)
else:
    print("Skipping clone/install — running inside dev container.")
```

**Cell 3** (code, tagged `parameters`): Papermill parameters
```python
# ── Papermill parameters ───────────────────────────────────────────
config_path = "configs/pouring.yaml"
fps = 15
image_width = 224
image_height = 224
```

**Cell 4** (code): Load storage config
```python
from tcc.storage import load_storage_config

storage = load_storage_config(config_path)

print(f"Storage backend: {storage.storage_backend}")
print(f"Dataset name:    {storage.dataset_name}")
print(f"Raw dir:         {storage.raw_dir}")
print(f"Processed dir:   {storage.processed_dir}")
if storage.cache_dir:
    print(f"Cache dir:       {storage.cache_dir}")
```

**Cell 5** (code): Derive local paths (always need local dirs for HF download + processing)
```python
if storage.storage_backend == "local":
    RAW_DIR = pathlib.Path(storage.raw_dir)
    PROCESSED_DIR = pathlib.Path(storage.processed_dir)
else:
    # S3 mode: use cache_dir for local staging
    RAW_DIR = pathlib.Path(storage.cache_dir) / "raw"
    PROCESSED_DIR = pathlib.Path(storage.cache_dir) / "processed" / storage.dataset_name

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.parent.mkdir(parents=True, exist_ok=True)

print(f"Local raw dir:       {RAW_DIR.resolve()}")
print(f"Local processed dir: {PROCESSED_DIR.resolve()}")
```

**Cell 6** (markdown): Download section header
```markdown
## 1. Download raw data from HuggingFace

The multiview pouring dataset is at
[`sermanet/multiview-pouring`](https://huggingface.co/datasets/sermanet/multiview-pouring).
```

**Cell 7** (code): Download from HuggingFace
```python
from huggingface_hub import snapshot_download

hf_cache_path = snapshot_download(
    repo_id="sermanet/multiview-pouring",
    repo_type="dataset",
    local_dir=str(RAW_DIR),
)

print("Dataset downloaded to:", hf_cache_path)

for split_dir in sorted(RAW_DIR.iterdir()):
    if split_dir.is_dir() and not split_dir.name.startswith("."):
        tfrecords = list(split_dir.glob("*.tfrecord*"))
        print(f"  {split_dir.name}/: {len(tfrecords)} TFRecord file(s)")
```

**Cell 8** (markdown): Conversion section header
```markdown
## 2. Convert videos to image-folder layout

Extract frames from `.mov` videos into the directory structure expected
by `tcc.datasets.VideoDataset`:

```
processed_dir/
└── pouring/
    ├── train/video_001/frame_0000.png
    └── val/video_050/frame_0000.png
```
```

**Cell 9** (code): Run video conversion
```python
VIDEO_INPUT_DIR = RAW_DIR / "videos"
print(f"Converting videos from {VIDEO_INPUT_DIR}...")
print(f"Output: {PROCESSED_DIR}")

subprocess.run(
    [sys.executable, "-m", "tcc.dataset_preparation.videos_to_dataset",
     "--input-dir", str(VIDEO_INPUT_DIR),
     "--output-dir", str(PROCESSED_DIR.parent),
     "--name", storage.dataset_name,
     "--fps", str(fps),
     "--width", str(image_width),
     "--height", str(image_height),
     "--file-pattern", "**/*.mov",
     "--rotate"],
    check=True)

total = sum(len(f) for _, _, f in os.walk(PROCESSED_DIR))
print(f"Conversion complete: {total} files")
```

**Cell 10** (markdown): Upload section header
```markdown
## 3. Upload to S3 (if S3 backend is configured)

When `storage_backend=s3`, upload the processed frames to the lakehouse
so that the training notebook can read directly from S3.
```

**Cell 11** (code): Upload to S3 if configured
```python
if storage.storage_backend == "s3":
    import fsspec

    fs = fsspec.filesystem("s3", **storage.s3_storage_options)

    local_processed = PROCESSED_DIR
    s3_dest = storage.processed_dir  # e.g. s3://landing/tcc/pouring/processed

    print(f"Uploading {local_processed} -> {s3_dest}")
    fs.put(str(local_processed), s3_dest, recursive=True)

    n_files = len(fs.ls(s3_dest, detail=False))
    print(f"Upload complete: {n_files} top-level entries at {s3_dest}")
else:
    print("Local backend — no S3 upload needed.")
    print(f"Processed data ready at: {PROCESSED_DIR.resolve()}")
```

**Cell 12** (code): Summary
```python
print("\n=== Data preparation complete ===")
print(f"Backend:       {storage.storage_backend}")
print(f"Processed at:  {storage.processed_dir}")
print("\nRun the training notebook next:")
print("  make train")
```

- [ ] **Step 2: Tag the parameters cell**

Ensure cell 3 has metadata `{"tags": ["parameters"]}` so papermill can inject overrides.

- [ ] **Step 3: Commit**

```bash
git add notebooks/self-supervised/tcc_data_prep.ipynb
git commit -m "feat: add data-prep notebook (stage 1 of pipeline split)"
```

---

## Chunk 3: Training Notebook

### Task 5: Create tcc_training.ipynb

**Files:**
- Create: `notebooks/self-supervised/tcc_training.ipynb`

This notebook is extracted from cells 21-51 of the monolithic notebook. It reads processed data from the configured storage backend and runs training, embedding extraction, and visualization.

**Papermill parameters cell** (tagged `parameters`):

```python
# ── Papermill parameters ───────────────────────────────────────────
config_path = "configs/pouring.yaml"
embed_dims = [32, 64, 128]
max_iters = 5000
wandb_enabled = True
wandb_project = "eng-ai-agents"
```

- [ ] **Step 1: Create the notebook**

Create `notebooks/self-supervised/tcc_training.ipynb` with the following cells:

**Cell 0** (markdown): Title
```markdown
# TCC Training & Evaluation
## Stage 2: Train embeddings, extract representations, visualize

This notebook loads the processed pouring dataset (from local disk or S3),
trains TCC models at multiple embedding dimensions, and produces
trajectory visualizations and segmentation analysis.

**Prerequisite:** Run `make data-prep` first (or `tcc_data_prep.ipynb`).

**Config:** Uses the same `configs/pouring.yaml` as the data-prep notebook.

**Usage:**
```bash
make train
make train NB_PARAMS="-p max_iters 10000"
```
```

**Cell 1** (code): Environment + imports
```python
import sys
import os
import pathlib
import subprocess

IN_COLAB = "google.colab" in sys.modules

print("Python:", sys.version)
print("Running in Colab:", IN_COLAB)
```

**Cell 2** (code): Colab setup
```python
if IN_COLAB:
    REPO_DIR = pathlib.Path("tcc")
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "https://github.com/aegean-ai/tcc", str(REPO_DIR)], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./tcc[notebooks]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "matplotlib", "scikit-learn", "umap-learn", "tqdm"], check=True)
else:
    print("Skipping clone/install — running inside dev container.")
```

**Cell 3** (code, tagged `parameters`): Papermill parameters
```python
# ── Papermill parameters ───────────────────────────────────────────
config_path = "configs/pouring.yaml"
embed_dims = [32, 64, 128]
max_iters = 5000
wandb_enabled = True
wandb_project = "eng-ai-agents"
```

**Cell 4** (code): Load storage config + resolve dataset path
```python
from tcc.storage import load_storage_config

storage = load_storage_config(config_path)

# Resolve the path that the training code will use.
# For local: direct path to processed_dir.
# For S3: download to cache_dir first, then point training at cache.
if storage.storage_backend == "s3":
    import fsspec

    fs = fsspec.filesystem("s3", **storage.s3_storage_options)
    cache_root = pathlib.Path(storage.cache_dir) / "processed" / storage.dataset_name
    cache_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading processed data from {storage.processed_dir} -> {cache_root}")
    fs.get(storage.processed_dir, str(cache_root), recursive=True)

    DATASET_ROOT = cache_root.parent  # parent of "pouring/" dir
    print(f"Dataset root: {DATASET_ROOT.resolve()}")
else:
    DATASET_ROOT = pathlib.Path(storage.processed_dir).parent
    print(f"Dataset root (local): {DATASET_ROOT.resolve()}")

# Verify data exists
dataset_dir = DATASET_ROOT / storage.dataset_name
assert dataset_dir.exists(), f"Dataset not found at {dataset_dir}. Run data-prep first."
splits = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
print(f"Found splits: {splits}")
```

**Cell 5** (code): Experiment settings
```python
EXPERIMENT_ROOT = pathlib.Path("runs_tutorial")
EXPERIMENT_ROOT.mkdir(exist_ok=True)

print("Experiments stored in:", EXPERIMENT_ROOT.resolve())
print("Embedding dims:", embed_dims)
print("Max iterations:", max_iters)
```

**Cell 6** (code): W&B setup
```python
WANDB_ENABLED = wandb_enabled
WANDB_PROJECT = wandb_project
WANDB_ENTITY = ""

if WANDB_ENABLED:
    try:
        import wandb
        print("wandb version:", wandb.__version__)
        print(f"W&B logging is ENABLED (project={WANDB_PROJECT})")
    except ImportError:
        print("wandb not installed — disabling W&B logging.")
        WANDB_ENABLED = False
else:
    print("W&B logging is DISABLED.")
```

**Cell 7** (code): Config helpers
```python
def set_if_exists(obj, path, value):
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if not hasattr(cur, p):
            return False
        cur = getattr(cur, p)
    if hasattr(cur, parts[-1]):
        setattr(cur, parts[-1], value)
        return True
    return False

def get_if_exists(obj, path, default=None):
    parts = path.split(".")
    cur = obj
    for p in parts:
        if not hasattr(cur, p):
            return default
        cur = getattr(cur, p)
    return cur
```

**Cell 8** (code): make_run_config
```python
def make_run_config(embed_dim=128, max_iters=5000, logdir=None):
    from tcc.config import get_default_config

    cfg = get_default_config()

    set_if_exists(cfg, "training_algo", "alignment")
    set_if_exists(cfg, "datasets", [storage.dataset_name])
    set_if_exists(cfg, "train.max_iters", max_iters)
    set_if_exists(cfg, "model.conv_embedder.embedding_size", embed_dim)

    ds_fmt = str((DATASET_ROOT / "%s").resolve())
    set_if_exists(cfg, "path_to_tfrecords", ds_fmt)

    if logdir is None:
        logdir = str((EXPERIMENT_ROOT / f"pouring_tcc_d{embed_dim}").resolve())
    set_if_exists(cfg, "logdir", logdir)

    set_if_exists(cfg, "logging.wandb_enabled", WANDB_ENABLED)
    set_if_exists(cfg, "logging.wandb_project", WANDB_PROJECT)
    set_if_exists(cfg, "logging.wandb_entity", WANDB_ENTITY)
    set_if_exists(cfg, "logging.wandb_run_name", f"pouring_d{embed_dim}")

    return cfg
```

**Cells 9-end**: Copy directly from the monolithic notebook cells 31-51 (training loop, checkpoint loading, embedding extraction, PCA/UMAP visualization, KMeans segmentation, dimension sweep, write-up questions). These cells are unchanged — they use `make_run_config()` which now reads from `DATASET_ROOT` resolved via the storage config.

The exact cells to copy are:
- Section 6: Training (run_training, debug run, full runs)
- Section 7: Checkpoint loading + embedding extraction
- Section 8: Representation diagnostics (PCA, UMAP projections)
- Section 9: Temporal segmentation (change-point, KMeans)
- Section 10: Embedding dimension sweep
- Section 11: Write-up questions
- Section 12: Final checklist

- [ ] **Step 2: Tag the parameters cell**

Ensure cell 3 has metadata `{"tags": ["parameters"]}`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/self-supervised/tcc_training.ipynb
git commit -m "feat: add training notebook (stage 2 of pipeline split)"
```

---

## Chunk 4: Makefile Targets + Final Integration

### Task 6: Add Makefile targets for the two-stage pipeline

**Files:**
- Modify: `Makefile:94-103`

- [ ] **Step 1: Add data-prep and train targets**

Append after the existing `execute-notebook` target:

```makefile
# Two-stage notebook pipeline
data-prep:
	@echo "Stage 1: Data preparation"
	docker compose run --rm torch.dev.gpu bash -c \
		"make venv-recreate && make install-notebooks && uv pip install papermill && \
		.venv/bin/python scripts/execute_notebook.py notebooks/self-supervised/tcc_data_prep.ipynb $(NB_PARAMS)"

train:
	@echo "Stage 2: Training & evaluation"
	docker compose run --rm torch.dev.gpu bash -c \
		"make venv-recreate && make install-notebooks && uv pip install papermill && \
		.venv/bin/python scripts/execute_notebook.py notebooks/self-supervised/tcc_training.ipynb $(NB_PARAMS)"

pipeline: data-prep train
	@echo "Pipeline complete (data-prep -> train)"
```

- [ ] **Step 2: Add new targets to .PHONY**

Update the first line of the Makefile to include the new targets:

```makefile
.PHONY: install install-dev install-notebooks format lint lint-check type-check test test-cov clean build venv venv-recreate start deps-update deps-sync quality style execute-notebook data-prep train pipeline
```

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add data-prep, train, and pipeline Makefile targets"
```

---

### Task 7: Run all tests and lint

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (including new `test_storage.py`)

- [ ] **Step 2: Run linter**

Run: `make lint-check`
Expected: No errors

- [ ] **Step 3: Run type checker**

Run: `make type-check`
Expected: No errors in `src/tcc/storage.py`

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address lint/type issues from notebook split"
```

---

### Task 8: Update beads issue

- [ ] **Step 1: Close tcc-xm2**

```bash
bd close tcc-xm2 --resolution "Implemented notebook split: tcc_data_prep.ipynb + tcc_training.ipynb with configs/pouring.yaml for local/S3 storage backends. Added src/tcc/storage.py, fsspec+s3fs deps, and Makefile pipeline targets."
```
