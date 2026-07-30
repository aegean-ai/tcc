# CLAUDE.md

## Shared GPU etiquette

The local card is a laptop GPU and it is shared. Before starting any long run:

- Check `nvidia-smi` for jobs already running. Never kill a process you did not start.
- Launch training or batch inference detached rather than blocking the session, and treat it
  as long-running: do not kill it on a timeout.
- Prefer a tailnet GPU node over the laptop card for long jobs. gpu-node-1 and gpu-node-3 are
  online; gpu-node-2 is offline, do not target it.
- Never `docker compose down`, prune, or restart containers without checking what is live
  first (`docker compose ps`, `nvidia-smi`).

## Project Overview

PyTorch port of Temporal Cycle-Consistency Learning (TCC), a self-supervised video representation learning method from CVPR 2019.

**Paper**: [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846) (Dwibedi et al., CVPR 2019)

## Branch Structure

- **`main`**: PyTorch implementation (in progress)
- **`tf2`**: Original Keras/TF2 codebase (archived reference)

When porting code, reference the `tf2` branch for the original implementation.

## Development Workflow

**CRITICAL**: This is a **container-first development environment**. All development workflows and make commands are designed to run **inside the devcontainer**, not on the host machine.

### Starting Development (CLI / Claude Code)
1. Build and start the container: `docker compose build && docker compose up -d`
2. Run commands inside the container: `docker compose exec torch.dev.gpu bash -c "make start"`
3. If port 8000 conflicts: `DEV_PORT=8001 docker compose up -d`

### Starting Development (VS Code)
1. Open the project in VS Code with the Dev Containers extension
2. Use "Dev Containers: Reopen in Container" command
3. Once inside the container, run `make start` to initialize the environment

### Key Constraints
- Python environment setup (`make start`, `make venv-recreate`) must run inside the container
- The Makefile uses `uv` package manager which is only available in the container
- Virtual environment at `.venv` has system-site-packages access to container-installed packages
- UV package manager respects container constraints (`/etc/pip/constraint.txt`)

### Host vs. Container Operations
- **Inside Container**: All make commands, Python development, package management
- **On Host**: Git operations, editing `.devcontainer/devcontainer.json`

## Development Commands

**Note**: All commands below must be executed from within the devcontainer.

### Initial Setup
```bash
make start                    # Create venv, sync deps, install package
source .venv/bin/activate     # Activate environment
```

### Code Quality
```bash
make format                   # Format with ruff
make lint                     # Lint with auto-fix
make lint-check               # Lint without modifications
make type-check               # MyPy strict type checking
make quality                  # All quality checks (lint-check + type-check)
make style                    # Format + lint combined
```

### Testing
```bash
make test                     # Run pytest
make test-cov                 # Pytest with coverage
```

### Dependencies
```bash
make deps-sync                # Sync dependencies from lock file
make deps-update              # Update dependencies
make venv-recreate            # Clean and recreate venv
```

## Training workflow

The Makefile owns the ML entry points. All of these run in Docker.

```bash
make data-prep    # Stage 1: notebooks/self-supervised/tcc_data_prep.ipynb
make train        # Stage 2: tcc_training.ipynb (training + evaluation)
make pipeline     # data-prep then train
make execute-notebook NOTEBOOK=<path> NB_PARAMS="-p key value"
```

- Override hyperparameters through `NB_PARAMS` (papermill `-p` flags). Never edit parameter
  defaults inside a notebook to change a run: the diff must show the override.
- Configs are in `configs/` (`default.yaml`, `demo.yaml`, `pouring.yaml`). Fetch the Pouring
  dataset with `scripts/download_pouring_data.sh`.
- Never commit datasets or `runs_tutorial/` outputs.
- `make train` recreates the venv on every invocation before training, so treat it as
  long-running: launch it in the background and monitor rather than killing it on a timeout.

## Architecture

### Package Structure
- `src/tcc/`: Main package (PyTorch implementation)
- `tests/`: Test suite
- `docs/`: currently only `gradient-issues.md`
- `docker/`: one Dockerfile, `Dockerfile.torch.dev.gpu`

### Package Management
Uses **uv** with constraints from base Docker image (`/etc/pip/constraint.txt`). Virtual environment at `.venv` with system-site-packages access.

## Code Style

- **Formatter/Linter**: Ruff (line length: 119)
- **Type Checker**: MyPy strict mode
- **Python**: >=3.11, <3.13 (do not test on 3.13)

## Issue Tracking

Issues live in Jira, project AURA at https://aegean-ai.atlassian.net. Use the Atlassian MCP tools (`mcp__plugin_atlassian_atlassian__*`) to search, read, file, comment on, and transition issues. Search for an existing issue before filing a new one.

beads and the `bd` CLI are retired. Never run `bd`, and never recreate a `.beads/` directory. The `.beads-archive.zip` at the repo root is a historical artifact, not a live tracker.

## Git Workflow

- `main` branch is protected: all changes via PRs
- Auto-merge is enabled on the repo
- Create feature branches, open PRs, and merge via `gh pr merge --auto`
