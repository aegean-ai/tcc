# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Temporal Cycle-Consistency Learning (TCC) — a self-supervised video representation learning codebase from CVPR 2019. Learns frame-level embeddings from unlabeled video by exploiting temporal cycle-consistency constraints.

**Paper**: [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846) (Dwibedi et al., CVPR 2019)

## Self-supervised Algorithms

- **TCC** (Temporal Cycle-Consistency): `algos/alignment.py`
- **Shuffle and Learn**: `algos/sal.py`
- **Time-Contrastive Networks (TCN)**: `algos/tcn.py`
- **Combined methods**: `algos/alignment_sal_tcn.py`
- **Supervised baseline**: `algos/classification.py`

Core TCC loss functions are in `tcc/alignment.py`, `tcc/deterministic_alignment.py`, `tcc/stochastic_alignment.py`.

## Evaluation Tasks

- Phase classification: `evaluation/classification.py`
- Few-shot phase classification: `evaluation/few_shot_classification.py`
- Phase progression: `evaluation/event_completion.py`
- Kendall's Tau: `evaluation/kendalls_tau.py`

## Key Files

- `train.py` — Training entry point
- `evaluate.py` — Evaluation entry point
- `extract_embeddings.py` — Extract per-frame embeddings
- `visualize_alignment.py` — Visualize nearest-neighbor alignments
- `config.py` — Configuration (uses `ml_collections`)
- `configs/demo.yml` — Demo config
- `models.py` — Model definitions (ResNetV2-50 backbone)
- `datasets.py` — TFRecord dataset loading
- `dataset_preparation/` — Scripts to convert videos/images to TFRecords

## Development Workflow

### Issue Tracking

This repo uses **beads** (`bd`) for issue tracking. Issue prefix: `tcc`.

```bash
bd list                       # List open issues
bd show tcc-<id>              # Show issue details
bd create                     # Create new issue
bd close tcc-<id>             # Close issue
```

## Code Style

- Follow existing code conventions
- The codebase uses TensorFlow 1.x patterns (`tf.compat.v1`)
- Configuration via `config.py` with `ml_collections`

## Git Workflow

- `main` branch is protected — all changes via PRs
- Auto-merge is enabled on the repo
- Create feature branches, open PRs, and merge via `gh pr merge --auto`
