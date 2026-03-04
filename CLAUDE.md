# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch port of Temporal Cycle-Consistency Learning (TCC) — a self-supervised video representation learning method from CVPR 2019.

**Paper**: [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846) (Dwibedi et al., CVPR 2019)

## Branch Structure

- **`main`** — PyTorch implementation (in progress)
- **`tf2`** — Original Keras/TF2 codebase (archived reference)

When porting code, reference the `tf2` branch for the original implementation.

## Self-supervised Algorithms (to be ported)

- **TCC** (Temporal Cycle-Consistency)
- **Shuffle and Learn**
- **Time-Contrastive Networks (TCN)**
- **Combined methods**
- **Supervised baseline** (per-frame classification)

## Evaluation Tasks (to be ported)

- Phase classification
- Few-shot phase classification
- Phase progression
- Kendall's Tau

## Issue Tracking

This repo uses **beads** (`bd`) for issue tracking. Issue prefix: `tcc`.

```bash
bd list                       # List open issues
bd show tcc-<id>              # Show issue details
bd create                     # Create new issue
bd close tcc-<id>             # Close issue
```

## Git Workflow

- `main` branch is protected — all changes via PRs
- Auto-merge is enabled on the repo
- Create feature branches, open PRs, and merge via `gh pr merge --auto`
