# Temporal Cycle-Consistency Learning

PyTorch implementation of [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846) (Dwibedi et al., CVPR 2019).

Self-supervised representation learning on videos by exploiting temporal cycle-consistency constraints. Useful for fine-grained sequential/temporal understanding tasks.

## Branch Structure

- **`main`** — PyTorch implementation (in progress)
- **`tf2`** — Original Keras/TF2 codebase (archived reference)

## Dataset

The **multiview pouring dataset** is sourced from HuggingFace
([`sermanet/multiview-pouring`](https://huggingface.co/datasets/sermanet/multiview-pouring))
and processed into per-frame images for PyTorch training.

### Two-notebook pipeline

The data and training workflow is split into two notebooks under
`notebooks/self-supervised/`:

| Notebook | Purpose |
|----------|---------|
| `tcc_data_prep.ipynb` | Download raw videos from HuggingFace, extract frames, optionally upload to S3 |
| `tcc_training.ipynb` | Load processed data, train TCC models (D=32/64/128), evaluate and visualize |

### Storage backends

Configured in `configs/pouring.yaml` via `storage_backend`:

| Backend | When to use | Data flow |
|---------|-------------|-----------|
| **`s3`** (default) | GPU nodes with MinIO access | Data-prep uploads to S3; training downloads from S3 to local cache |
| **`local`** | Colab, laptops, no S3 available | Both notebooks read/write under `data/` |

The `data/` directory is a **local cache** — it is regenerated automatically
by the notebooks and does not need to be checked in or preserved.

### Quick start

```bash
# Inside the devcontainer:
make data-prep          # Stage 1: download + process
make train              # Stage 2: train + evaluate
make pipeline           # Both stages sequentially
```

## Self-supervised Algorithms

- Temporal Cycle-Consistency (TCC)
- Shuffle and Learn
- Time-Contrastive Networks (TCN)
- Combined methods
- Supervised baseline (per-frame classification)

## Evaluation Tasks

- Phase classification
- Few-shot phase classification
- Phase progression
- Kendall's Tau

## Citation

If you found this paper/code useful in your research, please consider citing:

```bibtex
@InProceedings{Dwibedi_2019_CVPR,
  author = {Dwibedi, Debidatta and Aytar, Yusuf and Tompson, Jonathan and Sermanet, Pierre and Zisserman, Andrew},
  title = {Temporal Cycle-Consistency Learning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019},
}
```
