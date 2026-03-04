# Temporal Cycle-Consistency Learning

PyTorch implementation of [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846) (Dwibedi et al., CVPR 2019).

Self-supervised representation learning on videos by exploiting temporal cycle-consistency constraints. Useful for fine-grained sequential/temporal understanding tasks.

## Branch Structure

- **`main`** — PyTorch implementation (in progress)
- **`tf2`** — Original Keras/TF2 codebase (archived reference)

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
