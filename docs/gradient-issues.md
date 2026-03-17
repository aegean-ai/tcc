# Numerical Stability: TF2 vs PyTorch Port

This document records the gradient explosion / NaN weight corruption issue
discovered during the first full training runs on gpu-node-3
(NVIDIA RTX PRO 6000, 98 GB VRAM) and the fixes applied to the PyTorch port.

## Symptom

All three embedding-dimension sweeps (D=32, D=64, D=128) trained to 5 000
iterations without raising an error, yet **every checkpoint from iteration
1 000 onward contained NaN weights**.  The 50-iteration debug checkpoint was
clean, so NaN appeared between step 50 and step 1 000 of the first full run.

The corruption was discovered when the analysis cells in
`tcc_training.ipynb` called `sklearn.decomposition.PCA` on the extracted
embeddings and received:

```
ValueError: Input X contains NaN.
```

Inspection of the saved checkpoints confirmed that **172 of 278 tensors**
had NaN values, spanning every BatchNorm layer in the ResNet-50 backbone
(`cnn.backbone`) and all layers in the ConvEmbedder (`emb.*`).

## Root cause

The training loop had **no gradient clipping**, and the similarity scores
computed inside the cycle-consistency alignment could grow unbounded before
being passed to `F.softmax`.

The chain of events:

1. Similarity is computed as L2 distance (or cosine), then divided by the
   number of embedding channels **and** by the temperature (default 0.1,
   i.e. multiplied by 10).
2. With 32-dimensional embeddings and temperature 0.1, a raw L2 distance of
   ~160 becomes a similarity of `160 / 32 / 0.1 = 50`, which is already
   near the edge of float32 softmax stability.
3. `F.softmax` on values > ~88 produces `inf`; subsequent weighted averaging
   yields `NaN` query features.
4. NaN propagates through the cycle back to the loss, producing NaN
   gradients.
5. Because the config uses `train_base: only_bn`, only BatchNorm parameters
   receive gradient updates.  NaN gradients corrupt the BN weights first,
   then the BN running statistics, then the embedder layers via the next
   forward pass.
6. Checkpoints saved after NaN onset are permanently corrupted.

## Why TF2 did not exhibit this

The original TF2 implementation (branch `tf2`) uses **identical defaults**:

| Parameter            | TF2         | PyTorch     |
|----------------------|-------------|-------------|
| `loss_type`          | `regression_mse_var` | `regression_mse_var` |
| `temperature`        | 0.1         | 0.1         |
| `similarity_type`    | l2          | l2          |
| `label_smoothing`    | 0.1         | 0.1         |
| `huber_delta`        | 0.1         | 0.1         |
| `variance_lambda`    | 0.001       | 0.001       |
| `normalize_indices`  | True        | True        |

The TF2 code also has **no explicit gradient clipping or similarity
clamping**.  TensorFlow 1.x/2.x has several implicit numerical safeguards
that differ from PyTorch:

- `tf.nn.softmax` uses a numerically stable implementation that subtracts
  the max internally, reducing overflow risk.
- TF's automatic differentiation can produce `0` rather than `NaN` for
  certain degenerate gradient paths.
- TF's default float handling and kernel implementations can mask borderline
  instabilities that surface in PyTorch.

PyTorch's `F.softmax` is also numerically stable (subtracts max), but the
preceding divisions can still produce `inf` values that defeat the max-
subtraction trick when the dynamic range is extreme.

## Fixes applied

### 1. Gradient clipping (`algorithm.py`)

```python
# src/tcc/algos/algorithm.py  —  train_one_iter()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)  # NEW
optimizer.step()
```

This prevents any single gradient update from having an outsized effect on
the model weights, even if a single batch produces an unusually large loss.

### 2. Similarity clamping — stochastic alignment (`stochastic_alignment.py`)

```python
# src/tcc/stochastic_alignment.py  —  _align_single_cycle()
similarity = similarity / float(num_channels)
similarity = similarity / temperature
similarity = similarity.clamp(-50.0, 50.0)  # NEW
beta = F.softmax(similarity, dim=0)
```

The clamp range `[-50, 50]` is chosen so that `exp(50) ≈ 5.2e21`, which is
well within float32 range (`~3.4e38`).  For comparison, `exp(88) ≈ 1.6e38`
is the approximate overflow threshold.

### 3. Similarity clamping — deterministic alignment (`deterministic_alignment.py`)

```python
# src/tcc/deterministic_alignment.py  —  get_scaled_similarity()
similarity = similarity / channels
similarity = similarity / temperature
similarity = similarity.clamp(-50.0, 50.0)  # NEW
return similarity
```

Same rationale as above; this function feeds into `F.softmax` in
`align_pair_of_sequences`.

## Alternative loss types available

The codebase supports four loss types, configurable via the notebook's
papermill parameter `loss_type` or the config key
`alignment.loss_type`:

| Loss type              | Description                          | Numerical notes |
|------------------------|--------------------------------------|-----------------|
| `regression_mse`       | Mean squared error on predicted time | Can amplify outlier gradients |
| `regression_mse_var`   | MSE with learned variance weighting  | Default; `log(var)` can → `-inf` if variance collapses |
| `regression_huber`     | Huber loss (delta=0.1)               | More robust to outliers; clips large gradients inherently |
| `classification`       | Cross-entropy on softmax logits      | Different formulation; not directly comparable |

If gradient issues persist after the fixes above, switching to
`regression_huber` provides an additional safety margin because the Huber
loss transitions from quadratic to linear for residuals larger than `delta`,
effectively capping the gradient magnitude from the loss itself.

## Verification

After applying the fixes, re-run training and verify:

```bash
# Inside the devcontainer on gpu-node-3:
make train

# Then check checkpoints for NaN:
python3 -c "
import torch
for d in [32, 64, 128]:
    ckpt = torch.load(f'runs_tutorial/pouring_tcc_d{d}/checkpoint_5000.pt',
                       map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', {})
    nan_count = sum(1 for v in sd.values() if torch.isnan(v).any())
    print(f'd={d}: {nan_count}/{len(sd)} tensors with NaN')
"
```

Expected output after fix: `0/278 tensors with NaN` for all three
dimensions.
