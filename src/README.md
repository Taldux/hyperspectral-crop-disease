# Source

- `dataset.py` — `HyperspectralDataset` (PyTorch Dataset, loads `.npy` files, normalizes to [0,1])
- `train.py` — Training script for `ConditionalGlow`
- `train_classifier.py` — Training script for `HybridCNNTransformer`
- `evaluate.py` — Generate synthetic images and compute spectral + spatial FID scores
- `evaluate_classifier.py` — Evaluate generated images using a classifier trained on real data (TRTR vs TRTS)

## `models/`

- `flow.py` — `ConditionalGlow` normalizing flow (3 scales × 4 steps, class-conditional)
- `classifier.py` — `HybridCNNTransformer` (CNN feature extractor + Transformer encoder + linear head)

## `preprocessing/`

- `split.py` — Create train/val/eval split files
- `stats.py` — Compute global min/max statistics for normalization
