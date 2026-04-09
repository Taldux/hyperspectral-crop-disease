# Source

- `dataset.py` — `HyperspectralDataset` for loading `.npy` hyperspectral cubes from split files and normalizing them for PyTorch
- `train.py` — training loop for the class-conditional `ConditionalGlow` model
- `evaluate.py` — checkpoint evaluation and generation script using the competition-style FID pipeline

## `models/`

- `flow.py` — `ConditionalGlow` normalizing flow implementation with ActNorm, invertible `1x1` convolutions, affine coupling, and multi-scale sampling

## `preprocessing/`

- `split.py` — create `train.txt`, `val.txt`, `eval.txt`, and `stats.npz` from the repository data folders
- `stats.py` — compute global min/max and per-band normalization statistics over the training set
