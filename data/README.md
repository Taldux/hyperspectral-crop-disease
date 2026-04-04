# Data

Hyperspectral crop disease images — 128×128 pixels, 125 spectral bands, 10 severity classes (0–9). Each image is a `.npy` file (uint16)

## Structure

- `Train/` — Training images, organized by class
- `evaluation/` — Held-out evaluation images, organized by class
- `processed/`
  - `train.txt`, `val.txt`, `eval.txt` — file path lists for each split
  - `stats.npz` — global min/max used for normalization
