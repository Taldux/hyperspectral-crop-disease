"""
Compute normalization statistics across the training set.

Produces both legacy global min/max and per-band mean/std (Z-score).
Per-band Z-score is the correct normalisation for hyperspectral data:
each band has a different physical scale, and squashing 125 bands into
one global [0,1] range destroys the relative per-band signal that
distinguishes diseased from healthy pixels.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

NUM_CLASSES = 10
NUM_BANDS = 125


def compute_stats(data_dir: Path) -> dict:
    """
    Scan all training files and compute:
      - global_min / global_max  (kept for backward compatibility)
      - per_band_mean / per_band_std  (used for Z-score normalisation)

    Uses Welford's online algorithm so the full dataset never has to
    sit in memory at once.

    Args:
        data_dir: Root data directory containing a Train/ folder.

    Returns:
        Dict with keys: global_min, global_max, per_band_mean, per_band_std.
    """
    global_min = np.inf
    global_max = -np.inf

    # Welford accumulators — one slot per spectral band
    n = 0
    band_mean = np.zeros(NUM_BANDS, dtype=np.float64)
    band_M2   = np.zeros(NUM_BANDS, dtype=np.float64)

    all_files: list[Path] = []
    for class_id in range(NUM_CLASSES):
        class_dir = data_dir / "Train" / str(class_id)
        all_files.extend(sorted(class_dir.glob("*.npy")))

    for f in tqdm(all_files, desc="Computing per-band stats"):
        arr = np.load(f).astype(np.float64)   # (H, W, 125)
        global_min = min(global_min, float(arr.min()))
        global_max = max(global_max, float(arr.max()))
        # Batch Welford update (Chan parallel algorithm) — one image at a time
        pixels = arr.reshape(-1, NUM_BANDS)   # (H*W, 125)
        n_b = pixels.shape[0]
        mean_b = pixels.mean(axis=0)
        M2_b   = ((pixels - mean_b) ** 2).sum(axis=0)
        if n == 0:
            n, band_mean, band_M2 = n_b, mean_b, M2_b
        else:
            n_combined = n + n_b
            delta = mean_b - band_mean
            band_mean = band_mean + delta * n_b / n_combined
            band_M2   = band_M2 + M2_b + delta ** 2 * n * n_b / n_combined
            n = n_combined

    band_std = np.sqrt(band_M2 / max(n - 1, 1)).astype(np.float32)
    band_std = np.where(band_std < 1e-6, 1.0, band_std)   # guard /0

    return {
        "global_min":    float(global_min),
        "global_max":    float(global_max),
        "per_band_mean": band_mean.astype(np.float32),
        "per_band_std":  band_std,
    }


# Backward-compatible wrapper so split.py (and anything else) still works.
def compute_global_stats(data_dir: Path) -> tuple[float, float]:
    s = compute_stats(data_dir)
    return s["global_min"], s["global_max"]
