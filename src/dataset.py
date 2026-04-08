"""
PyTorch dataset that loads .npy hyperspectral images on-the-fly.

Each sample is a 128x128x125 uint16 image stored as an individual .npy file.
The dataset normalizes with precomputed statistics and returns tensors in
(C, H, W) format for PyTorch convolutions.
"""

import re
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def load_normalization_stats(stats_file: str) -> dict[str, np.ndarray | float | str]:
    """Load normalization statistics, preferring per-band standardization."""
    with np.load(stats_file) as stats:
        if "per_band_mean" in stats and "per_band_std" in stats:
            return {
                "mode": "per-band-standard",
                "per_band_mean": stats["per_band_mean"].astype(np.float32),
                "per_band_std": stats["per_band_std"].astype(np.float32),
            }
        global_min = float(stats["global_min"])
        global_max = float(stats["global_max"])
        return {
            "mode": "global-minmax",
            "global_min": global_min,
            "scale": global_max - global_min,
        }


def normalize_image(img: np.ndarray, stats: dict[str, np.ndarray | float | str]) -> np.ndarray:
    """Normalize an HWC hyperspectral image using the configured stats."""
    if stats["mode"] == "per-band-standard":
        per_band_mean = np.asarray(stats["per_band_mean"], dtype=np.float32)
        per_band_std = np.asarray(stats["per_band_std"], dtype=np.float32)
        return (img - per_band_mean) / (per_band_std + 1e-8)

    global_min = float(stats["global_min"])
    scale = float(stats["scale"])
    return (img - global_min) / (scale + 1e-8)


class HyperspectralDataset(Dataset):
    """Loads individual .npy files listed in a split file, normalizes on-the-fly.

    Args:
        split_file: Path to a .txt file with lines of "filepath\\tlabel".
        stats_file: Path to stats.npz containing global_min and global_max.
        data_root: Optional root directory. Paths in the split file that contain
            'data/Train' or 'data/evaluation' will be re-rooted here.
        transform: Optional callable applied to the tensor after normalization.
    """

    def __init__(self, split_file: str, stats_file: str, data_root: str | None = None, transform=None):
        self.files: list[str] = []
        self.labels: list[int] = []
        self.transform = transform

        self.norm_stats = load_normalization_stats(stats_file)

        # Load file list
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                filepath, label = line.split("\t")
                filepath = self._resolve_path(filepath, data_root)
                self.files.append(filepath)
                self.labels.append(int(label))

    @staticmethod
    def _resolve_path(filepath: str, data_root: str | None) -> str:
        """Re-root absolute paths so split files work across machines/OS."""
        if data_root is None:
            return filepath
        # Extract the relative portion starting from 'data/'
        m = re.search(r'[/\\](data[/\\].+)$', filepath)
        if m:
            rel = m.group(1).replace('\\', '/')
            return str(Path(data_root) / rel)
        return filepath

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load and normalize.
        img = np.load(self.files[idx]).astype(np.float32)
        img = normalize_image(img, self.norm_stats)

        # (H, W, C) → (C, H, W) for PyTorch
        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Returns (C, H, W) shape of a single sample."""
        sample = np.load(self.files[0])
        h, w, c = sample.shape
        return (c, h, w)
