"""
PyTorch dataset that loads .npy hyperspectral images on-the-fly.

Each sample is a 128x128x125 uint16 image stored as an individual .npy file.
The dataset normalizes to [0, 1] float32 using precomputed global min/max stats
and returns tensors in (C, H, W) format for PyTorch convolutions.
"""

import re
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


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

        # Load normalization stats
        stats = np.load(stats_file)
        self.global_min = float(stats["global_min"])
        self.global_max = float(stats["global_max"])
        self.scale = self.global_max - self.global_min

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
        # Load and normalize to [0, 1]
        img = np.load(self.files[idx]).astype(np.float32)
        img = (img - self.global_min) / (self.scale + 1e-8)

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
