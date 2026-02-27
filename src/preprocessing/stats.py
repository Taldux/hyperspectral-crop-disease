"""
Compute global normalization statistics across the training set.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

NUM_CLASSES = 10


def compute_global_stats(data_dir: Path) -> tuple[float, float]:
    """
    Scan all training files to find global min and max pixel values.

    These are needed to normalize images to [0, 1] during training.

    Args:
        data_dir: Root data directory containing Train/ folder.

    Returns:
        (global_min, global_max) as floats.
    """
    global_min = np.inf
    global_max = -np.inf

    for class_id in range(NUM_CLASSES):
        class_dir = data_dir / "Train" / str(class_id)
        files = sorted(class_dir.glob("*.npy"))
        for f in tqdm(files, desc=f"Class {class_id}", leave=False):
            arr = np.load(f)
            global_min = min(global_min, float(arr.min()))
            global_max = max(global_max, float(arr.max()))

    return global_min, global_max
