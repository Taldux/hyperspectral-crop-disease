"""
Create stratified train/val/eval split index files.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
NUM_CLASSES = 10
VAL_RATIO = 0.15


def create_split(data_dir: str | Path, out_dir: str | Path, val_ratio: float = VAL_RATIO):
    """
    Create train/val/eval splits as text files listing file paths and labels.

    No data is copied. Only index files pointing to the original .npy files
    are written, along with normalization statistics (stats.npz).

    Args:
        data_dir: Root data directory containing Train/ and evaluation/ folders.
        out_dir: Directory to write split files and stats into.
        val_ratio: Fraction of training data to reserve for validation.
    """
    from .stats import compute_global_stats

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect all training file paths and labels ---
    all_files: list[str] = []
    all_labels: list[int] = []

    for class_id in range(NUM_CLASSES):
        class_dir = data_dir / "Train" / str(class_id)
        for f in sorted(class_dir.glob("*.npy")):
            all_files.append(str(f.resolve()))
            all_labels.append(class_id)

    print(f"Found {len(all_files)} training samples across {NUM_CLASSES} classes")

    # --- Stratified train/val split ---
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels,
        test_size=val_ratio,
        stratify=all_labels,
        random_state=RANDOM_SEED,
    )

    for name, files, labels in [
        ("train", train_files, train_labels),
        ("val", val_files, val_labels),
    ]:
        split_path = out_dir / f"{name}.txt"
        with open(split_path, "w") as f:
            for filepath, label in zip(files, labels):
                f.write(f"{filepath}\t{label}\n")
        print(f"  {name}: {len(files)} samples -> {split_path}")

    # --- Evaluation index ---
    eval_files: list[str] = []
    eval_labels: list[int] = []
    for class_id in range(NUM_CLASSES):
        class_dir = data_dir / "evaluation" / str(class_id)
        for f in sorted(class_dir.glob("*.npy")):
            eval_files.append(str(f.resolve()))
            eval_labels.append(class_id)

    eval_path = out_dir / "eval.txt"
    with open(eval_path, "w") as f:
        for filepath, label in zip(eval_files, eval_labels):
            f.write(f"{filepath}\t{label}\n")
    print(f"  eval: {len(eval_files)} samples -> {eval_path}")

    # --- Normalization stats ---
    print("Computing per-band normalization stats (this may take a few minutes)...")
    from .stats import compute_stats
    stats = compute_stats(data_dir)
    np.savez(
        out_dir / "stats.npz",
        global_min=stats["global_min"],
        global_max=stats["global_max"],
        per_band_mean=stats["per_band_mean"],
        per_band_std=stats["per_band_std"],
    )
    print(f"  Stats saved: global_min={stats['global_min']:.1f}, global_max={stats['global_max']:.1f}")
    print(f"  Per-band mean range: [{stats['per_band_mean'].min():.1f}, {stats['per_band_mean'].max():.1f}]")
    print(f"  Per-band std  range: [{stats['per_band_std'].min():.1f},  {stats['per_band_std'].max():.1f}]")

    # --- Summary ---
    print(f"\nDone. Files written to {out_dir}/")
    print(f"  train.txt  ({len(train_files)} samples)")
    print(f"  val.txt    ({len(val_files)} samples)")
    print(f"  eval.txt   ({len(eval_files)} samples)")
    print(f"  stats.npz  (per-band Z-score + global min/max)")
