"""
Generate synthetic hyperspectral images and compute per-class FID.

Usage:
    python -m src.evaluate --checkpoint best.pt
    python -m src.evaluate --checkpoint best.pt --out-dir results/eval
    python -m src.evaluate --checkpoint best.pt --data-root /kaggle/working
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from src.models.flow import ConditionalGlow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images and compute FID")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--eval-file", type=str, default="data/processed/eval.txt",
                   help="Path to eval.txt listing real evaluation images")
    p.add_argument("--stats-file", type=str, default="data/processed/stats.npz",
                   help="Path to stats.npz with global_min/global_max")
    p.add_argument("--data-root", type=str, default=None,
                   help="Re-root paths in eval.txt to this directory")
    p.add_argument("--out-dir", type=str, default="results/eval",
                   help="Directory to save generated images and CSV")
    p.add_argument("--num-per-class", type=int, default=50)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=10,
                   help="Batch size for generation")
    # Architecture (must match training)
    p.add_argument("--num-scales", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--hidden-channels", type=int, default=256)
    return p.parse_args()


def resolve_path(filepath: str, data_root: str | None) -> str:
    if data_root is None:
        return filepath
    m = re.search(r'[/\\](data[/\\].+)$', filepath)
    if m:
        rel = m.group(1).replace('\\', '/')
        return str(Path(data_root) / rel)
    return filepath


def load_real_images(eval_file: str, stats_file: str, data_root: str | None
                     ) -> dict[int, list[np.ndarray]]:
    """Load real evaluation images grouped by class. Returns raw uint16 arrays."""
    real: dict[int, list[np.ndarray]] = {}
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filepath, label = line.split("\t")
            filepath = resolve_path(filepath, data_root)
            cls = int(label)
            arr = np.load(filepath)  # (H, W, C) uint16
            real.setdefault(cls, []).append(arr)
    return real


def denormalize(images: torch.Tensor, global_min: float, global_max: float
                ) -> np.ndarray:
    """Convert model output [0,1] (B,C,H,W) float32 -> (B,H,W,C) uint16."""
    # (B, C, H, W) -> (B, H, W, C)
    imgs = images.cpu().permute(0, 2, 3, 1).numpy()
    # [0,1] -> original range
    scale = global_max - global_min
    imgs = imgs * scale + global_min
    imgs = np.clip(imgs, 0, 65535).astype(np.uint16)
    return imgs


def flatten_for_fid(images: list[np.ndarray]) -> np.ndarray:
    """Flatten list of (H,W,C) images to (N, H*W*C) float64 for FID."""
    return np.array([img.astype(np.float64).ravel() for img in images])


def reduce_dimensions(real: np.ndarray, fake: np.ndarray,
                      n_components: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """PCA on combined data, then project both sets. Max components = min(N-1, n_components)."""
    from sklearn.decomposition import PCA

    n_components = min(n_components, real.shape[0] + fake.shape[0] - 1,
                       real.shape[1])
    combined = np.vstack([real, fake])
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(combined)
    return projected[:real.shape[0]], projected[real.shape[0]:]


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute FID between two sets of feature vectors.

    Uses PCA to reduce dimensionality (images are 2M-dim but only 50 samples),
    then applies the standard FID formula:
        FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r @ Sigma_f))
    """
    # Reduce dimensions to make covariance tractable
    real_features, fake_features = reduce_dimensions(real_features, fake_features)

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[1]) * 1e-6
    sigma_f = np.cov(fake_features, rowvar=False) + np.eye(fake_features.shape[1]) * 1e-6

    diff = mu_r - mu_f
    mean_term = np.dot(diff, diff)

    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    trace_term = np.trace(sigma_r + sigma_f - 2.0 * covmean)

    return float(mean_term + trace_term)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load stats
    stats = np.load(args.stats_file)
    global_min = float(stats["global_min"])
    global_max = float(stats["global_max"])
    print(f"Stats: min={global_min}, max={global_max}")

    # Load model
    model = ConditionalGlow(
        num_scales=args.num_scales,
        num_steps=args.num_steps,
        hidden_channels=args.hidden_channels,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, best NLL={ckpt['best_nll']:.4f}")

    # Load real evaluation images
    real_by_class = load_real_images(args.eval_file, args.stats_file, args.data_root)
    print(f"Real images: {sum(len(v) for v in real_by_class.values())} total, "
          f"{len(real_by_class)} classes")

    # Generate and evaluate per class
    fid_scores: dict[int, float] = {}

    for cls in range(args.num_classes):
        print(f"\nClass {cls}:")
        gen_dir = out_dir / f"generated/{cls}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Generate in batches
        generated: list[np.ndarray] = []
        remaining = args.num_per_class
        idx = 0
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            labels = torch.full((bs,), cls, dtype=torch.long, device=device)
            with torch.no_grad():
                images = model.generate(labels, temperature=args.temperature)
            imgs_np = denormalize(images, global_min, global_max)
            for i in range(bs):
                np.save(gen_dir / f"{idx:03d}.npy", imgs_np[i])
                generated.append(imgs_np[i])
                idx += 1
            remaining -= bs

        print(f"  Generated {len(generated)} images")

        # Get real images for this class
        real_imgs = real_by_class.get(cls, [])
        if len(real_imgs) == 0:
            print(f"  WARNING: no real images for class {cls}, skipping FID")
            continue

        # Compute FID
        real_flat = flatten_for_fid(real_imgs)
        fake_flat = flatten_for_fid(generated)
        fid = compute_fid(real_flat, fake_flat)
        fid_scores[cls] = fid
        print(f"  FID: {fid:.2f}")

    # Summary
    print("\n" + "=" * 50)
    print("Per-class FID scores:")
    for cls in range(args.num_classes):
        score = fid_scores.get(cls, float("nan"))
        print(f"  Class {cls}: {score:.2f}")
    mean_fid = np.mean(list(fid_scores.values()))
    print(f"  Mean FID: {mean_fid:.2f}")

    # Save CSV
    csv_path = out_dir / "submission.csv"
    with open(csv_path, "w") as f:
        f.write("class,fid\n")
        for cls in range(args.num_classes):
            score = fid_scores.get(cls, float("nan"))
            f.write(f"{cls},{score:.4f}\n")
    print(f"\nSaved submission to {csv_path}")


if __name__ == "__main__":
    main()
