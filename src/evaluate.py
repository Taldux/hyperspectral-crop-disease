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
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature (if unset, sweeps [0.9..1.2])")
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
    """Load real evaluation images grouped by class, normalized to [0,1] float32."""
    stats = np.load(stats_file)
    global_min = float(stats["global_min"])
    scale = float(stats["global_max"]) - global_min

    real: dict[int, list[np.ndarray]] = {}
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filepath, label = line.split("\t")
            filepath = resolve_path(filepath, data_root)
            cls = int(label)
            arr = np.load(filepath).astype(np.float32)  # (H, W, C)
            arr = (arr - global_min) / (scale + 1e-8)   # normalize to [0,1]
            real.setdefault(cls, []).append(arr)
    return real


def extract_spectral_features(images: list[np.ndarray]) -> np.ndarray:
    """Rich spectral features: mean, std, percentiles, inter-band correlation.

    Returns a 624-dim feature vector per image:
    - mean per band (125)
    - std per band (125)
    - 25th percentile per band (125)
    - 75th percentile per band (125)
    - adjacent-band correlations (124)
    """
    features = []
    for img in images:
        if img.shape[0] < img.shape[-1]:
            pixels = img.reshape(-1, img.shape[-1])  # (H*W, C)
        else:
            pixels = img.reshape(img.shape[0], -1).T  # (H*W, C)
        ch_mean = pixels.mean(axis=0)
        ch_std = pixels.std(axis=0)
        ch_p25 = np.percentile(pixels, 25, axis=0)
        ch_p75 = np.percentile(pixels, 75, axis=0)
        corr = np.array([
            np.corrcoef(pixels[:, i], pixels[:, i + 1])[0, 1]
            for i in range(pixels.shape[1] - 1)
        ])
        corr = np.nan_to_num(corr, nan=0.0)
        feat = np.concatenate([ch_mean, ch_std, ch_p25, ch_p75, corr])
        features.append(feat)
    return np.array(features)


def extract_spatial_features(images: list[np.ndarray], spatial_size: int = 32) -> np.ndarray:
    """Downsample to (spatial_size, spatial_size, C) and flatten."""
    from scipy.ndimage import zoom

    features = []
    for img in images:
        h, w = img.shape[:2]
        factors = (spatial_size / h, spatial_size / w, 1.0)
        img_small = zoom(img.astype(np.float64), factors, order=1)
        features.append(img_small.ravel())
    return np.array(features)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute FID between two sets of feature vectors.

    Applies PCA to reduce dimensionality (capped by sample count) before
    computing the standard FID formula:
        FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r @ Sigma_f))
    """
    from sklearn.decomposition import PCA

    # PCA to make covariance tractable: max components = min(N_total - 1, D, 64)
    n_total = real_features.shape[0] + fake_features.shape[0]
    n_components = min(64, n_total - 1, real_features.shape[1])
    n_real = real_features.shape[0]
    combined = np.vstack([real_features, fake_features])
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(combined)
    real_features = projected[:n_real]
    fake_features = projected[n_real:]

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[1]) * 1e-6
    sigma_f = np.cov(fake_features, rowvar=False) + np.eye(fake_features.shape[1]) * 1e-6

    diff = mu_r - mu_f
    mean_term = np.dot(diff, diff)

    product = sigma_r @ sigma_f
    covmean = linalg.sqrtm(product)

    if np.iscomplexobj(covmean):
        if np.max(np.abs(covmean.imag)) > 1e-3:
            # SVD-based fallback for poorly conditioned matrices
            U, s, Vt = np.linalg.svd(product)
            covmean = (U * np.sqrt(np.maximum(s, 0.0))) @ Vt
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

    # Determine temperatures to try
    if args.temperature is not None:
        temperatures = [args.temperature]
    else:
        temperatures = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
        print(f"Temperature sweep: {temperatures}")

    # Pre-extract real features (both spectral and spatial)
    real_spectral_by_class: dict[int, np.ndarray] = {}
    real_spatial_by_class: dict[int, np.ndarray] = {}
    for cls, imgs in real_by_class.items():
        real_spectral_by_class[cls] = extract_spectral_features(imgs)
        real_spatial_by_class[cls] = extract_spatial_features(imgs)

    # Generate and evaluate per class (with temperature sweep)
    fid_spectral: dict[int, float] = {}
    fid_spatial: dict[int, float] = {}
    best_temps: dict[int, float] = {}

    for cls in range(args.num_classes):
        print(f"\nClass {cls}:")
        rf_spec = real_spectral_by_class.get(cls)
        rf_spat = real_spatial_by_class.get(cls)
        if rf_spec is None:
            print(f"  WARNING: no real images for class {cls}, skipping")
            continue

        best_fid = float("inf")
        best_temp = temperatures[0]
        best_generated: list[np.ndarray] = []

        for temp in temperatures:
            # Generate images at this temperature
            generated: list[np.ndarray] = []
            remaining = args.num_per_class
            while remaining > 0:
                bs = min(args.batch_size, remaining)
                labels = torch.full((bs,), cls, dtype=torch.long, device=device)
                with torch.no_grad():
                    images = model.generate(labels, temperature=temp)
                imgs_01 = images.cpu().permute(0, 2, 3, 1).numpy()
                for i in range(bs):
                    generated.append(imgs_01[i])
                remaining -= bs

            ff_spec = extract_spectral_features(generated)
            fid_s = compute_fid(rf_spec, ff_spec)

            if len(temperatures) > 1:
                print(f"  temp={temp:.2f} -> Spectral FID={fid_s:.2f}")

            if fid_s < best_fid:
                best_fid = fid_s
                best_temp = temp
                best_generated = generated

        # Compute spatial FID for the best-temperature samples
        ff_spat = extract_spatial_features(best_generated)
        spatial_score = compute_fid(rf_spat, ff_spat)

        fid_spectral[cls] = best_fid
        fid_spatial[cls] = spatial_score
        best_temps[cls] = best_temp
        print(f"  Best: temp={best_temp:.2f}, Spectral FID={best_fid:.2f}, Spatial FID={spatial_score:.2f}")

        # Save best generated images (denormalized uint16)
        gen_dir = out_dir / f"generated/{cls}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_01 in enumerate(best_generated):
            img_raw = img_01 * (global_max - global_min) + global_min
            img_raw = np.clip(img_raw, 0, 65535).astype(np.uint16)
            np.save(gen_dir / f"{idx:03d}.npy", img_raw)

    # Summary
    print("\n" + "=" * 60)
    print("Per-class FID scores:")
    print(f"{'Class':<8} {'Spectral FID':<15} {'Spatial FID':<15} {'Temp':<6}")
    print("-" * 44)
    for cls in range(args.num_classes):
        spec = fid_spectral.get(cls, float("nan"))
        spat = fid_spatial.get(cls, float("nan"))
        temp = best_temps.get(cls, float("nan"))
        print(f"{cls:<8} {spec:<15.2f} {spat:<15.2f} {temp:<6.2f}")
    mean_spec = np.mean(list(fid_spectral.values()))
    mean_spat = np.mean(list(fid_spatial.values()))
    print("-" * 44)
    print(f"{'Mean':<8} {mean_spec:<15.2f} {mean_spat:<15.2f}")
    print(f"\nSpectral FID: rich spectral features (mean/std/percentiles/correlation, 624-dim)")
    print(f"Spatial FID: 32x32 downsample + PCA")

    # Save CSV
    csv_path = out_dir / "submission.csv"
    with open(csv_path, "w") as f:
        f.write("class,spectral_fid,spatial_fid,temperature\n")
        for cls in range(args.num_classes):
            spec = fid_spectral.get(cls, float("nan"))
            spat = fid_spatial.get(cls, float("nan"))
            temp = best_temps.get(cls, float("nan"))
            f.write(f"{cls},{spec:.4f},{spat:.4f},{temp:.2f}\n")
    print(f"\nSaved submission to {csv_path}")


if __name__ == "__main__":
    main()
