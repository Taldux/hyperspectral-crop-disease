"""Generate synthetic hyperspectral images and compute per-class FID.

Uses InceptionV3 pool3 features on SRF-projected RGB (Kaggle metric).

Usage:
    python -m src.evaluate --checkpoint results/flow/epoch_125.pt
    python -m src.evaluate --checkpoint best.pt --out-dir results/eval
    python -m src.evaluate --checkpoint best.pt --data-root /kaggle/working
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

from src.models.flow import ConditionalGlow

# Spectral Response Functions (Sentinel-2 green / red / NIR)
# Resampled from the competition's SRF table to 125 hyperspectral bands.
_SRF_GREEN = torch.tensor([
    0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0005, 0.0008, 0.0014, 0.0024, 0.0041,
    0.0069, 0.0113, 0.0180, 0.0279, 0.0414, 0.0583, 0.0783, 0.1008, 0.1252, 0.1507, 0.1766,
    0.2023, 0.2271, 0.2505, 0.2721, 0.2913, 0.3079, 0.3216, 0.3324, 0.3404, 0.3459, 0.3495,
    0.3516, 0.3528, 0.3533, 0.3535, 0.3536, 0.3538, 0.3539, 0.3541, 0.3542, 0.3542, 0.3541,
    0.3535, 0.3520, 0.3491, 0.3443, 0.3373, 0.3277, 0.3152, 0.2997, 0.2811, 0.2595, 0.2349,
    0.2076, 0.1778, 0.1462, 0.1140, 0.0823, 0.0524, 0.0259, 0.0037, 0.0003, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
]).float()

_SRF_RED = torch.tensor([
    0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0003, 0.0006, 0.0012, 0.0024, 0.0047,
    0.0087, 0.0154, 0.0255, 0.0395, 0.0575, 0.0786, 0.1020, 0.1265, 0.1505, 0.1732, 0.1940,
    0.2121, 0.2269, 0.2381, 0.2454, 0.2491, 0.2494, 0.2466, 0.2409, 0.2326, 0.2219, 0.2093,
    0.1952, 0.1799, 0.1639, 0.1476, 0.1314, 0.1157, 0.1008, 0.0870, 0.0744, 0.0629, 0.0525,
    0.0430, 0.0344, 0.0266, 0.0195, 0.0129, 0.0070, 0.0018, 0.0003, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
]).float()

_SRF_NIR = torch.tensor([
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0003, 0.0006, 0.0011, 0.0022,
    0.0041, 0.0073, 0.0125, 0.0204, 0.0317, 0.0470, 0.0666, 0.0905, 0.1185, 0.1500, 0.1841,
    0.2196, 0.2554, 0.2900, 0.3219, 0.3495, 0.3715, 0.3870, 0.3950, 0.3950, 0.3872, 0.3721,
    0.3503, 0.3228, 0.2912, 0.2573, 0.2228, 0.1888, 0.1563, 0.1261, 0.0990, 0.0755, 0.0557,
    0.0395, 0.0265, 0.0162, 0.0082, 0.0023, 0.0003, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
]).float()

_WAVELENGTHS = torch.linspace(450.0, 950.0, 125)


def _resample_srf(srf_1d: torch.Tensor, wl_axis: torch.Tensor) -> torch.Tensor:
    """Interpolate an SRF defined on its own grid to the hyperspectral band axis."""
    n = srf_1d.numel()
    xp = np.linspace(450.0, 950.0, n)
    values = np.interp(wl_axis.cpu().numpy(), xp, srf_1d.cpu().numpy())
    out = torch.from_numpy(values).float()
    return out / out.sum()


_SRF_RESAMPLED = {
    "green": _resample_srf(_SRF_GREEN, _WAVELENGTHS),
    "red":   _resample_srf(_SRF_RED,   _WAVELENGTHS),
    "nir":   _resample_srf(_SRF_NIR,   _WAVELENGTHS),
}


def hs_to_s2_rgb(hs_img: torch.Tensor) -> torch.Tensor:
    """Project (125, H, W) hyperspectral image to (3, H, W) Sentinel-2 RGB."""
    if hs_img.shape[0] != 125:
        raise ValueError(f"Expected 125 spectral bands, got {hs_img.shape[0]}")
    out = []
    for key in ("green", "red", "nir"):
        w = _SRF_RESAMPLED[key].view(125, 1, 1).to(hs_img.device)
        out.append((hs_img * w).sum(0))
    return torch.stack(out)  # (3, H, W)


# InceptionV3 feature extractor (pool3, 2048-d)

class InceptionPool3(nn.Module):
    """InceptionV3 trimmed to pool3 output (2048-d). Frozen, eval-only."""

    def __init__(self, device: torch.device):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, aux_logits=True, transform_input=False).to(device)
        net.eval()
        net.AuxLogits = nn.Identity()
        self.stem_and_blocks = nn.Sequential(*list(net.children())[:-2])
        self.avgpool = net.avgpool
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.stem_and_blocks(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


@torch.no_grad()
def get_activations(
    images: list[np.ndarray],
    model: InceptionPool3,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract InceptionV3 pool3 features from (H,W,125) images. Returns (N, 2048)."""
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        # (H, W, C) → (C, H, W) for each image, then stack to (N, 125, H, W)
        hs = torch.stack([
            torch.from_numpy(img.transpose(2, 0, 1))
            for img in batch
        ]).to(device)
        rgb = torch.stack([hs_to_s2_rgb(img) for img in hs])  # (N, 3, H, W)        rgb = F.interpolate(rgb, size=(299, 299), mode="bilinear", align_corners=False)
        feats.append(model(rgb).cpu().numpy())
    return np.concatenate(feats, axis=0)


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
    p.add_argument("--num-per-class", type=int, default=50,
                   help="Generated images per class (≥200 recommended for FID stability)")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature (if unset, sweeps [0.9..1.2])")
    p.add_argument("--batch-size", type=int, default=5,
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


def compute_fid(act_real: np.ndarray, act_gen: np.ndarray) -> float:
    """FID on InceptionV3 pool3 features with eps=1e-6 covariance regularization."""
    eps = 1e-6
    mu_r = act_real.mean(0)
    mu_g = act_gen.mean(0)
    sig_r = np.cov(act_real, rowvar=False) + np.eye(act_real.shape[1]) * eps
    sig_g = np.cov(act_gen,  rowvar=False) + np.eye(act_gen.shape[1])  * eps

    diff = mu_r - mu_g
    covmean = linalg.sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))


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
        temperatures = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(f"Temperature sweep: {temperatures}")

    # Pre-extract real InceptionV3 features per class
    print("\nLoading InceptionV3 (pool3)...")
    inc = InceptionPool3(device)
    inc.eval()

    print("Extracting real image features...")
    real_act_by_class: dict[int, np.ndarray] = {}
    for cls, imgs in real_by_class.items():
        real_act_by_class[cls] = get_activations(imgs, inc, device)
        print(f"  class {cls}: {len(imgs)} images → {real_act_by_class[cls].shape}")

    # Generate and evaluate per class (with temperature sweep)
    fid_scores: dict[int, float] = {}
    best_temps: dict[int, float] = {}

    for cls in range(args.num_classes):
        print(f"\nClass {cls}:")
        act_real = real_act_by_class.get(cls)
        if act_real is None:
            print(f"  WARNING: no real images for class {cls}, skipping")
            continue

        best_fid = float("inf")
        best_temp = temperatures[0]
        best_generated: list[np.ndarray] = []

        for temp in temperatures:
            # Generate images at this temperature
            generated: list[np.ndarray] = []
            remaining = args.num_per_class
            max_retries = remaining * 3  # avoid infinite loop
            attempts = 0
            while remaining > 0 and attempts < max_retries:
                bs = min(args.batch_size, remaining)
                labels = torch.full((bs,), cls, dtype=torch.long, device=device)
                try:
                    with torch.no_grad():
                        images = model.generate(labels, temperature=temp)
                    imgs_01 = images.cpu().permute(0, 2, 3, 1).numpy()
                    for i in range(bs):
                        generated.append(imgs_01[i])
                    remaining -= bs
                except RuntimeError as e:
                    attempts += 1
                    if attempts % 5 == 0:
                        print(f"    WARNING: {attempts} generation failures at temp={temp:.2f}: {e}")
                    continue
            if remaining > 0:
                print(f"    WARNING: only generated {len(generated)}/{args.num_per_class} at temp={temp:.2f}")

            act_gen = get_activations(generated, inc, device)
            fid_val = compute_fid(act_real, act_gen)

            if len(temperatures) > 1:
                print(f"  temp={temp:.2f} -> FID={fid_val:.2f}")

            if fid_val < best_fid:
                best_fid = fid_val
                best_temp = temp
                best_generated = generated

        fid_scores[cls] = best_fid
        best_temps[cls] = best_temp
        print(f"  Best: temp={best_temp:.2f}, FID={best_fid:.2f}")

        # Save best generated images (denormalized uint16)
        gen_dir = out_dir / f"generated/{cls}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_01 in enumerate(best_generated):
            img_raw = img_01 * (global_max - global_min) + global_min
            img_raw = np.clip(img_raw, 0, 65535).astype(np.uint16)
            np.save(gen_dir / f"{idx:03d}.npy", img_raw)

    # Summary
    print("\n" + "=" * 60)
    print("Per-class FID scores (InceptionV3 pool3, Kaggle metric):")
    print(f"{'Class':<8} {'FID':<12} {'Temp':<6}")
    print("-" * 26)
    for cls in range(args.num_classes):
        score = fid_scores.get(cls, float("nan"))
        temp  = best_temps.get(cls, float("nan"))
        print(f"{cls:<8} {score:<12.2f} {temp:<6.2f}")
    mean_fid = np.mean(list(fid_scores.values()))
    print("-" * 26)
    print(f"{'Mean':<8} {mean_fid:<12.2f}")

    # Save CSV
    csv_path = out_dir / "submission.csv"
    with open(csv_path, "w") as f:
        f.write("class,fid,temperature\n")
        for cls in range(args.num_classes):
            score = fid_scores.get(cls, float("nan"))
            temp  = best_temps.get(cls, float("nan"))
            f.write(f"{cls},{score:.4f},{temp:.2f}\n")
    print(f"\nSaved submission to {csv_path}")


if __name__ == "__main__":
    main()
