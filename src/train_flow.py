"""
Training script for the Conditional Glow normalizing flow.

Usage:
    python -m src.train_flow                                    # train from scratch
    python -m src.train_flow --resume results/flow/best.pt      # resume
    python -m src.train_flow --epochs 200 --batch-size 2 --grad-checkpoint
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HyperspectralDataset
from src.models.flow import ConditionalGlow, glow_nll_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train conditional Glow flow")
    p.add_argument("--data-dir", type=str, default="data/processed",
                   help="Directory containing split files and stats.npz")
    p.add_argument("--data-root", type=str, default=None,
                   help="Re-root .npy paths in split files to this directory")
    p.add_argument("--out-dir", type=str, default="results/flow",
                   help="Directory for checkpoints and logs")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Linear LR warmup epochs")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save-every", type=int, default=20)
    # Glow architecture
    p.add_argument("--num-scales", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--hidden-channels", type=int, default=256)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Use gradient checkpointing to save VRAM")
    # Generation
    p.add_argument("--sample-every", type=int, default=20,
                   help="Generate samples every N epochs (0 = never)")
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_nll: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_nll": best_nll,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_nll"]


def warmup_lr_lambda(epoch: int, warmup: int) -> float:
    """Linear warmup followed by constant 1.0 (cosine handled by scheduler)."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 1.0


def train_one_epoch(
    model: ConditionalGlow,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns: mean_nll, mean_prior, mean_logdet (all nats/dim)."""
    model.train()
    total_nll = 0.0
    total_prior = 0.0
    total_logdet = 0.0
    n_batches = 0
    nan_batches = 0

    for inputs, labels in tqdm(loader, desc="  train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        z_list, log_det = model(inputs, labels)
        nll, prior_nll, logdet_per_dim = glow_nll_loss(z_list, log_det)

        # Skip batches with NaN/Inf or exploding loss
        nll_val = nll.item()
        if torch.isnan(nll) or torch.isinf(nll) or nll_val > 100.0:
            nan_batches += 1
            continue  # skip this batch, don't update weights

        nll.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_nll += nll_val
        total_prior += prior_nll.item()
        total_logdet += logdet_per_dim.item()
        n_batches += 1

    if nan_batches > 0:
        print(f"  ⚠ skipped {nan_batches} bad batches (NaN/Inf/exploding)")

    return total_nll / n_batches, total_prior / n_batches, total_logdet / n_batches


@torch.no_grad()
def evaluate(
    model: ConditionalGlow,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_nll = 0.0
    total_prior = 0.0
    total_logdet = 0.0
    n_batches = 0

    for inputs, labels in tqdm(loader, desc="  eval ", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        z_list, log_det = model(inputs, labels)
        nll, prior_nll, logdet_per_dim = glow_nll_loss(z_list, log_det)

        total_nll += nll.item()
        total_prior += prior_nll.item()
        total_logdet += logdet_per_dim.item()
        n_batches += 1

    return total_nll / n_batches, total_prior / n_batches, total_logdet / n_batches


def generate_samples(
    model: ConditionalGlow,
    out_dir: Path,
    epoch: int,
    temperature: float,
    device: torch.device,
    num_per_class: int = 2,
    num_classes: int = 10,
):
    """Generate a few sample images and save as .npy for inspection."""
    import numpy as np

    sample_dir = out_dir / "samples" / f"epoch_{epoch:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for cls in range(num_classes):
        labels = torch.full((num_per_class,), cls, dtype=torch.long, device=device)
        images = model.generate(labels, temperature=temperature)
        for i, img in enumerate(images):
            # (C, H, W) -> (H, W, C) and save
            arr = img.cpu().permute(1, 2, 0).numpy()
            np.save(sample_dir / f"class{cls}_sample{i}.npy", arr)

    print(f"  → saved {num_classes * num_per_class} samples to {sample_dir}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds = HyperspectralDataset(
        split_file=str(data_dir / "train.txt"),
        stats_file=str(data_dir / "stats.npz"),
        data_root=args.data_root,
    )
    val_ds = HyperspectralDataset(
        split_file=str(data_dir / "val.txt"),
        stats_file=str(data_dir / "stats.npz"),
        data_root=args.data_root,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    # Model
    model = ConditionalGlow(
        num_scales=args.num_scales,
        num_steps=args.num_steps,
        hidden_channels=args.hidden_channels,
        use_grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    warmup = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: warmup_lr_lambda(e, args.warmup_epochs)
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs]
    )

    # NOTE: AMP (float16) is intentionally disabled for normalizing flows.
    # The chained exp/log operations in ActNorm + InvertibleConv overflow in fp16.

    # Resume
    start_epoch = 0
    best_nll = float("inf")
    if args.resume:
        start_epoch, best_nll = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        print(f"Resumed from epoch {start_epoch}, best NLL={best_nll:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_nll, train_prior, train_logdet = train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_nll, val_prior, val_logdet = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train NLL={train_nll:.4f} (prior={train_prior:.4f} logdet={train_logdet:.4f}) | "
            f"val NLL={val_nll:.4f} | lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Save best
        if val_nll < best_nll:
            best_nll = val_nll
            save_checkpoint(
                out_dir / "best.pt", model, optimizer, scheduler,
                epoch + 1, best_nll,
            )
            print(f"  → saved best (NLL={best_nll:.4f})")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch + 1:03d}.pt",
                model, optimizer, scheduler, epoch + 1, best_nll,
            )

        # Generate samples
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            generate_samples(
                model, out_dir, epoch + 1, args.temperature, device
            )

    # Final
    save_checkpoint(
        out_dir / "final.pt", model, optimizer, scheduler,
        args.epochs, best_nll,
    )
    print(f"\nDone. Best val NLL: {best_nll:.4f} nats/dim")


if __name__ == "__main__":
    main()
