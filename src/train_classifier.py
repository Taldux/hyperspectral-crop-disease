"""
Training script for the Hybrid CNN+Transformer classifier.

Usage:
    python -m src.train_classifier                          # train from scratch
    python -m src.train_classifier --resume results/classifier/best.pt  # resume
    python -m src.train_classifier --epochs 50 --batch-size 8
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HyperspectralDataset
from src.models.classifier import HybridCNNTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hyperspectral classifier")
    p.add_argument("--data-dir", type=str, default="data/processed",
                    help="Directory containing split files and stats.npz")
    p.add_argument("--out-dir", type=str, default="results/classifier",
                    help="Directory for checkpoints and logs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (0 = main process)")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=10,
                    help="Save checkpoint every N epochs")
    return p.parse_args()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_acc: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
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
    return ckpt["epoch"], ckpt["best_acc"]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="  train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="  eval ", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total * 100


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
    )
    val_ds = HyperspectralDataset(
        split_file=str(data_dir / "val.txt"),
        stats_file=str(data_dir / "stats.npz"),
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
    model = HybridCNNTransformer().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision on CUDA
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # Training
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.1f}% | "
            f"val loss={val_loss:.4f} acc={val_acc:.1f}% | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                out_dir / "best.pt", model, optimizer, scheduler, epoch + 1, best_acc
            )
            print(f"  → saved best ({best_acc:.1f}%)")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch + 1:03d}.pt",
                model, optimizer, scheduler, epoch + 1, best_acc,
            )

    # Final save
    save_checkpoint(
        out_dir / "final.pt", model, optimizer, scheduler, args.epochs, best_acc
    )
    print(f"\nDone. Best val accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    main()
