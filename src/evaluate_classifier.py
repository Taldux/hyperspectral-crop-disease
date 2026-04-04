"""
Evaluate generated images using a classifier trained on real data.

Trains a HybridCNNTransformer on real training images, then tests on:
  - Real evaluation images (TRTR baseline)
  - Generated images (TRTS — Train Real, Test Synthetic)

Usage:
    python -m src.evaluate_classifier
    python -m src.evaluate_classifier --gen-dir results/eval/generated --epochs 20
    python -m src.evaluate_classifier --classifier-checkpoint results/classifier/best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.dataset import HyperspectralDataset
from src.models.classifier import HybridCNNTransformer


class GeneratedDataset(Dataset):
    """Load generated .npy images from a directory tree of {class_id}/*.npy."""

    def __init__(self, gen_root: str, stats_file: str):
        stats = np.load(stats_file)
        self.global_min = float(stats["global_min"])
        self.scale = float(stats["global_max"]) - self.global_min
        self.files: list[Path] = []
        self.labels: list[int] = []

        root = Path(gen_root)
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            try:
                cls = int(cls_dir.name)
            except ValueError:
                continue
            for f in sorted(cls_dir.glob("*.npy")):
                self.files.append(f)
                self.labels.append(cls)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.files[idx]).astype(np.float32)
        img = (img - self.global_min) / (self.scale + 1e-8)
        img = np.clip(img, 0, 1)
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate generated images with a classifier")
    p.add_argument("--data-dir", type=str, default="data/processed",
                   help="Directory containing train.txt, eval.txt, stats.npz")
    p.add_argument("--gen-dir", type=str, default="results/eval/generated",
                   help="Root of generated images ({class}/*.npy)")
    p.add_argument("--classifier-checkpoint", type=str, default=None,
                   help="Skip training and load this classifier checkpoint")
    p.add_argument("--out-dir", type=str, default="results/eval",
                   help="Directory to save classification results CSV")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return running_loss / total, correct / total * 100


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model, returning overall accuracy, all predictions, and all labels."""
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in tqdm(loader, desc="  eval ", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean() * 100
    return acc, all_preds, all_labels


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> dict[int, float]:
    """Compute per-class accuracy (recall)."""
    result = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() == 0:
            result[cls] = float("nan")
        else:
            result[cls] = (preds[mask] == cls).mean() * 100
    return result


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Compute confusion matrix (rows=true, cols=predicted)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1
    return cm


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Datasets ----
    train_ds = HyperspectralDataset(
        split_file=str(data_dir / "train.txt"),
        stats_file=str(data_dir / "stats.npz"),
    )
    eval_ds = HyperspectralDataset(
        split_file=str(data_dir / "eval.txt"),
        stats_file=str(data_dir / "stats.npz"),
    )
    gen_ds = GeneratedDataset(args.gen_dir, str(data_dir / "stats.npz"))
    print(f"Train: {len(train_ds)}, Real Eval: {len(eval_ds)}, Generated: {len(gen_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    gen_loader = DataLoader(
        gen_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    # ---- Model ----
    model = HybridCNNTransformer().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.classifier_checkpoint:
        ckpt = torch.load(args.classifier_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded classifier from {args.classifier_checkpoint}")
    else:
        # ---- Train ----
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = GradScaler("cuda") if device.type == "cuda" else None

        print(f"\nTraining classifier for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>2}/{args.epochs}: "
                      f"loss={train_loss:.4f}, acc={train_acc:.1f}%")
        print("Training complete.")

    # ---- Evaluate ----
    print("\nEvaluating on real eval images (TRTR)...")
    real_acc, real_preds, real_labels = evaluate_split(model, eval_loader, device)
    print(f"  Real Eval Accuracy (TRTR): {real_acc:.1f}%")

    print("Evaluating on generated images (TRTS)...")
    gen_acc, gen_preds, gen_labels = evaluate_split(model, gen_loader, device)
    print(f"  Generated Accuracy (TRTS): {gen_acc:.1f}%")

    # ---- Per-class results ----
    real_per_cls = per_class_accuracy(real_preds, real_labels)
    gen_per_cls = per_class_accuracy(gen_preds, gen_labels)

    print(f"\n{'Class':<8} {'Real Acc%':<12} {'Gen Acc%':<12}")
    print("-" * 32)
    for cls in range(10):
        print(f"{cls:<8} {real_per_cls[cls]:<12.1f} {gen_per_cls[cls]:<12.1f}")
    print("-" * 32)
    print(f"{'Mean':<8} {real_acc:<12.1f} {gen_acc:<12.1f}")

    # ---- Confusion matrices ----
    real_cm = confusion_matrix(real_preds, real_labels)
    gen_cm = confusion_matrix(gen_preds, gen_labels)

    # ---- Save CSV ----
    csv_path = out_dir / "classifier_eval.csv"
    with open(csv_path, "w") as f:
        f.write("class,real_accuracy,generated_accuracy\n")
        for cls in range(10):
            f.write(f"{cls},{real_per_cls[cls]:.2f},{gen_per_cls[cls]:.2f}\n")
    print(f"\nSaved to {csv_path}")

    # Save confusion matrices
    np.savez(
        out_dir / "confusion_matrices.npz",
        real_cm=real_cm, gen_cm=gen_cm,
        real_acc=real_acc, gen_acc=gen_acc,
    )
    print(f"Saved confusion matrices to {out_dir / 'confusion_matrices.npz'}")


if __name__ == "__main__":
    main()
