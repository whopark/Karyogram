"""
Lightweight CNN classifier for chromosome identification (24 classes: chr1-22, X, Y).
Trains on cropped chromosome images produced by karyogram_parser.py.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

CLASS_NAMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
NUM_CLASSES = 24  # chr1-22, chrX, chrY
IMG_H, IMG_W = 64, 32  # height > width: chromosomes are taller than wide


class ChromosomeCNN(nn.Module):
    """Lightweight CNN for 24-class chromosome classification."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x32 -> 32x16
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32x16 -> 16x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 16x8 -> 4x2 (adaptive)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _build_augment_transform() -> transforms.Compose:
    """Augmentation pipeline applied during training."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_H, IMG_W)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # scales to [0, 1]
    ])


def _build_eval_transform() -> transforms.Compose:
    """Deterministic transform for validation and inference."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])


class ChromosomeDataset(Dataset):
    """Loads cropped chromosome images from data_dir/{chrN}/*.png layout."""

    def __init__(self, data_dir: str, transform=None) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label, name in enumerate(CLASS_NAMES):
            class_dir = Path(data_dir) / name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(f"No chromosome images found under '{data_dir}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")  # PIL needs RGB for ColorJitter
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> np.ndarray:
        """Return per-class sample counts for computing class weights."""
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _, label in self.samples:
            counts[label] += 1
        return counts


def stratified_split(
    dataset: ChromosomeDataset, val_ratio: float = 0.2
) -> tuple[list[int], list[int]]:
    """Return (train_indices, val_indices) with per-class stratification."""
    per_class: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(dataset.samples):
        per_class[label].append(idx)

    train_idx, val_idx = [], []
    rng = np.random.default_rng(42)
    for indices in per_class.values():
        if not indices:
            continue
        indices = rng.permutation(indices).tolist()
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return train_idx, val_idx


def compute_class_weights(counts: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced datasets."""
    counts = counts.astype(np.float32)
    counts = np.where(counts == 0, 1, counts)  # avoid division by zero
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= NUM_CLASSES  # normalise so mean weight == 1
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return overall accuracy and per-class correct/total arrays."""
    model.eval()
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    top3_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top1 = logits.argmax(dim=1)
        top3 = logits.topk(3, dim=1).indices

        for pred, tgt in zip(top1.cpu().numpy(), labels.cpu().numpy()):
            per_class_correct[tgt] += int(pred == tgt)
            per_class_total[tgt] += 1

        for preds_row, tgt in zip(top3.cpu().numpy(), labels.cpu().numpy()):
            top3_correct += int(tgt in preds_row)
        total += labels.size(0)

    top1_acc = per_class_correct.sum() / max(total, 1)
    top3_acc = top3_correct / max(total, 1)
    return top1_acc, top3_acc, per_class_correct, per_class_total


def confusion_summary(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    """Print the top-5 most confused class pairs from the confusion matrix."""
    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            for pred, tgt in zip(preds, labels.numpy()):
                matrix[tgt][pred] += 1

    # Collect off-diagonal errors
    errors = []
    for true_cls in range(NUM_CLASSES):
        for pred_cls in range(NUM_CLASSES):
            if true_cls != pred_cls and matrix[true_cls][pred_cls] > 0:
                errors.append((matrix[true_cls][pred_cls], true_cls, pred_cls))
    errors.sort(reverse=True)

    print("\n--- Top confused class pairs ---")
    for count, true_cls, pred_cls in errors[:5]:
        print(f"  {CLASS_NAMES[true_cls]:6s} -> {CLASS_NAMES[pred_cls]:6s}: {count} errors")


def train(args: argparse.Namespace) -> None:
    device = (
        torch.device(args.device)
        if args.device != "auto"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Build full dataset (no augmentation) for splitting
    base_dataset = ChromosomeDataset(args.data_dir, transform=_build_eval_transform())
    counts = base_dataset.class_counts()
    train_idx, val_idx = stratified_split(base_dataset, val_ratio=0.2)

    # Rebuild with augmentation applied only to training split
    train_dataset = ChromosomeDataset(args.data_dir, transform=_build_augment_transform())
    val_dataset = ChromosomeDataset(args.data_dir, transform=_build_eval_transform())

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = ChromosomeCNN().to(device)
    class_weights = compute_class_weights(counts, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience_counter = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = running_loss / max(len(train_loader.dataset), 1)
        val_top1, val_top3, _, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | loss {avg_loss:.4f} "
            f"| val top-1 {val_top1:.3f} | val top-3 {val_top3:.3f}"
        )

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model (val_acc={best_val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print("Early stopping triggered.")
                break

    # --- Final evaluation on best checkpoint ---
    model.load_state_dict(torch.load(output_path, map_location=device))
    top1_acc, top3_acc, per_cls_correct, per_cls_total = evaluate(model, val_loader, device)

    print("\n=== Final Validation Results ===")
    print(f"Overall top-1 accuracy : {top1_acc:.4f}")
    print(f"Overall top-3 accuracy : {top3_acc:.4f}")
    print("\nPer-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        total = per_cls_total[i]
        acc = per_cls_correct[i] / total if total > 0 else float("nan")
        print(f"  {name:6s}: {acc:.3f}  ({per_cls_correct[i]}/{total})")

    confusion_summary(model, val_loader, device)
    print(f"\nModel saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train lightweight CNN for chromosome classification (24 classes)."
    )
    p.add_argument("--data_dir", required=True,
                   help="Root dir with per-class subdirs (chr1/, ..., chrY/).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", default="models/chromosome_classifier.pth",
                   help="Path where the best model checkpoint is saved.")
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', or 'mps'.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
