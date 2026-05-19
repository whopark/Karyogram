"""
ResNet18 classifier for chromosome identification (24 classes: chr1-22, X, Y).
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

try:
    from training.chromosome_model import (
        ChromosomeResNet18, build_augment_transform, build_eval_transform,
        CLASS_NAMES, IMG_H, IMG_W, NUM_CLASSES,
    )
except ImportError:
    from chromosome_model import (
        ChromosomeResNet18, build_augment_transform, build_eval_transform,
        CLASS_NAMES, IMG_H, IMG_W, NUM_CLASSES,
    )


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
        img = Image.open(img_path).convert("RGB")
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
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= NUM_CLASSES
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

    errors = []
    for true_cls in range(NUM_CLASSES):
        for pred_cls in range(NUM_CLASSES):
            if true_cls != pred_cls and matrix[true_cls][pred_cls] > 0:
                errors.append((matrix[true_cls][pred_cls], true_cls, pred_cls))
    errors.sort(reverse=True)

    print("\n--- Top confused class pairs ---")
    for count, true_cls, pred_cls in errors[:5]:
        print(f"  {CLASS_NAMES[true_cls]:6s} -> {CLASS_NAMES[pred_cls]:6s}: {count} errors")


# @AX:WARN: [AUTO] two-phase training with separate optimizers — Phase 1 (frozen backbone) and Phase 2 (full fine-tune) use different optimizer/scheduler instances; ensure best checkpoint from warmup is not overwritten if Phase 2 never improves
def train(args: argparse.Namespace) -> None:
    device = (
        torch.device(args.device)
        if args.device != "auto"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # @AX:NOTE: [AUTO] magic constant — default warmup epoch count when attribute absent; exposed as CLI arg --warmup_epochs
    warmup_epochs = getattr(args, "warmup_epochs", 5)

    base_dataset = ChromosomeDataset(args.data_dir, transform=build_eval_transform())
    counts = base_dataset.class_counts()
    train_idx, val_idx = stratified_split(base_dataset, val_ratio=0.2)

    train_dataset = ChromosomeDataset(args.data_dir, transform=build_augment_transform())
    val_dataset = ChromosomeDataset(args.data_dir, transform=build_eval_transform())

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = ChromosomeResNet18().to(device)
    class_weights = compute_class_weights(counts, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    patience_counter = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: warmup with frozen backbone
    if warmup_epochs > 0:
        model.freeze_backbone()
        warmup_opt = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4,
        )
        warmup_sched = optim.lr_scheduler.CosineAnnealingLR(warmup_opt, T_max=warmup_epochs)
        print(f"\n--- Phase 1: Warmup ({warmup_epochs} epochs, backbone frozen) ---")
        for epoch in range(1, warmup_epochs + 1):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                warmup_opt.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                warmup_opt.step()
                running_loss += loss.item() * images.size(0)
            warmup_sched.step()
            avg_loss = running_loss / max(len(train_loader.dataset), 1)
            val_top1, val_top3, _, _ = evaluate(model, val_loader, device)
            print(
                f"Warmup {epoch:3d}/{warmup_epochs} | loss {avg_loss:.4f} "
                f"| val top-1 {val_top1:.3f} | val top-3 {val_top3:.3f}"
            )
            if val_top1 > best_val_acc:
                best_val_acc = val_top1
                torch.save(model.state_dict(), output_path)

    # Phase 2: fine-tune entire model
    model.unfreeze_backbone()
    finetune_epochs = args.epochs - warmup_epochs
    if finetune_epochs <= 0:
        finetune_epochs = 1

    # @AX:NOTE: [AUTO] magic constant — backbone LR is 1/10 of head LR; differential learning rate strategy for pretrained feature preservation
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr / 10},
        {"params": model.head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    print(f"\n--- Phase 2: Fine-tune ({finetune_epochs} epochs, full model) ---")
    for epoch in range(1, finetune_epochs + 1):
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

        global_epoch = warmup_epochs + epoch
        print(
            f"Epoch {global_epoch:3d}/{args.epochs} | loss {avg_loss:.4f} "
            f"| val top-1 {val_top1:.3f} | val top-3 {val_top3:.3f}"
        )

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model (val_acc={best_val_acc:.3f})")
        else:
            patience_counter += 1
            # @AX:NOTE: [AUTO] magic constant — early stopping patience threshold (15 non-improving epochs)
            if patience_counter >= 15:
                print("Early stopping triggered.")
                break

    # Final evaluation on best checkpoint
    model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))
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
        description="Train ResNet18 classifier for chromosome classification (24 classes)."
    )
    p.add_argument("--data_dir", required=True,
                   help="Root dir with per-class subdirs (chr1/, ..., chrY/).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Number of warmup epochs with frozen backbone (default: 5).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", default="models/chromosome_classifier.pth",
                   help="Path where the best model checkpoint is saved.")
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', or 'mps'.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
