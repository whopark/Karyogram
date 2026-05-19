"""Shared chromosome classifier models, constants, and transforms.

Single source of truth for ChromosomeResNet18 (new) and ChromosomeCNN (legacy).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

log = logging.getLogger(__name__)

# New ResNet18 constants
# @AX:NOTE: [AUTO] magic constant — cytogenetic class count (22 autosomes + X + Y)
NUM_CLASSES = 24
# @AX:NOTE: [AUTO] magic constant — ResNet18 input resolution; changing breaks pretrained weight compatibility
IMG_H, IMG_W = 128, 128

# Legacy CNN constants
LEGACY_IMG_H, LEGACY_IMG_W = 64, 32

CLASS_NAMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
IDX_TO_LABEL = CLASS_NAMES  # alias for backward compat

# ImageNet normalization
# @AX:NOTE: [AUTO] hardcoded ImageNet normalization values — must stay in sync with pretrained backbone weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# @AX:ANCHOR: [AUTO] public model API — instantiated by train_classifier.py, ml_pipeline.py, and predict.py; signature and output shape must not change without updating all three consumers
class ChromosomeResNet18(nn.Module):
    """ResNet18-based 24-class chromosome classifier with ImageNet pretrained backbone."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        # Replace the 1000-class fc with identity — we use our own head
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for warmup phase training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for full fine-tuning phase."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ChromosomeCNN(nn.Module):
    """Legacy 3-block CNN for backward-compatible weight loading."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
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


class FocalLoss(nn.Module):
    """Focal loss with label smoothing for imbalanced classification."""

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(counts: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced datasets."""
    counts = counts.astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32, device=device)


def mixup_data(x, y, alpha=0.2):
    """Apply Mixup augmentation to a batch."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def _grayscale_to_rgb(img):
    """Convert a grayscale PIL image to RGB by replicating the single channel."""
    return img.convert("RGB")


def build_augment_transform() -> transforms.Compose:
    """Training augmentation: resize, grayscale->RGB, augment, normalize."""
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.3),
    ])


def build_eval_transform() -> transforms.Compose:
    """Evaluation: resize, grayscale->RGB, normalize."""
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_legacy_eval_transform() -> transforms.Compose:
    """Legacy CNN evaluation: grayscale, resize to 64x32."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((LEGACY_IMG_H, LEGACY_IMG_W)),
        transforms.ToTensor(),
    ])


def detect_architecture(state_dict: dict) -> str:
    """Detect model architecture from state dict keys.

    Returns "resnet18" if ResNet keys found, "legacy_cnn" otherwise.
    """
    if "layer1.0.conv1.weight" in state_dict or "backbone.layer1.0.conv1.weight" in state_dict:
        return "resnet18"
    return "legacy_cnn"
