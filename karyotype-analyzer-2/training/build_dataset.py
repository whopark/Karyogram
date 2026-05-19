"""build_dataset.py — Assemble a YOLO training dataset from auto-annotated metaphase images.

Usage:
    python build_dataset.py --images_dir /path/to/images --labels_dir /path/to/labels \
                            --output_dir /path/to/dataset [--val_ratio 0.2] \
                            [--augment] [--aug_factor 3]
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# Supported image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Bounding-box transforms (YOLO normalised format: cx cy w h in [0, 1])
# ---------------------------------------------------------------------------

def _rotate_box_90(cx: float, cy: float, w: float, h: float) -> tuple:
    """Rotate bounding box 90 degrees clockwise. Image space: (0,0) top-left."""
    # 90 CW: (cx,cy) -> (1-cy, cx), swap w/h
    return 1.0 - cy, cx, h, w


def _rotate_boxes(boxes: list, degrees: int) -> list:
    """Apply rotation to a list of YOLO boxes. degrees must be 0/90/180/270."""
    steps = (degrees // 90) % 4
    result = [row[:] for row in boxes]
    for _ in range(steps):
        result = [[cls, *_rotate_box_90(cx, cy, w, h)] for cls, cx, cy, w, h in result]
    return result


def _flip_boxes_h(boxes: list) -> list:
    """Flip boxes horizontally: cx = 1 - cx."""
    return [[cls, 1.0 - cx, cy, w, h] for cls, cx, cy, w, h in boxes]


def _flip_boxes_v(boxes: list) -> list:
    """Flip boxes vertically: cy = 1 - cy."""
    return [[cls, cx, 1.0 - cy, w, h] for cls, cx, cy, w, h in boxes]


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def _rotate_image(img: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate image by 0/90/180/270 degrees clockwise."""
    steps = (degrees // 90) % 4
    # cv2.ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
    codes = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    code = codes[steps]
    return cv2.rotate(img, code) if code is not None else img


def _adjust_brightness_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Apply contrast (alpha) and brightness (beta) jitter. new_px = clip(alpha*px + beta)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Label I/O
# ---------------------------------------------------------------------------

def _read_labels(label_path: Path) -> list:
    """Read YOLO label file. Returns list of [class_id, cx, cy, w, h]."""
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append([int(parts[0])] + [float(v) for v in parts[1:]])
    return rows


def _write_labels(label_path: Path, boxes: list) -> None:
    """Write YOLO label file from list of [class_id, cx, cy, w, h]."""
    lines = [
        f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cls, cx, cy, w, h in boxes
    ]
    label_path.write_text("\n".join(lines) + "\n" if lines else "")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment_sample(img: np.ndarray, boxes: list, rng: random.Random) -> tuple:
    """Generate one random augmented (image, boxes) pair."""
    # Random rotation: 0, 90, 180, 270
    angle = rng.choice([0, 90, 180, 270])
    img = _rotate_image(img, angle)
    boxes = _rotate_boxes(boxes, angle)

    # Horizontal flip with 50% probability
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        boxes = _flip_boxes_h(boxes)

    # Vertical flip with 50% probability
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)
        boxes = _flip_boxes_v(boxes)

    # Brightness / contrast jitter (image only)
    alpha = rng.uniform(0.7, 1.3)
    beta = rng.uniform(-30, 30)
    img = _adjust_brightness_contrast(img, alpha, beta)

    return img, boxes


# ---------------------------------------------------------------------------
# Dataset split helpers
# ---------------------------------------------------------------------------

def _collect_pairs(images_dir: Path, labels_dir: Path) -> list:
    """Return list of (image_path, label_path) pairs that have both files."""
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs


def _split_pairs(pairs: list, val_ratio: float, seed: int = 42) -> tuple:
    """Shuffle and split pairs into (train, val) lists."""
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


# ---------------------------------------------------------------------------
# Copy / augment into output structure
# ---------------------------------------------------------------------------

def _copy_split(pairs: list, split: str, out_images: Path, out_labels: Path) -> int:
    """Copy image/label pairs into the output split directories. Returns annotation count."""
    total_annotations = 0
    for img_path, lbl_path in pairs:
        dst_img = out_images / img_path.name
        dst_lbl = out_labels / (img_path.stem + ".txt")
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)
        total_annotations += len(_read_labels(lbl_path))
    return total_annotations


def _augment_split(pairs: list,
                   out_images: Path,
                   out_labels: Path,
                   aug_factor: int) -> int:
    """Generate augmented copies for each pair in the train split."""
    rng = random.Random(0)
    added_annotations = 0
    for img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = _read_labels(lbl_path)
        for i in range(aug_factor):
            aug_img, aug_boxes = _augment_sample(img, [b[:] for b in boxes], rng)
            stem = f"{img_path.stem}_aug{i:02d}"
            cv2.imwrite(str(out_images / f"{stem}.png"), aug_img)
            _write_labels(out_labels / f"{stem}.txt", aug_boxes)
            added_annotations += len(aug_boxes)
    return added_annotations


# ---------------------------------------------------------------------------
# dataset.yaml
# ---------------------------------------------------------------------------

def _write_dataset_yaml(output_dir: Path) -> None:
    """Write dataset.yaml with absolute path for YOLOv8."""
    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "chromosome"},
    }
    yaml_path = output_dir / "dataset.yaml"
    with yaml_path.open("w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a YOLO dataset from auto-annotated metaphase images."
    )
    parser.add_argument("--images_dir", required=True,
                        help="Directory containing metaphase images.")
    parser.add_argument("--labels_dir", required=True,
                        help="Directory containing YOLO annotation .txt files.")
    parser.add_argument("--output_dir", required=True,
                        help="Root directory for the assembled YOLO dataset.")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of samples used for validation (default 0.2).")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation to the training split.")
    parser.add_argument("--aug_factor", type=int, default=3,
                        help="Number of augmented copies per training image (default 3).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not images_dir.is_dir():
        print(f"ERROR: --images_dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"ERROR: --labels_dir not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)
    if not (0.0 < args.val_ratio < 1.0):
        print("ERROR: --val_ratio must be in (0, 1).", file=sys.stderr)
        sys.exit(1)

    # Collect matched image/label pairs
    pairs = _collect_pairs(images_dir, labels_dir)
    if not pairs:
        print("ERROR: No matched image/label pairs found.", file=sys.stderr)
        sys.exit(1)

    # Split into train / val
    train_pairs, val_pairs = _split_pairs(pairs, args.val_ratio)

    # Create output directory structure
    out_train_img = output_dir / "images" / "train"
    out_val_img   = output_dir / "images" / "val"
    out_train_lbl = output_dir / "labels" / "train"
    out_val_lbl   = output_dir / "labels" / "val"
    for d in (out_train_img, out_val_img, out_train_lbl, out_val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    # Copy base samples
    train_ann = _copy_split(train_pairs, "train", out_train_img, out_train_lbl)
    val_ann   = _copy_split(val_pairs,   "val",   out_val_img,   out_val_lbl)

    train_count = len(train_pairs)
    val_count   = len(val_pairs)

    # Optional augmentation (train split only)
    aug_ann = 0
    if args.augment:
        aug_ann = _augment_split(train_pairs, out_train_img, out_train_lbl, args.aug_factor)
        train_count += len(train_pairs) * args.aug_factor

    # Write dataset.yaml
    _write_dataset_yaml(output_dir)

    # Summary
    total_ann = train_ann + val_ann + aug_ann
    print("Dataset assembled successfully.")
    print(f"  Output : {output_dir.resolve()}")
    print(f"  Train  : {train_count} images  ({train_ann + aug_ann} annotations)")
    print(f"  Val    : {val_count} images  ({val_ann} annotations)")
    print(f"  Total annotations: {total_ann}")
    if args.augment:
        print(f"  Augmentation: {args.aug_factor}x per training image "
              f"({len(train_pairs) * args.aug_factor} extra images, {aug_ann} extra annotations)")
    print(f"  YAML   : {(output_dir / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
