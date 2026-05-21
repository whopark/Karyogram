"""predict.py — Full inference pipeline: YOLO detection -> crop -> classify -> karyotype.

Usage:
    python predict.py --image_dir /path/to/images --output results.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from training.chromosome_model import (
        ChromosomeResNet18, ChromosomeCNN, detect_architecture,
        IMG_H, IMG_W, LEGACY_IMG_H, LEGACY_IMG_W,
        IMAGENET_MEAN, IMAGENET_STD, IDX_TO_LABEL, NUM_CLASSES,
    )
except ImportError:
    from chromosome_model import (
        ChromosomeResNet18, ChromosomeCNN, detect_architecture,
        IMG_H, IMG_W, LEGACY_IMG_H, LEGACY_IMG_W,
        IMAGENET_MEAN, IMAGENET_STD, IDX_TO_LABEL, NUM_CLASSES,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Denver cytogenetic group mapping (members are int for autosomes, str for sex chr)
DENVER_GROUPS = {
    "A": [1, 2, 3], "B": [4, 5],
    "C": [6, 7, 8, 9, 10, 11, 12, "chrX"],
    "D": [13, 14, 15], "E": [16, 17, 18], "F": [19, 20],
    "G": [21, 22, "chrY"],
}

# IDX_TO_LABEL and NUM_CLASSES imported from chromosome_model (chrX/chrY labels)
AUTOSOME_LABELS = IDX_TO_LABEL[:22]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_classifier(weights_path: str, device: torch.device):
    """Load classifier with automatic architecture detection.

    Returns (model, crop_size) tuple.
    """
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    arch = detect_architecture(state)
    if arch == "resnet18":
        model = ChromosomeResNet18(pretrained=False)
        crop_size = (IMG_H, IMG_W)
        log.info("Detected ResNet18 architecture")
    else:
        log.warning("Loading legacy CNN weights — consider retraining with ResNet18")
        model = ChromosomeCNN()
        crop_size = (LEGACY_IMG_H, LEGACY_IMG_W)

    model.load_state_dict(state)
    model.to(device).eval()
    log.info("Classifier loaded from %s (arch=%s)", weights_path, arch)
    return model, crop_size


def _crop_tensor(image: np.ndarray, bbox: list, crop_size=None) -> torch.Tensor:
    """Crop bbox [x,y,w,h] from RGB array; return appropriately shaped tensor."""
    if crop_size is None:
        crop_size = (LEGACY_IMG_H, LEGACY_IMG_W)

    x, y, w, h = [int(v) for v in bbox]
    ch, cw = crop_size

    if ch == LEGACY_IMG_H and cw == LEGACY_IMG_W:
        # Legacy: grayscale (1,1,H,W)
        pil = Image.fromarray(image[y: y + h, x: x + w]).convert("L").resize((cw, ch))
        arr = np.array(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    else:
        # ResNet18: RGB (1,3,H,W) with ImageNet normalization
        pil = Image.fromarray(image[y: y + h, x: x + w]).convert("RGB").resize((cw, ch))
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        for c in range(3):
            arr[c] = (arr[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Pair-based refinement
# ---------------------------------------------------------------------------

# @AX:WARN: [AUTO] iterative reassignment algorithm — parallel to _pair_refine in ml_pipeline.py; duplication increases maintenance risk if the algorithm is corrected in one place only
def _refine_predictions(labels: list, max_iter: int = 10) -> list:
    """Reassign over-represented autosomes to under-represented ones (target=2 each)."""
    labels = list(labels)
    for _ in range(max_iter):
        counts = {lbl: labels.count(lbl) for lbl in IDX_TO_LABEL}
        changed = False
        for lbl in AUTOSOME_LABELS:
            while counts.get(lbl, 0) > 2:
                under = min(
                    (l for l in AUTOSOME_LABELS if counts.get(l, 0) < 2),
                    key=lambda l: counts.get(l, 0),
                    default=None,
                )
                if under is None:
                    break
                idx = len(labels) - 1 - labels[::-1].index(lbl)
                labels[idx] = under
                counts[lbl] -= 1
                counts[under] = counts.get(under, 0) + 1
                changed = True
        if not changed:
            break
    return labels


# ---------------------------------------------------------------------------
# Karyotype generation
# ---------------------------------------------------------------------------

def _build_karyotype(labels: list) -> dict:
    """Derive sex chromosomes, abnormalities, ISCN notation, and Denver groups."""
    counts = {lbl: labels.count(lbl) for lbl in IDX_TO_LABEL}
    x, y = counts.get("chrX", 0), counts.get("chrY", 0)

    if x >= 2 and y == 0:
        sex = "XX"
    elif x >= 1 and y >= 1:
        sex = "XY"
    else:
        sex = "X" * x + "Y" * y

    abnormalities = []
    for i in range(1, 23):
        n = counts.get(f"chr{i}", 0)
        if n == 3:
            abnormalities.append(f"+{i}")
        elif n == 1:
            abnormalities.append(f"-{i}")
    if x == 1 and y == 0:
        abnormalities.append("monosomy X")
    if x == 3:
        abnormalities.append("triple X")

    iscn_parts = [str(len(labels)), sex] + [a for a in abnormalities if a[0] in ("+", "-")]
    notation = ",".join(iscn_parts)

    denver = {}
    for group, members in DENVER_GROUPS.items():
        denver[group] = sum(counts.get(f"chr{m}" if isinstance(m, int) else m, 0) for m in members)

    return {"sex_chromosomes": sex, "notation": notation,
            "abnormalities": abnormalities, "denver_groups": denver}


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

def analyze_image(image_path: Path, detector, classifier,
                  device: torch.device, conf_threshold: float,
                  crop_size=None) -> dict:
    """Run full pipeline on one image and return result dict."""
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)

    results = detector(img_pil, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    crop_tensors, valid_bboxes = [], []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        crop_tensors.append(_crop_tensor(img_np, [x1, y1, w, h], crop_size))
        valid_bboxes.append([x1, y1, w, h])

    if not crop_tensors:
        log.warning("%s: no chromosomes detected", image_path.name)
        return {"file": image_path.name, "chromosome_count": 0, "notation": "0,?",
                "sex_chromosomes": "?", "classifications": [], "abnormalities": [],
                "denver_groups": {}}

    batch = torch.cat(crop_tensors, dim=0).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(batch), dim=1)
        confs, preds = probs.max(dim=1)

    raw_labels = [IDX_TO_LABEL[p.item()] for p in preds]
    raw_confs = confs.tolist()
    refined = _refine_predictions(raw_labels)

    classifications = [
        {"bbox": bbox, "class": lbl, "confidence": round(conf, 4)}
        for bbox, lbl, conf in zip(valid_bboxes, refined, raw_confs)
    ]
    kt = _build_karyotype(refined)
    return {
        "file": image_path.name,
        "chromosome_count": len(refined),
        "notation": kt["notation"],
        "sex_chromosomes": kt["sex_chromosomes"],
        "classifications": classifications,
        "abnormalities": kt["abnormalities"],
        "denver_groups": kt["denver_groups"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Karyotype inference pipeline")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--detector",
                   default="runs/detect/chromosome_detector/weights/best.pt")
    p.add_argument("--classifier", default="models/chromosome_classifier.pth")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics is not installed. Run: pip install ultralytics")
        raise SystemExit(1)

    device = _detect_device(args.device)
    log.info("Device: %s", device)
    detector = YOLO(args.detector)
    classifier, crop_size = _load_classifier(args.classifier, device)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = sorted(p for p in Path(args.image_dir).iterdir()
                         if p.suffix.lower() in exts)
    if not image_paths:
        log.warning("No images found in %s", args.image_dir)
        return

    log.info("Analyzing %d image(s)...", len(image_paths))
    records = []
    for img_path in image_paths:
        log.info("  -> %s", img_path.name)
        try:
            records.append(analyze_image(
                img_path, detector, classifier, device, args.conf, crop_size,
            ))
        except Exception as exc:
            log.error("Failed on %s: %s", img_path.name, exc)
            records.append({"file": img_path.name, "error": str(exc)})

    Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")
    log.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
