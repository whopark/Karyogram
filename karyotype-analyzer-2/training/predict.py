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
import torch.nn as nn
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Denver cytogenetic group mapping (members are int for autosomes, str for sex chr)
DENVER_GROUPS = {
    "A": [1, 2, 3], "B": [4, 5],
    "C": [6, 7, 8, 9, 10, 11, 12, "X"],
    "D": [13, 14, 15], "E": [16, 17, 18], "F": [19, 20],
    "G": [21, 22, "Y"],
}

IDX_TO_LABEL = [f"chr{i}" for i in range(1, 23)] + ["X", "Y"]
AUTOSOME_LABELS = IDX_TO_LABEL[:22]
NUM_CLASSES = 24
CROP_H, CROP_W = 64, 32


# ---------------------------------------------------------------------------
# Model definition — must match train_classifier.py exactly
# ---------------------------------------------------------------------------

class ChromosomeNet(nn.Module):
    """Small CNN for 24-class chromosome classification (grayscale 64x32 crops)."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_classifier(weights_path: str, device: torch.device) -> ChromosomeNet:
    model = ChromosomeNet()
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    log.info("Classifier loaded from %s", weights_path)
    return model


def _crop_tensor(image: np.ndarray, bbox: list) -> torch.Tensor:
    """Crop bbox [x,y,w,h] from RGB array; return (1,1,H,W) float tensor."""
    x, y, w, h = [int(v) for v in bbox]
    pil = Image.fromarray(image[y: y + h, x: x + w]).convert("L").resize((CROP_W, CROP_H))
    arr = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Pair-based refinement
# ---------------------------------------------------------------------------

def _refine_predictions(labels: list, max_iter: int = 10) -> list:
    """Reassign over-represented autosomes to under-represented ones (target=2 each)."""
    labels = list(labels)
    for _ in range(max_iter):
        counts = {lbl: labels.count(lbl) for lbl in IDX_TO_LABEL}
        changed = False
        for lbl in AUTOSOME_LABELS:
            while counts.get(lbl, 0) > 2:
                # Pick most under-represented autosome
                under = min(
                    (l for l in AUTOSOME_LABELS if counts.get(l, 0) < 2),
                    key=lambda l: counts.get(l, 0),
                    default=None,
                )
                if under is None:
                    break
                # Reassign last occurrence (proxy for lowest confidence)
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
    x, y = counts.get("X", 0), counts.get("Y", 0)

    if x >= 2 and y == 0:
        sex = "XX"
    elif x >= 1 and y >= 1:
        sex = "XY"
    else:
        sex = "X" * x + "Y" * y  # handles Turner (X), triple-X, etc.

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

    # ISCN: total,sex[,numeric abnormalities]
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

def analyze_image(image_path: Path, detector, classifier: ChromosomeNet,
                  device: torch.device, conf_threshold: float) -> dict:
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
        crop_tensors.append(_crop_tensor(img_np, [x1, y1, w, h]))
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
    p.add_argument("--conf", type=float, default=0.25)
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
    classifier = _load_classifier(args.classifier, device)

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
            records.append(analyze_image(img_path, detector, classifier, device, args.conf))
        except Exception as exc:
            log.error("Failed on %s: %s", img_path.name, exc)
            records.append({"file": img_path.name, "error": str(exc)})

    Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")
    log.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
