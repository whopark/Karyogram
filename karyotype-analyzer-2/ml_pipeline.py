"""ml_pipeline.py — Standalone ML inference orchestrator for karyogram generation.

Combines YOLO detection (T1), ChromosomeNet CNN classification (T2), and ISCN
karyotype derivation (T3). Does not import from app.py or training/.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

NUM_CLASSES = 24
CROP_H, CROP_W = 64, 32
IDX_TO_LABEL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

DENVER_GROUPS = {
    "A": ["chr1", "chr2", "chr3"],
    "B": ["chr4", "chr5"],
    "C": ["chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chrX"],
    "D": ["chr13", "chr14", "chr15"],
    "E": ["chr16", "chr17", "chr18"],
    "F": ["chr19", "chr20"],
    "G": ["chr21", "chr22", "chrY"],
}

# Resolve default paths relative to this file's directory, not CWD
_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DETECTOR_PATH = str(_MODULE_DIR / "runs" / "detect" / "training" / "runs" / "chromosome_v2" / "weights" / "best.pt")
DEFAULT_CLASSIFIER_PATH = str(_MODULE_DIR / "training" / "models" / "chromosome_classifier.pth")

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Model architecture must match training/train_classifier.py exactly
if _TORCH_AVAILABLE:
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

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(x))
else:
    class ChromosomeNet:  # type: ignore[no-redef]
        """Stub when torch is unavailable."""
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is not installed")


def load_detector(weights_path: Optional[str] = None) -> dict:
    """Load YOLOv8 model. Returns {"model": YOLO} or {"error": str}."""
    path = weights_path or DEFAULT_DETECTOR_PATH
    if not Path(path).exists():
        return {"error": f"Detector weights not found at {path}. Run: python training/train_detector.py"}
    try:
        from ultralytics import YOLO  # noqa: PLC0415
        model = YOLO(path)
        log.info("Detector loaded from %s", path)
        return {"model": model}
    except ImportError:
        return {"error": "ultralytics is not installed. Run: pip install ultralytics"}
    except Exception as exc:
        return {"error": f"Failed to load detector: {exc}"}


def load_classifier(weights_path: Optional[str] = None, device=None) -> dict:
    """Load CNN classifier. Returns {"model": ChromosomeNet, "device": device} or {"error": str}."""
    if not _TORCH_AVAILABLE:
        return {"error": "torch is not installed. Run: pip install torch"}
    path = weights_path or DEFAULT_CLASSIFIER_PATH
    if not Path(path).exists():
        return {"error": f"Classifier weights not found at {path}. Run: python training/train_classifier.py"}
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChromosomeNet()
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device).eval()
        log.info("Classifier loaded from %s on %s", path, device)
        return {"model": model, "device": device}
    except Exception as exc:
        return {"error": f"Failed to load classifier: {exc}"}


def detect_chromosomes(detector: dict, image_pil, conf: float = 0.25) -> list:
    """Run YOLO on image_pil. Returns list of {"bbox": [x,y,w,h], "crop": PIL.Image}."""
    from PIL import Image  # noqa: PLC0415
    img_np = np.array(image_pil.convert("RGB"))
    results = detector["model"](image_pil, conf=conf, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue  # skip degenerate boxes
        detections.append({"bbox": [x1, y1, w, h], "crop": Image.fromarray(img_np[y1:y1 + h, x1:x1 + w])})
    log.info("Detected %d chromosomes (conf>=%.2f)", len(detections), conf)
    return detections


def _crop_to_tensor(crop_pil, device):
    """Resize crop to CROP_HxCROP_W grayscale, normalize [0,1], return (1,1,H,W) tensor."""
    arr = np.array(crop_pil.convert("L").resize((CROP_W, CROP_H)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _pair_refine(labels: list, confs: list, max_iter: int = 10) -> list:
    """Reassign over-represented autosomes (target=2) to under-represented ones.

    Uses lowest-confidence occurrence of the over-represented class as swap candidate.
    """
    labels, confs = list(labels), list(confs)
    autosome_labels = IDX_TO_LABEL[:22]
    for _ in range(max_iter):
        counts = {lbl: labels.count(lbl) for lbl in IDX_TO_LABEL}
        changed = False
        for lbl in autosome_labels:
            while counts.get(lbl, 0) > 2:
                under = min(
                    (l for l in autosome_labels if counts.get(l, 0) < 2),
                    key=lambda l: counts.get(l, 0),
                    default=None,
                )
                if under is None:
                    break
                lbl_indices = [i for i, l in enumerate(labels) if l == lbl]
                worst = min(lbl_indices, key=lambda i: confs[i])
                labels[worst] = under
                counts[lbl] -= 1
                counts[under] = counts.get(under, 0) + 1
                changed = True
        if not changed:
            break
    return labels


def classify_chromosomes(classifier_info: dict, detections: list) -> list:
    """Classify each detected crop. Returns list of enriched detection dicts."""
    if not detections:
        return []
    model, device = classifier_info["model"], classifier_info["device"]
    batch = torch.cat([_crop_to_tensor(d["crop"], device) for d in detections], dim=0)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)
        confs_t, preds_t = probs.max(dim=1)
    raw_labels = [IDX_TO_LABEL[p.item()] for p in preds_t]
    raw_confs = confs_t.tolist()
    refined = _pair_refine(raw_labels, raw_confs)
    return [
        {"bbox": d["bbox"], "label": lbl, "confidence": round(c, 4), "crop": d["crop"]}
        for d, lbl, c in zip(detections, refined, raw_confs)
    ]


def build_karyotype(classifications: list) -> dict:
    """Derive ISCN 2020 notation and metadata from classification results."""
    labels = [c["label"] for c in classifications]
    counts = {lbl: labels.count(lbl) for lbl in IDX_TO_LABEL}
    x_n, y_n = counts.get("chrX", 0), counts.get("chrY", 0)

    if x_n >= 2 and y_n == 0:
        sex = "XX"
    elif x_n >= 1 and y_n >= 1:
        sex = "XY"
    else:
        sex = "X" * x_n + "Y" * y_n  # handles Turner (X), triple-X, etc.

    abnormalities = []
    for i in range(1, 23):
        n = counts.get(f"chr{i}", 0)
        if n == 3:
            abnormalities.append(f"+{i}")
        elif n == 1:
            abnormalities.append(f"-{i}")
    if x_n == 1 and y_n == 0:
        abnormalities.append("monosomy X")
    if x_n == 3:
        abnormalities.append("triple X")

    iscn_parts = [str(len(labels)), sex] + [a for a in abnormalities if a[0] in ("+", "-")]
    return {
        "notation": ",".join(iscn_parts),
        "count": len(labels),
        "sex": sex,
        "abnormalities": abnormalities,
        "denver_groups": {g: sum(counts.get(m, 0) for m in members) for g, members in DENVER_GROUPS.items()},
    }


def run_pipeline(
    image_pil,
    detector=None,
    classifier_info=None,
    conf: float = 0.25,
) -> dict:
    """Full ML inference pipeline: detect -> classify -> karyotype.

    Args:
        image_pil: PIL Image to analyze.
        detector: Pre-loaded detector dict from load_detector(), or None to auto-load.
        classifier_info: Pre-loaded classifier dict from load_classifier(), or None to auto-load.
        conf: YOLO confidence threshold.

    Returns a result dict with all fields needed by karyogram_generator.
    On failure returns a dict with an "error" key.
    """
    if detector is None:
        detector = load_detector()
    if "error" in detector:
        return {"error": detector["error"], "count": 0, "classifications": []}

    if classifier_info is None:
        classifier_info = load_classifier()
    if "error" in classifier_info:
        return {"error": classifier_info["error"], "count": 0, "classifications": []}

    detections = detect_chromosomes(detector, image_pil, conf=conf)
    if not detections:
        return {
            "warning": "No chromosomes detected",
            "count": 0,
            "notation": "0,?",
            "sex": "?",
            "abnormalities": [],
            "denver_groups": {g: 0 for g in DENVER_GROUPS},
            "classifications": [],
        }

    classifications = classify_chromosomes(classifier_info, detections)
    karyotype = build_karyotype(classifications)
    return {
        "count": karyotype["count"],
        "notation": karyotype["notation"],
        "sex": karyotype["sex"],
        "abnormalities": karyotype["abnormalities"],
        "denver_groups": karyotype["denver_groups"],
        "classifications": classifications,
    }
