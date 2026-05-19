"""
train_detector.py — Fine-tune YOLOv8 for chromosome detection in metaphase spreads.

Usage:
    python train_detector.py --dataset dataset.yaml --model yolov8n.pt --epochs 100
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for chromosome detection"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Pretrained weights: yolov8n.pt (speed) or yolov8s.pt (accuracy)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size; -1 = auto (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", default=None, help="Device: 'cpu' or '0' for GPU (auto-detect)")
    parser.add_argument("--project", default="runs/detect", help="Output root dir (default: runs/detect)")
    parser.add_argument("--name", default="chromosome_detector", help="Run sub-dir name")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience (default: 20)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--freeze", type=int, default=10, help="Backbone layers to freeze (default: 10)")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_dataset(dataset_path: str) -> Path:
    """Verify that the dataset YAML file exists and contains required keys."""
    import yaml  # PyYAML ships with ultralytics

    p = Path(dataset_path)
    if not p.exists():
        print(f"[ERROR] dataset.yaml not found: {p.resolve()}", file=sys.stderr)
        sys.exit(1)

    with p.open() as fh:
        cfg = yaml.safe_load(fh)

    required_keys = {"path", "train", "val", "names"}
    missing = required_keys - set(cfg.keys())
    if missing:
        print(
            f"[ERROR] dataset.yaml is missing required keys: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Dataset config loaded: {p.resolve()}")
    print(f"       Classes ({len(cfg['names'])}): {cfg['names']}")
    return p


def resolve_device(requested: str | None) -> str:
    """Auto-detect CUDA; fall back to CPU when requested device is unavailable."""
    if requested is not None:
        return requested

    try:
        import torch
        if torch.cuda.is_available():
            device = "0"
            name = torch.cuda.get_device_name(0)
            print(f"[INFO] CUDA detected — using GPU 0: {name}")
        else:
            device = "cpu"
            print("[INFO] No CUDA device found — using CPU")
    except ImportError:
        device = "cpu"
        print("[INFO] torch not importable for device check — defaulting to CPU")

    return device


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, dataset_path: Path, device: str) -> Path:
    """Run YOLOv8 fine-tuning and return the path to the best weights file."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[ERROR] ultralytics is not installed. Run: pip install ultralytics",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load pretrained model or resume from checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume checkpoint not found: {resume_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        model = YOLO(str(resume_path))
    else:
        print(f"[INFO] Loading pretrained weights: {args.model}")
        model = YOLO(args.model)

    # Chromosome-specific training configuration
    # - mosaic=1.0: mosaic augmentation helps with varying chromosome density
    # - iou=0.5: higher IoU threshold because chromosomes can be close together
    # - multi_scale=True: robustness across image resolutions
    # - freeze: freeze backbone layers for small datasets to preserve pretrained features
    results = model.train(
        data=str(dataset_path.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        lr0=args.lr0,
        freeze=args.freeze,
        # Augmentation settings tuned for chromosome images
        mosaic=1.0,
        multi_scale=True,
        # NMS IoU threshold — chromosomes cluster tightly in metaphase spreads
        iou=0.5,
        # Determinism helpers
        deterministic=False,
        verbose=True,
    )

    # Locate best weights produced by the run
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        # Fallback: last.pt when best.pt was not saved (e.g. training interrupted)
        best_weights = Path(results.save_dir) / "weights" / "last.pt"

    return best_weights, results


# ---------------------------------------------------------------------------
# Post-training reporting and export
# ---------------------------------------------------------------------------

def print_metrics_summary(results, best_weights: Path) -> None:
    """Print a compact training metrics summary to stdout."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    try:
        # results.results_dict is available after training completes
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", float("nan"))
        map5095 = metrics.get("metrics/mAP50-95(B)", float("nan"))
        print(f"  Best mAP@0.5       : {map50:.4f}")
        print(f"  Best mAP@0.5:0.95  : {map5095:.4f}")
    except Exception:
        print("  (metrics not available in results object)")

    print(f"  Best weights       : {best_weights.resolve()}")
    print("=" * 60 + "\n")


def run_validation(best_weights: Path, dataset_path: Path, device: str) -> None:
    """Validate the best model on the val split and print results."""
    from ultralytics import YOLO

    print("[INFO] Running validation on val split …")
    model = YOLO(str(best_weights))
    val_results = model.val(
        data=str(dataset_path.resolve()),
        device=device,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    try:
        metrics = val_results.results_dict
        for key, value in metrics.items():
            if "mAP" in key or "precision" in key.lower() or "recall" in key.lower():
                print(f"  {key:<30}: {value:.4f}")
    except Exception:
        print("  (detailed metrics unavailable)")
    print("=" * 60 + "\n")


def export_onnx(best_weights: Path, imgsz: int) -> None:
    """Export best weights to ONNX format for deployment."""
    from ultralytics import YOLO

    print("[INFO] Exporting best model to ONNX …")
    model = YOLO(str(best_weights))
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
    )
    print(f"[INFO] ONNX model saved to: {export_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Step 1: validate inputs
    dataset_path = validate_dataset(args.dataset)
    device = resolve_device(args.device)

    print(f"[INFO] Model        : {args.model}")
    print(f"[INFO] Epochs       : {args.epochs}")
    print(f"[INFO] Batch size   : {args.batch}")
    print(f"[INFO] Image size   : {args.imgsz}")
    print(f"[INFO] Device       : {device}")
    print(f"[INFO] Freeze layers: {args.freeze}")
    print(f"[INFO] Output       : {args.project}/{args.name}")

    # Step 2: train
    best_weights, results = train(args, dataset_path, device)

    # Step 3: print metrics summary
    print_metrics_summary(results, best_weights)

    # Step 4: validate on val split
    run_validation(best_weights, dataset_path, device)

    # Step 5: export to ONNX (optional, may fail without onnx package)
    try:
        export_onnx(best_weights, args.imgsz)
    except Exception as e:
        print(f"[WARN] ONNX export skipped: {e}")

    print("[DONE] Training pipeline complete.")


if __name__ == "__main__":
    main()
