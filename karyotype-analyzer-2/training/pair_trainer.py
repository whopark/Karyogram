"""pair_trainer.py — Orchestrate paired metaphase-karyogram training.

Scans directories for filename-matched pairs, extracts labeled chromosome
crops via karyogram_parser, generates YOLO annotations via auto_annotate,
then trains a YOLO detector and CNN classifier.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def normalize_stem(filename: str) -> str:
    """Normalize filename stem by stripping leading zeros from numeric segments.

    Example: '26-k-0659' -> '26-k-659', '26-k-0659-1' -> '26-k-659-1'
    """
    stem = Path(filename).stem
    return re.sub(r"-0+(\d)", r"-\1", stem)


def find_matched_pairs(metaphase_dir: str, karyogram_dir: str) -> list:
    """Find filename-matched image pairs between two directories.

    Returns list of (metaphase_path, karyogram_path) tuples.
    Uses normalize_stem() for fuzzy comparison. Logs unmatched files to stderr.
    """
    meta_dir = Path(metaphase_dir)
    kary_dir = Path(karyogram_dir)

    meta_map = {normalize_stem(p.name): p for p in meta_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTS}
    kary_map = {normalize_stem(p.name): p for p in kary_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTS}

    matched = []
    for key, meta_path in sorted(meta_map.items()):
        if key in kary_map:
            matched.append((meta_path, kary_map[key]))
        else:
            print(f"[WARN] No karyogram match for metaphase: {meta_path.name}", file=sys.stderr)

    for key, kary_path in sorted(kary_map.items()):
        if key not in meta_map:
            print(f"[WARN] No metaphase match for karyogram: {kary_path.name}", file=sys.stderr)

    return matched


def extract_labeled_crops(karyogram_dir: str, output_dir: str, padding: int = 5) -> dict:
    """Extract labeled chromosome crops from karyogram images.

    Calls karyogram_parser.parse_karyogram() per image and accumulates
    per-class counts. Per-image errors are logged and skipped.
    Returns dict mapping label -> count.
    """
    try:
        from karyogram_parser import parse_karyogram
    except ImportError as exc:
        print(f"[ERROR] karyogram_parser not available: {exc}", file=sys.stderr)
        return {}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    totals: dict = defaultdict(int)

    for img_path in sorted(Path(karyogram_dir).iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            counts = parse_karyogram(str(img_path), str(out_dir), padding)
            for label, cnt in counts.items():
                totals[label] += cnt
        except Exception as exc:
            print(f"[WARN] Skipping {img_path.name}: {exc}", file=sys.stderr)

    return dict(totals)


def generate_yolo_annotations(metaphase_dir: str, output_dir: str) -> int:
    """Generate YOLO bounding-box annotations for metaphase images.

    Imports process_image from auto_annotate and processes each image.
    Returns total annotation count.
    """
    try:
        from auto_annotate import process_image
    except ImportError as exc:
        print(f"[ERROR] auto_annotate not available: {exc}", file=sys.stderr)
        return 0

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for img_path in sorted(Path(metaphase_dir).iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            total += process_image(img_path, out_dir)
        except Exception as exc:
            print(f"[WARN] Annotation failed for {img_path.name}: {exc}", file=sys.stderr)

    return total


def run_training(
    crops_dir: str,
    annotations_dir: str,
    metaphase_dir: str,
    output_dir: str,
    epochs_detector: int = 50,
    epochs_classifier: int = 30,
    aug_factor: int = 5,
    batch_detector: int = 4,
    batch_classifier: int = 16,
    device: str = "cpu",
) -> dict:
    """Run YOLO detector + CNN classifier training sequentially.

    Builds YOLO dataset, trains detector (subprocess) and CNN classifier.
    Returns metrics dict with detector mAP and classifier accuracy values.
    """
    import subprocess

    out = Path(output_dir)
    dataset_dir = out / "yolo_dataset"
    metrics: dict = {}

    # Step 1: build YOLO dataset
    print("[INFO] Building YOLO dataset ...")
    build_argv = [
        "--images_dir", metaphase_dir,
        "--labels_dir", str(Path(annotations_dir) / "labels"),
        "--output_dir", str(dataset_dir),
        "--augment", "--aug_factor", str(aug_factor),
    ]
    try:
        from build_dataset import main as build_dataset_main
        build_dataset_main(build_argv)
    except Exception as exc:
        print(f"[WARN] Dataset build failed: {exc}", file=sys.stderr)

    # Step 2: train YOLO detector
    dataset_yaml = dataset_dir / "dataset.yaml"
    detector_out = out / "detector"
    if dataset_yaml.exists():
        try:
            import ultralytics  # noqa: F401
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "train_detector.py"),
                "--dataset", str(dataset_yaml),
                "--epochs", str(epochs_detector),
                "--batch", str(batch_detector),
                "--device", device,
                "--project", str(detector_out),
                "--name", "run",
            ]
            subprocess.run(cmd, check=True)
            metrics["detector_map50"] = "see detector/run/results.csv"
        except ImportError:
            print("[WARN] ultralytics not installed — skipping YOLO training.", file=sys.stderr)
            metrics["detector_map50"] = None
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] YOLO training subprocess failed: {exc}", file=sys.stderr)
            metrics["detector_map50"] = None
    else:
        print("[WARN] dataset.yaml not found — skipping YOLO training.", file=sys.stderr)
        metrics["detector_map50"] = None

    # Step 3: train CNN classifier
    classifier_out = out / "classifier_model.pth"
    try:
        import torch  # noqa: F401
        from train_classifier import train as train_classifier
        # @AX:NOTE: [AUTO] magic constant — warmup_epochs hardcoded to 5 here; not exposed as a pair_trainer CLI argument
        clf_args = argparse.Namespace(
            data_dir=crops_dir,
            epochs=epochs_classifier,
            warmup_epochs=5,
            batch_size=batch_classifier,
            lr=1e-3,
            output=str(classifier_out),
            device=device,
        )
        train_classifier(clf_args)
        metrics["classifier_top1"] = "see training log"
        metrics["classifier_top3"] = "see training log"
    except ImportError:
        print("[WARN] torch not installed — skipping CNN training.", file=sys.stderr)
        metrics["classifier_top1"] = None
        metrics["classifier_top3"] = None
    except Exception as exc:
        print(f"[WARN] CNN training failed: {exc}", file=sys.stderr)
        metrics["classifier_top1"] = None
        metrics["classifier_top3"] = None

    return metrics


def print_summary(pairs: int, crops: dict, metrics: dict) -> None:
    """Print structured training summary with pair count, crops per class, and model metrics."""
    sep = "=" * 60
    print(f"\n{sep}\n  PAIRED TRAINING SUMMARY\n{sep}")
    print(f"  Matched pairs processed : {pairs}")

    if crops:
        print("\n  Crops per class:")
        for label, count in sorted(crops.items(), key=lambda x: str(x[0])):
            print(f"    chr{label:<4} : {count}")
    else:
        print("  Crops per class         : none extracted")

    map50 = metrics.get("detector_map50")
    top1 = metrics.get("classifier_top1")
    top3 = metrics.get("classifier_top3")
    print(f"\n  YOLO mAP@0.5            : {map50 if map50 is not None else 'N/A'}")
    print(f"  CNN top-1 accuracy      : {top1 if top1 is not None else 'N/A'}")
    print(f"  CNN top-3 accuracy      : {top3 if top3 is not None else 'N/A'}")
    print(f"{sep}\n")


def main() -> None:
    """CLI entry point for paired metaphase-karyogram training."""
    default_device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            default_device = "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            default_device = "mps"
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Orchestrate paired metaphase-karyogram training pipeline."
    )
    parser.add_argument("--metaphase_dir", required=True,
                        help="Directory containing metaphase spread images.")
    parser.add_argument("--karyogram_dir", required=True,
                        help="Directory containing arranged karyogram images.")
    parser.add_argument("--output_dir", default="training/paired_output",
                        help="Root output directory (default: training/paired_output).")
    parser.add_argument("--epochs_detector", type=int, default=50)
    parser.add_argument("--epochs_classifier", type=int, default=30)
    parser.add_argument("--aug_factor", type=int, default=5)
    parser.add_argument("--batch_detector", type=int, default=4)
    parser.add_argument("--batch_classifier", type=int, default=16)
    parser.add_argument("--device", default=default_device,
                        help=f"Compute device: 'cpu', '0' (GPU), 'mps' (default: {default_device}).")
    parser.add_argument("--skip_detector", action="store_true",
                        help="Skip YOLO detector training.")
    parser.add_argument("--skip_classifier", action="store_true",
                        help="Skip CNN classifier training.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    crops_dir = out_dir / "crops"
    annotations_dir = out_dir / "annotations"

    print("[INFO] Scanning for matched pairs ...")
    pairs = find_matched_pairs(args.metaphase_dir, args.karyogram_dir)
    print(f"[INFO] Found {len(pairs)} matched pair(s).")
    if not pairs:
        print("[ERROR] No matched pairs found — aborting.", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Extracting labeled crops from karyogram images ...")
    crop_counts = extract_labeled_crops(args.karyogram_dir, str(crops_dir))

    print("[INFO] Generating YOLO annotations from metaphase images ...")
    annotation_count = generate_yolo_annotations(args.metaphase_dir, str(annotations_dir))
    print(f"[INFO] Total annotations generated: {annotation_count}")

    metrics: dict = {}
    if not args.skip_detector or not args.skip_classifier:
        metrics = run_training(
            crops_dir=str(crops_dir),
            annotations_dir=str(annotations_dir),
            metaphase_dir=args.metaphase_dir,
            output_dir=str(out_dir),
            epochs_detector=args.epochs_detector if not args.skip_detector else 0,
            epochs_classifier=args.epochs_classifier if not args.skip_classifier else 0,
            aug_factor=args.aug_factor,
            batch_detector=args.batch_detector,
            batch_classifier=args.batch_classifier,
            device=args.device,
        )
    else:
        print("[INFO] Both detector and classifier skipped.")

    print_summary(len(pairs), crop_counts, metrics)


if __name__ == "__main__":
    main()
