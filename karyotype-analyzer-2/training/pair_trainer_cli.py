"""CLI entry point for paired metaphase-karyogram training."""

import argparse
import sys
from pathlib import Path

from pair_trainer import (
    find_matched_pairs,
    extract_labeled_crops,
    generate_yolo_annotations,
    run_training,
    print_summary,
)


def main() -> None:
    """Parse arguments and run the paired training pipeline."""
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
                        help=f"Compute device: 'cpu', '0' (GPU), 'mps' "
                             f"(default: {default_device}).")
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
    annotation_count = generate_yolo_annotations(
        args.metaphase_dir, str(annotations_dir),
    )
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
