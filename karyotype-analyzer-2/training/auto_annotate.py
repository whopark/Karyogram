"""
auto_annotate.py — Auto-generate YOLO bounding box annotations from metaphase spread images.

Uses multi-threshold OpenCV contour detection to locate chromosomes and outputs
YOLO-format .txt annotation files alongside visualization images.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# Detection parameters
MIN_AREA_RATIO = 0.0002      # Minimum contour area as fraction of image area
MAX_AREA_RATIO = 0.15        # Maximum contour area as fraction of image area
BASELINE_AREA = 1_000_000    # Baseline image area (1000x1000) for area scaling
MIN_AREA_ABS = 200           # Absolute minimum area (for small images)
MAX_AREA_ABS = 150_000       # Absolute maximum area cap
NUCLEUS_CIRCULARITY = 0.85   # Circularity above this is likely a nucleus
NUCLEUS_AREA_FACTOR = 3.0    # Nucleus must also be > median_area * this factor
TARGET_COUNT = 46            # Expected chromosome count (normal human karyotype)
TARGET_RANGE = (40, 50)      # Acceptable chromosome count range for selection


def load_image_as_gray(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load image and return both the original BGR and grayscale versions."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        # Try RGBA path
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, gray


def enhance_with_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def compute_area_bounds(img_h: int, img_w: int) -> tuple[int, int]:
    """Compute absolute area bounds scaled to actual image size vs baseline."""
    scale = (img_h * img_w) / BASELINE_AREA
    min_area = max(MIN_AREA_ABS, int(MIN_AREA_RATIO * img_h * img_w))
    max_area = min(MAX_AREA_ABS, int(MAX_AREA_RATIO * img_h * img_w))
    # Clamp min to scaled baseline
    min_area = max(min_area, int(200 * scale))
    return min_area, max_area


def build_binary_masks(enhanced: np.ndarray) -> list[tuple]:
    """Generate binary masks via Otsu, adaptive Gaussian, and fixed-level thresholding."""
    # Invert so chromosomes become white foreground on dark background
    inverted = cv2.bitwise_not(enhanced)
    masks = []
    # Otsu global threshold
    _, otsu_mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(("otsu", otsu_mask))
    # Adaptive Gaussian threshold (blocksize=21, C=5)
    adaptive_mask = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    masks.append(("adaptive", adaptive_mask))
    # Fixed-level thresholds
    for level in [100, 127, 150]:
        _, fixed_mask = cv2.threshold(inverted, level, 255, cv2.THRESH_BINARY)
        masks.append((f"fixed_{level}", fixed_mask))
    return masks


def extract_contours_from_mask(mask: np.ndarray) -> list:
    """Find external contours from a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def circularity(contour) -> float:
    """Compute circularity = 4*pi*area / perimeter^2. Range [0, 1]."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-6:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def filter_contours(contours: list, min_area: int, max_area: int) -> list:
    """Filter by area bounds, then remove interphase nuclei (high-circularity + large area)."""
    area_filtered = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    if not area_filtered:
        return []
    areas = np.array([cv2.contourArea(c) for c in area_filtered])
    median_area = float(np.median(areas))
    result = []
    for c in area_filtered:
        c_area = cv2.contourArea(c)
        c_circ = circularity(c)
        is_nucleus = (c_circ > NUCLEUS_CIRCULARITY) and (c_area > median_area * NUCLEUS_AREA_FACTOR)
        if not is_nucleus:
            result.append(c)
    return result


def contours_to_yolo(contours: list, img_h: int, img_w: int) -> list[str]:
    """Convert contours to YOLO lines: '0 cx cy w h' normalized to [0, 1]."""
    lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_w - x), min(h, img_h - y)
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return lines


def select_best_mask_result(mask_results: list[tuple]) -> tuple:
    """
    Pick the (method_name, contours) result closest to TARGET_COUNT.

    Prefers results within TARGET_RANGE; falls back to globally nearest count.
    """
    best_name, best_contours = mask_results[0]
    best_dist = abs(len(best_contours) - TARGET_COUNT)
    in_range = [
        (n, c) for n, c in mask_results
        if TARGET_RANGE[0] <= len(c) <= TARGET_RANGE[1]
    ]
    if in_range:
        best_name, best_contours = min(in_range, key=lambda x: abs(len(x[1]) - TARGET_COUNT))
    else:
        for name, contours in mask_results:
            dist = abs(len(contours) - TARGET_COUNT)
            if dist < best_dist:
                best_dist, best_name, best_contours = dist, name, contours
    return best_name, best_contours


def draw_detections(img_bgr: np.ndarray, contours: list) -> np.ndarray:
    """Draw bounding boxes and contour count on a copy of the image."""
    vis = img_bgr.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f"Detected: {len(contours)}"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def process_image(image_path: Path, output_dir: Path) -> int:
    """
    Process a single image: detect chromosomes, write YOLO annotation, save visualization.

    Returns the number of chromosomes detected.
    """
    img_bgr, gray = load_image_as_gray(image_path)
    img_h, img_w = gray.shape[:2]

    enhanced = enhance_with_clahe(gray)
    min_area, max_area = compute_area_bounds(img_h, img_w)

    masks = build_binary_masks(enhanced)

    # Collect (method_name, filtered_contours) for each threshold method
    mask_results = []
    for method_name, mask in masks:
        raw_contours = extract_contours_from_mask(mask)
        filtered = filter_contours(raw_contours, min_area, max_area)
        mask_results.append((method_name, filtered))

    best_method, best_contours = select_best_mask_result(mask_results)

    # Write YOLO annotation file
    stem = image_path.stem
    ann_path = output_dir / "labels" / f"{stem}.txt"
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    yolo_lines = contours_to_yolo(best_contours, img_h, img_w)
    ann_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    # Save visualization
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis = draw_detections(img_bgr, best_contours)
    cv2.imwrite(str(vis_dir / f"{stem}_annot.png"), vis)

    return len(best_contours)


def collect_image_paths(input_dir: Path) -> list[Path]:
    """Collect all PNG and JPG image paths from input directory."""
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
    return paths


def print_summary(counts: list[int]) -> None:
    """Print summary statistics for the annotation run."""
    if not counts:
        print("No images processed.")
        return
    print(f"\n--- Annotation Summary ---")
    print(f"Images processed : {len(counts)}")
    print(f"Min detections   : {min(counts)}")
    print(f"Max detections   : {max(counts)}")
    print(f"Avg detections   : {np.mean(counts):.1f}")
    print(f"Median detections: {int(np.median(counts))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-annotate metaphase images with YOLO bounding boxes."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing metaphase PNG/JPG images.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for YOLO labels/ and visualizations/.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: input_dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        print(f"No PNG/JPG images found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s) in {input_dir}")

    counts: list[int] = []
    for img_path in image_paths:
        try:
            n = process_image(img_path, output_dir)
            print(f"  {img_path.name}: {n} chromosomes detected")
            counts.append(n)
        except Exception as exc:
            print(f"  ERROR processing {img_path.name}: {exc}", file=sys.stderr)

    print_summary(counts)


if __name__ == "__main__":
    main()
