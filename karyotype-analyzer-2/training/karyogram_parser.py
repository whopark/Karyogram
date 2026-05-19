"""
karyogram_parser.py — Extract labeled chromosome crops from arranged karyogram images.

Karyogram images show chromosomes sorted by number in a grid layout.
This module identifies each chromosome, assigns the correct label based on
grid position, and saves crops as classification training data.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Chromosome label order as they appear left-to-right, top-to-bottom in a karyogram.
# Each number appears twice (homologous pair), then X and Y (or X,X in female).
KARYOGRAM_ORDER = (
    [1, 1, 2, 2, 3, 3]           # Row 1 Group A
    + [4, 4, 5, 5]               # Row 1 Group B
    + [6, 6, 7, 7, 8, 8, 9, 9]  # Row 1 Group C (first part)
    + [10, 10, 11, 11, 12, 12]   # Row 2 Group C (rest)
    + [13, 13, 14, 14, 15, 15]   # Row 2 Group D
    + [16, 16, 17, 17, 18, 18]   # Row 2 Group E
    + [19, 19, 20, 20]           # Row 3 Group F
    + [21, 21, 22, 22]           # Row 3 Group G
    + ["X", "X", "Y"]            # Row 3 sex chromosomes (Y absent in female)
)

MIN_CONTOUR_AREA = 300   # Pixels; filters noise and debris
ROW_GAP_FACTOR = 0.5     # Fraction of median chromosome height to detect row breaks


def load_as_gray(image_path: str) -> np.ndarray:
    """Load an image in any mode and return a uint8 grayscale array."""
    pil_img = Image.open(image_path)
    if pil_img.mode == "RGBA":
        # Composite onto white background before converting
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    return np.array(pil_img.convert("L"))


def threshold_image(gray: np.ndarray) -> np.ndarray:
    """Return a binary mask isolating dark chromosome objects on light background."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def find_chromosome_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes (x, y, w, h) for chromosome-sized contours."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            boxes.append(cv2.boundingRect(cnt))
    return boxes


def cluster_rows(
    boxes: list[tuple[int, int, int, int]],
) -> list[list[tuple[int, int, int, int]]]:
    """
    Group bounding boxes into horizontal rows using adaptive gap detection.

    Boxes are first sorted by their vertical center. A new row starts whenever
    the vertical gap between consecutive centers exceeds ROW_GAP_FACTOR times
    the median chromosome height.
    """
    if not boxes:
        return []

    heights = [h for _, _, _, h in boxes]
    median_h = float(np.median(heights))
    gap_threshold = ROW_GAP_FACTOR * median_h

    # Sort by vertical center
    sorted_boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)

    rows: list[list[tuple[int, int, int, int]]] = [[sorted_boxes[0]]]
    for box in sorted_boxes[1:]:
        prev_center_y = rows[-1][-1][1] + rows[-1][-1][3] / 2
        curr_center_y = box[1] + box[3] / 2
        if curr_center_y - prev_center_y > gap_threshold:
            rows.append([])
        rows[-1].append(box)

    # Within each row, sort left-to-right by x
    for row in rows:
        row.sort(key=lambda b: b[0])

    return rows


def build_label_sequence(chromosome_count: int) -> list[str]:
    """
    Return the ordered list of string labels for the detected chromosomes.

    Normal karyotype has 46 chromosomes. Abnormal counts (45, 47 …) are handled
    by truncating or extending the standard order near the sex chromosomes.
    """
    # Base order covers up to 47 (XY karyotype with one extra)
    base = list(KARYOGRAM_ORDER)  # length 47 including Y

    if chromosome_count <= len(base):
        sequence = [str(lbl) for lbl in base[:chromosome_count]]
    else:
        # More than 47: repeat the last sex chromosome label
        sequence = [str(lbl) for lbl in base]
        extra = chromosome_count - len(base)
        sequence += [sequence[-1]] * extra

    return sequence


def assign_labels(
    rows: list[list[tuple[int, int, int, int]]],
) -> list[tuple[tuple[int, int, int, int], str, str]]:
    """
    Pair each bounding box with a chromosome label and a/b suffix.

    Returns a list of (box, label, suffix) tuples in reading order.
    """
    flat_boxes = [box for row in rows for box in row]
    labels = build_label_sequence(len(flat_boxes))

    # Track how many times each label has been assigned to set 'a'/'b' suffix
    seen: dict[str, int] = defaultdict(int)
    result = []
    for box, label in zip(flat_boxes, labels):
        seen[label] += 1
        suffix = "a" if seen[label] == 1 else "b"
        result.append((box, label, suffix))
    return result


def crop_chromosome(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int,
) -> np.ndarray:
    """Return a padded grayscale crop of one chromosome."""
    x, y, w, h = box
    img_h, img_w = gray.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    return gray[y1:y2, x1:x2]


def parse_karyogram(
    image_path: str,
    output_dir: str,
    padding: int = 5,
) -> dict[str, int]:
    """
    Extract chromosome crops from one karyogram image and save them.

    Returns a dict mapping label strings to the number of crops saved.
    """
    stem = Path(image_path).stem
    gray = load_as_gray(image_path)
    binary = threshold_image(gray)
    boxes = find_chromosome_boxes(binary)

    if not boxes:
        print(f"  [WARN] No chromosomes detected in {image_path}")
        return {}

    rows = cluster_rows(boxes)
    labeled = assign_labels(rows)

    counts: dict[str, int] = defaultdict(int)
    for box, label, suffix in labeled:
        class_dir = os.path.join(output_dir, f"chr{label}")
        os.makedirs(class_dir, exist_ok=True)
        crop = crop_chromosome(gray, box, padding)
        filename = f"{stem}_chr{label}_{suffix}.png"
        out_path = os.path.join(class_dir, filename)
        cv2.imwrite(out_path, crop)
        counts[label] += 1

    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract labeled chromosome crops from arranged karyogram images."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing karyogram image files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory for labeled crop subdirectories.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Pixel padding added around each chromosome crop (default: 5).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input_dir '{args.input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_files = [
        p for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in '{args.input_dir}'.", file=sys.stderr)
        sys.exit(1)

    total_per_class: dict[str, int] = defaultdict(int)
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        try:
            counts = parse_karyogram(str(img_path), args.output_dir, args.padding)
            for label, n in counts.items():
                total_per_class[label] += n
        except Exception as e:
            print(f"  SKIP: {e}")

    # Print summary sorted by chromosome number
    print("\n=== Extraction Summary ===")
    print(f"{'Class':<8} {'Crops':>6}")
    print("-" * 16)

    def sort_key(label: str) -> tuple[int, str]:
        try:
            return (int(label), "")
        except ValueError:
            return (100, label)  # X and Y sort after numerics

    for label in sorted(total_per_class.keys(), key=sort_key):
        print(f"chr{label:<5} {total_per_class[label]:>6}")

    print("-" * 16)
    print(f"{'TOTAL':<8} {sum(total_per_class.values()):>6}")


if __name__ == "__main__":
    main()
