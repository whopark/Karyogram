"""
workstation_parser.py — Split clinical workstation screenshots into metaphase and
karyogram halves, invert dark backgrounds, and delegate chromosome extraction to
karyogram_parser.py.

Input: ~1800x950px RGBA screenshots. Left half = metaphase spread; right half =
arranged karyogram grid. A vertical toolbar column separates the two halves.
"""

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from karyogram_parser import parse_karyogram

# Minimum rows to strip from top (toolbar) and bottom (ISCN text)
TOOLBAR_STRIP_PX = 50
ISCN_STRIP_PX = 40
# Row std-dev below which a row is classified as uniform toolbar
TOOLBAR_STD_THRESHOLD = 10.0
# Row variance above which bottom rows are classified as text content
ISCN_VARIANCE_THRESHOLD = 20.0


def detect_split_boundary(gray: np.ndarray) -> int:
    """Find the vertical split between metaphase (left) and karyogram (right).

    Computes per-column intensity standard deviation and returns the column with
    the lowest variance within the center 40 % of the image (cols 30 %–70 %).
    That low-variance column corresponds to the dividing toolbar or separator line.
    """
    width = gray.shape[1]
    col_start = int(width * 0.30)
    col_end = int(width * 0.70)
    col_stds = gray[:, col_start:col_end].std(axis=0)
    return col_start + int(np.argmin(col_stds))


def invert_if_dark(gray_array: np.ndarray) -> np.ndarray:
    """Invert pixel intensities when median < 128 (dark background).

    karyogram_parser expects dark chromosomes on a light background.
    Clinical screenshots use the opposite convention.
    """
    if np.median(gray_array) < 128:
        return 255 - gray_array
    return gray_array


def _toolbar_height(gray: np.ndarray) -> int:
    """Return row count of the uniform top toolbar (std < threshold per row)."""
    for row_idx in range(gray.shape[0]):
        if gray[row_idx, :].std() > TOOLBAR_STD_THRESHOLD:
            return row_idx
    return 0


def _iscn_text_start(gray: np.ndarray) -> int:
    """Return the top row index of the bottom ISCN text band.

    Scans upward from the bottom; a run of high-variance rows signals text.
    Returns image height (no trimming) when no text region is found.
    """
    height = gray.shape[0]
    for row_idx in range(height - 1, -1, -1):
        if float(gray[row_idx, :].var()) > ISCN_VARIANCE_THRESHOLD:
            text_start = row_idx
            for scan in range(row_idx - 1, max(row_idx - 80, -1), -1):
                if float(gray[scan, :].var()) > ISCN_VARIANCE_THRESHOLD:
                    text_start = scan
                else:
                    break
            return text_start
    return height


def crop_karyogram_region(image_path: str) -> Image.Image:
    """Load a workstation screenshot and return the karyogram (right) half.

    Pipeline:
    1. Load RGBA, composite onto white, convert to grayscale.
    2. Detect vertical split; keep right half (split_col onward).
    3. Strip uniform toolbar rows from top (at least TOOLBAR_STRIP_PX).
    4. Strip ISCN text rows from bottom (at least ISCN_STRIP_PX).
    5. Return cropped region as a grayscale PIL Image.
    """
    pil_img = Image.open(image_path)
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg

    gray_full = np.array(pil_img.convert("L"))
    split_col = detect_split_boundary(gray_full)
    kgram = gray_full[:, split_col:]

    top_trim = max(_toolbar_height(kgram), TOOLBAR_STRIP_PX)
    height = kgram.shape[0]
    bottom_trim = height - min(_iscn_text_start(kgram), height - ISCN_STRIP_PX)

    return Image.fromarray(kgram[top_trim : height - bottom_trim, :])


def parse_workstation_image(
    image_path: str,
    output_dir: str,
    padding: int = 5,
) -> dict:
    """Full pipeline: crop karyogram half -> invert -> extract chromosome crops.

    1. Crop the karyogram (right) region.
    2. Invert if background is dark.
    3. Save to a temporary PNG file.
    4. Call karyogram_parser.parse_karyogram() on the temp file.
    5. Remove temp file and return label-count dict.
    """
    kgram_pil = crop_karyogram_region(image_path)
    normalised = invert_if_dark(np.array(kgram_pil))

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem

    with tempfile.NamedTemporaryFile(
        suffix=".png", prefix=f"ws_{stem}_", delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, normalised)
        counts = parse_karyogram(tmp_path, output_dir, padding)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return counts


def main() -> None:
    """Process all images in input_dir and write chromosome crops to output_dir."""
    arg_parser = argparse.ArgumentParser(
        description="Parse clinical workstation karyogram screenshots"
    )
    arg_parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing workstation screenshot images.",
    )
    arg_parser.add_argument(
        "--output_dir", required=True,
        help="Directory where chromosome crop sub-folders will be written.",
    )
    arg_parser.add_argument(
        "--padding", type=int, default=5,
        help="Pixel padding around each chromosome crop (default: 5).",
    )
    args = arg_parser.parse_args()

    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [
        p for p in Path(args.input_dir).iterdir()
        if p.suffix.lower() in supported
    ]

    if not image_paths:
        print(f"[WARN] No supported images found in {args.input_dir}")
        return

    total_crops = 0
    for img_path in sorted(image_paths):
        print(f"Processing: {img_path.name}")
        counts = parse_workstation_image(str(img_path), args.output_dir, args.padding)
        n = sum(counts.values())
        total_crops += n
        print(f"  -> {n} crops  {dict(counts)}")

    print(f"\nDone. Total chromosome crops saved: {total_crops}")


if __name__ == "__main__":
    main()
