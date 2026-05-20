"""Digital preprocessor for chromosome image denoising and straightening.

Implements cascaded denoising (background subtraction, debris removal,
CLAHE, bilateral filter) and medial-axis chromosome straightening.
"""

import numpy as np
from typing import Dict, List, Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2

from cv.preprocessor_helpers import (
    extract_skeleton,
    order_skeleton_points,
    smooth_curve,
    compute_curvature,
)


class DigitalPreprocessor:
    """Cascaded denoising and medial-axis chromosome straightening."""

    def __init__(self):
        # Denoising parameters
        self.clahe_clip_limit = 3.0
        self.clahe_grid_size = (8, 8)
        self.bilateral_d = 7
        self.bilateral_sigma_color = 50
        self.bilateral_sigma_space = 50
        self.debris_max_area_ratio = 0.0005
        # Straightening parameters
        self.straighten_width = 30

    # -- Cascaded Denoising Pipeline ------------------------------------
    def denoise(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Multi-stage cascaded denoising pipeline.

        Returns denoised image and metadata about what was cleaned.
        """
        if not CV2_AVAILABLE:
            return gray, {"error": "OpenCV not available"}

        h, w = gray.shape
        metadata: Dict = {
            "stages_applied": [],
            "original_stats": {
                "mean": float(np.mean(gray)),
                "std": float(np.std(gray)),
            },
        }
        current = gray.copy()

        # Stage 1: Background estimation & subtraction
        local_means = cv2.blur(current, (h // 4 | 1, w // 4 | 1))
        illumination_var = float(np.std(local_means[current > 20]))
        if illumination_var > 30:
            bg_kernel_size = max(51, min(h, w) // 8) | 1
            bg_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size)
            )
            background = cv2.morphologyEx(current, cv2.MORPH_OPEN, bg_kernel)
            current = cv2.subtract(current, background)
            current = cv2.normalize(current, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            metadata["stages_applied"].append("background_subtraction")
        else:
            metadata["stages_applied"].append("background_subtraction_skipped")

        # Stage 2: Debris / artifact removal
        _, debris_thresh = cv2.threshold(
            current, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        debris_contours, _ = cv2.findContours(
            debris_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        max_debris_area = h * w * self.debris_max_area_ratio
        debris_mask = np.zeros((h, w), dtype=np.uint8)
        debris_count = 0
        for c in debris_contours:
            if cv2.contourArea(c) < max_debris_area:
                cv2.drawContours(debris_mask, [c], -1, 255, -1)
                debris_count += 1
        if debris_count > 0:
            current = cv2.inpaint(current, debris_mask, 3, cv2.INPAINT_TELEA)
        metadata["stages_applied"].append("debris_removal")
        metadata["debris_removed"] = debris_count

        # Stage 3: CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size
        )
        current = clahe.apply(current)
        metadata["stages_applied"].append("clahe_enhancement")

        # Stage 4: Bilateral filter
        current = cv2.bilateralFilter(
            current,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )
        metadata["stages_applied"].append("bilateral_smoothing")
        metadata["denoised_stats"] = {
            "mean": float(np.mean(current)),
            "std": float(np.std(current)),
        }
        return current, metadata

    # -- Chromosome Straightening ---------------------------------------
    def straighten_chromosome(
        self, gray: np.ndarray, contour, box: Tuple
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Straighten a single curved chromosome using medial axis transform."""
        x, y, w, h = box
        if w < 3 or h < 3:
            return np.zeros((1, self.straighten_width), dtype=np.uint8), np.array([])

        roi = gray[y : y + h, x : x + w].copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted], -1, 255, -1)

        skeleton = extract_skeleton(mask)
        if np.sum(skeleton) == 0:
            return roi, np.array([])

        ordered_points = order_skeleton_points(skeleton)
        if len(ordered_points) < 3:
            return roi, np.array([])

        smoothed = smooth_curve(
            ordered_points, num_points=max(10, len(ordered_points))
        )

        half_width = self.straighten_width // 2
        strip = np.zeros((len(smoothed), self.straighten_width), dtype=np.uint8)
        for i in range(len(smoothed)):
            cx, cy = smoothed[i]
            if i == 0:
                dx = smoothed[1][0] - smoothed[0][0]
                dy = smoothed[1][1] - smoothed[0][1]
            elif i == len(smoothed) - 1:
                dx = smoothed[-1][0] - smoothed[-2][0]
                dy = smoothed[-1][1] - smoothed[-2][1]
            else:
                dx = smoothed[i + 1][0] - smoothed[i - 1][0]
                dy = smoothed[i + 1][1] - smoothed[i - 1][1]

            length = np.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                continue
            nx, ny = -dy / length, dx / length

            for j in range(-half_width, half_width):
                sx = int(round(cx + j * nx))
                sy = int(round(cy + j * ny))
                if 0 <= sx < w and 0 <= sy < h:
                    strip[i, j + half_width] = roi[sy, sx]

        axis_global = smoothed + np.array([x, y])
        return strip, axis_global

    def straighten_all(
        self, gray: np.ndarray, contours: List, boxes: List[Tuple]
    ) -> List[Dict]:
        """Straighten all detected chromosomes."""
        results = []
        for contour, box in zip(contours, boxes):
            strip, axis_pts = self.straighten_chromosome(gray, contour, box)
            results.append(
                {
                    "strip": strip,
                    "medial_axis": axis_pts,
                    "curvature": compute_curvature(axis_pts)
                    if len(axis_pts) > 2
                    else 0.0,
                }
            )
        return results
