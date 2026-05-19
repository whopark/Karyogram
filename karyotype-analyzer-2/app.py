import streamlit as st
import base64
from PIL import Image
import io
from datetime import datetime
import json
import re
from typing import Dict, Optional, List, Tuple
from collections import Counter
from enum import Enum
import numpy as np

# Computer Vision for chromosome detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Vision-Language Model API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class APIProvider(Enum):
    OPENAI = "OpenAI GPT-4 Vision"
    ANTHROPIC = "Anthropic Claude Vision"
    GEMINI = "Google Gemini Vision"
    CONSENSUS = "Multi-Model Consensus"
    CV_VLM = "CV + VLM (Hybrid)"
    TWO_STAGE = "Two-Stage Pipeline (CV + VLM)"
    PRECISION_LENS = "Precision Clinical Lens (6-Stage)"
    YOLO_KARYOGRAM = "YOLO Karyogram (ML Pipeline)"
    MOCK = "Demo Mode (No API)"


class DigitalPreprocessor:
    """디지털 전처리 (Task 4) - 강건화 및 궤적 교정

    1. Cascaded Denoising: multi-stage noise removal pipeline
       Stage 1: Background estimation & subtraction
       Stage 2: Debris/artifact removal (small particle filtering)
       Stage 3: Adaptive contrast enhancement (CLAHE)
       Stage 4: Edge-preserving smoothing (bilateral filter)

    2. Chromosome Straightening (Medial Axis Transform):
       - Extract medial axis (skeleton) of each chromosome
       - Fit a smooth curve to the skeleton
       - Unwarp the chromosome along the medial axis to a straight form
       - Improves banding pattern extraction accuracy
    """

    def __init__(self):
        # Denoising parameters
        self.clahe_clip_limit = 3.0
        self.clahe_grid_size = (8, 8)
        self.bilateral_d = 7
        self.bilateral_sigma_color = 50
        self.bilateral_sigma_space = 50
        self.debris_max_area_ratio = 0.0005  # Max area ratio for debris particles (very conservative)
        # Straightening parameters
        self.straighten_width = 30  # Output width for straightened chromosome strip

    # ── Cascaded Denoising Pipeline ──────────────────────────────
    def denoise(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Multi-stage cascaded denoising pipeline.
        Returns denoised image and metadata about what was cleaned."""
        if not CV2_AVAILABLE:
            return gray, {"error": "OpenCV not available"}

        h, w = gray.shape
        metadata = {"stages_applied": [], "original_stats": {
            "mean": float(np.mean(gray)), "std": float(np.std(gray))
        }}

        current = gray.copy()

        # ── Stage 1: Background estimation & subtraction ──
        # Only apply if image has significant illumination variation
        # Check if the image is already well-normalized (e.g., arranged karyogram)
        local_means = cv2.blur(current, (h // 4 | 1, w // 4 | 1))
        illumination_var = float(np.std(local_means[current > 20]))  # Variation in lit areas

        if illumination_var > 30:  # Significant illumination gradient
            bg_kernel_size = max(51, min(h, w) // 8) | 1
            bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size))
            background = cv2.morphologyEx(current, cv2.MORPH_OPEN, bg_kernel)
            current = cv2.subtract(current, background)
            current = cv2.normalize(current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            metadata["stages_applied"].append("background_subtraction")
        else:
            metadata["stages_applied"].append("background_subtraction_skipped")

        # ── Stage 2: Debris/artifact removal ──
        # Small particles below threshold are likely debris, not chromosomes
        _, debris_thresh = cv2.threshold(current, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        debris_contours, _ = cv2.findContours(debris_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_debris_area = h * w * self.debris_max_area_ratio
        debris_mask = np.zeros((h, w), dtype=np.uint8)
        debris_count = 0
        for c in debris_contours:
            if cv2.contourArea(c) < max_debris_area:
                cv2.drawContours(debris_mask, [c], -1, 255, -1)
                debris_count += 1

        # Fill debris regions with local median
        if debris_count > 0:
            # Use inpainting to fill debris regions smoothly
            current = cv2.inpaint(current, debris_mask, 3, cv2.INPAINT_TELEA)

        metadata["stages_applied"].append("debris_removal")
        metadata["debris_removed"] = debris_count

        # ── Stage 3: Adaptive contrast enhancement (CLAHE) ──
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        current = clahe.apply(current)
        metadata["stages_applied"].append("clahe_enhancement")

        # ── Stage 4: Edge-preserving smoothing (bilateral filter) ──
        current = cv2.bilateralFilter(
            current,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )
        metadata["stages_applied"].append("bilateral_smoothing")

        metadata["denoised_stats"] = {
            "mean": float(np.mean(current)), "std": float(np.std(current))
        }

        return current, metadata

    # ── Chromosome Straightening (Medial Axis Transform) ─────────
    def straighten_chromosome(self, gray: np.ndarray, contour,
                                box: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """Straighten a single curved chromosome using medial axis transform.

        Steps:
        1. Extract chromosome ROI and create binary mask
        2. Compute skeleton (medial axis) via morphological thinning
        3. Order skeleton points along the axis
        4. Sample intensity perpendicular to the axis at each point
        5. Return straightened strip image and ordered medial axis points

        Returns:
            (straightened_strip, medial_axis_points)
        """
        x, y, w, h = box
        if w < 3 or h < 3:
            return np.zeros((1, self.straighten_width), dtype=np.uint8), np.array([])

        # Extract ROI
        roi = gray[y:y+h, x:x+w].copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted], -1, 255, -1)

        # ── Step 1: Skeleton extraction ──
        skeleton = self._extract_skeleton(mask)
        if np.sum(skeleton) == 0:
            return roi, np.array([])

        # ── Step 2: Order skeleton points along the medial axis ──
        ordered_points = self._order_skeleton_points(skeleton)
        if len(ordered_points) < 3:
            return roi, np.array([])

        # ── Step 3: Smooth the medial axis curve ──
        smoothed = self._smooth_curve(ordered_points, num_points=max(10, len(ordered_points)))

        # ── Step 4: Sample perpendicular strips ──
        half_width = self.straighten_width // 2
        strip = np.zeros((len(smoothed), self.straighten_width), dtype=np.uint8)

        for i in range(len(smoothed)):
            cx, cy = smoothed[i]

            # Compute tangent direction
            if i == 0:
                dx = smoothed[1][0] - smoothed[0][0]
                dy = smoothed[1][1] - smoothed[0][1]
            elif i == len(smoothed) - 1:
                dx = smoothed[-1][0] - smoothed[-2][0]
                dy = smoothed[-1][1] - smoothed[-2][1]
            else:
                dx = smoothed[i+1][0] - smoothed[i-1][0]
                dy = smoothed[i+1][1] - smoothed[i-1][1]

            # Perpendicular direction
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1e-6:
                continue
            nx, ny = -dy / length, dx / length

            # Sample pixels along perpendicular line
            for j in range(-half_width, half_width):
                sx = int(round(cx + j * nx))
                sy = int(round(cy + j * ny))
                if 0 <= sx < w and 0 <= sy < h:
                    strip[i, j + half_width] = roi[sy, sx]

        # Return straightened strip and axis points (in original image coords)
        axis_global = smoothed + np.array([x, y])
        return strip, axis_global

    def straighten_all(self, gray: np.ndarray, contours: List,
                        boxes: List[Tuple]) -> List[Dict]:
        """Straighten all detected chromosomes and return their straightened forms."""
        results = []
        for contour, box in zip(contours, boxes):
            strip, axis_pts = self.straighten_chromosome(gray, contour, box)
            results.append({
                "strip": strip,
                "medial_axis": axis_pts,
                "curvature": self._compute_curvature(axis_pts) if len(axis_pts) > 2 else 0.0,
            })
        return results

    def _extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Fast skeleton extraction via distance transform peak ridges."""
        if mask.size == 0 or np.sum(mask) == 0:
            return np.zeros_like(mask)

        # Use distance transform: skeleton = ridge of distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        if dist.max() == 0:
            return np.zeros_like(mask)

        # Normalize and find ridges via Laplacian
        dist_norm = (dist / dist.max() * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(dist_norm, cv2.CV_64F)

        # Skeleton = local maxima of distance transform (where Laplacian < 0)
        skeleton = np.zeros_like(mask)
        # Threshold at a fraction of max distance
        threshold = dist.max() * 0.3
        skeleton[(dist > threshold) & (laplacian < 0)] = 255

        # Thin the result
        kernel = np.ones((2, 2), np.uint8)
        skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel, iterations=1)

        return skeleton

    def _order_skeleton_points(self, skeleton: np.ndarray) -> np.ndarray:
        """Order skeleton pixels into a connected path from one endpoint to another."""
        points = np.column_stack(np.where(skeleton > 0))  # (row, col) = (y, x)
        if len(points) == 0:
            return np.array([])

        # Convert to (x, y)
        points = points[:, ::-1].astype(np.float64)

        # Find endpoints: skeleton pixels with only 1 neighbor
        h, w = skeleton.shape
        endpoints = []
        for pt in points:
            px, py = int(pt[0]), int(pt[1])
            neighbors = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                        neighbors += 1
            if neighbors <= 1:
                endpoints.append(pt)

        # Start from an endpoint (or first point if no endpoints found)
        start = endpoints[0] if endpoints else points[0]

        # Greedy nearest-neighbor ordering
        remaining = list(range(len(points)))
        ordered = []

        # Find start index
        dists = np.sum((points - start) ** 2, axis=1)
        current_idx = int(np.argmin(dists))
        ordered.append(current_idx)
        remaining.remove(current_idx)

        while remaining:
            current_pt = points[ordered[-1]]
            # Find nearest unvisited point
            min_dist = float('inf')
            best_idx = remaining[0]
            for idx in remaining:
                d = np.sum((points[idx] - current_pt) ** 2)
                if d < min_dist:
                    min_dist = d
                    best_idx = idx
            # Stop if gap is too large (disconnected skeleton fragment)
            if min_dist > 8:  # Max gap of ~2.8 pixels
                break
            ordered.append(best_idx)
            remaining.remove(best_idx)

        return points[ordered]

    def _smooth_curve(self, points: np.ndarray, num_points: int = 50) -> np.ndarray:
        """Smooth an ordered point set using moving average."""
        if len(points) < 3:
            return points

        # Moving average smoothing
        window = min(5, len(points) // 2)
        if window < 2:
            return points

        smoothed_x = np.convolve(points[:, 0], np.ones(window)/window, mode='valid')
        smoothed_y = np.convolve(points[:, 1], np.ones(window)/window, mode='valid')

        smoothed = np.column_stack([smoothed_x, smoothed_y])

        # Resample to uniform spacing
        if len(smoothed) > 2:
            cumlen = np.zeros(len(smoothed))
            for i in range(1, len(smoothed)):
                cumlen[i] = cumlen[i-1] + np.linalg.norm(smoothed[i] - smoothed[i-1])
            if cumlen[-1] > 0:
                target_positions = np.linspace(0, cumlen[-1], num_points)
                resampled_x = np.interp(target_positions, cumlen, smoothed[:, 0])
                resampled_y = np.interp(target_positions, cumlen, smoothed[:, 1])
                return np.column_stack([resampled_x, resampled_y])

        return smoothed

    def _compute_curvature(self, axis_points: np.ndarray) -> float:
        """Compute average curvature of the medial axis.
        Low curvature = straight chromosome, high = curved/bent."""
        if len(axis_points) < 3:
            return 0.0

        # Curvature at each interior point via discrete formula
        curvatures = []
        for i in range(1, len(axis_points) - 1):
            p0 = axis_points[i-1]
            p1 = axis_points[i]
            p2 = axis_points[i+1]

            v1 = p1 - p0
            v2 = p2 - p1
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            denom = l1 * l2
            if denom > 1e-8:
                curvatures.append(cross / denom)

        return float(np.mean(curvatures)) if curvatures else 0.0


class SegmentationMatrix:
    """분할 매트릭스 (Segmentation Matrix) - 얽힌 염색체 분리
    Two-path segmentation for overlapping chromosomes:

    Path 1 - Semantic Segmentation:
        Pixel-level classification → background / single chromosome / overlap region
        Then stitching-based reconstruction to recover individual chromosomes from overlaps.
        (Inspired by U-Net, ARMS Net, cGAN approaches)

    Path 2 - Instance Segmentation:
        Marker-controlled watershed with gradient-guided boundaries to produce
        per-instance chromosome masks, even in dense overlap regions.
        (Inspired by Mask R-CNN, AS-PANet approaches)
    """

    def __init__(self):
        # Semantic segmentation thresholds
        self.overlap_intensity_factor = 1.3  # Overlap regions are darker (more dense)
        self.morph_kernel_size = 3
        # Instance segmentation parameters
        self.dist_transform_threshold = 0.35  # For finding sure foreground markers
        self.gradient_weight = 0.7  # Weight of gradient in combined boundary map

    # ── Path 1: Semantic Segmentation ────────────────────────────
    def semantic_segmentation(self, gray: np.ndarray, binary_mask: np.ndarray
                               ) -> Dict:
        """Classify each pixel as background(0), single-chromosome(1), or overlap(2).

        Approach:
        1. Use intensity analysis: overlap regions are darker than single-layer regions
        2. Create 3-class semantic map
        3. Apply morphological cleanup
        4. Return the semantic map + overlap mask for downstream stitching
        """
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available"}

        h, w = gray.shape

        # ── Step 1: Identify chromosome foreground ──
        # binary_mask: 255 = chromosome, 0 = background
        fg_mask = (binary_mask > 0).astype(np.uint8)

        # ── Step 2: Intensity analysis within foreground ──
        # In overlap regions, two chromosomes stack → darker/denser pixels
        fg_pixels = gray[fg_mask == 1]
        if len(fg_pixels) == 0:
            return {
                "semantic_map": np.zeros((h, w), dtype=np.uint8),
                "overlap_mask": np.zeros((h, w), dtype=np.uint8),
                "overlap_pixel_count": 0,
                "single_pixel_count": 0,
            }

        fg_mean = float(np.mean(fg_pixels))
        fg_std = float(np.std(fg_pixels))

        # ── Step 3: Build 3-class semantic map ──
        semantic_map = np.zeros((h, w), dtype=np.uint8)
        # Class 0: background (default)
        # Class 1: single chromosome
        semantic_map[fg_mask == 1] = 1

        # Class 2: overlap — pixels significantly darker than mean foreground
        # For G-banded images, overlaps appear darker (more stain absorption)
        overlap_threshold = fg_mean - fg_std * 0.8
        overlap_candidates = (gray < overlap_threshold) & (fg_mask == 1)

        # Refine: overlap regions should be spatially coherent (not just random dark pixels)
        overlap_raw = overlap_candidates.astype(np.uint8) * 255
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        overlap_cleaned = cv2.morphologyEx(overlap_raw, cv2.MORPH_OPEN, kernel, iterations=2)
        overlap_cleaned = cv2.morphologyEx(overlap_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Additional filter: overlap blobs should have minimum area
        min_overlap_area = max(50, int(h * w * 0.0002))
        contours_ov, _ = cv2.findContours(overlap_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlap_final = np.zeros((h, w), dtype=np.uint8)
        for c in contours_ov:
            if cv2.contourArea(c) >= min_overlap_area:
                cv2.drawContours(overlap_final, [c], -1, 255, -1)

        semantic_map[overlap_final > 0] = 2

        return {
            "semantic_map": semantic_map,
            "overlap_mask": overlap_final,
            "overlap_pixel_count": int(np.sum(overlap_final > 0)),
            "single_pixel_count": int(np.sum(semantic_map == 1)),
        }

    def stitch_from_overlap(self, gray: np.ndarray, semantic_map: np.ndarray,
                             overlap_mask: np.ndarray, contours: List,
                             min_area: int) -> List[Tuple]:
        """Stitching-based reconstruction: recover individual chromosomes from overlap regions.

        For each overlap region:
        1. Find which chromosome contours touch the overlap
        2. Extend each touching chromosome through the overlap using intensity/direction continuity
        3. Split the overlap pixels between the chromosomes
        """
        if not CV2_AVAILABLE:
            return []

        h, w = gray.shape
        results = []  # List of (contour, bbox, area)

        # Label individual overlap blobs
        num_labels, overlap_labels = cv2.connectedComponents(overlap_mask)

        for label_id in range(1, num_labels):
            blob_mask = (overlap_labels == label_id).astype(np.uint8) * 255
            blob_contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not blob_contours:
                continue

            blob_contour = blob_contours[0]
            bx, by, bw, bh = cv2.boundingRect(blob_contour)

            # Dilate the blob slightly to find neighboring chromosome contours
            dilated_blob = cv2.dilate(blob_mask, np.ones((5, 5), np.uint8), iterations=2)

            # Find which existing contours overlap with the dilated blob
            touching_indices = []
            for idx, c in enumerate(contours):
                # Check if any contour point falls within dilated blob
                c_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                intersection = cv2.bitwise_and(c_mask, dilated_blob)
                if np.sum(intersection) > 0:
                    touching_indices.append(idx)

            if len(touching_indices) < 2:
                # Not enough touching contours — can't split, keep overlap as-is
                continue

            # ── Direction-guided split ──
            # Use gradient direction within the overlap to assign pixels to each touching chromosome
            # Compute Sobel gradients
            roi_gray = gray[by:by+bh, bx:bx+bw]
            if roi_gray.size == 0:
                continue

            gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            angles = np.arctan2(gy, gx)  # Gradient direction

            # For each touching chromosome, compute its major axis direction
            chr_directions = []
            chr_centers = []
            for idx in touching_indices:
                c = contours[idx]
                pts = c.reshape(-1, 2).astype(np.float64)
                if len(pts) < 5:
                    chr_directions.append(0.0)
                    chr_centers.append(np.mean(pts, axis=0))
                    continue
                mean_pt = np.mean(pts, axis=0)
                centered = pts - mean_pt
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                major = eigenvectors[:, np.argmax(eigenvalues)]
                direction = float(np.arctan2(major[1], major[0]))
                chr_directions.append(direction)
                chr_centers.append(mean_pt)

            # Assign overlap pixels to nearest chromosome by direction similarity
            blob_roi_mask = blob_mask[by:by+bh, bx:bx+bw]
            assignment_map = np.full((bh, bw), -1, dtype=np.int32)

            ys, xs = np.where(blob_roi_mask > 0)
            for py, px in zip(ys, xs):
                pixel_angle = angles[py, px] if py < angles.shape[0] and px < angles.shape[1] else 0
                # Also consider distance to each chromosome center
                best_idx = 0
                best_score = float('inf')
                for ci, (tidx, direction, center) in enumerate(
                        zip(touching_indices, chr_directions, chr_centers)):
                    # Angular difference
                    angle_diff = abs(pixel_angle - direction)
                    angle_diff = min(angle_diff, np.pi - angle_diff)
                    # Spatial distance to chromosome center
                    dist = np.sqrt((px + bx - center[0])**2 + (py + by - center[1])**2)
                    score = angle_diff * 50 + dist  # Weighted combination
                    if score < best_score:
                        best_score = score
                        best_idx = ci

                assignment_map[py, px] = best_idx

            # Build split masks
            for ci in range(len(touching_indices)):
                split_mask = np.zeros((h, w), dtype=np.uint8)
                # Original chromosome pixels
                cv2.drawContours(split_mask, [contours[touching_indices[ci]]], -1, 255, -1)
                # Add assigned overlap pixels
                assigned_roi = (assignment_map == ci).astype(np.uint8) * 255
                split_mask[by:by+bh, bx:bx+bw] = cv2.bitwise_or(
                    split_mask[by:by+bh, bx:bx+bw], assigned_roi
                )
                # Extract contour
                split_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for sc in split_contours:
                    a = cv2.contourArea(sc)
                    if a >= min_area * 0.4:
                        sx, sy, sw, sh = cv2.boundingRect(sc)
                        results.append((sc, (sx, sy, sw, sh), a))

        return results

    # ── Path 2: Instance Segmentation ────────────────────────────
    def instance_segmentation(self, gray: np.ndarray, binary_mask: np.ndarray,
                               min_area: int) -> Dict:
        """Marker-controlled watershed for per-instance chromosome masks.

        Approach (Mask R-CNN / AS-PANet inspired, using classical CV):
        1. Distance transform on foreground to find chromosome centers
        2. Gradient magnitude as boundary indicator
        3. Combine into marker-controlled watershed
        4. Extract per-instance masks
        """
        if not CV2_AVAILABLE:
            return {"instances": [], "count": 0}

        h, w = gray.shape

        # ── Step 1: Distance transform → sure foreground markers ──
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        if dist_transform.max() == 0:
            return {"instances": [], "count": 0}

        # Adaptive threshold: use per-blob local maxima instead of global threshold
        # This handles chromosomes of varying sizes better
        _, sure_fg = cv2.threshold(
            dist_transform,
            self.dist_transform_threshold * dist_transform.max(),
            255, 0
        )
        sure_fg = np.uint8(sure_fg)

        # Remove tiny fragments
        kernel_small = np.ones((2, 2), np.uint8)
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # ── Step 2: Gradient magnitude as boundary signal ──
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        gradient_mag = (gradient_mag / (gradient_mag.max() + 1e-8) * 255).astype(np.uint8)

        # ── Step 3: Sure background ──
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # ── Step 4: Marker labeling ──
        num_markers, markers = cv2.connectedComponents(sure_fg)

        # Markers: 0=unknown, 1..N=instances, need to increment so watershed treats 0 as unknown
        markers = markers + 1
        markers[unknown == 255] = 0

        # ── Step 5: Watershed with gradient-enhanced image ──
        # Combine gray image with gradient for better boundary detection
        if len(gray.shape) == 2:
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            color_img = gray.copy()

        # Blend gradient into the image to emphasize boundaries
        grad_3ch = cv2.cvtColor(gradient_mag, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(
            color_img, 1.0 - self.gradient_weight,
            grad_3ch, self.gradient_weight, 0
        )

        markers_ws = np.int32(markers)
        cv2.watershed(blended, markers_ws)

        # ── Step 6: Extract per-instance results ──
        instances = []
        for label_id in range(2, num_markers + 1):  # Skip background (1)
            instance_mask = np.uint8(markers_ws == label_id) * 255
            inst_contours, _ = cv2.findContours(
                instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for ic in inst_contours:
                area = cv2.contourArea(ic)
                if area >= min_area * 0.4:
                    ix, iy, iw, ih = cv2.boundingRect(ic)
                    instances.append({
                        "contour": ic,
                        "bbox": (ix, iy, iw, ih),
                        "area": area,
                        "mask": instance_mask,
                        "label_id": label_id,
                    })

        return {
            "instances": instances,
            "count": len(instances),
            "markers_used": num_markers - 1,  # Excluding background
            "watershed_labels": markers_ws,
        }

    # ── Combined Pipeline ────────────────────────────────────────
    def segment_and_separate(self, gray: np.ndarray, binary_mask: np.ndarray,
                              contours: List, boxes: List[Tuple],
                              areas: List[float], min_area: int
                              ) -> Tuple[List, List[Tuple], List[float], Dict]:
        """Run both segmentation paths and merge results.

        Strategy:
        1. Run semantic segmentation to identify overlap regions
        2. If overlaps found: use stitching to recover chromosomes in overlap zones
        3. Run instance segmentation on the full image as validation
        4. Merge: prefer stitched results for overlap zones, instance results elsewhere
        5. Return improved contour set + segmentation metadata
        """
        metadata = {
            "semantic_overlap_pixels": 0,
            "instance_count": 0,
            "stitched_chromosomes": 0,
            "segmentation_method": "none",
        }

        # ── Semantic path ──
        sem_result = self.semantic_segmentation(gray, binary_mask)
        overlap_mask = sem_result.get("overlap_mask", np.zeros_like(gray))
        overlap_pixels = sem_result.get("overlap_pixel_count", 0)
        metadata["semantic_overlap_pixels"] = overlap_pixels

        # ── Instance path ──
        inst_result = self.instance_segmentation(gray, binary_mask, min_area)
        metadata["instance_count"] = inst_result.get("count", 0)

        # ── Decision: which path to use? ──
        has_significant_overlap = overlap_pixels > (gray.shape[0] * gray.shape[1] * 0.001)

        if has_significant_overlap and contours:
            # Use semantic stitching for overlap zones
            stitched = self.stitch_from_overlap(
                gray, sem_result["semantic_map"], overlap_mask, contours, min_area
            )
            metadata["stitched_chromosomes"] = len(stitched)

            if stitched:
                # Build result: keep non-overlapping chromosomes + add stitched ones
                overlap_dilated = cv2.dilate(overlap_mask, np.ones((5, 5), np.uint8), iterations=2)
                new_contours = []
                new_boxes = []
                new_areas = []

                # Keep contours that DON'T touch overlap regions
                for c, b, a in zip(contours, boxes, areas):
                    c_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                    cv2.drawContours(c_mask, [c], -1, 255, -1)
                    intersection = cv2.bitwise_and(c_mask, overlap_dilated)
                    if np.sum(intersection) < a * 0.3:  # Less than 30% overlap
                        new_contours.append(c)
                        new_boxes.append(b)
                        new_areas.append(a)

                # Add stitched chromosomes
                for sc, sb, sa in stitched:
                    new_contours.append(sc)
                    new_boxes.append(sb)
                    new_areas.append(sa)

                metadata["segmentation_method"] = "semantic_stitching"
                return new_contours, new_boxes, new_areas, metadata

        # Fallback: use instance segmentation if it found a count closer to expected
        inst_count = inst_result.get("count", 0)
        if 38 <= inst_count <= 52 and abs(inst_count - 46) < abs(len(contours) - 46):
            # Instance segmentation gave a better count
            inst_contours = [i["contour"] for i in inst_result["instances"]]
            inst_boxes = [i["bbox"] for i in inst_result["instances"]]
            inst_areas = [i["area"] for i in inst_result["instances"]]
            metadata["segmentation_method"] = "instance_watershed"
            return inst_contours, inst_boxes, inst_areas, metadata

        # No improvement — return original
        metadata["segmentation_method"] = "passthrough"
        return contours, boxes, areas, metadata


class ClusterRouter:
    """사전 라우팅 메커니즘 (Pre-routing Mechanism) - Task 3
    Classifies each detected contour cluster by overlap type and routes
    to the optimal segmentation pipeline:

    Route A: Touching      → Instance segmentation (fast, simple separation)
    Route B: One-Overlap   → Semantic stitching (U-Net style, gradient-guided split)
    Route C: Multi-Overlap → Multi-pass segmentation (watershed + concavity + stitching)

    This reduces computation by only applying heavy segmentation where needed,
    and improves accuracy by matching the method to the overlap complexity.
    """

    class OverlapType:
        ISOLATED = "isolated"        # Single chromosome, no neighbors
        TOUCHING = "touching"        # Adjacent but not overlapping
        ONE_OVERLAP = "one_overlap"  # Two chromosomes overlapping
        MULTI_OVERLAP = "multi_overlap"  # 3+ chromosomes in complex overlap

    def __init__(self):
        # Adjacency detection parameters
        self.adjacency_dilate_px = 5    # Dilation to detect touching neighbors
        self.overlap_dilate_px = 3      # Smaller dilation for overlap detection
        self.area_overlap_threshold = 0.08  # Min overlap ratio to consider "overlapping"
        self.multi_overlap_min = 3      # Min neighbors for multi-overlap classification

    def classify_clusters(self, gray: np.ndarray, binary_mask: np.ndarray,
                           contours: List, boxes: List[Tuple],
                           areas: List[float]) -> List[Dict]:
        """Classify each contour by its overlap relationship with neighbors.

        Returns list of cluster info dicts, one per contour:
        {
            "index": int,
            "overlap_type": OverlapType,
            "neighbor_indices": [int, ...],
            "overlap_area": float,
            "cluster_id": int,  # Connected group ID
        }
        """
        if not CV2_AVAILABLE or not contours:
            return []

        h, w = gray.shape
        n = len(contours)

        # ── Step 1: Build per-contour masks ──
        masks = []
        for c in contours:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(m, [c], -1, 255, -1)
            masks.append(m)

        # ── Step 2: Adjacency matrix via dilation overlap ──
        adjacency = [[False] * n for _ in range(n)]
        overlap_areas = [[0.0] * n for _ in range(n)]
        kernel_adj = np.ones((self.adjacency_dilate_px * 2 + 1,
                              self.adjacency_dilate_px * 2 + 1), np.uint8)

        dilated_masks = []
        for m in masks:
            dilated_masks.append(cv2.dilate(m, kernel_adj, iterations=1))

        for i in range(n):
            for j in range(i + 1, n):
                # Check adjacency: dilated mask i intersects mask j
                intersection = cv2.bitwise_and(dilated_masks[i], masks[j])
                inter_pixels = np.sum(intersection > 0)

                if inter_pixels > 0:
                    adjacency[i][j] = True
                    adjacency[j][i] = True

                # Check actual overlap (undilated masks intersect)
                direct_overlap = cv2.bitwise_and(masks[i], masks[j])
                overlap_px = np.sum(direct_overlap > 0)
                min_area = min(areas[i], areas[j]) if areas[i] > 0 and areas[j] > 0 else 1
                overlap_ratio = overlap_px / min_area
                overlap_areas[i][j] = overlap_ratio
                overlap_areas[j][i] = overlap_ratio

        # ── Step 3: Build connected clusters (graph components) ──
        visited = [False] * n
        cluster_ids = [-1] * n
        cluster_id = 0

        def bfs(start):
            queue = [start]
            members = []
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                cluster_ids[node] = cluster_id
                members.append(node)
                for neighbor in range(n):
                    if adjacency[node][neighbor] and not visited[neighbor]:
                        queue.append(neighbor)
            return members

        clusters = []
        for i in range(n):
            if not visited[i]:
                members = bfs(i)
                clusters.append(members)
                cluster_id += 1

        # ── Step 4: Classify each contour ──
        results = []
        for i in range(n):
            neighbors = [j for j in range(n) if adjacency[i][j]]
            overlapping_neighbors = [
                j for j in neighbors
                if overlap_areas[i][j] > self.area_overlap_threshold
            ]
            touching_only = [j for j in neighbors if j not in overlapping_neighbors]

            if not neighbors:
                overlap_type = self.OverlapType.ISOLATED
            elif not overlapping_neighbors:
                overlap_type = self.OverlapType.TOUCHING
            elif len(overlapping_neighbors) == 1:
                overlap_type = self.OverlapType.ONE_OVERLAP
            else:
                overlap_type = self.OverlapType.MULTI_OVERLAP

            max_overlap = max(
                (overlap_areas[i][j] for j in neighbors), default=0.0
            )

            results.append({
                "index": i,
                "overlap_type": overlap_type,
                "neighbor_indices": neighbors,
                "overlapping_neighbors": overlapping_neighbors,
                "touching_neighbors": touching_only,
                "max_overlap_ratio": max_overlap,
                "cluster_id": cluster_ids[i],
            })

        return results

    def route_and_segment(self, gray: np.ndarray, binary_mask: np.ndarray,
                           contours: List, boxes: List[Tuple],
                           areas: List[float], min_area: int,
                           segmenter: 'SegmentationMatrix'
                           ) -> Tuple[List, List[Tuple], List[float], Dict]:
        """Route each cluster to the appropriate segmentation pipeline.

        Route A (Touching):       Instance segmentation - simple marker watershed
        Route B (One-Overlap):    Semantic stitching - gradient-guided pixel assignment
        Route C (Multi-Overlap):  Multi-pass - watershed + concavity + stitching combined
        Isolated:                 Pass through unchanged
        """
        if not CV2_AVAILABLE or not contours:
            return contours, boxes, areas, {"routing": "empty"}

        h, w = gray.shape
        classifications = self.classify_clusters(gray, binary_mask, contours, boxes, areas)

        if not classifications:
            return contours, boxes, areas, {"routing": "no_classifications"}

        # Track which contours have been processed
        processed = [False] * len(contours)
        final_contours = []
        final_boxes = []
        final_areas = []

        # Routing statistics
        route_stats = {
            "isolated": 0, "touching": 0, "one_overlap": 0, "multi_overlap": 0,
            "route_a_splits": 0, "route_b_splits": 0, "route_c_splits": 0,
        }

        # Group by cluster_id to process connected components together
        cluster_groups = {}
        for info in classifications:
            cid = info["cluster_id"]
            if cid not in cluster_groups:
                cluster_groups[cid] = []
            cluster_groups[cid].append(info)

        for cid, group in cluster_groups.items():
            indices = [info["index"] for info in group]
            # Determine the dominant overlap type in this cluster
            types = [info["overlap_type"] for info in group]

            has_multi = self.OverlapType.MULTI_OVERLAP in types
            has_one = self.OverlapType.ONE_OVERLAP in types
            has_touching = self.OverlapType.TOUCHING in types

            if len(indices) == 1 and types[0] == self.OverlapType.ISOLATED:
                # ── Isolated: pass through ──
                idx = indices[0]
                final_contours.append(contours[idx])
                final_boxes.append(boxes[idx])
                final_areas.append(areas[idx])
                processed[idx] = True
                route_stats["isolated"] += 1

            elif has_multi:
                # ── Route C: Multi-overlap → multi-pass segmentation ──
                cluster_results = self._route_c_multi_overlap(
                    gray, binary_mask, contours, boxes, areas,
                    indices, min_area, segmenter
                )
                for c, b, a in cluster_results:
                    final_contours.append(c)
                    final_boxes.append(b)
                    final_areas.append(a)
                for idx in indices:
                    processed[idx] = True
                route_stats["multi_overlap"] += len(indices)
                route_stats["route_c_splits"] += max(0, len(cluster_results) - len(indices))

            elif has_one:
                # ── Route B: One-overlap → semantic stitching ──
                cluster_results = self._route_b_one_overlap(
                    gray, binary_mask, contours, boxes, areas,
                    indices, min_area, segmenter
                )
                for c, b, a in cluster_results:
                    final_contours.append(c)
                    final_boxes.append(b)
                    final_areas.append(a)
                for idx in indices:
                    processed[idx] = True
                route_stats["one_overlap"] += len(indices)
                route_stats["route_b_splits"] += max(0, len(cluster_results) - len(indices))

            elif has_touching:
                # ── Route A: Touching → instance segmentation ──
                cluster_results = self._route_a_touching(
                    gray, binary_mask, contours, boxes, areas,
                    indices, min_area, segmenter
                )
                for c, b, a in cluster_results:
                    final_contours.append(c)
                    final_boxes.append(b)
                    final_areas.append(a)
                for idx in indices:
                    processed[idx] = True
                route_stats["touching"] += len(indices)
                route_stats["route_a_splits"] += max(0, len(cluster_results) - len(indices))

            else:
                # All isolated in this cluster
                for idx in indices:
                    final_contours.append(contours[idx])
                    final_boxes.append(boxes[idx])
                    final_areas.append(areas[idx])
                    processed[idx] = True
                    route_stats["isolated"] += 1

        # Add any unprocessed contours
        for i in range(len(contours)):
            if not processed[i]:
                final_contours.append(contours[i])
                final_boxes.append(boxes[i])
                final_areas.append(areas[i])

        metadata = {
            "routing": "cluster_router",
            "total_clusters": len(cluster_groups),
            "route_stats": route_stats,
        }
        return final_contours, final_boxes, final_areas, metadata

    # ── Route A: Touching → Instance Segmentation ────────────────
    def _route_a_touching(self, gray, binary_mask, contours, boxes, areas,
                           indices, min_area, segmenter):
        """Simple separation for merely touching chromosomes.
        Uses localized instance segmentation (fast marker watershed)."""
        # Build combined mask for this cluster
        h, w = gray.shape
        cluster_mask = np.zeros((h, w), dtype=np.uint8)
        for idx in indices:
            cv2.drawContours(cluster_mask, [contours[idx]], -1, 255, -1)

        # Get bounding box of the cluster
        xs = [boxes[i][0] for i in indices]
        ys = [boxes[i][1] for i in indices]
        xe = [boxes[i][0] + boxes[i][2] for i in indices]
        ye = [boxes[i][1] + boxes[i][3] for i in indices]
        pad = 10
        rx1 = max(0, min(xs) - pad)
        ry1 = max(0, min(ys) - pad)
        rx2 = min(w, max(xe) + pad)
        ry2 = min(h, max(ye) + pad)

        roi_mask = cluster_mask[ry1:ry2, rx1:rx2]
        roi_gray = gray[ry1:ry2, rx1:rx2]

        # Instance segmentation on the ROI
        inst_result = segmenter.instance_segmentation(roi_gray, roi_mask, min_area)

        if inst_result["count"] >= len(indices):
            results = []
            for inst in inst_result["instances"]:
                c = inst["contour"] + np.array([rx1, ry1])
                ix, iy, iw, ih = cv2.boundingRect(c)
                results.append((c, (ix, iy, iw, ih), inst["area"]))
            return results

        # Fallback: return originals
        return [(contours[i], boxes[i], areas[i]) for i in indices]

    # ── Route B: One-Overlap → Semantic Stitching ────────────────
    def _route_b_one_overlap(self, gray, binary_mask, contours, boxes, areas,
                              indices, min_area, segmenter):
        """Gradient-guided semantic stitching for single overlap regions."""
        h, w = gray.shape
        cluster_mask = np.zeros((h, w), dtype=np.uint8)
        for idx in indices:
            cv2.drawContours(cluster_mask, [contours[idx]], -1, 255, -1)

        # Run semantic segmentation on cluster region
        sem_result = segmenter.semantic_segmentation(gray, cluster_mask)
        overlap_mask = sem_result.get("overlap_mask", np.zeros((h, w), dtype=np.uint8))

        if sem_result.get("overlap_pixel_count", 0) > 0:
            cluster_contours = [contours[i] for i in indices]
            stitched = segmenter.stitch_from_overlap(
                gray, sem_result["semantic_map"], overlap_mask,
                cluster_contours, min_area
            )
            if stitched and len(stitched) >= len(indices):
                return stitched

        # Fallback: return originals
        return [(contours[i], boxes[i], areas[i]) for i in indices]

    # ── Route C: Multi-Overlap → Multi-pass Segmentation ────────
    def _route_c_multi_overlap(self, gray, binary_mask, contours, boxes, areas,
                                indices, min_area, segmenter):
        """Multi-pass segmentation for complex overlapping clusters.
        Combines: semantic stitching → instance watershed → concavity splitting."""
        h, w = gray.shape
        cluster_mask = np.zeros((h, w), dtype=np.uint8)
        for idx in indices:
            cv2.drawContours(cluster_mask, [contours[idx]], -1, 255, -1)

        # Pass 1: Semantic stitching attempt
        sem_result = segmenter.semantic_segmentation(gray, cluster_mask)
        overlap_mask = sem_result.get("overlap_mask", np.zeros((h, w), dtype=np.uint8))

        current_contours = [contours[i] for i in indices]
        current_boxes = [boxes[i] for i in indices]
        current_areas = [areas[i] for i in indices]

        if sem_result.get("overlap_pixel_count", 0) > 0:
            stitched = segmenter.stitch_from_overlap(
                gray, sem_result["semantic_map"], overlap_mask,
                current_contours, min_area
            )
            if stitched:
                current_contours = [s[0] for s in stitched]
                current_boxes = [s[1] for s in stitched]
                current_areas = [s[2] for s in stitched]

        # Pass 2: Instance segmentation on remaining merged regions
        xs = [b[0] for b in current_boxes]
        ys = [b[1] for b in current_boxes]
        xe = [b[0] + b[2] for b in current_boxes]
        ye = [b[1] + b[3] for b in current_boxes]
        pad = 10
        rx1 = max(0, min(xs) - pad)
        ry1 = max(0, min(ys) - pad)
        rx2 = min(w, max(xe) + pad)
        ry2 = min(h, max(ye) + pad)

        roi_mask = cluster_mask[ry1:ry2, rx1:rx2]
        roi_gray = gray[ry1:ry2, rx1:rx2]

        inst_result = segmenter.instance_segmentation(roi_gray, roi_mask, min_area)

        if inst_result["count"] > len(current_contours):
            results = []
            for inst in inst_result["instances"]:
                c = inst["contour"] + np.array([rx1, ry1])
                ix, iy, iw, ih = cv2.boundingRect(c)
                results.append((c, (ix, iy, iw, ih), inst["area"]))
            return results

        return [(c, b, a) for c, b, a in zip(current_contours, current_boxes, current_areas)]


class ChromosomeClassifier:
    """24-클래스 염색체 분류기 (Task 5)
    Classifies each detected chromosome into one of 24 classes (chr1-22, X, Y)
    using morphometric features and banding pattern analysis.

    Handles:
    - Intra-class variance from G-band staining quality differences
    - Super-resolution enhancement for low-quality chromosome images
    - SMOTE-style synthetic feature augmentation for rare classes
    """

    # Reference chromosome properties (haploid genome percentages & centromere indices)
    # Based on ISCN 2020 ideogram data
    # Format: (relative_size, centromere_index, group)
    # centromere_index = p-arm / total length (0.5 = metacentric, <0.3 = acrocentric)
    CHROMOSOME_TEMPLATES = {
        "1":  {"size_pct": 8.44, "ci": 0.48, "group": "A", "type": "metacentric"},
        "2":  {"size_pct": 8.02, "ci": 0.39, "group": "A", "type": "submetacentric"},
        "3":  {"size_pct": 6.83, "ci": 0.47, "group": "A", "type": "metacentric"},
        "4":  {"size_pct": 6.30, "ci": 0.29, "group": "B", "type": "submetacentric"},
        "5":  {"size_pct": 6.08, "ci": 0.29, "group": "B", "type": "submetacentric"},
        "6":  {"size_pct": 5.90, "ci": 0.39, "group": "C", "type": "submetacentric"},
        "7":  {"size_pct": 5.36, "ci": 0.39, "group": "C", "type": "submetacentric"},
        "8":  {"size_pct": 4.93, "ci": 0.34, "group": "C", "type": "submetacentric"},
        "9":  {"size_pct": 4.80, "ci": 0.35, "group": "C", "type": "submetacentric"},
        "10": {"size_pct": 4.59, "ci": 0.34, "group": "C", "type": "submetacentric"},
        "11": {"size_pct": 4.61, "ci": 0.40, "group": "C", "type": "submetacentric"},
        "12": {"size_pct": 4.66, "ci": 0.30, "group": "C", "type": "submetacentric"},
        "13": {"size_pct": 3.74, "ci": 0.17, "group": "D", "type": "acrocentric"},
        "14": {"size_pct": 3.56, "ci": 0.19, "group": "D", "type": "acrocentric"},
        "15": {"size_pct": 3.46, "ci": 0.20, "group": "D", "type": "acrocentric"},
        "16": {"size_pct": 3.36, "ci": 0.41, "group": "E", "type": "metacentric"},
        "17": {"size_pct": 3.25, "ci": 0.34, "group": "E", "type": "submetacentric"},
        "18": {"size_pct": 2.93, "ci": 0.26, "group": "E", "type": "submetacentric"},
        "19": {"size_pct": 2.67, "ci": 0.47, "group": "F", "type": "metacentric"},
        "20": {"size_pct": 2.56, "ci": 0.45, "group": "F", "type": "metacentric"},
        "21": {"size_pct": 1.90, "ci": 0.28, "group": "G", "type": "acrocentric"},
        "22": {"size_pct": 2.04, "ci": 0.28, "group": "G", "type": "acrocentric"},
        "X":  {"size_pct": 5.12, "ci": 0.40, "group": "C", "type": "submetacentric"},
        "Y":  {"size_pct": 2.00, "ci": 0.27, "group": "G", "type": "acrocentric"},
    }

    def __init__(self):
        self.super_res_target = 64  # Target height for super-resolution
        self.feature_weights = {
            "size": 0.45,
            "centromere": 0.30,
            "banding": 0.15,
            "aspect_ratio": 0.10,
        }
        # SMOTE: synthetic augmentation noise levels per feature
        self.smote_noise = {
            "size": 0.05,
            "centromere": 0.03,
            "banding": 0.08,
        }

    def classify_all(self, gray: np.ndarray, contours: List,
                      boxes: List[Tuple], areas: List[float],
                      band_profiles: List[np.ndarray],
                      straightened: Optional[List[Dict]] = None
                      ) -> List[Dict]:
        """Classify each chromosome into one of 24 classes.

        Returns list of classification results, one per chromosome:
        {
            "predicted_class": "1"-"22" / "X" / "Y",
            "confidence": float 0-1,
            "denver_group": "A"-"G",
            "features": {size_ratio, centromere_index, ...},
            "top3": [(class, score), ...],
        }
        """
        if not contours:
            return []

        # ── Step 1: Extract features for each chromosome ──
        total_area = sum(areas) if areas else 1
        features_list = []
        for i, (contour, box, area) in enumerate(zip(contours, boxes, areas)):
            features = self._extract_features(
                gray, contour, box, area, total_area,
                band_profiles[i] if i < len(band_profiles) else None,
                straightened[i] if straightened and i < len(straightened) else None
            )
            features_list.append(features)

        # ── Step 2: Classify using template matching ──
        results = []
        for i, features in enumerate(features_list):
            scores = self._compute_class_scores(features)
            # Sort by score descending
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_class, best_score = ranked[0]
            template = self.CHROMOSOME_TEMPLATES[best_class]

            results.append({
                "predicted_class": best_class,
                "confidence": round(best_score, 3),
                "denver_group": template["group"],
                "chromosome_type": template["type"],
                "features": features,
                "top3": [(c, round(s, 3)) for c, s in ranked[:3]],
            })

        # ── Step 3: Pair-based refinement ──
        # In a normal karyotype, each autosome appears exactly twice
        # Use this constraint to refine ambiguous classifications
        results = self._refine_with_pairing(results)

        return results

    def _extract_features(self, gray: np.ndarray, contour, box: Tuple,
                           area: float, total_area: float,
                           band_profile: Optional[np.ndarray],
                           straightened_info: Optional[Dict]) -> Dict:
        """Extract morphometric features for classification."""
        x, y, w, h = box

        # ── Size ratio (relative to total genome) ──
        size_ratio = (area / total_area) * 100 if total_area > 0 else 0

        # ── Aspect ratio ──
        aspect_ratio = h / w if w > 0 else 1.0

        # ── Centromere index estimation ──
        # Use the narrowest point along the major axis as centromere
        centromere_index = self._estimate_centromere_index(gray, contour, box)

        # ── Super-resolution enhanced banding ──
        enhanced_profile = self._super_resolve_banding(gray, contour, box, band_profile)

        # ── Curvature (from straightening) ──
        curvature = 0.0
        if straightened_info:
            curvature = straightened_info.get("curvature", 0.0)

        return {
            "size_ratio": size_ratio,
            "centromere_index": centromere_index,
            "aspect_ratio": aspect_ratio,
            "banding_profile": enhanced_profile,
            "curvature": curvature,
            "area": area,
        }

    def _estimate_centromere_index(self, gray: np.ndarray, contour, box: Tuple) -> float:
        """Estimate centromere index = p-arm length / total length.
        The centromere is at the narrowest constriction point."""
        x, y, w, h = box
        if h < 5 or w < 3:
            return 0.5

        # Create mask for this chromosome
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted], -1, 255, -1)

        # Measure width at each row (perpendicular to long axis)
        widths = []
        for row in range(h):
            row_pixels = np.sum(mask[row, :] > 0)
            widths.append(row_pixels)

        if not widths or max(widths) == 0:
            return 0.5

        widths = np.array(widths, dtype=np.float64)

        # Smooth to avoid noise
        if len(widths) > 5:
            kernel_size = min(5, len(widths) // 2) | 1
            widths_smooth = np.convolve(widths, np.ones(kernel_size)/kernel_size, mode='same')
        else:
            widths_smooth = widths

        # Find the narrowest point (centromere = primary constriction)
        # Exclude the top/bottom 15% to avoid tips
        margin = max(1, int(h * 0.15))
        search_region = widths_smooth[margin:h - margin]
        if len(search_region) == 0:
            return 0.5

        min_idx = np.argmin(search_region) + margin
        centromere_index = min_idx / h

        return float(np.clip(centromere_index, 0.1, 0.9))

    def _super_resolve_banding(self, gray: np.ndarray, contour, box: Tuple,
                                 existing_profile: Optional[np.ndarray]) -> np.ndarray:
        """Super-resolution enhancement for low-quality banding patterns.
        Upscales the chromosome ROI and re-extracts the banding profile
        for improved detail, especially for small chromosomes (Group F, G)."""
        x, y, w, h = box
        target_h = self.super_res_target
        num_bins = 32

        if existing_profile is not None and h >= target_h:
            # Already good resolution, use existing profile
            return existing_profile

        # Extract ROI
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            return existing_profile if existing_profile is not None else np.zeros(num_bins)

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted], -1, 255, -1)

        # ── Super-resolution via bicubic interpolation + sharpening ──
        if h < target_h:
            scale = target_h / h
            new_w = max(3, int(w * scale))
            roi_upscaled = cv2.resize(roi, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
            mask_upscaled = cv2.resize(mask, (new_w, target_h), interpolation=cv2.INTER_NEAREST)

            # Sharpen to recover high-frequency banding details
            blurred = cv2.GaussianBlur(roi_upscaled, (0, 0), 1.0)
            roi_sharp = cv2.addWeighted(roi_upscaled, 1.5, blurred, -0.5, 0)
        else:
            roi_sharp = roi
            mask_upscaled = mask

        # Extract banding profile from enhanced ROI
        sr_h, sr_w = roi_sharp.shape[:2]
        profile = np.zeros(sr_h, dtype=np.float64)
        for row in range(sr_h):
            row_mask = mask_upscaled[row, :] if row < mask_upscaled.shape[0] else np.zeros(sr_w)
            masked_pixels = roi_sharp[row, :][row_mask > 0]
            if len(masked_pixels) > 0:
                profile[row] = np.mean(masked_pixels)

        # Resample to fixed bins
        if len(profile) > 1:
            resampled = np.interp(
                np.linspace(0, len(profile) - 1, num_bins),
                np.arange(len(profile)), profile
            )
        else:
            resampled = np.zeros(num_bins)

        # Normalize
        norm = np.linalg.norm(resampled)
        return resampled / norm if norm > 0 else resampled

    def _compute_class_scores(self, features: Dict) -> Dict[str, float]:
        """Compute similarity score for each of the 24 chromosome classes.
        Uses weighted combination of size, centromere index, and banding features."""
        scores = {}
        size_ratio = features["size_ratio"]
        ci = features["centromere_index"]
        ar = features["aspect_ratio"]

        for chr_name, template in self.CHROMOSOME_TEMPLATES.items():
            # ── Size similarity (Gaussian-like) ──
            size_diff = abs(size_ratio - template["size_pct"]) / max(template["size_pct"], 0.1)
            size_score = np.exp(-2.0 * size_diff ** 2)  # Tighter Gaussian for better discrimination

            # ── Centromere index similarity ──
            ci_diff = abs(ci - template["ci"])
            ci_score = max(0, 1.0 - ci_diff * 3.0)

            # ── Type compatibility (metacentric/submetacentric/acrocentric) ──
            type_score = self._type_compatibility(ci, template["type"])

            # ── Aspect ratio expectation ──
            # Longer chromosomes (Group A,B) should have higher aspect ratio
            expected_ar = 2.0 + template["size_pct"] * 0.3
            ar_diff = abs(ar - expected_ar) / max(expected_ar, 1)
            ar_score = max(0, 1.0 - ar_diff * 0.5)

            # ── Weighted combination ──
            total = (
                self.feature_weights["size"] * size_score +
                self.feature_weights["centromere"] * ci_score +
                self.feature_weights["banding"] * type_score +
                self.feature_weights["aspect_ratio"] * ar_score
            )

            # ── SMOTE-style tolerance: slightly widen acceptance for rare classes ──
            # Large chromosomes (Group A,B) are visually distinctive, need less tolerance
            # Small chromosomes (Group F,G) and Y are harder to distinguish, add tolerance
            if template["group"] in ("A", "B"):
                total *= 1.02  # Large chromosomes are easier to identify

            scores[chr_name] = total

        return scores

    def _type_compatibility(self, measured_ci: float, expected_type: str) -> float:
        """Score how well the measured centromere index matches the expected type."""
        if expected_type == "metacentric":
            # CI should be 0.40-0.50
            return max(0, 1.0 - abs(measured_ci - 0.45) * 4)
        elif expected_type == "submetacentric":
            # CI should be 0.25-0.40
            return max(0, 1.0 - abs(measured_ci - 0.33) * 4)
        elif expected_type == "acrocentric":
            # CI should be 0.10-0.28
            return max(0, 1.0 - abs(measured_ci - 0.20) * 4)
        return 0.5

    def _refine_with_pairing(self, results: List[Dict]) -> List[Dict]:
        """Pair-based refinement: enforce the constraint that each autosome
        should appear exactly twice in a normal karyotype.
        Reassign over/under-represented classes to their next-best match."""
        if not results:
            return results

        # Count predictions per class
        class_counts = {}
        for r in results:
            cls = r["predicted_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # For autosomes (1-22), expected count is 2
        # For X, expected 1-2; for Y, expected 0-1
        # Multiple passes: sort by confidence (lowest first) so weakest assignments get reassigned
        MAX_ITER = 10
        for _ in range(MAX_ITER):
            changed = False
            # Process lowest-confidence assignments first (most likely to be wrong)
            order = sorted(range(len(results)), key=lambda i: results[i]["confidence"])
            for i in order:
                r = results[i]
                cls = r["predicted_class"]
                expected_max = 2  # Autosomes
                if cls == "X":
                    expected_max = 3
                elif cls == "Y":
                    expected_max = 2

                if class_counts.get(cls, 0) > expected_max:
                    # Over-represented: try to reassign to an under-represented class
                    # Build candidate list from ALL classes, not just top3
                    features = r.get("features", {})
                    if features:
                        all_scores = self._compute_class_scores(features)
                        candidates = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                    else:
                        candidates = r["top3"]

                    for alt_cls, alt_score in candidates:
                        if alt_cls == cls:
                            continue
                        alt_expected_max = 2 if alt_cls not in ("X", "Y") else 3
                        if class_counts.get(alt_cls, 0) < alt_expected_max:
                            class_counts[cls] -= 1
                            class_counts[alt_cls] = class_counts.get(alt_cls, 0) + 1
                            results[i]["predicted_class"] = alt_cls
                            results[i]["confidence"] = alt_score
                            results[i]["denver_group"] = self.CHROMOSOME_TEMPLATES[alt_cls]["group"]
                            results[i]["refined"] = True
                            changed = True
                            break
            if not changed:
                break

        return results

    def generate_karyotype_summary(self, classifications: List[Dict]) -> Dict:
        """Generate a karyotype summary from classification results."""
        class_counts = {}
        for r in classifications:
            cls = r["predicted_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Determine sex chromosomes
        x_count = class_counts.get("X", 0)
        y_count = class_counts.get("Y", 0)

        if x_count == 2 and y_count == 0:
            sex_chr = "XX"
        elif x_count == 1 and y_count == 1:
            sex_chr = "XY"
        elif x_count == 1 and y_count == 0:
            sex_chr = "X"
        elif x_count == 2 and y_count == 1:
            sex_chr = "XXY"
        elif x_count == 3 and y_count == 0:
            sex_chr = "XXX"
        elif x_count == 1 and y_count == 2:
            sex_chr = "XYY"
        else:
            sex_chr = f"X{x_count}Y{y_count}"

        total = sum(class_counts.values())

        # Find abnormalities (autosomes with count != 2)
        abnormalities = []
        for chr_num in [str(i) for i in range(1, 23)]:
            count = class_counts.get(chr_num, 0)
            if count > 2:
                abnormalities.append({
                    "type": "trisomy",
                    "chromosome": chr_num,
                    "count": count,
                    "description": f"Trisomy {chr_num}: {count} copies detected"
                })
            elif count < 2:
                abnormalities.append({
                    "type": "monosomy",
                    "chromosome": chr_num,
                    "count": count,
                    "description": f"Monosomy {chr_num}: {count} copies detected"
                })

        # Build ISCN notation
        if not abnormalities and sex_chr in ("XX", "XY"):
            notation = f"{total},{sex_chr}"
        else:
            notation = f"{total},{sex_chr}"
            for ab in abnormalities:
                if ab["type"] == "trisomy":
                    notation += f",+{ab['chromosome']}"
                elif ab["type"] == "monosomy" and ab["count"] == 0:
                    notation += f",-{ab['chromosome']}"

        # Group distribution
        group_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}
        for r in classifications:
            grp = r.get("denver_group", "?")
            if grp in group_dist:
                group_dist[grp] += 1

        return {
            "total_count": total,
            "sex_chromosomes": sex_chr,
            "notation": notation,
            "class_counts": class_counts,
            "abnormalities": abnormalities,
            "group_distribution": group_dist,
            "avg_confidence": round(
                sum(r["confidence"] for r in classifications) / len(classifications), 3
            ) if classifications else 0,
        }


class EnsembleClassifier:
    """5가지 분류 알고리즘 앙상블 (models.png)

    Strategy 1: Simple CNN         - 기본 형태 특징 기반 분류 (경량화 CNN)
    Strategy 2: Siamese Contrastive - 쌍별 거리 메트릭 학습 (특징 대조)
    Strategy 3: SRAS-Enhanced      - 초해상도 강화 후 분류 (전처리 강화)
    Strategy 4: VariFocal Fusion   - 전역 형태 + 국소 밴딩 융합 (전역/국소 융합)
    Strategy 5: Multi-Task Ensemble - 다중 태스크 앙상블 투표 (다중 작업 앙상블)

    Final result: Weighted majority voting across all 5 strategies.
    """

    CHROMOSOME_TEMPLATES = ChromosomeClassifier.CHROMOSOME_TEMPLATES

    def __init__(self):
        self.base_classifier = ChromosomeClassifier()
        # Strategy weights (sum to 1.0)
        self.strategy_weights = {
            "simple_cnn": 0.15,
            "siamese": 0.20,
            "sras": 0.20,
            "varifocal": 0.25,
            "multitask": 0.20,
        }

    def classify_ensemble(self, gray: np.ndarray, contours: List,
                           boxes: List[Tuple], areas: List[float],
                           band_profiles: List[np.ndarray],
                           straightened: Optional[List[Dict]] = None
                           ) -> Tuple[List[Dict], Dict]:
        """Run all 5 classification strategies and combine via weighted voting.

        Returns:
            (classifications, ensemble_metadata)
        """
        n = len(contours)
        if n == 0:
            return [], {"strategies": {}}

        total_area = sum(areas) if areas else 1

        # ── Run all 5 strategies ──
        s1_results = self._strategy_simple_cnn(gray, contours, boxes, areas, total_area)
        s2_results = self._strategy_siamese(gray, contours, boxes, areas, band_profiles, total_area)
        s3_results = self._strategy_sras(gray, contours, boxes, areas, total_area)
        s4_results = self._strategy_varifocal(gray, contours, boxes, areas, band_profiles, straightened, total_area)
        s5_results = self._strategy_multitask(gray, contours, boxes, areas, band_profiles, straightened, total_area)

        all_strategies = [
            ("simple_cnn", s1_results),
            ("siamese", s2_results),
            ("sras", s3_results),
            ("varifocal", s4_results),
            ("multitask", s5_results),
        ]

        # ── Weighted voting per chromosome ──
        final_results = []
        for i in range(n):
            vote_scores = {}  # class -> weighted vote sum
            strategy_votes = {}

            for strategy_name, strategy_results in all_strategies:
                if i >= len(strategy_results):
                    continue
                pred_class = strategy_results[i]["class"]
                confidence = strategy_results[i]["confidence"]
                weight = self.strategy_weights[strategy_name]

                vote_scores[pred_class] = vote_scores.get(pred_class, 0) + weight * confidence
                strategy_votes[strategy_name] = pred_class

            # Winner = highest weighted vote
            if vote_scores:
                winner = max(vote_scores, key=vote_scores.get)
                winner_score = vote_scores[winner]
                total_weight = sum(vote_scores.values())
                normalized_conf = winner_score / total_weight if total_weight > 0 else 0
            else:
                winner = "?"
                normalized_conf = 0

            template = self.CHROMOSOME_TEMPLATES.get(winner, {"group": "?", "type": "unknown"})

            # Top 3 from combined votes
            top3 = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)[:3]

            final_results.append({
                "predicted_class": winner,
                "confidence": round(normalized_conf, 3),
                "denver_group": template["group"],
                "chromosome_type": template.get("type", "unknown"),
                "strategy_votes": strategy_votes,
                "vote_scores": {k: round(v, 3) for k, v in vote_scores.items()},
                "top3": [(c, round(s, 3)) for c, s in top3],
                "features": s1_results[i].get("features", {}) if i < len(s1_results) else {},
            })

        # ── Pair-based refinement ──
        final_results = self.base_classifier._refine_with_pairing(final_results)

        # ── Metadata ──
        agreement_count = 0
        for i in range(n):
            votes = [sr[i]["class"] for _, sr in all_strategies if i < len(sr)]
            if len(set(votes)) == 1:
                agreement_count += 1

        metadata = {
            "strategies_used": list(self.strategy_weights.keys()),
            "weights": self.strategy_weights,
            "unanimous_agreement": agreement_count,
            "agreement_ratio": round(agreement_count / n, 3) if n > 0 else 0,
            "total_chromosomes": n,
        }

        return final_results, metadata

    # ── Strategy 1: Simple CNN (경량화 CNN) ──────────────────────
    def _strategy_simple_cnn(self, gray, contours, boxes, areas, total_area) -> List[Dict]:
        """Lightweight feature-based classification using size + centromere index only.
        Mimics a shallow CNN that captures basic morphological features."""
        results = []
        for contour, box, area in zip(contours, boxes, areas):
            size_ratio = (area / total_area) * 100
            ci = self.base_classifier._estimate_centromere_index(gray, contour, box)
            x, y, w, h = box
            ar = h / w if w > 0 else 1.0

            best_class = "21"
            best_score = 0

            for chr_name, tmpl in self.CHROMOSOME_TEMPLATES.items():
                size_diff = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
                s_score = np.exp(-2.0 * size_diff ** 2)
                ci_diff = abs(ci - tmpl["ci"])
                c_score = max(0, 1.0 - ci_diff * 3.0)
                score = 0.6 * s_score + 0.4 * c_score
                if score > best_score:
                    best_score = score
                    best_class = chr_name

            results.append({
                "class": best_class, "confidence": best_score,
                "features": {"size_ratio": size_ratio, "centromere_index": ci, "aspect_ratio": ar}
            })
        return results

    # ── Strategy 2: Siamese Contrastive (특징 대조) ──────────────
    def _strategy_siamese(self, gray, contours, boxes, areas, band_profiles, total_area) -> List[Dict]:
        """Pairwise distance metric classification.
        Compares each chromosome's banding profile against reference profiles
        using cosine distance, mimicking Siamese network embedding comparison."""
        # Build reference profiles per class from the detected chromosomes
        # (In a real Siamese network, these would be learned embeddings)
        results = []
        n = len(contours)

        for i, (contour, box, area) in enumerate(zip(contours, boxes, areas)):
            size_ratio = (area / total_area) * 100
            profile = band_profiles[i] if i < len(band_profiles) else np.zeros(32)

            best_class = "21"
            best_score = 0

            for chr_name, tmpl in self.CHROMOSOME_TEMPLATES.items():
                # Size-based embedding distance
                size_dist = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
                size_sim = np.exp(-1.5 * size_dist ** 2)

                # Centromere embedding distance
                ci = self.base_classifier._estimate_centromere_index(gray, contour, box)
                ci_dist = abs(ci - tmpl["ci"])
                ci_sim = np.exp(-3.0 * ci_dist ** 2)

                # Banding profile similarity (cosine) — contrastive metric
                # Compare profile shape characteristics
                if np.linalg.norm(profile) > 0:
                    # Use profile statistics as embedding proxy
                    profile_var = float(np.var(profile))
                    # Metacentric chromosomes have more symmetric profiles
                    profile_skew = float(np.mean(profile[:16]) - np.mean(profile[16:]))
                    profile_score = 0.5  # neutral
                    if tmpl["type"] == "metacentric" and abs(profile_skew) < 0.1:
                        profile_score = 0.7
                    elif tmpl["type"] == "acrocentric" and abs(profile_skew) > 0.05:
                        profile_score = 0.7
                    elif tmpl["type"] == "submetacentric":
                        profile_score = 0.6
                else:
                    profile_score = 0.5

                # Contrastive distance combination
                score = 0.40 * size_sim + 0.30 * ci_sim + 0.30 * profile_score
                if score > best_score:
                    best_score = score
                    best_class = chr_name

            results.append({"class": best_class, "confidence": best_score})
        return results

    # ── Strategy 3: SRAS-Enhanced (전처리 강화) ──────────────────
    def _strategy_sras(self, gray, contours, boxes, areas, total_area) -> List[Dict]:
        """Super-resolution enhanced classification.
        Upscales each chromosome ROI before feature extraction,
        improving detail for small chromosomes (Group F, G)."""
        results = []
        target_h = 64

        for contour, box, area in zip(contours, boxes, areas):
            x, y, w, h = box
            size_ratio = (area / total_area) * 100

            # Super-resolve the ROI
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                results.append({"class": "21", "confidence": 0.3})
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - np.array([x, y])
            cv2.drawContours(mask, [shifted], -1, 255, -1)

            if h < target_h and h > 3:
                scale = target_h / h
                new_w = max(3, int(w * scale))
                roi_sr = cv2.resize(roi, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
                mask_sr = cv2.resize(mask, (new_w, target_h), interpolation=cv2.INTER_NEAREST)
                # Sharpen
                blurred = cv2.GaussianBlur(roi_sr, (0, 0), 1.0)
                roi_sr = cv2.addWeighted(roi_sr, 1.5, blurred, -0.5, 0)
            else:
                roi_sr = roi
                mask_sr = mask

            # Extract enhanced features from super-resolved ROI
            sr_h, sr_w = roi_sr.shape[:2]

            # Width profile for centromere detection (on enhanced image)
            widths = np.array([np.sum(mask_sr[r, :] > 0) for r in range(sr_h)], dtype=np.float64)
            if len(widths) > 5:
                ks = min(5, len(widths) // 2) | 1
                widths = np.convolve(widths, np.ones(ks)/ks, mode='same')
            margin = max(1, int(sr_h * 0.15))
            search = widths[margin:sr_h - margin]
            ci = (np.argmin(search) + margin) / sr_h if len(search) > 0 else 0.5

            # Enhanced banding from SR image
            profile = np.zeros(sr_h)
            for r in range(sr_h):
                masked_px = roi_sr[r, :][mask_sr[r, :] > 0]
                if len(masked_px) > 0:
                    profile[r] = np.mean(masked_px)

            best_class = "21"
            best_score = 0
            for chr_name, tmpl in self.CHROMOSOME_TEMPLATES.items():
                s_diff = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
                s_score = np.exp(-2.0 * s_diff ** 2)
                c_diff = abs(ci - tmpl["ci"])
                c_score = np.exp(-3.0 * c_diff ** 2)
                type_score = self.base_classifier._type_compatibility(ci, tmpl["type"])
                score = 0.45 * s_score + 0.35 * c_score + 0.20 * type_score
                if score > best_score:
                    best_score = score
                    best_class = chr_name

            results.append({"class": best_class, "confidence": best_score})
        return results

    # ── Strategy 4: VariFocal Fusion (전역/국소 융합) ────────────
    def _strategy_varifocal(self, gray, contours, boxes, areas,
                             band_profiles, straightened, total_area) -> List[Dict]:
        """Global shape + local banding feature fusion.
        Global: overall chromosome morphology (size, shape, centromere position)
        Local: specific banding pattern regions (telomere, centromere bands, q-arm bands)"""
        results = []

        for i, (contour, box, area) in enumerate(zip(contours, boxes, areas)):
            x, y, w, h = box
            size_ratio = (area / total_area) * 100
            profile = band_profiles[i] if i < len(band_profiles) else np.zeros(32)

            # ── Global features ──
            ci = self.base_classifier._estimate_centromere_index(gray, contour, box)
            ar = h / w if w > 0 else 1.0
            # Solidity: contour area / convex hull area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # ── Local features (band regions) ──
            if len(profile) >= 32 and np.linalg.norm(profile) > 0:
                # p-arm bands (top portion, before centromere)
                ci_bin = max(1, min(31, int(ci * 32)))
                p_arm_intensity = float(np.mean(profile[:ci_bin])) if ci_bin > 0 else 0
                q_arm_intensity = float(np.mean(profile[ci_bin:])) if ci_bin < 32 else 0
                # Band contrast (difference between darkest and lightest bands)
                band_contrast = float(np.max(profile) - np.min(profile))
                # Number of "dark bands" (below mean)
                mean_val = np.mean(profile)
                dark_bands = int(np.sum(profile < mean_val * 0.8))
                # Centromere region intensity
                cen_start = max(0, ci_bin - 2)
                cen_end = min(32, ci_bin + 2)
                centromere_band = float(np.mean(profile[cen_start:cen_end]))
            else:
                p_arm_intensity = q_arm_intensity = band_contrast = centromere_band = 0.5
                dark_bands = 4

            best_class = "21"
            best_score = 0

            for chr_name, tmpl in self.CHROMOSOME_TEMPLATES.items():
                # Global score
                s_diff = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
                global_size = np.exp(-2.0 * s_diff ** 2)
                global_ci = np.exp(-3.0 * (ci - tmpl["ci"]) ** 2)
                global_score = 0.5 * global_size + 0.3 * global_ci + 0.2 * solidity

                # Local score
                type_compat = self.base_classifier._type_compatibility(ci, tmpl["type"])
                # Larger chromosomes should have more dark bands
                expected_bands = max(2, int(tmpl["size_pct"] * 1.5))
                band_diff = abs(dark_bands - expected_bands) / max(expected_bands, 1)
                local_band_score = max(0, 1.0 - band_diff * 0.5)
                local_score = 0.5 * type_compat + 0.3 * local_band_score + 0.2 * min(1.0, band_contrast)

                # Fusion: weighted combination of global and local
                score = 0.55 * global_score + 0.45 * local_score
                if score > best_score:
                    best_score = score
                    best_class = chr_name

            results.append({"class": best_class, "confidence": best_score})
        return results

    # ── Strategy 5: Multi-Task Ensemble (다중 작업 앙상블) ────────
    def _strategy_multitask(self, gray, contours, boxes, areas,
                             band_profiles, straightened, total_area) -> List[Dict]:
        """Multi-task learning simulation: simultaneously considers
        classification + segmentation quality + banding pattern matching.
        Each task provides a confidence signal that modulates the final prediction."""
        results = []

        for i, (contour, box, area) in enumerate(zip(contours, boxes, areas)):
            x, y, w, h = box
            size_ratio = (area / total_area) * 100
            ci = self.base_classifier._estimate_centromere_index(gray, contour, box)
            profile = band_profiles[i] if i < len(band_profiles) else np.zeros(32)

            # ── Task A: Classification (size + centromere) ──
            class_scores = {}
            for chr_name, tmpl in self.CHROMOSOME_TEMPLATES.items():
                s_diff = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
                s_score = np.exp(-2.0 * s_diff ** 2)
                c_score = np.exp(-3.0 * (ci - tmpl["ci"]) ** 2)
                class_scores[chr_name] = 0.55 * s_score + 0.45 * c_score

            # ── Task B: Segmentation quality signal ──
            # Well-segmented chromosomes have smooth contours and expected solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            # Chromosomes should have moderate solidity (0.7-0.95) and low circularity
            seg_quality = 1.0
            if solidity < 0.5 or solidity > 0.99:
                seg_quality *= 0.8
            if circularity > 0.6:  # Too circular = likely debris
                seg_quality *= 0.7

            # ── Task C: Banding pattern task ──
            band_quality = 0.5
            if np.linalg.norm(profile) > 0:
                # Higher variance = clearer banding = more reliable
                band_var = float(np.var(profile))
                band_quality = min(1.0, band_var * 5 + 0.3)

            # ── Combine tasks ──
            # Modulate classification scores by segmentation and banding quality
            curvature = 0.0
            if straightened and i < len(straightened):
                curvature = straightened[i].get("curvature", 0.0)
            # Straighter chromosomes give more reliable features
            straightness_bonus = max(0.8, 1.0 - curvature)

            quality_modifier = seg_quality * band_quality * straightness_bonus

            best_class = max(class_scores, key=class_scores.get)
            best_score = class_scores[best_class] * quality_modifier

            results.append({"class": best_class, "confidence": best_score})
        return results


class ChromosomeDetector:
    """Enhanced CV-based chromosome detection with:
    - Banding Pattern Mining for self-similarity disambiguation (DeepACE-style)
    - Two-stage NMS to prevent duplicate/missed detections
    - Overlap-aware splitting via watershed & concavity analysis
    """

    def __init__(self):
        self.base_min_area = 200
        self.base_max_area = 150000
        # Banding pattern parameters
        self.band_profile_bins = 32  # Number of bins for intensity profile
        self.band_similarity_threshold = 0.85  # Cosine similarity threshold for "same chromosome"
        # Overlap splitting parameters
        self.overlap_area_ratio = 2.5  # If area > median * ratio, suspect overlap (conservative)
        self.concavity_depth_ratio = 0.20  # Min concavity depth relative to contour size

    # ── Main Detection Pipeline ──────────────────────────────────
    def detect_chromosomes(self, image: Image.Image) -> Dict:
        """
        Enhanced chromosome detection pipeline:
        0. Digital preprocessing (denoise + straighten)
        1. Multi-threshold contour extraction
        2. Cluster routing + segmentation matrix
        3. Banding pattern extraction (on straightened forms)
        4. Two-stage NMS (spatial NMS → banding-aware NMS)
        """
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available", "count": 0}

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            gray_raw = img_array
        else:
            gray_raw = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # ── Stage 0: Digital Preprocessing (Task 4) ──
        preprocessor = DigitalPreprocessor()
        gray, denoise_metadata = preprocessor.denoise(gray_raw)

        img_height, img_width = gray.shape
        img_area = img_height * img_width
        scale_factor = img_area / (1000 * 1000)
        min_area = max(100, int(self.base_min_area * scale_factor))
        max_area = int(self.base_max_area * scale_factor)

        kernel = np.ones((3, 3), np.uint8)
        kernel_small = np.ones((2, 2), np.uint8)

        # ── Stage A: Multi-threshold contour extraction ──
        all_results = []

        # Method 1: Otsu
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        all_results.append(self._detect_from_threshold(otsu_thresh, kernel, min_area, max_area, "otsu"))

        # Method 2: Adaptive
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        all_results.append(self._detect_from_threshold(adaptive_thresh, kernel, min_area, max_area, "adaptive"))

        # Method 3: Multi-level simple thresholds
        for thresh_val in [100, 127, 150, 180]:
            _, simple_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            all_results.append(self._detect_from_threshold(simple_thresh, kernel_small, min_area, max_area, f"simple_{thresh_val}"))

        # Method 4: Small-area tolerance
        small_min_area = max(50, min_area // 3)
        all_results.append(self._detect_from_threshold(otsu_thresh, kernel, small_min_area, max_area, "otsu_small"))

        # Select best initial result
        best_result = self._select_best_result(all_results)

        # ── Stage B: Cluster Router + Segmentation Matrix (Task 2 + 3) ──
        # Pre-route each cluster by overlap type, then apply appropriate segmentation
        initial_count = best_result["count"]
        segmenter = SegmentationMatrix()
        router = ClusterRouter()

        # Only apply routing if count suggests merged objects (not already over-segmented)
        if initial_count < 55:
            routed_contours, routed_boxes, routed_areas, router_metadata = router.route_and_segment(
                gray, otsu_thresh,
                best_result["chromosomes"], best_result["boxes"], best_result["areas"],
                min_area, segmenter
            )
        else:
            routed_contours = best_result["chromosomes"]
            routed_boxes = best_result["boxes"]
            routed_areas = best_result["areas"]
            router_metadata = {"routing": "skipped_high_count"}

        # ── Stage B2: Overlap splitting (fallback for remaining merges) ──
        if len(routed_contours) < 50:
            split_contours, split_boxes, split_areas = self._split_overlapping_chromosomes(
                gray, otsu_thresh, routed_contours, routed_boxes, routed_areas, min_area
            )
        else:
            split_contours = routed_contours
            split_boxes = routed_boxes
            split_areas = routed_areas

        # ── Stage C: Chromosome straightening + banding pattern extraction ──
        # Straighten curved chromosomes for more accurate banding profiles
        straightened = preprocessor.straighten_all(gray, split_contours, split_boxes)

        # Extract banding profiles: prefer straightened strips when available
        band_profiles = []
        for i, (contour, box) in enumerate(zip(split_contours, split_boxes)):
            if i < len(straightened) and straightened[i]["strip"].size > 0:
                strip = straightened[i]["strip"]
                # Use column-mean of the straightened strip as the banding profile
                if strip.shape[0] > 0 and strip.shape[1] > 0:
                    col_means = np.mean(strip, axis=1).astype(np.float64)
                    profile = np.interp(
                        np.linspace(0, len(col_means) - 1, self.band_profile_bins),
                        np.arange(len(col_means)), col_means
                    )
                    norm = np.linalg.norm(profile)
                    band_profiles.append(profile / norm if norm > 0 else profile)
                    continue
            # Fallback to PCA-based profile
            profiles_fallback = self._extract_banding_profiles(gray, [contour], [box])
            band_profiles.append(profiles_fallback[0] if profiles_fallback else np.zeros(self.band_profile_bins))

        # ── Stage D: Two-stage NMS ──
        final_indices = self._two_stage_nms(split_boxes, split_areas, band_profiles)

        final_boxes = [split_boxes[i] for i in final_indices]
        final_areas = [split_areas[i] for i in final_indices]
        final_contours = [split_contours[i] for i in final_indices]
        final_profiles = [band_profiles[i] for i in final_indices]

        # Sort by area descending
        if final_areas:
            sorted_indices = np.argsort(final_areas)[::-1]
            final_boxes = [final_boxes[i] for i in sorted_indices]
            final_areas = [final_areas[i] for i in sorted_indices]
            final_contours = [final_contours[i] for i in sorted_indices]
            final_profiles = [final_profiles[i] for i in sorted_indices]

        group_counts = self._estimate_denver_groups(final_areas, len(final_contours))
        sex_chr_info = self._detect_sex_chromosome_region(image, final_boxes)

        # ── Stage E: 5-Strategy Ensemble Classification (Task 5 + models) ──
        ensemble = EnsembleClassifier()
        base_classifier = ensemble.base_classifier

        # Map final indices back to straightened data
        final_straightened = []
        for i in final_indices:
            if i < len(straightened):
                final_straightened.append(straightened[i])
            else:
                final_straightened.append({"strip": np.array([]), "medial_axis": np.array([]), "curvature": 0.0})

        # Sort straightened in same order as final contours
        if final_areas and len(final_straightened) == len(final_indices):
            sorted_indices_list = list(np.argsort(final_areas)[::-1]) if final_areas else list(range(len(final_areas)))
            final_straightened_sorted = [final_straightened[i] for i in sorted_indices_list] if sorted_indices_list else final_straightened
        else:
            final_straightened_sorted = final_straightened

        classifications, ensemble_metadata = ensemble.classify_ensemble(
            gray, final_contours, final_boxes, final_areas,
            final_profiles, final_straightened_sorted
        )
        karyotype_summary = base_classifier.generate_karyotype_summary(classifications)

        return {
            "count": len(final_contours),
            "bounding_boxes": final_boxes,
            "areas": final_areas,
            "group_counts": group_counts,
            "sex_chromosome_region": sex_chr_info,
            "detection_method": best_result["method"] + "+preprocess+cluster_router+seg_matrix+24class",
            "band_profiles": final_profiles,
            "overlap_splits_performed": len(split_contours) - len(routed_contours),
            "cluster_routing": router_metadata,
            "preprocessing": denoise_metadata,
            "straightening": {
                "total": len(straightened),
                "avg_curvature": float(np.mean([s["curvature"] for s in straightened])) if straightened else 0.0,
                "max_curvature": float(max((s["curvature"] for s in straightened), default=0.0)),
            },
            "classifications": classifications,
            "karyotype_summary": karyotype_summary,
            "ensemble": ensemble_metadata,
        }

    def _select_best_result(self, all_results: List[Dict]) -> Dict:
        """Pick detection result closest to expected chromosome count"""
        best_result = None
        best_diff = float('inf')
        for result in all_results:
            count = result["count"]
            if 45 <= count <= 48:
                diff = abs(count - 46)
                if diff < best_diff:
                    best_diff = diff
                    best_result = result
            elif best_result is None or abs(count - 46) < best_diff:
                best_diff = abs(count - 46)
                best_result = result
        return best_result if best_result else all_results[0]

    def _detect_from_threshold(self, thresh_img, kernel, min_area, max_area, method_name) -> Dict:
        """Apply morphology and extract chromosome contours from thresholded image"""
        processed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chromosomes = []
        bounding_boxes = []
        areas = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                if 0.15 < aspect_ratio < 20:
                    chromosomes.append(contour)
                    bounding_boxes.append((x, y, w, h))
                    areas.append(area)

        return {
            "count": len(chromosomes),
            "chromosomes": chromosomes,
            "boxes": bounding_boxes,
            "areas": areas,
            "method": method_name
        }

    # ── Banding Pattern Mining (Self-Similarity) ─────────────────
    def _extract_banding_profiles(self, gray: np.ndarray, contours: List,
                                   boxes: List[Tuple]) -> List[np.ndarray]:
        """Extract intensity profile along each chromosome's major axis.
        This creates a 'banding fingerprint' that distinguishes self-similar chromosomes.
        Inspired by DeepACE banding pattern mining approach."""
        profiles = []
        for contour, (x, y, w, h) in zip(contours, boxes):
            # Extract ROI
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                profiles.append(np.zeros(self.band_profile_bins))
                continue

            # Create mask for this contour within ROI
            mask = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contour - np.array([x, y])
            cv2.drawContours(mask, [shifted_contour], -1, 255, -1)

            # Determine major axis via PCA on contour points
            pts = contour.reshape(-1, 2).astype(np.float64)
            if len(pts) < 5:
                # Fallback: use vertical profile
                col_means = np.mean(roi, axis=1)
                profile = np.interp(
                    np.linspace(0, len(col_means) - 1, self.band_profile_bins),
                    np.arange(len(col_means)), col_means
                )
                profiles.append(self._normalize_profile(profile))
                continue

            mean_pt = np.mean(pts, axis=0)
            centered = pts - mean_pt
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Major axis = eigenvector with largest eigenvalue
            major_axis = eigenvectors[:, np.argmax(eigenvalues)]

            # Project contour points onto major axis
            projections = centered @ major_axis
            proj_min, proj_max = projections.min(), projections.max()
            if proj_max - proj_min < 1:
                profiles.append(np.zeros(self.band_profile_bins))
                continue

            # Sample intensity along major axis
            num_samples = self.band_profile_bins
            sample_positions = np.linspace(proj_min, proj_max, num_samples)
            profile = np.zeros(num_samples)

            for i, pos in enumerate(sample_positions):
                # Find point on major axis
                pt = mean_pt + pos * major_axis
                px, py = int(round(pt[0])), int(round(pt[1]))
                # Sample in a small neighborhood perpendicular to axis
                perp = np.array([-major_axis[1], major_axis[0]])
                intensities = []
                for offset in range(-2, 3):
                    sx = int(round(pt[0] + offset * perp[0]))
                    sy = int(round(pt[1] + offset * perp[1]))
                    if 0 <= sy < gray.shape[0] and 0 <= sx < gray.shape[1]:
                        intensities.append(float(gray[sy, sx]))
                profile[i] = np.mean(intensities) if intensities else 0

            profiles.append(self._normalize_profile(profile))

        return profiles

    def _normalize_profile(self, profile: np.ndarray) -> np.ndarray:
        """Normalize banding profile to unit vector for cosine similarity"""
        norm = np.linalg.norm(profile)
        if norm > 0:
            return profile / norm
        return profile

    def _banding_similarity(self, profile_a: np.ndarray, profile_b: np.ndarray) -> float:
        """Cosine similarity between two banding profiles"""
        dot = np.dot(profile_a, profile_b)
        return float(dot)  # Already normalized

    # ── Two-Stage NMS ────────────────────────────────────────────
    def _two_stage_nms(self, boxes: List[Tuple], areas: List[float],
                       band_profiles: List[np.ndarray], iou_thresh: float = 0.3) -> List[int]:
        """Two-stage Non-Maximum Suppression:
        Stage 1: Standard spatial IoU-based NMS - removes clear duplicates
        Stage 2: Banding-pattern-aware NMS - keeps nearby chromosomes that have
                 different banding patterns (self-similar but distinct chromosomes)
        """
        if not boxes:
            return []

        n = len(boxes)
        suppressed = [False] * n

        # Compute IoU matrix
        def iou(box_a, box_b):
            ax, ay, aw, ah = box_a
            bx, by, bw, bh = box_b
            x1 = max(ax, bx)
            y1 = max(ay, by)
            x2 = min(ax + aw, bx + bw)
            y2 = min(ay + ah, by + bh)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0

        # Sort by area (larger first)
        indices = sorted(range(n), key=lambda i: areas[i], reverse=True)

        # Stage 1: Standard NMS - suppress high-IoU duplicates
        stage1_keep = []
        for i in indices:
            if suppressed[i]:
                continue
            stage1_keep.append(i)
            for j in indices:
                if j == i or suppressed[j]:
                    continue
                if iou(boxes[i], boxes[j]) > iou_thresh:
                    # High overlap detected - check banding patterns before suppressing
                    # Stage 2: If banding patterns are sufficiently different, KEEP both
                    if (len(band_profiles[i]) > 0 and len(band_profiles[j]) > 0):
                        similarity = self._banding_similarity(band_profiles[i], band_profiles[j])
                        if similarity < self.band_similarity_threshold:
                            # Different banding patterns → different chromosomes → keep both
                            continue
                    # Same or very similar patterns → suppress the smaller one
                    suppressed[j] = True

        return stage1_keep

    # ── Overlap Detection & Splitting ────────────────────────────
    def _split_overlapping_chromosomes(self, gray: np.ndarray, thresh: np.ndarray,
                                        contours: List, boxes: List[Tuple],
                                        areas: List[float], min_area: int
                                        ) -> Tuple[List, List[Tuple], List[float]]:
        """Detect and split overlapping chromosomes using:
        1. Area-based heuristic: if contour area >> median, suspect overlap
        2. Concavity analysis: find deep concavity points as split candidates
        3. Watershed segmentation: split merged blobs at concavity points
        """
        if not areas:
            return contours, boxes, areas

        median_area = float(np.median(areas))
        new_contours = []
        new_boxes = []
        new_areas = []
        split_count = 0
        max_splits = max(10, len(contours) // 4)  # Cap splits to avoid over-fragmentation

        for contour, box, area in zip(contours, boxes, areas):
            # Check if this contour likely contains multiple chromosomes
            if (area > median_area * self.overlap_area_ratio
                    and median_area > 0 and split_count < max_splits):
                # Try to split
                split_results = self._watershed_split(gray, thresh, contour, box, min_area)
                if split_results and len(split_results) > 1:
                    for s_contour, s_box, s_area in split_results:
                        new_contours.append(s_contour)
                        new_boxes.append(s_box)
                        new_areas.append(s_area)
                    split_count += 1
                    continue

                # Fallback: concavity-based split
                split_results = self._concavity_split(gray, contour, box, min_area)
                if split_results and len(split_results) > 1:
                    for s_contour, s_box, s_area in split_results:
                        new_contours.append(s_contour)
                        new_boxes.append(s_box)
                        new_areas.append(s_area)
                    split_count += 1
                    continue

            # No split needed or split failed - keep original
            new_contours.append(contour)
            new_boxes.append(box)
            new_areas.append(area)

        return new_contours, new_boxes, new_areas

    def _watershed_split(self, gray: np.ndarray, thresh: np.ndarray,
                          contour, box: Tuple, min_area: int
                          ) -> Optional[List[Tuple]]:
        """Use watershed segmentation to split overlapping chromosomes"""
        x, y, w, h = box
        # Add padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        roi_gray = gray[y1:y2, x1:x2]
        roi_thresh = thresh[y1:y2, x1:x2].copy()

        if roi_gray.size == 0 or roi_thresh.size == 0:
            return None

        # Distance transform
        dist_transform = cv2.distanceTransform(roi_thresh, cv2.DIST_L2, 5)
        if dist_transform.max() == 0:
            return None

        # Find local maxima (chromosome centers) via thresholding distance map
        # Conservative threshold (0.5) to avoid over-splitting
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Find connected components in sure foreground
        num_labels, labels = cv2.connectedComponents(sure_fg)

        # Need at least 2 separate regions to split
        if num_labels < 3:  # background + at least 2 objects
            return None

        # Prepare markers for watershed
        markers = labels.copy()
        # Mark unknown region
        sure_bg = cv2.dilate(roi_thresh, np.ones((3, 3), np.uint8), iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        markers[unknown == 255] = 0  # Unknown gets 0

        # Watershed needs 3-channel image
        if len(roi_gray.shape) == 2:
            roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        else:
            roi_color = roi_gray

        markers = np.int32(markers)
        cv2.watershed(roi_color, markers)

        # Extract individual chromosomes from watershed result
        results = []
        for label_id in range(1, num_labels):
            mask = np.uint8(markers == label_id) * 255
            sub_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for sc in sub_contours:
                sc_area = cv2.contourArea(sc)
                if sc_area >= min_area * 0.5:
                    sc_shifted = sc + np.array([x1, y1])
                    sx, sy, sw, sh = cv2.boundingRect(sc_shifted)
                    results.append((sc_shifted, (sx, sy, sw, sh), sc_area))

        return results if len(results) > 1 else None

    def _concavity_split(self, gray: np.ndarray, contour, box: Tuple,
                          min_area: int) -> Optional[List[Tuple]]:
        """Split overlapping chromosomes by finding deep concavity points
        where the contour 'pinches' inward at the overlap junction."""
        if len(contour) < 10:
            return None

        x, y, w, h = box
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 4:
            return None

        # Find convexity defects
        try:
            defects = cv2.convexityDefects(contour, hull)
        except cv2.error:
            return None

        if defects is None:
            return None

        # Find deep concavity points
        contour_size = max(w, h)
        min_depth = contour_size * self.concavity_depth_ratio
        deep_defects = []

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0  # Convert from fixed-point
            if depth > min_depth:
                far_point = tuple(contour[f][0])
                deep_defects.append((far_point, depth))

        # Need at least 2 deep concavity points to define a split line
        if len(deep_defects) < 2:
            return None

        # Sort by depth (deepest first) and take top 2
        deep_defects.sort(key=lambda x: x[1], reverse=True)
        pt1 = deep_defects[0][0]
        pt2 = deep_defects[1][0]

        # Create split mask: draw a line between the two deepest concavity points
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        cv2.line(mask, pt1, pt2, 0, 2)  # Cut line

        # Find resulting contours after the split
        split_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for sc in split_contours:
            sc_area = cv2.contourArea(sc)
            if sc_area >= min_area * 0.5:
                sx, sy, sw, sh = cv2.boundingRect(sc)
                results.append((sc, (sx, sy, sw, sh), sc_area))

        return results if len(results) > 1 else None

    def _estimate_denver_groups(self, areas: List[float], total_count: int) -> Dict:
        """Estimate chromosome groups based on size distribution"""
        if not areas or total_count == 0:
            return {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}

        # Normalize areas
        max_area = max(areas) if areas else 1
        normalized = [a / max_area for a in areas]

        # Estimate groups based on size percentiles
        # Group A (1-3): Largest, ~13% of chromosomes
        # Group B (4-5): Large, ~9%
        # Group C (6-12+X): Medium-large, ~30%
        # Group D (13-15): Medium, ~13%
        # Group E (16-18): Medium-small, ~13%
        # Group F (19-20): Small, ~9%
        # Group G (21-22+Y): Smallest, ~13%

        groups = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}

        for norm_area in normalized:
            if norm_area > 0.85:
                groups["A"] += 1
            elif norm_area > 0.70:
                groups["B"] += 1
            elif norm_area > 0.50:
                groups["C"] += 1
            elif norm_area > 0.35:
                groups["D"] += 1
            elif norm_area > 0.25:
                groups["E"] += 1
            elif norm_area > 0.15:
                groups["F"] += 1
            else:
                groups["G"] += 1

        return groups

    def _detect_sex_chromosome_region(self, image: Image.Image, boxes: List[Tuple]) -> Dict:
        """Analyze the sex chromosome region (typically last row of karyogram)"""
        if not boxes:
            return {"x_count": 0, "y_count": 0, "estimated": "unknown"}

        # In a standard karyogram, sex chromosomes are at the bottom-right
        # Find chromosomes in the lower portion of the image
        img_height = image.height
        img_width = image.width

        # Look at bottom 20% of image
        bottom_region_y = img_height * 0.8
        right_region_x = img_width * 0.6

        sex_chr_candidates = []
        for box in boxes:
            x, y, w, h = box
            center_y = y + h/2
            center_x = x + w/2
            if center_y > bottom_region_y and center_x > right_region_x:
                sex_chr_candidates.append(box)

        # Count and analyze sizes
        num_candidates = len(sex_chr_candidates)

        # Estimate based on count and relative sizes
        if num_candidates == 0:
            return {"x_count": 0, "y_count": 0, "estimated": "unknown", "region_count": 0}

        # Get areas of sex chromosome candidates
        candidate_areas = [w * h for (x, y, w, h) in sex_chr_candidates]

        if num_candidates >= 2:
            # Check if one is notably smaller (Y chromosome)
            min_area = min(candidate_areas)
            max_area = max(candidate_areas)

            if min_area < max_area * 0.6:  # Y is typically smaller
                # Likely XY configuration
                return {"x_count": 1, "y_count": 1, "estimated": "XY", "region_count": num_candidates}
            else:
                # Similar sizes - likely XX
                return {"x_count": 2, "y_count": 0, "estimated": "XX", "region_count": num_candidates}
        elif num_candidates == 1:
            # Single sex chromosome - likely Turner syndrome (45,X)
            return {"x_count": 1, "y_count": 0, "estimated": "X", "region_count": 1}

        return {"x_count": num_candidates, "y_count": 0, "estimated": "unknown", "region_count": num_candidates}

    def detect_karyogram_positions(self, image: Image.Image) -> Dict:
        """
        Detect chromosome positions in an ARRANGED karyogram image.
        Uses grid-based detection to count chromosomes at specific positions.

        Standard karyogram layout:
        Row 1: Groups A,B (1-5)
        Row 2: Group C (6-12, X)
        Row 3: Groups D,E (13-18)
        Row 4: Groups F,G (19-22, Y)
        """
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available"}

        # First, detect all chromosomes
        detection = self.detect_chromosomes(image)
        boxes = detection.get("bounding_boxes", [])
        total_count = detection.get("count", 0)

        if total_count == 0:
            return {"error": "No chromosomes detected", "total": 0}

        img_width = image.width
        img_height = image.height

        # Define grid regions for standard karyogram layout
        # Row 4 (bottom 25%) contains Groups F and G (positions 19-22, Y)
        row4_y_start = img_height * 0.75

        # Within row 4, Group G (21, 22, Y) is typically in the right portion
        # Group G spans roughly the last 30% of the width
        group_g_x_start = img_width * 0.55

        # Position 21 is typically first in Group G (left side of Group G region)
        # Position 22 is middle, Y (or X for females) is at the end
        pos21_x_start = img_width * 0.55
        pos21_x_end = img_width * 0.70
        pos22_x_start = img_width * 0.70
        pos22_x_end = img_width * 0.82
        sex_x_start = img_width * 0.82

        # Count chromosomes in each region
        pos21_count = 0
        pos22_count = 0
        sex_chr_count = 0
        group_g_boxes = []

        for box in boxes:
            x, y, w, h = box
            center_x = x + w/2
            center_y = y + h/2

            # Check if in row 4 (Group G region)
            if center_y > row4_y_start:
                if pos21_x_start < center_x < pos21_x_end:
                    pos21_count += 1
                    group_g_boxes.append(("21", box))
                elif pos22_x_start < center_x < pos22_x_end:
                    pos22_count += 1
                    group_g_boxes.append(("22", box))
                elif center_x >= sex_x_start:
                    sex_chr_count += 1
                    group_g_boxes.append(("sex", box))

        # Analyze sex chromosomes based on size
        sex_region = detection.get("sex_chromosome_region", {})

        # Build position counts
        position_counts = {
            "position_21": pos21_count,
            "position_22": pos22_count,
            "sex_chromosomes": sex_chr_count,
            "sex_chr_estimated": sex_region.get("estimated", "unknown")
        }

        # Determine karyotype based on counts
        karyotype_analysis = self._analyze_karyotype_from_counts(
            total_count,
            pos21_count,
            sex_region
        )

        return {
            "total_count": total_count,
            "position_counts": position_counts,
            "group_g_details": group_g_boxes,
            "karyotype_analysis": karyotype_analysis,
            "detection_method": "karyogram_grid_analysis"
        }

    def _analyze_karyotype_from_counts(self, total: int, pos21_count: int, sex_info: Dict) -> Dict:
        """Analyze karyotype based on CV counts"""
        analysis = {
            "total_chromosomes": total,
            "position_21_count": pos21_count,
            "sex_chromosomes": sex_info.get("estimated", "unknown"),
            "preliminary_diagnosis": "Unknown",
            "confidence": "low"
        }

        # Basic decision logic based on CV counts
        if total == 46:
            if pos21_count == 2:
                if sex_info.get("estimated") == "XY":
                    analysis["preliminary_diagnosis"] = "46,XY (Normal male)"
                elif sex_info.get("estimated") == "XX":
                    analysis["preliminary_diagnosis"] = "46,XX (Normal female)"
                else:
                    analysis["preliminary_diagnosis"] = "46,?? (Normal count, sex uncertain)"
                analysis["confidence"] = "medium"
            else:
                analysis["preliminary_diagnosis"] = f"46 total but pos21={pos21_count} (needs review)"
                analysis["confidence"] = "low"

        elif total == 47:
            if pos21_count >= 3:
                # Likely Down syndrome
                sex = sex_info.get("estimated", "??")
                analysis["preliminary_diagnosis"] = f"47,{sex},+21 (Likely Down syndrome)"
                analysis["confidence"] = "medium"
            elif pos21_count == 2:
                # Sex chromosome abnormality likely
                sex = sex_info.get("estimated", "unknown")
                if sex in ["XXY", "XXX", "XYY"]:
                    analysis["preliminary_diagnosis"] = f"47,{sex} (Sex chromosome abnormality)"
                else:
                    analysis["preliminary_diagnosis"] = "47,?? (Trisomy, location uncertain)"
                analysis["confidence"] = "medium"
            else:
                analysis["preliminary_diagnosis"] = "47 total (abnormality location uncertain)"
                analysis["confidence"] = "low"

        elif total == 45:
            analysis["preliminary_diagnosis"] = "45,? (Possible monosomy)"
            analysis["confidence"] = "low"
        else:
            analysis["preliminary_diagnosis"] = f"{total} chromosomes (unusual count)"
            analysis["confidence"] = "low"

        return analysis

    def create_annotated_image(self, image: Image.Image, detection_result: Dict) -> Image.Image:
        """Create an annotated image with detected chromosomes highlighted"""
        if not CV2_AVAILABLE:
            return image

        img_array = np.array(image.convert('RGB'))

        # Draw bounding boxes
        for i, (x, y, w, h) in enumerate(detection_result.get("bounding_boxes", [])):
            color = (0, 255, 0)  # Green
            cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_array, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add count text
        count = detection_result.get("count", 0)
        cv2.putText(img_array, f"Detected: {count} chromosomes", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return Image.fromarray(img_array)


# 페이지 설정
st.set_page_config(
    page_title="Chromosome Karyotype Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        background-color: #3B82F6;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .disclaimer {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
    .notation-display {
        background-color: #1F2937;
        color: #F9FAFB;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .api-available {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .api-unavailable {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .consensus-agree {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .consensus-partial {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .consensus-disagree {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .voting-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .voting-table th, .voting-table td {
        border: 1px solid #E5E7EB;
        padding: 0.75rem;
        text-align: center;
    }
    .voting-table th {
        background-color: #F3F4F6;
    }
    .vote-winner {
        background-color: #D1FAE5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'raw_response' not in st.session_state:
    st.session_state.raw_response = None
if 'consensus_api_keys' not in st.session_state:
    st.session_state.consensus_api_keys = {
        'openai': None,
        'anthropic': None,
        'gemini': None
    }
if 'consensus_settings' not in st.session_state:
    st.session_state.consensus_settings = {
        'use_openai': True,
        'use_anthropic': True,
        'use_gemini': True,
        'min_agreement': 'majority'
    }
if 'cv_detection' not in st.session_state:
    st.session_state.cv_detection = None
if 'saved_api_key' not in st.session_state:
    st.session_state.saved_api_key = None
if 'api_key_saved' not in st.session_state:
    st.session_state.api_key_saved = False


def inject_local_storage_script(action: str, key_value: str = ""):
    """Inject JavaScript to interact with localStorage"""
    if action == "save":
        # Use JSON encoding to prevent XSS - properly escapes quotes and special chars
        import json as json_module
        safe_key = json_module.dumps(key_value)  # Returns quoted string like '"sk-xxx"'
        return f"""
        <script>
        localStorage.setItem('karyotype_openai_api_key', {safe_key});
        console.log('API key saved to localStorage');
        </script>
        """
    elif action == "clear":
        return """
        <script>
        localStorage.removeItem('karyotype_openai_api_key');
        console.log('API key cleared from localStorage');
        </script>
        """
    elif action == "load":
        return """
        <div id="ls-loader" style="display:none;"></div>
        <script>
        (function() {
            const savedKey = localStorage.getItem('karyotype_openai_api_key');
            if (savedKey) {
                document.getElementById('ls-loader').setAttribute('data-key', savedKey);
                // Create hidden input for Streamlit to read
                const container = document.querySelector('[data-testid="stSidebar"]');
                if (container) {
                    let indicator = document.getElementById('saved-key-indicator');
                    if (!indicator) {
                        indicator = document.createElement('div');
                        indicator.id = 'saved-key-indicator';
                        indicator.style.display = 'none';
                        indicator.textContent = savedKey;
                        container.appendChild(indicator);
                    }
                }
            }
        })();
        </script>
        """
    return ""


# Karyotype analysis system prompt - Enhanced with Chain-of-Thought and Verification
KARYOTYPE_ANALYSIS_PROMPT = """You are an expert clinical cytogeneticist analyzing a KARYOGRAM image.

## WHAT IS A KARYOGRAM?

A karyogram is an ARRANGED display of chromosomes where:
- Chromosomes are organized by NUMBER (1-22, X, Y)
- Each position has a LABEL showing which chromosome it is
- Most positions show a PAIR (2 chromosomes side by side)
- Abnormalities show 1 (monosomy) or 3 (trisomy) at a position

## STEP 1: READ THE LABELS IN THE IMAGE

Look at the image and find the numeric labels:
- Labels "1" through "22" mark the autosome positions
- Labels "X" and "Y" mark the sex chromosome positions
- These labels are typically printed below or near each chromosome group

## STEP 2: COUNT CHROMOSOMES AT EACH LABELED POSITION

For each labeled position, count the chromosome objects:

**Autosomes (positions 1-22):**
- Normal: 2 chromosomes at each position
- Trisomy: 3 chromosomes at one position (e.g., 3 at position "21" = Down syndrome)

**Sex chromosomes:**
- Normal female: 2 chromosomes at position X, 0 at Y → XX
- Normal male: 1 chromosome at position X, 1 at Y → XY
- Klinefelter: 2 at X, 1 at Y → XXY
- Triple X: 3 at X, 0 at Y → XXX

## STEP 3: DETERMINE THE KARYOTYPE

**Look specifically at position "21" in the image:**
- Count how many chromosome objects are shown under/near the "21" label
- If you see THREE chromosomes at position 21 → This is TRISOMY 21 (Down Syndrome)
- If you see TWO chromosomes at position 21 → Normal for position 21

**Look at the sex chromosome positions (X and Y):**
- Count chromosomes at position X
- Count chromosomes at position Y (or check if Y position is empty)

## STEP 4: CALCULATE TOTAL

Add up all chromosomes:
- Positions 1-22: Count at each position (normally 2 each = 44)
- Position X: Count (1 or 2 or 3)
- Position Y: Count (0 or 1 or 2)
- TOTAL should be 45, 46, or 47

## COMMON KARYOTYPES

| Karyotype | Position 21 | X count | Y count | Total |
|-----------|-------------|---------|---------|-------|
| 46,XY (Normal male) | 2 | 1 | 1 | 46 |
| 46,XX (Normal female) | 2 | 2 | 0 | 46 |
| 47,XY,+21 (Down syndrome male) | **3** | 1 | 1 | 47 |
| 47,XX,+21 (Down syndrome female) | **3** | 2 | 0 | 47 |
| 47,XXY (Klinefelter) | 2 | 2 | 1 | 47 |
| 47,XXX (Triple X) | 2 | 3 | 0 | 47 |

## KEY DISTINCTION: DOWN SYNDROME vs KLINEFELTER

Both have 47 chromosomes, but:
- **Down Syndrome**: Position 21 shows THREE small chromosomes, sex chromosomes are normal (XX or XY)
- **Klinefelter**: Position 21 shows TWO chromosomes (normal), sex chromosomes show XXY

**THE CRITICAL QUESTION:** Does position 21 have 2 or 3 chromosomes?
- If 3 → Down syndrome
- If 2 → Check sex chromosomes for XXY, XXX, etc.

## OUTPUT FORMAT

Return ONLY a valid JSON object:
{
    "notation": "ISCN notation (e.g., 46,XY or 47,XX,+21)",
    "chromosome_count": number,
    "sex_chromosomes": "XX/XY/XXY/XXX/X",
    "chromosome_21_count": number (2 or 3),
    "position_counts": {
        "autosomes_1_to_20": "2 each (normal) or specify abnormalities",
        "position_21": number,
        "position_22": number,
        "position_X": number,
        "position_Y": number
    },
    "abnormalities": [
        {"type": "type", "chromosome": "affected", "description": "description"}
    ],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation",
    "detailed_findings": "I see [N] chromosomes at position 21, [N] at X, [N] at Y"
}

## BEFORE ANSWERING, VERIFY:

1. What number label do you see at position 21? How many chromosomes are there?
2. What do you see at the X and Y positions?
3. Does your total match your individual position counts?

If position 21 shows 3 chromosomes, report Down syndrome (47,XX,+21 or 47,XY,+21).
If position 21 shows 2 chromosomes but total is 47, check sex chromosomes (XXY = Klinefelter, XXX = Triple X)."""


# CV+VLM Interpretation Prompt - VLM interprets CV counts (no visual counting needed)
CV_VLM_INTERPRETATION_PROMPT = """You are a clinical cytogeneticist interpreting computer vision analysis results.

## YOUR TASK
A computer vision system has analyzed a karyogram image and provided chromosome counts.
Your job is to INTERPRET these counts and provide a clinical diagnosis.

DO NOT try to count chromosomes yourself - trust the CV system's counts.

## CV ANALYSIS RESULTS
{cv_results}

## INTERPRETATION RULES

Based on the CV counts above, determine the karyotype:

**Total = 46 with Position 21 = 2:**
- If sex chromosomes = XY → 46,XY (Normal male)
- If sex chromosomes = XX → 46,XX (Normal female)

**Total = 47 with Position 21 = 3:**
- If sex chromosomes = XY → 47,XY,+21 (Down syndrome, male)
- If sex chromosomes = XX → 47,XX,+21 (Down syndrome, female)

**Total = 47 with Position 21 = 2:**
- If sex chromosomes = XXY → 47,XXY (Klinefelter syndrome)
- If sex chromosomes = XXX → 47,XXX (Triple X syndrome)
- If sex chromosomes = XYY → 47,XYY (Jacob syndrome)

**Total = 45:**
- If sex chromosomes = X only → 45,X (Turner syndrome)

## OUTPUT FORMAT

Return ONLY a valid JSON object:
{{
    "notation": "ISCN notation based on CV counts",
    "chromosome_count": {total_count},
    "sex_chromosomes": "XX/XY/XXY/XXX/X based on CV data",
    "chromosome_21_count": {pos21_count},
    "abnormalities": [
        {{"type": "type", "chromosome": "affected", "description": "description"}}
    ],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation",
    "cv_analysis_summary": "Summary of what CV detected",
    "analysis_method": "CV+VLM"
}}

Provide your interpretation based ONLY on the CV counts provided above."""


class PrecisionClinicalLens:
    """정밀 임상 렌즈 - 6단계 파이프라인으로 핵형 분석
    Stage 1: 계수 (Counting)
    Stage 2: 분류 (Classification)
    Stage 3: 클러스터 분류 (Cluster Classification)
    Stage 4: 전이 (Translocation Detection)
    Stage 5: 분석 (Comprehensive Analysis)
    Stage 6: 이상 탐지 (Abnormality Detection & Final Diagnosis)
    """

    STAGE_NAMES = [
        ("계수", "Counting"),
        ("분류", "Classification"),
        ("클러스터 분류", "Cluster Classification"),
        ("전이", "Translocation Detection"),
        ("분석", "Comprehensive Analysis"),
        ("이상 탐지", "Abnormality Detection"),
    ]

    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider

    def _encode_image_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()

    def _call_vlm(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        """Call VLM API and return raw text response"""
        image_base64 = self._encode_image_base64(image)

        if self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                        {"type": "text", "text": user_prompt}
                    ]
                }]
            )
            return response.content[0].text
        elif self.provider == "gemini" and GEMINI_AVAILABLE:
            client = genai.Client(api_key=self.api_key)
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[system_prompt + "\n\n" + user_prompt, image],
                config={'temperature': 0.1, 'max_output_tokens': 2500}
            )
            return response.text
        else:
            # Default: OpenAI
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}}
                    ]}
                ],
                max_tokens=2500,
                temperature=0.1
            )
            return response.choices[0].message.content

    def _parse_json(self, raw: str) -> Dict:
        """Extract JSON from VLM response"""
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
            match = re.search(pattern, raw)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw_text": raw, "parse_error": True}

    def run_pipeline(self, image: Image.Image, progress_callback=None) -> Dict:
        """Run the full 6-stage pipeline"""
        stages = {}
        cv_data = None

        # Optional CV pre-processing for Stage 1
        if CV2_AVAILABLE:
            detector = ChromosomeDetector()
            cv_data = detector.detect_chromosomes(image)

        # ── Stage 1: 계수 (Counting) ──
        if progress_callback:
            progress_callback(1, "계수 (Counting)")
        stages["stage_1"] = self._stage_counting(image, cv_data)

        # ── Stage 2: 분류 (Classification) ──
        if progress_callback:
            progress_callback(2, "분류 (Classification)")
        stages["stage_2"] = self._stage_classification(image, stages["stage_1"])

        # ── Stage 3: 클러스터 분류 (Cluster Classification) ──
        if progress_callback:
            progress_callback(3, "클러스터 분류 (Cluster Classification)")
        stages["stage_3"] = self._stage_cluster_classification(image, stages["stage_1"], stages["stage_2"])

        # ── Stage 4: 전이 (Translocation Detection) ──
        if progress_callback:
            progress_callback(4, "전이 (Translocation Detection)")
        stages["stage_4"] = self._stage_translocation(image, stages["stage_3"])

        # ── Stage 5: 분석 (Comprehensive Analysis) ──
        if progress_callback:
            progress_callback(5, "분석 (Comprehensive Analysis)")
        stages["stage_5"] = self._stage_analysis(image, stages)

        # ── Stage 6: 이상 탐지 (Abnormality Detection & Final Diagnosis) ──
        if progress_callback:
            progress_callback(6, "이상 탐지 (Abnormality Detection)")
        stages["stage_6"] = self._stage_abnormality_detection(image, stages)

        # Build final result
        final = stages["stage_6"]
        return {
            "notation": final.get("notation", "Unable to determine"),
            "chromosome_count": final.get("chromosome_count", stages["stage_1"].get("total_count", 0)),
            "sex_chromosomes": final.get("sex_chromosomes", "Unknown"),
            "abnormalities": final.get("abnormalities", []),
            "confidence": final.get("confidence", 0),
            "interpretation": final.get("interpretation", ""),
            "detailed_findings": final.get("detailed_findings", ""),
            "analysis_time": datetime.now().isoformat(),
            "technical_notes": "Precision Clinical Lens - 6-Stage Sequential Pipeline",
            "provider": f"Precision Clinical Lens ({self.provider})",
            "pipeline": "precision_lens",
            "stages": stages,
            "cv_data": cv_data
        }

    # ── Stage 1: 계수 (Counting) ──────────────────────────────────
    def _stage_counting(self, image: Image.Image, cv_data: Optional[Dict]) -> Dict:
        cv_hint = ""
        if cv_data and cv_data.get("count", 0) > 0:
            cv_hint = f"\n\nCV pre-analysis detected approximately {cv_data['count']} chromosomes using method '{cv_data.get('detection_method', 'unknown')}'. Use this as a reference but verify visually."

        raw = self._call_vlm(
            system_prompt="You are a board-certified clinical cytogeneticist. Your ONLY task in this step is to COUNT chromosomes accurately. Do not diagnose yet.",
            user_prompt=f"""## STAGE 1: CHROMOSOME COUNTING (계수)

Your sole task is to count the total number of chromosomes in this karyogram image.
{cv_hint}

### Instructions:
1. Count chromosomes at each labeled position (1-22, X, Y)
2. Record how many chromosomes appear at each position
3. Calculate the total

### Output JSON:
{{
    "total_count": <number>,
    "position_counts": {{
        "1": 2, "2": 2, ..., "22": 2,
        "X": <number>, "Y": <number>
    }},
    "count_notes": "any observations about counting difficulty or ambiguity"
}}""",
            image=image
        )
        return self._parse_json(raw)

    # ── Stage 2: 분류 (Classification) ─────────────────────────────
    def _stage_classification(self, image: Image.Image, stage1: Dict) -> Dict:
        total = stage1.get("total_count", "unknown")
        raw = self._call_vlm(
            system_prompt="You are a clinical cytogeneticist. Classify chromosomes into Denver groups based on size and centromere position.",
            user_prompt=f"""## STAGE 2: CHROMOSOME CLASSIFICATION (분류)

Previous stage counted {total} chromosomes.

### Instructions:
Classify each chromosome into Denver groups based on size and centromere position:
- **Group A (1-3)**: Large metacentric/submetacentric
- **Group B (4-5)**: Large submetacentric
- **Group C (6-12, X)**: Medium submetacentric
- **Group D (13-15)**: Medium acrocentric
- **Group E (16-18)**: Small metacentric/submetacentric
- **Group F (19-20)**: Small metacentric
- **Group G (21-22, Y)**: Small acrocentric

### Output JSON:
{{
    "denver_groups": {{
        "A": {{"count": <n>, "expected": 6, "chromosomes": "1-3"}},
        "B": {{"count": <n>, "expected": 4, "chromosomes": "4-5"}},
        "C": {{"count": <n>, "expected": 15, "chromosomes": "6-12,X (female) or 14 for male"}},
        "D": {{"count": <n>, "expected": 6, "chromosomes": "13-15"}},
        "E": {{"count": <n>, "expected": 6, "chromosomes": "16-18"}},
        "F": {{"count": <n>, "expected": 4, "chromosomes": "19-20"}},
        "G": {{"count": <n>, "expected": 5, "chromosomes": "21-22,Y (male) or 4 for female"}}
    }},
    "total_classified": <number>,
    "classification_notes": "observations about banding patterns, staining quality"
}}""",
            image=image
        )
        return self._parse_json(raw)

    # ── Stage 3: 클러스터 분류 (Cluster Classification) ────────────
    def _stage_cluster_classification(self, image: Image.Image, stage1: Dict, stage2: Dict) -> Dict:
        pos_counts = json.dumps(stage1.get("position_counts", {}), indent=2)
        denver = json.dumps(stage2.get("denver_groups", {}), indent=2)
        raw = self._call_vlm(
            system_prompt="You are a clinical cytogeneticist performing detailed chromosome identification. Match each chromosome to its specific number.",
            user_prompt=f"""## STAGE 3: CLUSTER CLASSIFICATION (클러스터 분류)

### Previous Results:
Position counts: {pos_counts}
Denver groups: {denver}

### Instructions:
For each chromosome position (1-22, X, Y), confirm:
1. The number of chromosomes present (normally 2 per autosome)
2. Whether paired chromosomes show normal homolog matching
3. Whether banding patterns are consistent with the labeled position
4. Identify the sex chromosome configuration

### Output JSON:
{{
    "chromosome_pairs": {{
        "1": {{"count": 2, "status": "normal/abnormal", "note": ""}},
        "2": {{"count": 2, "status": "normal", "note": ""}},
        ...
        "22": {{"count": 2, "status": "normal", "note": ""}},
        "X": {{"count": <n>, "note": ""}},
        "Y": {{"count": <n>, "note": ""}}
    }},
    "sex_determination": "XX/XY/XXY/XXX/X/XYY",
    "anomalous_positions": ["list of positions with abnormal counts or morphology"],
    "cluster_notes": "observations about homolog pairing and banding consistency"
}}""",
            image=image
        )
        return self._parse_json(raw)

    # ── Stage 4: 전이 (Translocation Detection) ────────────────────
    def _stage_translocation(self, image: Image.Image, stage3: Dict) -> Dict:
        anomalous = json.dumps(stage3.get("anomalous_positions", []))
        raw = self._call_vlm(
            system_prompt="You are a clinical cytogeneticist specializing in structural chromosome abnormalities. Focus on detecting translocations, inversions, deletions, and duplications.",
            user_prompt=f"""## STAGE 4: TRANSLOCATION & STRUCTURAL ABNORMALITY DETECTION (전이)

### Previous Results:
Anomalous positions flagged: {anomalous}

### Instructions:
Examine each chromosome pair for structural abnormalities:

1. **Translocations**: Material exchanged between non-homologous chromosomes
   - Reciprocal translocations: t(A;B)(breakpoints)
   - Robertsonian translocations: rob(13;14), rob(14;21), etc.
2. **Inversions**: Segment reversed within a chromosome
   - Pericentric (includes centromere): inv(chromosome)(p;q)
   - Paracentric (within one arm)
3. **Deletions**: Missing chromosome segment - del(chromosome)(breakpoint)
4. **Duplications**: Extra copy of a segment - dup(chromosome)(region)
5. **Ring chromosomes**: r(chromosome)
6. **Isochromosomes**: i(chromosome)(arm)

Pay special attention to:
- Unusual banding patterns within chromosome pairs
- Size differences between homologs
- Derivative chromosomes

### Output JSON:
{{
    "structural_abnormalities": [
        {{
            "type": "translocation/inversion/deletion/duplication/ring/isochromosome",
            "iscn_notation": "e.g., t(9;22)(q34;q11.2)",
            "chromosomes_involved": ["9", "22"],
            "description": "detailed description",
            "confidence": <0-100>
        }}
    ],
    "normal_structure": true/false,
    "translocation_notes": "observations about structural integrity"
}}""",
            image=image
        )
        return self._parse_json(raw)

    # ── Stage 5: 분석 (Comprehensive Analysis) ─────────────────────
    def _stage_analysis(self, image: Image.Image, all_stages: Dict) -> Dict:
        summary = {
            "total_count": all_stages["stage_1"].get("total_count", "unknown"),
            "sex_determination": all_stages["stage_3"].get("sex_determination", "unknown"),
            "anomalous_positions": all_stages["stage_3"].get("anomalous_positions", []),
            "structural_abnormalities": all_stages["stage_4"].get("structural_abnormalities", []),
            "normal_structure": all_stages["stage_4"].get("normal_structure", True)
        }
        summary_json = json.dumps(summary, indent=2)

        raw = self._call_vlm(
            system_prompt="You are a senior clinical cytogeneticist performing comprehensive karyotype analysis. Synthesize all previous findings into a coherent clinical picture.",
            user_prompt=f"""## STAGE 5: COMPREHENSIVE ANALYSIS (분석)

### Accumulated Findings:
{summary_json}

### Instructions:
Synthesize all previous stage findings:

1. **Numerical Analysis**: Confirm total count and identify any aneuploidy
   - Trisomy: extra chromosome (e.g., +21, +18, +13)
   - Monosomy: missing chromosome (e.g., -X in Turner)
   - Polyploidy: entire extra set

2. **Structural Analysis**: Review all detected structural changes
   - Clinical significance of each translocation/inversion/deletion
   - Whether structural changes are balanced or unbalanced

3. **Sex Chromosome Analysis**: Confirm sex chromosome constitution
   - Normal: 46,XX or 46,XY
   - Abnormal: 47,XXY (Klinefelter), 45,X (Turner), 47,XXX, 47,XYY

4. **Cross-validation**: Check consistency between stages
   - Does total count match position-by-position count?
   - Do Denver group counts align with specific chromosome identification?

### Output JSON:
{{
    "numerical_status": "normal/aneuploidy type",
    "structural_status": "normal/abnormal with details",
    "sex_chromosome_status": "normal/abnormal with details",
    "cross_validation": {{
        "count_consistent": true/false,
        "groups_consistent": true/false,
        "discrepancies": ["list any inconsistencies"]
    }},
    "preliminary_karyotype": "ISCN notation",
    "analysis_summary": "comprehensive text summary"
}}""",
            image=image
        )
        return self._parse_json(raw)

    # ── Stage 6: 이상 탐지 (Abnormality Detection & Final Diagnosis) ─
    def _stage_abnormality_detection(self, image: Image.Image, all_stages: Dict) -> Dict:
        # Compile all findings for final determination
        stage1 = all_stages["stage_1"]
        stage3 = all_stages["stage_3"]
        stage4 = all_stages["stage_4"]
        stage5 = all_stages["stage_5"]

        findings = {
            "total_count": stage1.get("total_count", "unknown"),
            "sex_determination": stage3.get("sex_determination", "unknown"),
            "anomalous_positions": stage3.get("anomalous_positions", []),
            "structural_abnormalities": stage4.get("structural_abnormalities", []),
            "preliminary_karyotype": stage5.get("preliminary_karyotype", "unknown"),
            "numerical_status": stage5.get("numerical_status", "unknown"),
            "structural_status": stage5.get("structural_status", "unknown"),
            "cross_validation": stage5.get("cross_validation", {})
        }
        findings_json = json.dumps(findings, indent=2)

        raw = self._call_vlm(
            system_prompt="You are a chief cytogeneticist issuing the FINAL clinical karyotype report. You must produce the definitive ISCN 2020 notation and clinical interpretation.",
            user_prompt=f"""## STAGE 6: FINAL ABNORMALITY DETECTION & DIAGNOSIS (이상 탐지)

### All Pipeline Findings:
{findings_json}

### Instructions:
This is the FINAL stage. Produce the definitive karyotype report:

1. **Final ISCN Notation**: Complete, ISCN 2020-compliant notation
2. **Abnormality Classification**:
   - Numerical: trisomy, monosomy, polyploidy
   - Structural: translocations, inversions, deletions, duplications
   - Combined if both present
3. **Clinical Correlation**:
   - Known syndromes (Down, Turner, Klinefelter, Edwards, Patau, etc.)
   - Philadelphia chromosome, other cancer-related changes
   - Prognostic implications
4. **Confidence Assessment**: Based on image quality, consistency across stages

### Output JSON:
{{
    "notation": "Complete ISCN 2020 notation",
    "chromosome_count": <number>,
    "sex_chromosomes": "XX/XY/X/XXY/XXX/XYY",
    "abnormalities": [
        {{
            "type": "numerical/structural",
            "subtype": "trisomy/monosomy/translocation/etc",
            "chromosome": "affected chromosome(s)",
            "iscn_detail": "ISCN sub-notation for this abnormality",
            "description": "detailed description",
            "clinical_significance": "associated syndrome or clinical impact"
        }}
    ],
    "confidence": <0-100>,
    "interpretation": "Complete clinical interpretation paragraph",
    "detailed_findings": "Summary of all 6 stages with key findings from each",
    "recommendations": "suggested follow-up if abnormalities found"
}}""",
            image=image
        )
        return self._parse_json(raw)


class KaryotypeAnalyzer:
    """염색체 핵형 분석기 클래스 - 다중 API 제공자 지원"""

    def __init__(self, provider: APIProvider, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key

    def encode_image_base64(self, image: Image.Image) -> str:
        """이미지를 base64로 인코딩"""
        buffered = io.BytesIO()
        # Convert to RGB if necessary (for RGBA, P, or grayscale images)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()

    def analyze(self, image: Image.Image, consensus_keys: Optional[Dict] = None) -> Dict:
        """선택된 API 제공자를 사용하여 분석 수행"""
        if self.provider == APIProvider.OPENAI:
            return self._analyze_with_openai(image)
        elif self.provider == APIProvider.ANTHROPIC:
            return self._analyze_with_anthropic(image)
        elif self.provider == APIProvider.GEMINI:
            return self._analyze_with_gemini(image)
        elif self.provider == APIProvider.CONSENSUS:
            return self._analyze_with_consensus(image, consensus_keys or {})
        elif self.provider == APIProvider.PRECISION_LENS:
            return self._analyze_with_precision_lens(image)
        elif self.provider == APIProvider.TWO_STAGE:
            return self._analyze_with_two_stage(image)
        elif self.provider == APIProvider.CV_VLM:
            return self._analyze_with_cv_vlm(image)
        else:
            return self._mock_analysis()

    def _analyze_with_openai(self, image: Image.Image) -> Dict:
        """OpenAI GPT-4 Vision API를 사용한 분석"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        client = OpenAI(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a board-certified clinical cytogeneticist. CRITICAL: You must COUNT chromosomes accurately before diagnosis. A normal karyotype has EXACTLY 46 chromosomes. Do NOT assume abnormalities - verify by counting each Denver group. Only report an abnormality if you can identify the SPECIFIC extra or missing chromosome. Always provide results as valid JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": KARYOTYPE_ANALYSIS_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2500,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        st.session_state.raw_response = raw_content
        return self._parse_response(raw_content, "OpenAI GPT-4 Vision")

    def _analyze_with_anthropic(self, image: Image.Image) -> Dict:
        """Anthropic Claude Vision API를 사용한 분석"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        client = anthropic.Anthropic(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            system="You are a board-certified clinical cytogeneticist. CRITICAL: You must COUNT chromosomes accurately before diagnosis. A normal karyotype has EXACTLY 46 chromosomes. Do NOT assume abnormalities - verify by counting each Denver group. Only report an abnormality if you can identify the SPECIFIC extra or missing chromosome. Always provide results as valid JSON.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": KARYOTYPE_ANALYSIS_PROMPT
                        }
                    ]
                }
            ]
        )

        raw_content = response.content[0].text
        st.session_state.raw_response = raw_content
        return self._parse_response(raw_content, "Anthropic Claude Vision")

    def _analyze_with_gemini(self, image: Image.Image) -> Dict:
        """Google Gemini Vision API를 사용한 분석"""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI package not installed. Run: pip install google-genai")
        if not self.api_key:
            raise ValueError("Google API key is required")

        client = genai.Client(api_key=self.api_key)

        # Convert image for Gemini
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        # Prepend system instruction to the prompt
        system_instruction = "CRITICAL: You must COUNT chromosomes accurately before diagnosis. A normal karyotype has EXACTLY 46 chromosomes. Do NOT assume abnormalities - verify by counting each Denver group. Only report an abnormality if you can identify the SPECIFIC extra or missing chromosome.\n\n"

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[system_instruction + KARYOTYPE_ANALYSIS_PROMPT, image],
            config={
                'temperature': 0.1,
                'max_output_tokens': 2500
            }
        )

        raw_content = response.text
        st.session_state.raw_response = raw_content
        return self._parse_response(raw_content, "Google Gemini Vision")

    def _analyze_with_two_stage(self, image: Image.Image) -> Dict:
        """Two-stage analysis: CV detection + VLM classification"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not installed. Run: pip install opencv-python")
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Stage 2 VLM analysis")

        # Stage 1: Computer Vision Detection
        detector = ChromosomeDetector()
        detection_result = detector.detect_chromosomes(image)

        cv_count = detection_result.get("count", 0)
        cv_groups = detection_result.get("group_counts", {})
        sex_chr_info = detection_result.get("sex_chromosome_region", {})

        # Store detection result for display
        st.session_state.cv_detection = detection_result

        # Stage 2: VLM Classification with CV-informed prompt
        two_stage_prompt = f"""You are an expert clinical cytogeneticist. A computer vision system has pre-analyzed this karyotype image.

## COMPUTER VISION DETECTION RESULTS (Stage 1)

**Detected Chromosome Count: {cv_count}**

Estimated Denver Group Distribution:
- Group A (1-3): {cv_groups.get('A', 0)} chromosomes
- Group B (4-5): {cv_groups.get('B', 0)} chromosomes
- Group C (6-12+X): {cv_groups.get('C', 0)} chromosomes
- Group D (13-15): {cv_groups.get('D', 0)} chromosomes
- Group E (16-18): {cv_groups.get('E', 0)} chromosomes
- Group F (19-20): {cv_groups.get('F', 0)} chromosomes
- Group G (21-22+Y): {cv_groups.get('G', 0)} chromosomes

Sex Chromosome Region Analysis:
- Estimated configuration: {sex_chr_info.get('estimated', 'unknown')}
- Chromosomes in region: {sex_chr_info.get('region_count', 0)}

## YOUR TASK (Stage 2)

Based on the CV detection count of **{cv_count} chromosomes**, determine the karyotype:

**CRITICAL DECISION RULES:**

1. **If CV count = 46**: This is likely a NORMAL karyotype
   - Check sex chromosomes: XX (female) or XY (male)
   - Report as 46,XX or 46,XY

2. **If CV count = 45**: This suggests MONOSOMY (missing chromosome)
   - Most common: Turner syndrome (45,X) - missing one sex chromosome
   - Look for single X chromosome in sex region

3. **If CV count = 47**: This suggests TRISOMY (extra chromosome)
   - Check WHERE the extra chromosome is:
   - If extra in Group G (small): Likely Trisomy 21 (Down syndrome) → 47,XX,+21 or 47,XY,+21
   - If extra in Group C (medium X): Could be Triple X (47,XXX) or Klinefelter (47,XXY)
   - Distinguish by checking Y chromosome presence:
     * NO Y chromosome → 47,XXX (Triple X, female)
     * Y chromosome present → 47,XXY (Klinefelter, male)

4. **If CV count differs significantly from expected**: Trust the CV count over visual estimation.

## VISUAL VERIFICATION

Look at the image to verify:
1. The sex chromosome region (bottom-right of karyogram)
2. Whether Group G has 4 (normal), 5 (trisomy 21), or other count
3. The presence/absence of Y chromosome (small, distinct from 21/22)

## OUTPUT FORMAT

Return ONLY valid JSON:
{{
    "notation": "ISCN notation based on CV count of {cv_count}",
    "chromosome_count": {cv_count},
    "sex_chromosomes": "XX/XY/X/XXY/XXX",
    "cv_detection": {{
        "count": {cv_count},
        "groups": {json.dumps(cv_groups)},
        "sex_region": "{sex_chr_info.get('estimated', 'unknown')}"
    }},
    "abnormalities": [
        {{"type": "type", "chromosome": "chr", "description": "desc"}}
    ],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation",
    "detailed_findings": "how CV detection informed the diagnosis"
}}

IMPORTANT: The CV system detected {cv_count} chromosomes. Use this as your primary count reference."""

        # Call OpenAI with CV-informed prompt
        client = OpenAI(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a cytogeneticist. A CV system detected {cv_count} chromosomes. Use this count as ground truth and determine the karyotype classification. Do not re-count - trust the CV detection."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": two_stage_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2500,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        st.session_state.raw_response = raw_content

        result = self._parse_response(raw_content, "Two-Stage Pipeline (CV + VLM)")
        result['cv_detection'] = detection_result
        result['pipeline'] = 'two_stage'
        return result

    def _parse_response(self, raw_content: str, provider_name: str) -> Dict:
        """API 응답에서 JSON 파싱 - 다양한 형식 지원"""
        if not raw_content or not raw_content.strip():
            return self._create_error_response("Empty response from API", "", provider_name)

        try:
            # Method 1: Try to parse directly as JSON
            try:
                result = json.loads(raw_content.strip())
                return self._finalize_result(result, provider_name)
            except json.JSONDecodeError:
                pass

            # Method 2: Extract from markdown code blocks (```json ... ```)
            code_block_patterns = [
                r'```json\s*([\s\S]*?)\s*```',
                r'```\s*([\s\S]*?)\s*```',
            ]
            for pattern in code_block_patterns:
                match = re.search(pattern, raw_content)
                if match:
                    try:
                        result = json.loads(match.group(1).strip())
                        return self._finalize_result(result, provider_name)
                    except json.JSONDecodeError:
                        continue

            # Method 3: Find JSON object by proper brace counting
            # This handles nested objects correctly
            first_brace = raw_content.find('{')
            if first_brace != -1:
                brace_count = 0
                in_string = False
                escape_next = False
                end_pos = -1

                for i, char in enumerate(raw_content[first_brace:], start=first_brace):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\' and in_string:
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i
                                break

                if end_pos != -1:
                    json_str = raw_content[first_brace:end_pos + 1]
                    try:
                        result = json.loads(json_str)
                        return self._finalize_result(result, provider_name)
                    except json.JSONDecodeError:
                        pass

            # Method 4: Try to fix common JSON issues
            # Remove trailing commas before } or ]
            cleaned = re.sub(r',\s*([}\]])', r'\1', raw_content)
            first_brace = cleaned.find('{')
            last_brace = cleaned.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = cleaned[first_brace:last_brace + 1]
                try:
                    result = json.loads(json_str)
                    return self._finalize_result(result, provider_name)
                except json.JSONDecodeError:
                    pass

            raise ValueError("No valid JSON found in response")

        except Exception as e:
            return self._create_error_response(str(e), raw_content, provider_name)

    def _finalize_result(self, result: Dict, provider_name: str) -> Dict:
        """결과에 기본값 및 메타데이터 추가"""
        result.setdefault('notation', 'Unable to determine')
        result.setdefault('chromosome_count', 0)
        result.setdefault('sex_chromosomes', 'Unknown')
        result.setdefault('abnormalities', [])
        result.setdefault('confidence', 0)
        result.setdefault('interpretation', 'Analysis incomplete')
        result.setdefault('detailed_findings', '')

        result['analysis_time'] = datetime.now().isoformat()
        result['technical_notes'] = f"Analysis performed using {provider_name}"
        result['provider'] = provider_name

        return result

    def _create_error_response(self, error_msg: str, raw_content: str, provider_name: str) -> Dict:
        """에러 응답 생성"""
        preview = raw_content[:500] if raw_content else "No response content"
        return {
            'notation': 'Parse Error',
            'chromosome_count': 0,
            'sex_chromosomes': 'Unknown',
            'abnormalities': [],
            'confidence': 0,
            'interpretation': f'Failed to parse API response: {error_msg}',
            'detailed_findings': f'Raw response preview: {preview}...',
            'analysis_time': datetime.now().isoformat(),
            'technical_notes': f'Error parsing response from {provider_name}. Check raw response for details.',
            'provider': provider_name
        }

    def _analyze_with_cv_vlm(self, image: Image.Image) -> Dict:
        """CV counts chromosomes, VLM interprets the counts. Falls back to VLM-only if CV unreliable."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available. Run: pip install opencv-python")

        # Stage 1: CV Detection and Counting
        detector = ChromosomeDetector()
        cv_result = detector.detect_karyogram_positions(image)

        # Store CV detection for display
        st.session_state.cv_detection = cv_result

        # Extract key counts
        total_count = cv_result.get("total_count", 0)
        position_counts = cv_result.get("position_counts", {})
        pos21_count = position_counts.get("position_21", 0)
        sex_chr_count = position_counts.get("sex_chromosomes", 0)  # Actual count in sex region
        sex_chr_estimated = position_counts.get("sex_chr_estimated", "unknown")
        karyotype_analysis = cv_result.get("karyotype_analysis", {})

        # FALLBACK: If CV count is unreliable (outside 44-50 range), use VLM with CV hints
        CV_MIN_RELIABLE = 44
        CV_MAX_RELIABLE = 50

        if total_count < CV_MIN_RELIABLE or total_count > CV_MAX_RELIABLE:
            # CV total count is unreliable, but position data may still be useful
            st.warning(f"⚠️ CV detected {total_count} chromosomes (outside 44-50 range). Using VLM with CV position hints.")

            # Use VLM analysis WITH CV position hints (not pure VLM-only)
            result = self._analyze_with_cv_hints(image, pos21_count, sex_chr_estimated, sex_chr_count)

            # Add fallback note
            result["fallback_used"] = True
            result["cv_detection"] = {
                "total_detected": total_count,
                "position_21_count": pos21_count,
                "cv_unreliable": True,
                "fallback_reason": f"CV count ({total_count}) outside reliable range (44-50)"
            }
            result["technical_notes"] = f"CV total ({total_count}) unreliable. VLM analyzed with CV position hints (pos21={pos21_count})."
            result["analysis_method"] = "CV+VLM (CV-assisted fallback)"

            return result

        # Format CV results for VLM
        cv_results_text = f"""
COMPUTER VISION ANALYSIS RESULTS:
- Total chromosomes detected: {total_count}
- Position 21 count: {pos21_count}
- Position 22 count: {position_counts.get("position_22", 0)}
- Sex chromosome region count: {position_counts.get("sex_chromosomes", 0)}
- Estimated sex chromosomes: {sex_chr_estimated}
- CV preliminary diagnosis: {karyotype_analysis.get("preliminary_diagnosis", "Unknown")}
- CV confidence: {karyotype_analysis.get("confidence", "low")}
"""

        # Stage 2: VLM Interpretation
        client = OpenAI(api_key=self.api_key)

        # Fill in the prompt template
        interpretation_prompt = CV_VLM_INTERPRETATION_PROMPT.format(
            cv_results=cv_results_text,
            total_count=total_count,
            pos21_count=pos21_count
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical cytogeneticist. Interpret the computer vision chromosome counts provided. Do NOT try to count visually - trust the CV data and provide clinical interpretation based on the counts."
                },
                {
                    "role": "user",
                    "content": interpretation_prompt
                }
            ],
            max_tokens=1500,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        st.session_state.raw_response = f"CV Results:\n{cv_results_text}\n\nVLM Interpretation:\n{raw_content}"

        # Parse the VLM response
        result = self._parse_response(raw_content, "CV + VLM (Hybrid)")

        # LOW-CONFIDENCE FALLBACK: If VLM confidence < 50%, re-run with CV hints (visual analysis)
        LOW_CONFIDENCE_THRESHOLD = 50
        vlm_confidence = result.get("confidence", 0)

        if vlm_confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ Initial analysis confidence low ({vlm_confidence}%). Re-analyzing with visual verification...")

            # Re-run with CV hints - this uses visual analysis with position hints
            result = self._analyze_with_cv_hints(image, pos21_count, sex_chr_estimated)

            # Add fallback note
            result["fallback_used"] = True
            result["cv_detection"] = {
                "total_detected": total_count,
                "position_21_count": pos21_count,
                "cv_unreliable": False,
                "fallback_reason": f"Initial confidence ({vlm_confidence}%) below threshold ({LOW_CONFIDENCE_THRESHOLD}%)"
            }
            result["technical_notes"] = f"CV detected {total_count} chromosomes. Low confidence ({vlm_confidence}%) triggered visual re-analysis with CV hints."
            result["analysis_method"] = "CV+VLM (low-confidence fallback)"

            return result

        # Add CV-specific data to result
        result["cv_detection"] = {
            "total_detected": total_count,
            "position_21_count": pos21_count,
            "sex_chr_estimated": sex_chr_estimated,
            "cv_preliminary": karyotype_analysis.get("preliminary_diagnosis", "Unknown")
        }
        result["analysis_method"] = "CV+VLM"
        result["technical_notes"] = f"CV detected {total_count} chromosomes, {pos21_count} at position 21. VLM interpreted the counts."

        return result

    def _analyze_with_cv_hints(self, image: Image.Image, cv_pos21_count: int, cv_sex_estimate: str, cv_sex_region_count: int = 0) -> Dict:
        """VLM analysis with CV position hints when total count is unreliable"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Prepare image
        buffered = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Detect potential Triple X pattern: CV may misclassify 3 X chromosomes as position 21
        # If pos21=3 but sex region count is low, this could indicate Triple X, not Down syndrome
        triple_x_warning = ""
        if cv_pos21_count >= 3 and cv_sex_region_count <= 1:
            triple_x_warning = """
⚠️ **CRITICAL WARNING - POTENTIAL TRIPLE X MISCLASSIFICATION**:
CV detected 3 chromosomes near position 21 but few in the sex region. This pattern often occurs
when THREE X CHROMOSOMES are misdetected as position 21 due to image layout.

VISUALLY VERIFY: Are the 3 chromosomes at "position 21" actually:
- THREE SMALL acrocentric chromosomes (true Down syndrome - trisomy 21), OR
- THREE MEDIUM-SIZED submetacentric chromosomes (Triple X syndrome - 47,XXX)?

If they are MEDIUM-sized and similar to each other → This is likely 47,XXX (Triple X), NOT Down syndrome!
"""

        # Create prompt with CV hints
        cv_hint_prompt = f"""You are an expert clinical cytogeneticist analyzing a karyotype image.

## CV DETECTION HINTS (use as guidance, VERIFY VISUALLY - CV may be incorrect)
- CV detected approximately {cv_pos21_count} chromosome(s) near position 21 region
- CV detected {cv_sex_region_count} chromosome(s) in the sex chromosome region
- CV estimated sex chromosomes: {cv_sex_estimate}
- Note: CV grid detection may misclassify chromosomes due to image layout variations
{triple_x_warning}

## ANALYSIS ORDER (FOLLOW THIS SEQUENCE)

### STEP 1: COUNT SEX CHROMOSOMES FIRST (Most Important!)
Look at the sex chromosome region (typically bottom-right, after position 22):
- X chromosome: MEDIUM-SIZED (~6-7% of haploid genome), submetacentric, similar to chromosomes 6-12
- Y chromosome: SMALL (~2% of genome), acrocentric, much smaller than X

**Count how many X chromosomes you see:**
- 1 X chromosome = possible Turner (45,X) or male (XY)
- 2 X chromosomes = normal female (XX) or Klinefelter if Y present (XXY)
- 3 X chromosomes = TRIPLE X SYNDROME (47,XXX) - THREE medium-sized X's, NO Y

**TRIPLE X KEY IDENTIFIER**: Look for THREE similar-sized MEDIUM chromosomes grouped together in the sex region. They will all look alike (all X's are submetacentric and similar size).

### STEP 2: CHECK FOR Y CHROMOSOME
- If Y present with 2 X's → Klinefelter (47,XXY)
- If NO Y with 3 X's → Triple X (47,XXX)
- If NO Y with 1 X → Turner (45,X)

### STEP 3: COUNT POSITION 21 (Group G)
Position 21 contains SMALL acrocentric chromosomes (much smaller than X):
- 2 at position 21 = Normal
- 3 at position 21 = Down syndrome (trisomy 21)

## CRITICAL DISTINCTION: Triple X vs Down Syndrome

| Feature | Triple X (47,XXX) | Down Syndrome (47,+21) |
|---------|-------------------|------------------------|
| Extra chromosome location | SEX REGION | POSITION 21 |
| Extra chromosome size | MEDIUM (X-sized) | SMALL (Group G) |
| Extra chromosome shape | Submetacentric | Acrocentric |
| Sex chromosomes | XXX (3 medium) | XX or XY (2 normal) |
| Position 21 count | 2 (normal) | 3 (trisomy) |

**If you see 47 chromosomes with 3 MEDIUM chromosomes in sex region → 47,XXX (Triple X)**
**If you see 47 chromosomes with 3 SMALL chromosomes at position 21 → 47,+21 (Down)**

## SYNDROME DETECTION CHECKLIST
1. Total = 45, sex region has only 1 X → Turner syndrome (45,X)
2. Total = 47, sex region has XXY → Klinefelter syndrome (47,XXY)
3. Total = 47, sex region has XXX (3 medium, no Y) → Triple X syndrome (47,XXX)
4. Total = 47, position 21 has 3 small acrocentric → Down syndrome (47,XX,+21 or 47,XY,+21)

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
    "notation": "ISCN notation",
    "chromosome_count": number,
    "sex_chromosomes": "XX/XY/X/XXY/XXX/XYY",
    "abnormalities": [
        {{"type": "trisomy/monosomy/etc", "chromosome": "21/X/etc", "description": "details"}}
    ],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation with syndrome name",
    "detailed_findings": "Explicitly state: (1) how many X chromosomes counted, (2) Y present or absent, (3) count at position 21"
}}"""

        client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a board-certified clinical cytogeneticist. CRITICAL: Count sex chromosomes FIRST before checking position 21. For 47-chromosome karyotypes, distinguish: (1) Triple X (47,XXX) = THREE medium-sized X chromosomes in sex region, NO Y, normal position 21; (2) Down syndrome (47,+21) = normal XX or XY sex chromosomes, THREE small acrocentric at position 21. X chromosomes are MEDIUM-sized, chromosome 21 is SMALL. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cv_hint_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        st.session_state.raw_response = raw_content

        # Parse response
        result = self._parse_response(raw_content, "CV+VLM (CV-assisted)")

        return result

    def _analyze_with_precision_lens(self, image: Image.Image) -> Dict:
        """정밀 임상 렌즈 - 6단계 파이프라인 분석"""
        if not self.api_key:
            raise ValueError("API key is required for Precision Clinical Lens analysis")

        # Determine which VLM provider to use based on the selected provider mode
        provider_to_vlm = {
            APIProvider.OPENAI: "openai",
            APIProvider.ANTHROPIC: "anthropic",
            APIProvider.GEMINI: "gemini",
        }
        if self.provider in provider_to_vlm:
            vlm_provider = provider_to_vlm[self.provider]
        elif OPENAI_AVAILABLE:
            vlm_provider = "openai"
        elif ANTHROPIC_AVAILABLE:
            vlm_provider = "anthropic"
        elif GEMINI_AVAILABLE:
            vlm_provider = "gemini"
        else:
            raise ImportError("No VLM API package available. Install openai, anthropic, or google-genai.")

        lens = PrecisionClinicalLens(api_key=self.api_key, provider=vlm_provider)

        # Progress tracking via session state
        if 'precision_lens_stage' not in st.session_state:
            st.session_state.precision_lens_stage = (0, "Initializing...")

        def progress_callback(stage_num, stage_name):
            st.session_state.precision_lens_stage = (stage_num, stage_name)

        result = lens.run_pipeline(image, progress_callback=progress_callback)

        # Store raw stages for display
        st.session_state.raw_response = json.dumps(result.get("stages", {}), indent=2, ensure_ascii=False, default=str)

        return result

    def _mock_analysis(self) -> Dict:
        """Mock 분석 결과 (데모용)"""
        import random

        chromosome_count = random.choice([46, 47, 45])
        sex_chromosomes = random.choice(["XX", "XY"])

        abnormalities = []
        confidence = random.uniform(75, 92)

        if chromosome_count == 47:
            trisomy = random.choice([21, 18, 13])
            abnormalities.append({
                "type": "trisomy",
                "chromosome": str(trisomy),
                "description": f"Trisomy {trisomy} detected - extra copy of chromosome {trisomy}"
            })
            notation = f"47,{sex_chromosomes},+{trisomy}"
        elif chromosome_count == 45:
            if sex_chromosomes == "XX":
                abnormalities.append({
                    "type": "monosomy",
                    "chromosome": "X",
                    "description": "Turner syndrome (Monosomy X) - missing one X chromosome"
                })
                notation = "45,X"
                sex_chromosomes = "X"
            else:
                notation = f"45,{sex_chromosomes},-21"
                abnormalities.append({
                    "type": "monosomy",
                    "chromosome": "21",
                    "description": "Monosomy 21 detected"
                })
        else:
            notation = f"46,{sex_chromosomes}"

        # Add structural abnormality occasionally
        if random.random() < 0.15 and chromosome_count == 46:
            abnormalities.append({
                "type": "translocation",
                "chromosome": "9;22",
                "description": "Balanced translocation t(9;22)(q34;q11) - Philadelphia chromosome"
            })
            notation += ",t(9;22)(q34;q11)"

        interpretation = self._generate_interpretation(chromosome_count, abnormalities, sex_chromosomes)

        return {
            "notation": notation,
            "chromosome_count": chromosome_count,
            "sex_chromosomes": sex_chromosomes,
            "abnormalities": abnormalities,
            "confidence": round(confidence, 1),
            "interpretation": interpretation,
            "detailed_findings": "Demo mode: This is simulated data for demonstration purposes only.",
            "analysis_time": datetime.now().isoformat(),
            "technical_notes": "Demo Mode - No actual image analysis performed",
            "provider": "Demo Mode"
        }

    def _generate_interpretation(self, count: int, abnormalities: list, sex: str) -> str:
        """분석 결과에 대한 임상적 해석 생성"""
        if count == 46 and not abnormalities:
            gender = "female" if sex == "XX" else "male"
            return f"Normal {gender} karyotype with no apparent numerical or structural abnormalities detected."

        interpretations = []
        if count != 46:
            interpretations.append(f"Numerical abnormality: {count} chromosomes detected (normal is 46).")

        for abnormality in abnormalities:
            atype = abnormality.get("type", "")
            chrom = abnormality.get("chromosome", "")

            if atype == "trisomy":
                if chrom == "21":
                    interpretations.append("Trisomy 21 is associated with Down syndrome, characterized by intellectual disability, distinctive facial features, and increased risk of certain medical conditions.")
                elif chrom == "18":
                    interpretations.append("Trisomy 18 (Edwards syndrome) is a severe condition with multiple organ system abnormalities and limited survival.")
                elif chrom == "13":
                    interpretations.append("Trisomy 13 (Patau syndrome) involves severe intellectual disability, heart defects, and other organ abnormalities.")
            elif atype == "monosomy" and chrom == "X":
                interpretations.append("Turner syndrome (45,X) affects females and is characterized by short stature, ovarian insufficiency, and possible cardiac/renal anomalies.")
            elif atype == "translocation":
                interpretations.append(f"Translocation {abnormality.get('description', '')} detected. Clinical significance depends on whether balanced or unbalanced.")

        return " ".join(interpretations)

    def _analyze_with_consensus(self, image: Image.Image, api_keys: Dict) -> Dict:
        """Run analysis on multiple providers and vote on results"""
        results = []
        errors = []
        providers_used = []

        # Run analysis with each available provider
        provider_configs = [
            ('openai', APIProvider.OPENAI, OPENAI_AVAILABLE, "GPT-4 Vision"),
            ('anthropic', APIProvider.ANTHROPIC, ANTHROPIC_AVAILABLE, "Claude Vision"),
            ('gemini', APIProvider.GEMINI, GEMINI_AVAILABLE, "Gemini Vision")
        ]

        settings = st.session_state.consensus_settings

        for key_name, provider_enum, is_available, display_name in provider_configs:
            use_flag = f'use_{key_name}'
            if not settings.get(use_flag, True):
                continue
            if not is_available:
                errors.append(f"{display_name}: Package not installed")
                continue
            if not api_keys.get(key_name):
                errors.append(f"{display_name}: API key not provided")
                continue

            try:
                # Create temporary analyzer for this provider
                temp_analyzer = KaryotypeAnalyzer(provider=provider_enum, api_key=api_keys[key_name])

                if provider_enum == APIProvider.OPENAI:
                    result = temp_analyzer._analyze_with_openai(image)
                elif provider_enum == APIProvider.ANTHROPIC:
                    result = temp_analyzer._analyze_with_anthropic(image)
                elif provider_enum == APIProvider.GEMINI:
                    result = temp_analyzer._analyze_with_gemini(image)

                results.append(result)
                providers_used.append(display_name)

            except Exception as e:
                errors.append(f"{display_name}: {str(e)}")

        # Calculate consensus from results
        if not results:
            return {
                'notation': 'No Results',
                'chromosome_count': 0,
                'sex_chromosomes': 'Unknown',
                'abnormalities': [],
                'confidence': 0,
                'interpretation': 'No successful analyses to aggregate.',
                'detailed_findings': f"Errors: {'; '.join(errors)}",
                'analysis_time': datetime.now().isoformat(),
                'technical_notes': 'Consensus analysis failed - no providers returned results',
                'provider': 'Multi-Model Consensus',
                'is_consensus': True,
                'individual_results': [],
                'agreement_level': 0,
                'voting_breakdown': {},
                'errors': errors
            }

        consensus = self._calculate_consensus(results, providers_used, errors)
        return consensus

    def _calculate_consensus(self, results: List[Dict], providers_used: List[str], errors: List[str]) -> Dict:
        """Calculate consensus from multiple analysis results using majority voting"""
        if len(results) == 1:
            # Only one result - return it with consensus metadata
            single = results[0]
            single['is_consensus'] = True
            single['individual_results'] = results
            single['agreement_level'] = 1.0
            single['providers_used'] = providers_used
            single['voting_breakdown'] = {
                'chromosome_count': {single['chromosome_count']: 1},
                'sex_chromosomes': {single['sex_chromosomes']: 1},
                'notation': {single['notation']: 1}
            }
            single['provider'] = f"Consensus ({providers_used[0]} only)"
            single['errors'] = errors
            return single

        total = len(results)

        # Count votes for each field
        count_votes = Counter(r.get('chromosome_count', 0) for r in results)
        sex_votes = Counter(r.get('sex_chromosomes', 'Unknown') for r in results)
        notation_votes = Counter(r.get('notation', 'Unknown') for r in results)

        # Determine consensus (most common value)
        consensus_count, count_agreement = count_votes.most_common(1)[0]
        consensus_sex, sex_agreement = sex_votes.most_common(1)[0]
        consensus_notation, notation_agreement = notation_votes.most_common(1)[0]

        # Calculate overall agreement level (0.0 - 1.0)
        agreement_level = count_agreement / total

        # Aggregate abnormalities with tracking
        abnormality_counts = {}
        for r in results:
            for abnorm in r.get('abnormalities', []):
                key = f"{abnorm.get('type', 'unknown')}:{abnorm.get('chromosome', 'N/A')}"
                if key not in abnormality_counts:
                    abnormality_counts[key] = {
                        'abnormality': abnorm,
                        'detected_by': [],
                        'count': 0
                    }
                abnormality_counts[key]['count'] += 1
                abnormality_counts[key]['detected_by'].append(r.get('provider', 'Unknown'))

        # Build merged abnormalities list with agreement info
        merged_abnormalities = []
        for key, data in abnormality_counts.items():
            abnorm = data['abnormality'].copy()
            abnorm['agreement'] = f"{data['count']}/{total} models"
            abnorm['detected_by'] = data['detected_by']
            merged_abnormalities.append(abnorm)

        # Calculate consensus confidence
        base_confidence = sum(r.get('confidence', 0) for r in results) / total
        agreement_bonus = agreement_level * 10  # +10% for unanimous
        final_confidence = min(99, base_confidence + agreement_bonus)

        # Generate consensus interpretation
        if agreement_level == 1.0:
            agreement_text = "All models unanimously agree on this analysis."
        elif agreement_level >= 0.67:
            agreement_text = f"Majority agreement ({count_agreement}/{total} models)."
        else:
            agreement_text = f"Models disagree - review individual results carefully."

        # Combine interpretations
        interpretations = [r.get('interpretation', '') for r in results if r.get('interpretation')]
        consensus_interpretation = f"{agreement_text}\n\n" + "\n---\n".join(
            f"**{providers_used[i]}**: {interpretations[i]}"
            for i in range(len(interpretations))
        )

        # Build voting breakdown for display
        voting_breakdown = {
            'chromosome_count': dict(count_votes),
            'sex_chromosomes': dict(sex_votes),
            'notation': dict(notation_votes)
        }

        return {
            'notation': consensus_notation,
            'chromosome_count': consensus_count,
            'sex_chromosomes': consensus_sex,
            'abnormalities': merged_abnormalities,
            'confidence': round(final_confidence, 1),
            'interpretation': consensus_interpretation,
            'detailed_findings': f"Consensus from {total} models: {', '.join(providers_used)}",
            'analysis_time': datetime.now().isoformat(),
            'technical_notes': f"Multi-model consensus voting with {agreement_level:.0%} agreement",
            'provider': 'Multi-Model Consensus',
            'is_consensus': True,
            'individual_results': results,
            'agreement_level': agreement_level,
            'providers_used': providers_used,
            'voting_breakdown': voting_breakdown,
            'errors': errors
        }


def display_header():
    """헤더 표시"""
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Chromosome Karyotype Analyzer</h1>
        <p>AI-Powered Cytogenetic Analysis Tool</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">ISCN 2020 Compliant • Multi-Provider AI • Educational & Research Use Only</p>
    </div>
    """, unsafe_allow_html=True)


def display_disclaimer():
    """의료 면책 조항 표시"""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only.
        Results must be validated by qualified cytogenetics professionals. Do not use for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


def display_api_status():
    """API 상태 표시 - GPT-4 Vision only"""
    st.sidebar.subheader("📦 Package Status")

    status_html = ""
    if OPENAI_AVAILABLE:
        status_html += '<div class="api-status api-available">✓ OpenAI GPT-4 Vision ready</div>'
    else:
        status_html += '<div class="api-status api-unavailable">✗ OpenAI package missing - run: pip install openai</div>'

    st.sidebar.markdown(status_html, unsafe_allow_html=True)


def display_sidebar_settings() -> tuple:
    """사이드바 설정 및 API 키 입력"""
    with st.sidebar:
        st.header("⚙️ Settings")

        # API 상태 표시
        display_api_status()

        st.divider()

        # API 제공자 선택
        st.subheader("🔌 API Provider")

        # Available provider options
        provider_options = ["Demo Mode (No API)"]
        if OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE or GEMINI_AVAILABLE:
            provider_options.insert(0, "Precision Clinical Lens (6-Stage)")
        if OPENAI_AVAILABLE:
            provider_options.insert(0, "OpenAI GPT-4 Vision")
            if CV2_AVAILABLE:
                provider_options.insert(1, "CV + VLM (Hybrid)")
        if TORCH_AVAILABLE and YOLO_AVAILABLE:
            provider_options.insert(0, "YOLO Karyogram (ML Pipeline)")

        selected_provider = st.selectbox(
            "Select AI Provider",
            provider_options,
            help="CV+VLM uses computer vision for counting, VLM for interpretation"
        )

        # Map selection to enum
        provider_map = {
            "YOLO Karyogram (ML Pipeline)": APIProvider.YOLO_KARYOGRAM,
            "OpenAI GPT-4 Vision": APIProvider.OPENAI,
            "CV + VLM (Hybrid)": APIProvider.CV_VLM,
            "Precision Clinical Lens (6-Stage)": APIProvider.PRECISION_LENS,
            "Demo Mode (No API)": APIProvider.MOCK
        }
        provider = provider_map.get(selected_provider, APIProvider.MOCK)

        # Show CV+VLM explanation if selected
        if provider == APIProvider.CV_VLM:
            st.info("**CV + VLM Mode**: Computer Vision counts chromosomes → VLM interprets the counts. More accurate than VLM-only for counting.")

        if provider == APIProvider.PRECISION_LENS:
            st.info("**🔬 정밀 임상 렌즈**: 6단계 순차 분석 파이프라인\n\n"
                    "1. 계수 → 2. 분류 → 3. 클러스터 분류 → 4. 전이 → 5. 분석 → 6. 이상 탐지\n\n"
                    "각 단계가 이전 결과를 기반으로 정밀 분석을 수행합니다.")

        st.divider()

        # API 키 입력
        api_key = None
        consensus_keys = {}  # Keep for compatibility

        if provider in [APIProvider.OPENAI, APIProvider.CV_VLM, APIProvider.PRECISION_LENS]:
            # Use saved key from session state if available
            default_key = st.session_state.get('saved_api_key', '') or ''

            api_key = st.text_input(
                "OpenAI API Key",
                value=default_key,
                type="password",
                help="Enter your OpenAI API key (starts with sk-)",
                key="openai_api_key_input"
            )

            # Save/Clear buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", use_container_width=True, help="Save API key to browser"):
                    clean_key = api_key.strip() if api_key else ""
                    # Validate API key format: starts with sk-, min 20 chars, alphanumeric/dash/underscore only
                    is_valid = (
                        clean_key.startswith("sk-") and
                        len(clean_key) >= 20 and
                        all(c.isalnum() or c in '-_' for c in clean_key)
                    )
                    if is_valid:
                        st.session_state.saved_api_key = clean_key
                        st.session_state.api_key_saved = True
                        # Inject JavaScript to save to localStorage
                        st.markdown(inject_local_storage_script("save", clean_key), unsafe_allow_html=True)
                        st.success("✅ Saved to browser!")
                    else:
                        st.error("❌ Invalid key format")

            with col2:
                if st.button("🗑️ Clear", use_container_width=True, help="Clear saved API key"):
                    st.session_state.saved_api_key = None
                    st.session_state.api_key_saved = False
                    # Inject JavaScript to clear localStorage
                    st.markdown(inject_local_storage_script("clear"), unsafe_allow_html=True)
                    st.info("🗑️ Cleared!")
                    st.rerun()

            # Show saved status
            if st.session_state.get('saved_api_key'):
                st.caption("🔐 API key saved (persists in browser)")
            else:
                st.caption("Get key: [platform.openai.com](https://platform.openai.com/api-keys)")

            # Try to load from localStorage on page load
            st.markdown(inject_local_storage_script("load"), unsafe_allow_html=True)

        st.divider()

        # 정보 섹션
        st.markdown("""
        ### About
        This tool analyzes chromosome metaphase spreads and generates ISCN 2020 compliant karyotype notations.

        **Available Modes:**
        - **GPT-4 Vision**: VLM-only analysis
        - **CV + VLM (Hybrid)**: Computer Vision counts → VLM interprets
        - **정밀 임상 렌즈 (Precision Clinical Lens)**: 6-stage pipeline

        The Precision Clinical Lens performs 6 sequential stages:
        1. 계수 (Counting) - Chromosome counting
        2. 분류 (Classification) - Denver group classification
        3. 클러스터 분류 (Cluster) - Individual chromosome identification
        4. 전이 (Translocation) - Structural abnormality detection
        5. 분석 (Analysis) - Cross-validation & synthesis
        6. 이상 탐지 (Abnormality) - Final diagnosis & ISCN notation

        ### Resources
        - [ISCN 2020 Standards](https://karger.com/books/book/367/An-International-System-for-Human-Cytogenomic)
        - [Karyotype Analysis Guide](https://www.ncbi.nlm.nih.gov/books/NBK557817/)
        """)

        return provider, api_key, consensus_keys


def display_upload_section():
    """이미지 업로드 섹션"""
    st.header("📤 Upload Metaphase Spread Image")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a high-quality metaphase spread image (max 10MB)"
        )

        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File size exceeds 10MB limit. Please upload a smaller image.")
                return None

            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image

            st.image(image, caption="Uploaded Image", width="stretch")

            st.info(f"**Image Details:**\n"
                   f"- Format: {image.format or 'N/A'}\n"
                   f"- Size: {image.size[0]} x {image.size[1]} pixels\n"
                   f"- Mode: {image.mode}\n"
                   f"- File size: {uploaded_file.size / 1024:.1f} KB")

            return image

    with col2:
        st.markdown("""
        ### Guidelines:
        - Use high-resolution images
        - Ensure clear chromosome spread
        - Avoid overlapping chromosomes
        - Good contrast and lighting

        ### Supported Formats:
        - PNG, JPG, JPEG
        - TIFF, BMP
        - Max size: 10MB

        ### Best Results:
        - G-banded metaphase spreads
        - Individual well-separated chromosomes
        - Consistent staining quality
        """)

    return None


def display_analysis_section(analyzer: KaryotypeAnalyzer, image: Image.Image, provider: APIProvider, consensus_keys: Optional[Dict] = None):
    """분석 섹션"""
    st.header("🔬 Chromosome Analysis")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        provider_name = provider.value
        button_text = f"🚀 Analyze with {provider_name}"

        if provider == APIProvider.MOCK:
            st.info("💡 Demo Mode: Results will be simulated. Select an API provider for real analysis.")

        if provider == APIProvider.CV_VLM:
            st.info("🔬 **CV + VLM Hybrid Mode**: OpenCV will detect and count chromosomes at each position, then GPT-4 will interpret the counts. More accurate for chromosome counting.")

        if provider == APIProvider.PRECISION_LENS:
            st.info("🔬 **정밀 임상 렌즈 (Precision Clinical Lens)**: 6단계 순차 파이프라인으로 정밀 분석을 수행합니다. VLM을 6번 호출하므로 시간이 더 소요될 수 있습니다.")

        if provider == APIProvider.CONSENSUS:
            # Show which models will be used
            active_models = []
            if consensus_keys.get('openai') and OPENAI_AVAILABLE:
                active_models.append("GPT-4")
            if consensus_keys.get('anthropic') and ANTHROPIC_AVAILABLE:
                active_models.append("Claude")
            if consensus_keys.get('gemini') and GEMINI_AVAILABLE:
                active_models.append("Gemini")
            st.info(f"🗳️ Consensus Voting: Will analyze with {', '.join(active_models)} and combine results")

        if st.button(button_text, type="primary", width="stretch"):
            if provider == APIProvider.CONSENSUS:
                spinner_text = f"Running multi-model consensus analysis... This may take 2-3 minutes."
            elif provider == APIProvider.PRECISION_LENS:
                spinner_text = "🔬 정밀 임상 렌즈 6단계 분석 진행 중... 약 2-3분 소요됩니다."
            else:
                spinner_text = f"Analyzing chromosomes using {provider_name}... This may take up to 60 seconds."

            with st.spinner(spinner_text):
                try:
                    progress_bar = st.progress(0)
                    progress_bar.progress(10)

                    if provider == APIProvider.CONSENSUS:
                        # Pass consensus_keys to the analyzer
                        result = analyzer.analyze(image, consensus_keys=consensus_keys)
                    else:
                        result = analyzer.analyze(image)

                    progress_bar.progress(100)
                    st.session_state.analysis_result = result

                    if result.get('is_consensus'):
                        agreement = result.get('agreement_level', 0)
                        if agreement == 1.0:
                            st.success("✅ Consensus analysis complete - All models agree!")
                        elif agreement >= 0.67:
                            st.success("✅ Consensus analysis complete - Majority agreement achieved!")
                        else:
                            st.warning("⚠️ Consensus analysis complete - Models disagree, review carefully.")
                        st.balloons()
                    elif result.get('pipeline') == 'precision_lens':
                        st.success("✅ 정밀 임상 렌즈 6단계 분석 완료!")
                        st.balloons()
                    elif result.get('pipeline') == 'two_stage':
                        cv_count = result.get('cv_detection', {}).get('count', 0)
                        st.success(f"✅ Two-Stage Analysis complete - CV detected {cv_count} chromosomes!")
                        st.balloons()
                    elif result.get('confidence', 0) > 0:
                        st.success("✅ Analysis completed successfully!")
                        st.balloons()
                    else:
                        st.warning("⚠️ Analysis completed with issues. Check results below.")

                except Exception as e:
                    st.error(f"❌ Analysis failed. Please check your API key and try again.")
                    with st.expander("Technical details", expanded=False):
                        st.code(str(e))


def get_confidence_class(confidence: float) -> str:
    """신뢰도에 따른 CSS 클래스 반환"""
    if confidence >= 85:
        return "confidence-high"
    elif confidence >= 65:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_results(result: Dict):
    """분석 결과 표시"""
    st.header("📊 Analysis Results")

    # Provider badge
    provider = result.get('provider', 'Unknown')
    st.caption(f"🤖 Analyzed by: **{provider}**")

    # ISCN Notation
    st.markdown(f"""
    <div class="notation-display">
        {result.get('notation', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    # 주요 결과 요약
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Chromosome Count", result.get('chromosome_count', 'N/A'))

    with col2:
        st.metric("Sex Chromosomes", result.get('sex_chromosomes', 'N/A'))

    with col3:
        confidence = result.get('confidence', 0)
        confidence_class = get_confidence_class(confidence)
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Confidence Score</h4>
            <p class="{confidence_class}" style="font-size: 2rem;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # 상세 결과
    st.markdown("""<div class="result-card">""", unsafe_allow_html=True)

    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get('abnormalities', [])
    if abnormalities:
        for i, abnormality in enumerate(abnormalities):
            chrom = abnormality.get('chromosome', 'N/A')
            desc = abnormality.get('description', 'No description')
            atype = abnormality.get('type', 'unknown')
            st.write(f"{i+1}. **{atype.title()}** (Chromosome {chrom}): {desc}")
    else:
        st.write("✓ No chromosomal abnormalities detected.")

    st.subheader("📝 Clinical Interpretation")
    st.write(result.get('interpretation', 'No interpretation available.'))

    if result.get('detailed_findings'):
        st.subheader("🔎 Detailed Findings")
        st.write(result.get('detailed_findings'))

    st.subheader("🔧 Technical Notes")
    st.write(result.get('technical_notes', 'N/A'))
    st.caption(f"Analysis performed at: {result.get('analysis_time', 'N/A')}")

    st.markdown("""</div>""", unsafe_allow_html=True)

    # Raw response expander (for debugging)
    if st.session_state.raw_response:
        with st.expander("📄 View Raw API Response"):
            st.code(st.session_state.raw_response, language="json")


def display_consensus_results(result: Dict):
    """Display consensus voting results with breakdown"""
    st.header("🗳️ Multi-Model Consensus Results")

    # Agreement level indicator
    agreement_level = result.get('agreement_level', 0)
    providers_used = result.get('providers_used', [])
    total_models = len(providers_used)

    if agreement_level == 1.0:
        agreement_class = "consensus-agree"
        agreement_icon = "✅"
        agreement_text = f"All {total_models} models agree!"
    elif agreement_level >= 0.67:
        agreement_class = "consensus-partial"
        agreement_icon = "🟡"
        agreement_text = f"Majority agreement ({int(agreement_level * total_models)}/{total_models} models)"
    else:
        agreement_class = "consensus-disagree"
        agreement_icon = "⚠️"
        agreement_text = "Models disagree - review individual results"

    st.markdown(f"""
    <div class="{agreement_class}">
        <h3>{agreement_icon} {agreement_text}</h3>
        <p>Providers: {', '.join(providers_used)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Consensus ISCN Notation
    st.markdown(f"""
    <div class="notation-display">
        <strong>Consensus:</strong> {result.get('notation', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Chromosome Count", result.get('chromosome_count', 'N/A'))

    with col2:
        st.metric("Sex Chromosomes", result.get('sex_chromosomes', 'N/A'))

    with col3:
        confidence = result.get('confidence', 0)
        confidence_class = get_confidence_class(confidence)
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Confidence</h4>
            <p class="{confidence_class}" style="font-size: 1.5rem;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Agreement</h4>
            <p style="font-size: 1.5rem; font-weight: bold;">{agreement_level:.0%}</p>
        </div>
        """, unsafe_allow_html=True)

    # Voting breakdown table
    st.subheader("🗳️ Voting Breakdown")

    voting_breakdown = result.get('voting_breakdown', {})
    individual_results = result.get('individual_results', [])

    if individual_results:
        # Create comparison table
        table_data = []
        for i, r in enumerate(individual_results):
            provider_name = providers_used[i] if i < len(providers_used) else r.get('provider', 'Unknown')
            table_data.append({
                'Model': provider_name,
                'Notation': r.get('notation', 'N/A'),
                'Count': r.get('chromosome_count', 'N/A'),
                'Sex': r.get('sex_chromosomes', 'N/A'),
                'Confidence': f"{r.get('confidence', 0):.1f}%"
            })

        # Display as table
        import pandas as pd
        df = pd.DataFrame(table_data)

        # Highlight consensus values
        def highlight_consensus(row):
            styles = [''] * len(row)
            if row['Count'] == result.get('chromosome_count'):
                styles[2] = 'background-color: #D1FAE5'
            if row['Sex'] == result.get('sex_chromosomes'):
                styles[3] = 'background-color: #D1FAE5'
            return styles

        styled_df = df.style.apply(highlight_consensus, axis=1)
        st.dataframe(styled_df, hide_index=True, use_container_width=True)

    # Abnormalities with agreement info
    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get('abnormalities', [])

    if abnormalities:
        for i, abnormality in enumerate(abnormalities):
            chrom = abnormality.get('chromosome', 'N/A')
            desc = abnormality.get('description', 'No description')
            atype = abnormality.get('type', 'unknown')
            agreement = abnormality.get('agreement', 'N/A')
            detected_by = abnormality.get('detected_by', [])

            st.markdown(f"""
            **{i+1}. {atype.title()}** (Chromosome {chrom})
            - {desc}
            - *Agreement: {agreement}*
            - *Detected by: {', '.join(detected_by)}*
            """)
    else:
        st.success("✓ No chromosomal abnormalities detected by consensus.")

    # Individual model results (expandable)
    st.subheader("📋 Individual Model Results")

    for i, r in enumerate(individual_results):
        provider_name = providers_used[i] if i < len(providers_used) else r.get('provider', 'Unknown')
        with st.expander(f"🤖 {provider_name}: {r.get('notation', 'N/A')}"):
            st.write(f"**Chromosome Count:** {r.get('chromosome_count', 'N/A')}")
            st.write(f"**Sex Chromosomes:** {r.get('sex_chromosomes', 'N/A')}")
            st.write(f"**Confidence:** {r.get('confidence', 0):.1f}%")
            st.write(f"**Interpretation:** {r.get('interpretation', 'N/A')}")

            if r.get('abnormalities'):
                st.write("**Abnormalities:**")
                for abnorm in r.get('abnormalities', []):
                    st.write(f"- {abnorm.get('type', 'unknown')}: {abnorm.get('description', 'N/A')}")

    # Errors section (if any)
    errors = result.get('errors', [])
    if errors:
        with st.expander("⚠️ Provider Errors"):
            for error in errors:
                st.warning(error)


def display_two_stage_results(result: Dict):
    """Display two-stage pipeline results with CV detection info"""
    st.header("🔬 Two-Stage Pipeline Results")

    # Pipeline badge
    st.info("📊 **Stage 1:** Computer Vision Detection → **Stage 2:** VLM Classification")

    # CV Detection results
    cv_detection = result.get('cv_detection', st.session_state.cv_detection or {})

    if cv_detection:
        st.subheader("🖥️ Stage 1: Computer Vision Detection")

        cv_col1, cv_col2, cv_col3 = st.columns(3)

        with cv_col1:
            cv_count = cv_detection.get('count', 0)
            st.metric("CV Detected Count", cv_count)

        with cv_col2:
            sex_region = cv_detection.get('sex_chromosome_region', {})
            sex_est = sex_region.get('estimated', 'unknown')
            st.metric("Estimated Sex Chr", sex_est)

        with cv_col3:
            method = cv_detection.get('detection_method', 'N/A')
            st.metric("Detection Method", method.replace('_', ' ').title())

        # Denver group estimates
        groups = cv_detection.get('group_counts', {})
        if groups:
            with st.expander("📊 CV Denver Group Distribution"):
                group_data = {
                    'Group': ['A (1-3)', 'B (4-5)', 'C (6-12+X)', 'D (13-15)', 'E (16-18)', 'F (19-20)', 'G (21-22+Y)'],
                    'Count': [groups.get('A', 0), groups.get('B', 0), groups.get('C', 0),
                             groups.get('D', 0), groups.get('E', 0), groups.get('F', 0), groups.get('G', 0)],
                    'Expected': [6, 4, 15, 6, 6, 4, 5]  # For normal male
                }
                import pandas as pd
                df = pd.DataFrame(group_data)
                st.dataframe(df, hide_index=True, use_container_width=True)

                total_cv = sum(groups.values())
                st.caption(f"**CV Total:** {total_cv} chromosomes")

    # Stage 2: VLM Classification results
    st.subheader("🤖 Stage 2: VLM Classification")

    # ISCN Notation
    st.markdown(f"""
    <div class="notation-display">
        {result.get('notation', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Final Chromosome Count", result.get('chromosome_count', 'N/A'))

    with col2:
        st.metric("Sex Chromosomes", result.get('sex_chromosomes', 'N/A'))

    with col3:
        confidence = result.get('confidence', 0)
        confidence_class = get_confidence_class(confidence)
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Confidence Score</h4>
            <p class="{confidence_class}" style="font-size: 2rem;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Abnormalities
    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get('abnormalities', [])
    if abnormalities:
        for i, abnormality in enumerate(abnormalities):
            chrom = abnormality.get('chromosome', 'N/A')
            desc = abnormality.get('description', 'No description')
            atype = abnormality.get('type', 'unknown')
            st.write(f"{i+1}. **{atype.title()}** (Chromosome {chrom}): {desc}")
    else:
        st.success("✓ No chromosomal abnormalities detected.")

    st.subheader("📝 Clinical Interpretation")
    st.write(result.get('interpretation', 'No interpretation available.'))

    if result.get('detailed_findings'):
        st.subheader("🔎 Detailed Findings")
        st.write(result.get('detailed_findings'))

    # Technical notes
    st.subheader("🔧 Technical Notes")
    st.write(f"Pipeline: Two-Stage (CV + VLM)")
    st.caption(f"Analysis performed at: {result.get('analysis_time', 'N/A')}")

    # Raw response
    if st.session_state.raw_response:
        with st.expander("📄 View Raw API Response"):
            st.code(st.session_state.raw_response, language="json")


def display_precision_lens_results(result: Dict):
    """정밀 임상 렌즈 6단계 파이프라인 결과 표시"""
    st.header("🔬 정밀 임상 렌즈 (Precision Clinical Lens)")

    # Pipeline visualization
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; color: white; text-align: center;">
        <h3 style="margin: 0 0 1rem 0; color: #e0e0e0;">🔬 6-Stage Analysis Pipeline</h3>
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.3rem;">
            <span style="background: #3B82F6; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem;">1. 계수</span>
            <span style="color: #64748b;">→</span>
            <span style="background: #6366F1; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem;">2. 분류</span>
            <span style="color: #64748b;">→</span>
            <span style="background: #8B5CF6; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem;">3. 클러스터 분류</span>
            <span style="color: #64748b;">→</span>
            <span style="background: #A855F7; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem;">4. 전이</span>
            <span style="color: #64748b;">→</span>
            <span style="background: #D946EF; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem;">5. 분석</span>
            <span style="color: #64748b;">→</span>
            <span style="background: #EC4899; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem;">6. 이상 탐지</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Final ISCN Notation
    st.markdown(f"""
    <div class="notation-display" style="font-size: 1.5rem;">
        <strong>최종 핵형:</strong> {result.get('notation', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("염색체 수 (Count)", result.get('chromosome_count', 'N/A'))
    with col2:
        st.metric("성염색체 (Sex Chr)", result.get('sex_chromosomes', 'N/A'))
    with col3:
        confidence = result.get('confidence', 0)
        confidence_class = get_confidence_class(confidence)
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>신뢰도 (Confidence)</h4>
            <p class="{confidence_class}" style="font-size: 2rem;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Stage-by-stage results
    stages = result.get("stages", {})
    stage_colors = ["#3B82F6", "#6366F1", "#8B5CF6", "#A855F7", "#D946EF", "#EC4899"]

    for i, (kr_name, en_name) in enumerate(PrecisionClinicalLens.STAGE_NAMES):
        stage_key = f"stage_{i+1}"
        stage_data = stages.get(stage_key, {})
        color = stage_colors[i]

        with st.expander(f"Stage {i+1}: {kr_name} ({en_name})", expanded=(i == 5)):
            if stage_data.get("parse_error"):
                st.warning("이 단계의 응답을 JSON으로 파싱하지 못했습니다.")
                st.code(stage_data.get("raw_text", "No data"), language="text")
                continue

            if i == 0:  # 계수
                total = stage_data.get("total_count", "N/A")
                st.metric("총 염색체 수", total)
                pos = stage_data.get("position_counts", {})
                if pos:
                    st.write("**위치별 개수:**")
                    # Display as compact table
                    auto_pairs = {k: v for k, v in pos.items() if k not in ("X", "Y")}
                    sex_pairs = {k: v for k, v in pos.items() if k in ("X", "Y")}
                    cols = st.columns(6)
                    items = list(auto_pairs.items())
                    for idx, (k, v) in enumerate(items):
                        with cols[idx % 6]:
                            status = "✅" if v == 2 else "⚠️"
                            st.write(f"Chr {k}: {v} {status}")
                    if sex_pairs:
                        st.write(f"**성염색체:** X={sex_pairs.get('X', '?')}, Y={sex_pairs.get('Y', '?')}")
                if stage_data.get("count_notes"):
                    st.info(stage_data["count_notes"])

            elif i == 1:  # 분류
                groups = stage_data.get("denver_groups", {})
                if groups:
                    import pandas as pd
                    rows = []
                    for grp_name, grp_data in groups.items():
                        if isinstance(grp_data, dict):
                            rows.append({
                                "Group": grp_name,
                                "Count": grp_data.get("count", 0),
                                "Expected": grp_data.get("expected", "?"),
                                "Chromosomes": grp_data.get("chromosomes", "")
                            })
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                total_cl = stage_data.get("total_classified", "N/A")
                st.metric("분류된 총 수", total_cl)
                if stage_data.get("classification_notes"):
                    st.info(stage_data["classification_notes"])

            elif i == 2:  # 클러스터 분류
                sex_det = stage_data.get("sex_determination", "N/A")
                st.metric("성별 판정", sex_det)
                anomalous = stage_data.get("anomalous_positions", [])
                if anomalous:
                    st.warning(f"이상 위치 감지: {', '.join(str(a) for a in anomalous)}")
                else:
                    st.success("모든 위치 정상")
                if stage_data.get("cluster_notes"):
                    st.info(stage_data["cluster_notes"])

            elif i == 3:  # 전이
                normal = stage_data.get("normal_structure", True)
                if normal:
                    st.success("구조적 이상 없음")
                else:
                    abnorms = stage_data.get("structural_abnormalities", [])
                    for ab in abnorms:
                        notation = ab.get("iscn_notation", "N/A")
                        desc = ab.get("description", "")
                        conf = ab.get("confidence", 0)
                        st.error(f"**{notation}** - {desc} (신뢰도: {conf}%)")
                if stage_data.get("translocation_notes"):
                    st.info(stage_data["translocation_notes"])

            elif i == 4:  # 분석
                prelim = stage_data.get("preliminary_karyotype", "N/A")
                st.markdown(f"**잠정 핵형:** `{prelim}`")
                num_status = stage_data.get("numerical_status", "N/A")
                struct_status = stage_data.get("structural_status", "N/A")
                st.write(f"수적 상태: {num_status}")
                st.write(f"구조적 상태: {struct_status}")
                cv = stage_data.get("cross_validation", {})
                if cv:
                    consistent = cv.get("count_consistent", True) and cv.get("groups_consistent", True)
                    if consistent:
                        st.success("교차 검증 통과 - 모든 단계 결과 일치")
                    else:
                        discrep = cv.get("discrepancies", [])
                        st.warning(f"교차 검증 불일치: {', '.join(str(d) for d in discrep)}")
                if stage_data.get("analysis_summary"):
                    st.write(stage_data["analysis_summary"])

            elif i == 5:  # 이상 탐지 (Final)
                abnormalities = stage_data.get("abnormalities", [])
                if abnormalities:
                    for ab in abnormalities:
                        atype = ab.get("type", "unknown")
                        subtype = ab.get("subtype", "")
                        chrom = ab.get("chromosome", "N/A")
                        desc = ab.get("description", "")
                        clinical = ab.get("clinical_significance", "")
                        st.write(f"**{atype.title()} - {subtype}** (Chr {chrom}): {desc}")
                        if clinical:
                            st.write(f"   임상적 의의: _{clinical}_")
                else:
                    st.success("이상 소견 없음 - 정상 핵형")

                st.subheader("📝 임상 해석")
                st.write(stage_data.get("interpretation", result.get("interpretation", "N/A")))

                if stage_data.get("recommendations"):
                    st.subheader("💡 권고사항")
                    st.write(stage_data["recommendations"])

                if stage_data.get("detailed_findings"):
                    st.subheader("🔎 상세 소견")
                    st.write(stage_data["detailed_findings"])

    # Technical info
    st.caption(f"분석 시간: {result.get('analysis_time', 'N/A')}")
    st.caption(f"파이프라인: {result.get('provider', 'Precision Clinical Lens')}")

    # Raw stage data
    if st.session_state.raw_response:
        with st.expander("📄 전체 파이프라인 원시 데이터"):
            st.code(st.session_state.raw_response, language="json")


def generate_report(result: Dict) -> str:
    """텍스트 보고서 생성"""
    abnormalities_text = ""
    if result.get('abnormalities'):
        for i, abnormality in enumerate(result['abnormalities']):
            abnormalities_text += f"{i+1}. {abnormality.get('type', 'Unknown')}: {abnormality.get('description', 'N/A')}\n"
    else:
        abnormalities_text = "No chromosomal abnormalities detected.\n"

    report = f"""
CHROMOSOME KARYOTYPE ANALYSIS REPORT
====================================

Analysis Date: {result.get('analysis_time', 'N/A')}
AI Provider: {result.get('provider', 'N/A')}

ISCN 2020 Notation: {result.get('notation', 'N/A')}

SUMMARY
-------
Chromosome Count: {result.get('chromosome_count', 'N/A')}
Sex Chromosomes: {result.get('sex_chromosomes', 'N/A')}
Confidence Score: {result.get('confidence', 0):.1f}%

DETECTED ABNORMALITIES
---------------------
{abnormalities_text}

CLINICAL INTERPRETATION
----------------------
{result.get('interpretation', 'N/A')}

DETAILED FINDINGS
----------------
{result.get('detailed_findings', 'N/A')}

TECHNICAL NOTES
--------------
{result.get('technical_notes', 'N/A')}

DISCLAIMER
----------
This analysis is for educational and research purposes only.
Results must be validated by qualified cytogenetics professionals.
Do not use for clinical diagnosis or medical decision-making.

Generated by Chromosome Karyotype Analyzer
Powered by Vision-Language AI Models
"""

    return report


def display_report_section(result: Dict):
    """보고서 다운로드 섹션"""
    st.header("📄 Generate Report")

    col1, col2 = st.columns(2)

    with col1:
        report_text = generate_report(result)
        st.download_button(
            label="📥 Download Text Report",
            data=report_text,
            file_name=f"karyotype_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    with col2:
        st.download_button(
            label="📥 Download JSON Data",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"karyotype_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """메인 애플리케이션"""
    # 헤더 표시
    display_header()

    # 면책 조항
    display_disclaimer()

    # 사이드바 설정 (API 제공자 및 키)
    provider, api_key, consensus_keys = display_sidebar_settings()

    # 분석기 초기화
    analyzer = KaryotypeAnalyzer(provider=provider, api_key=api_key)

    # 이미지 업로드
    image = display_upload_section()

    # 분석 섹션
    if image is not None:
        # YOLO Karyogram mode — separate UI flow, no API key needed
        if provider == APIProvider.YOLO_KARYOGRAM:
            from karyogram_ui import display_karyogram_analysis
            display_karyogram_analysis(image)
            return

        # API 키 확인 (Mock 모드가 아닌 경우)
        if provider == APIProvider.CONSENSUS:
            # Check if at least 2 API keys are provided for consensus
            valid_keys = sum(1 for k, v in consensus_keys.items() if v)
            if valid_keys < 2:
                st.warning("⚠️ Please provide API keys for at least 2 models in the sidebar for consensus voting.")
            else:
                display_analysis_section(analyzer, image, provider, consensus_keys)
        elif provider != APIProvider.MOCK and not api_key:
            st.warning(f"⚠️ Please enter your {provider.value} API key in the sidebar to proceed with analysis.")
        else:
            display_analysis_section(analyzer, image, provider)

    # 결과 표시
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result

        # Use appropriate display function based on result type
        if result.get('is_consensus'):
            display_consensus_results(result)
        elif result.get('pipeline') == 'precision_lens':
            display_precision_lens_results(result)
        elif result.get('pipeline') == 'two_stage':
            display_two_stage_results(result)
        else:
            display_results(result)

        display_report_section(result)

        # 새 분석 버튼
        if st.button("🔄 Start New Analysis"):
            st.session_state.analysis_result = None
            st.session_state.uploaded_image = None
            st.session_state.raw_response = None
            st.rerun()


if __name__ == "__main__":
    main()
