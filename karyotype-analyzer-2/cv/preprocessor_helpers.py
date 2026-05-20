"""Helper functions for DigitalPreprocessor: skeleton extraction,
point ordering, curve smoothing, and curvature computation."""

import numpy as np
from typing import Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


def extract_skeleton(mask: np.ndarray) -> np.ndarray:
    """Fast skeleton extraction via distance transform peak ridges."""
    if mask.size == 0 or np.sum(mask) == 0:
        return np.zeros_like(mask)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if dist.max() == 0:
        return np.zeros_like(mask)

    dist_norm = (dist / dist.max() * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(dist_norm, cv2.CV_64F)

    skeleton = np.zeros_like(mask)
    threshold = dist.max() * 0.3
    skeleton[(dist > threshold) & (laplacian < 0)] = 255

    kernel = np.ones((2, 2), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel, iterations=1)
    return skeleton


def order_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
    """Order skeleton pixels into a connected path from one endpoint to another."""
    points = np.column_stack(np.where(skeleton > 0))
    if len(points) == 0:
        return np.array([])

    points = points[:, ::-1].astype(np.float64)

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

    start = endpoints[0] if endpoints else points[0]

    remaining = list(range(len(points)))
    ordered = []
    dists = np.sum((points - start) ** 2, axis=1)
    current_idx = int(np.argmin(dists))
    ordered.append(current_idx)
    remaining.remove(current_idx)

    while remaining:
        current_pt = points[ordered[-1]]
        min_dist = float("inf")
        best_idx = remaining[0]
        for idx in remaining:
            d = np.sum((points[idx] - current_pt) ** 2)
            if d < min_dist:
                min_dist = d
                best_idx = idx
        if min_dist > 8:
            break
        ordered.append(best_idx)
        remaining.remove(best_idx)

    return points[ordered]


def smooth_curve(points: np.ndarray, num_points: int = 50) -> np.ndarray:
    """Smooth an ordered point set using moving average."""
    if len(points) < 3:
        return points

    window = min(5, len(points) // 2)
    if window < 2:
        return points

    smoothed_x = np.convolve(points[:, 0], np.ones(window) / window, mode="valid")
    smoothed_y = np.convolve(points[:, 1], np.ones(window) / window, mode="valid")
    smoothed = np.column_stack([smoothed_x, smoothed_y])

    if len(smoothed) > 2:
        cumlen = np.zeros(len(smoothed))
        for i in range(1, len(smoothed)):
            cumlen[i] = cumlen[i - 1] + np.linalg.norm(smoothed[i] - smoothed[i - 1])
        if cumlen[-1] > 0:
            target_positions = np.linspace(0, cumlen[-1], num_points)
            resampled_x = np.interp(target_positions, cumlen, smoothed[:, 0])
            resampled_y = np.interp(target_positions, cumlen, smoothed[:, 1])
            return np.column_stack([resampled_x, resampled_y])

    return smoothed


def compute_curvature(axis_points: np.ndarray) -> float:
    """Compute average curvature of the medial axis."""
    if len(axis_points) < 3:
        return 0.0

    curvatures = []
    for i in range(1, len(axis_points) - 1):
        p0 = axis_points[i - 1]
        p1 = axis_points[i]
        p2 = axis_points[i + 1]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        denom = l1 * l2
        if denom > 1e-8:
            curvatures.append(cross / denom)

    return float(np.mean(curvatures)) if curvatures else 0.0
