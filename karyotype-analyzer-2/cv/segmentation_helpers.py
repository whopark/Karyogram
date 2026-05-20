"""Helper for SegmentationMatrix: stitching-based overlap recovery."""

import numpy as np
from typing import List, Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


def stitch_from_overlap(
    gray: np.ndarray,
    semantic_map: np.ndarray,
    overlap_mask: np.ndarray,
    contours: List,
    min_area: int,
) -> List[Tuple]:
    """Recover individual chromosomes from overlap regions via
    gradient-direction-guided pixel assignment."""
    if not CV2_AVAILABLE:
        return []

    h, w = gray.shape
    results: List[Tuple] = []

    num_labels, overlap_labels = cv2.connectedComponents(overlap_mask)

    for label_id in range(1, num_labels):
        blob_mask = (overlap_labels == label_id).astype(np.uint8) * 255
        blob_contours, _ = cv2.findContours(
            blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not blob_contours:
            continue

        blob_contour = blob_contours[0]
        bx, by, bw, bh = cv2.boundingRect(blob_contour)

        dilated_blob = cv2.dilate(
            blob_mask, np.ones((5, 5), np.uint8), iterations=2
        )

        touching_indices = []
        for idx, c in enumerate(contours):
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            intersection = cv2.bitwise_and(c_mask, dilated_blob)
            if np.sum(intersection) > 0:
                touching_indices.append(idx)

        if len(touching_indices) < 2:
            continue

        roi_gray = gray[by : by + bh, bx : bx + bw]
        if roi_gray.size == 0:
            continue

        gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(gy, gx)

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

        blob_roi_mask = blob_mask[by : by + bh, bx : bx + bw]
        assignment_map = np.full((bh, bw), -1, dtype=np.int32)

        ys, xs = np.where(blob_roi_mask > 0)
        for py, px in zip(ys, xs):
            pixel_angle = (
                angles[py, px]
                if py < angles.shape[0] and px < angles.shape[1]
                else 0
            )
            best_idx = 0
            best_score = float("inf")
            for ci, (tidx, direction, center) in enumerate(
                zip(touching_indices, chr_directions, chr_centers)
            ):
                angle_diff = abs(pixel_angle - direction)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                dist = np.sqrt(
                    (px + bx - center[0]) ** 2 + (py + by - center[1]) ** 2
                )
                score = angle_diff * 50 + dist
                if score < best_score:
                    best_score = score
                    best_idx = ci
            assignment_map[py, px] = best_idx

        for ci in range(len(touching_indices)):
            split_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(
                split_mask, [contours[touching_indices[ci]]], -1, 255, -1
            )
            assigned_roi = (assignment_map == ci).astype(np.uint8) * 255
            split_mask[by : by + bh, bx : bx + bw] = cv2.bitwise_or(
                split_mask[by : by + bh, bx : bx + bw], assigned_roi
            )
            split_contours, _ = cv2.findContours(
                split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for sc in split_contours:
                a = cv2.contourArea(sc)
                if a >= min_area * 0.4:
                    sx, sy, sw, sh = cv2.boundingRect(sc)
                    results.append((sc, (sx, sy, sw, sh), a))

    return results
