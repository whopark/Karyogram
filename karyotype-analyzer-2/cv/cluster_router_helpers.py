"""Cluster router helpers: route A/B/C segmentation dispatch functions."""

import numpy as np
from typing import List, Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


def route_a_touching(
    gray, binary_mask, contours, boxes, areas, indices, min_area, segmenter
) -> List[Tuple]:
    """Simple separation for merely touching chromosomes via localized
    instance segmentation (fast marker watershed)."""
    h, w = gray.shape
    cluster_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in indices:
        cv2.drawContours(cluster_mask, [contours[idx]], -1, 255, -1)

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

    inst_result = segmenter.instance_segmentation(roi_gray, roi_mask, min_area)

    if inst_result["count"] >= len(indices):
        results = []
        for inst in inst_result["instances"]:
            c = inst["contour"] + np.array([rx1, ry1])
            ix, iy, iw, ih = cv2.boundingRect(c)
            results.append((c, (ix, iy, iw, ih), inst["area"]))
        return results

    return [(contours[i], boxes[i], areas[i]) for i in indices]


def route_b_one_overlap(
    gray, binary_mask, contours, boxes, areas, indices, min_area, segmenter
) -> List[Tuple]:
    """Gradient-guided semantic stitching for single overlap regions."""
    h, w = gray.shape
    cluster_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in indices:
        cv2.drawContours(cluster_mask, [contours[idx]], -1, 255, -1)

    sem_result = segmenter.semantic_segmentation(gray, cluster_mask)
    overlap_mask = sem_result.get(
        "overlap_mask", np.zeros((h, w), dtype=np.uint8)
    )

    if sem_result.get("overlap_pixel_count", 0) > 0:
        cluster_contours = [contours[i] for i in indices]
        from cv.segmentation_helpers import stitch_from_overlap

        stitched = stitch_from_overlap(
            gray, sem_result["semantic_map"], overlap_mask, cluster_contours, min_area
        )
        if stitched and len(stitched) >= len(indices):
            return stitched

    return [(contours[i], boxes[i], areas[i]) for i in indices]


def route_c_multi_overlap(
    gray, binary_mask, contours, boxes, areas, indices, min_area, segmenter
) -> List[Tuple]:
    """Multi-pass segmentation for complex overlapping clusters.
    Combines semantic stitching then instance watershed."""
    h, w = gray.shape
    cluster_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in indices:
        cv2.drawContours(cluster_mask, [contours[idx]], -1, 255, -1)

    sem_result = segmenter.semantic_segmentation(gray, cluster_mask)
    overlap_mask = sem_result.get(
        "overlap_mask", np.zeros((h, w), dtype=np.uint8)
    )

    current_contours = [contours[i] for i in indices]
    current_boxes = [boxes[i] for i in indices]
    current_areas = [areas[i] for i in indices]

    if sem_result.get("overlap_pixel_count", 0) > 0:
        from cv.segmentation_helpers import stitch_from_overlap

        stitched = stitch_from_overlap(
            gray, sem_result["semantic_map"], overlap_mask, current_contours, min_area
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

    return [
        (c, b, a)
        for c, b, a in zip(current_contours, current_boxes, current_areas)
    ]
