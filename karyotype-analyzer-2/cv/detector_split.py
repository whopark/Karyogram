"""Chromosome overlap splitting helpers: watershed and concavity split."""

import numpy as np
from typing import List, Optional, Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


def watershed_split(gray, thresh, contour, box, min_area):
    """Watershed-based splitting of overlapping chromosomes."""
    x, y, w, h = box
    pad = 5
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi_g = gray[y1:y2, x1:x2]
    roi_t = thresh[y1:y2, x1:x2].copy()
    if roi_g.size == 0 or roi_t.size == 0:
        return None
    dt = cv2.distanceTransform(roi_t, cv2.DIST_L2, 5)
    if dt.max() == 0:
        return None
    _, sf = cv2.threshold(dt, 0.5 * dt.max(), 255, 0)
    sf = np.uint8(sf)
    nl, labels = cv2.connectedComponents(sf)
    if nl < 3:
        return None
    markers = labels.copy()
    sb = cv2.dilate(roi_t, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sb, sf)
    markers[unknown == 255] = 0
    color = cv2.cvtColor(roi_g, cv2.COLOR_GRAY2BGR) if len(roi_g.shape) == 2 else roi_g
    markers = np.int32(markers)
    cv2.watershed(color, markers)
    results = []
    for lid in range(1, nl):
        m = np.uint8(markers == lid) * 255
        scs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sc in scs:
            a = cv2.contourArea(sc)
            if a >= min_area * 0.5:
                sc_s = sc + np.array([x1, y1])
                sx, sy, sw, sh = cv2.boundingRect(sc_s)
                results.append((sc_s, (sx, sy, sw, sh), a))
    return results if len(results) > 1 else None


def concavity_split(gray, contour, box, min_area, depth_ratio):
    """Concavity-based splitting at deep inward pinch points."""
    if len(contour) < 10:
        return None
    x, y, w, h = box
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return None
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return None
    if defects is None:
        return None
    contour_size = max(w, h)
    min_depth = contour_size * depth_ratio
    deep = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth > min_depth:
            deep.append((tuple(contour[f][0]), depth))
    if len(deep) < 2:
        return None
    deep.sort(key=lambda x: x[1], reverse=True)
    pt1, pt2 = deep[0][0], deep[1][0]
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    cv2.line(mask, pt1, pt2, 0, 2)
    scs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for sc in scs:
        a = cv2.contourArea(sc)
        if a >= min_area * 0.5:
            sx, sy, sw, sh = cv2.boundingRect(sc)
            results.append((sc, (sx, sy, sw, sh), a))
    return results if len(results) > 1 else None
