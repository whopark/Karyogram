"""Detector helpers: threshold detection, banding extraction, NMS,
overlap splitting, Denver group estimation, sex chromosome detection,
and karyotype count analysis."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2

from cv.detector_split import watershed_split, concavity_split


def select_best_result(all_results: List[Dict]) -> Dict:
    """Pick detection result closest to expected chromosome count."""
    best_result = None
    best_diff = float("inf")
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


def detect_from_threshold(thresh_img, kernel, min_area, max_area, method_name) -> Dict:
    """Apply morphology and extract chromosome contours from thresholded image."""
    processed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chromosomes, bounding_boxes, areas = [], [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            if 0.15 < aspect_ratio < 20:
                chromosomes.append(contour)
                bounding_boxes.append((x, y, w, h))
                areas.append(area)
    return {"count": len(chromosomes), "chromosomes": chromosomes, "boxes": bounding_boxes, "areas": areas, "method": method_name}


def extract_banding_profiles(gray, contours, boxes, band_bins) -> List[np.ndarray]:
    """Extract intensity profile along each chromosome's major axis."""
    profiles = []
    for contour, (x, y, w, h) in zip(contours, boxes):
        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            profiles.append(np.zeros(band_bins))
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted], -1, 255, -1)
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) < 5:
            col_means = np.mean(roi, axis=1)
            profile = np.interp(np.linspace(0, len(col_means) - 1, band_bins), np.arange(len(col_means)), col_means)
            profiles.append(_normalize_profile(profile))
            continue
        mean_pt = np.mean(pts, axis=0)
        centered = pts - mean_pt
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        major_axis = eigenvectors[:, np.argmax(eigenvalues)]
        projections = centered @ major_axis
        proj_min, proj_max = projections.min(), projections.max()
        if proj_max - proj_min < 1:
            profiles.append(np.zeros(band_bins))
            continue
        sample_positions = np.linspace(proj_min, proj_max, band_bins)
        profile = np.zeros(band_bins)
        perp = np.array([-major_axis[1], major_axis[0]])
        for i, pos in enumerate(sample_positions):
            pt = mean_pt + pos * major_axis
            intensities = []
            for offset in range(-2, 3):
                sx = int(round(pt[0] + offset * perp[0]))
                sy = int(round(pt[1] + offset * perp[1]))
                if 0 <= sy < gray.shape[0] and 0 <= sx < gray.shape[1]:
                    intensities.append(float(gray[sy, sx]))
            profile[i] = np.mean(intensities) if intensities else 0
        profiles.append(_normalize_profile(profile))
    return profiles


def two_stage_nms(boxes, areas, band_profiles, band_sim_thresh, iou_thresh=0.3) -> List[int]:
    """Two-stage NMS: spatial IoU then banding-pattern-aware."""
    if not boxes:
        return []
    n = len(boxes)
    suppressed = [False] * n
    indices = sorted(range(n), key=lambda i: areas[i], reverse=True)
    stage1_keep = []
    for i in indices:
        if suppressed[i]:
            continue
        stage1_keep.append(i)
        for j in indices:
            if j == i or suppressed[j]:
                continue
            if _iou(boxes[i], boxes[j]) > iou_thresh:
                if len(band_profiles[i]) > 0 and len(band_profiles[j]) > 0:
                    similarity = float(np.dot(band_profiles[i], band_profiles[j]))
                    if similarity < band_sim_thresh:
                        continue
                suppressed[j] = True
    return stage1_keep


def split_overlapping_chromosomes(gray, thresh, contours, boxes, areas, min_area, detector) -> Tuple[List, List[Tuple], List[float]]:
    """Detect and split overlapping chromosomes using area heuristic + watershed + concavity."""
    if not areas:
        return contours, boxes, areas
    median_area = float(np.median(areas))
    new_c, new_b, new_a = [], [], []
    split_count = 0
    max_splits = max(10, len(contours) // 4)
    for contour, box, area in zip(contours, boxes, areas):
        if area > median_area * detector.overlap_area_ratio and median_area > 0 and split_count < max_splits:
            sr = watershed_split(gray, thresh, contour, box, min_area)
            if sr and len(sr) > 1:
                for sc, sb, sa in sr:
                    new_c.append(sc); new_b.append(sb); new_a.append(sa)
                split_count += 1
                continue
            sr = concavity_split(gray, contour, box, min_area, detector.concavity_depth_ratio)
            if sr and len(sr) > 1:
                for sc, sb, sa in sr:
                    new_c.append(sc); new_b.append(sb); new_a.append(sa)
                split_count += 1
                continue
        new_c.append(contour); new_b.append(box); new_a.append(area)
    return new_c, new_b, new_a


def estimate_denver_groups(areas, total_count) -> Dict:
    """Estimate chromosome groups based on size distribution."""
    if not areas or total_count == 0:
        return {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}
    max_a = max(areas)
    normalized = [a / max_a for a in areas]
    groups = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}
    for na in normalized:
        if na > 0.85: groups["A"] += 1
        elif na > 0.70: groups["B"] += 1
        elif na > 0.50: groups["C"] += 1
        elif na > 0.35: groups["D"] += 1
        elif na > 0.25: groups["E"] += 1
        elif na > 0.15: groups["F"] += 1
        else: groups["G"] += 1
    return groups


def detect_sex_chromosome_region(image: Image.Image, boxes) -> Dict:
    """Analyze the sex chromosome region of the karyogram."""
    if not boxes:
        return {"x_count": 0, "y_count": 0, "estimated": "unknown"}
    ih, iw = image.height, image.width
    bottom_y, right_x = ih * 0.8, iw * 0.6
    candidates = []
    for box in boxes:
        x, y, w, h = box
        if y + h / 2 > bottom_y and x + w / 2 > right_x:
            candidates.append(box)
    n = len(candidates)
    if n == 0:
        return {"x_count": 0, "y_count": 0, "estimated": "unknown", "region_count": 0}
    c_areas = [w * h for (x, y, w, h) in candidates]
    if n >= 2:
        if min(c_areas) < max(c_areas) * 0.6:
            return {"x_count": 1, "y_count": 1, "estimated": "XY", "region_count": n}
        return {"x_count": 2, "y_count": 0, "estimated": "XX", "region_count": n}
    if n == 1:
        return {"x_count": 1, "y_count": 0, "estimated": "X", "region_count": 1}
    return {"x_count": n, "y_count": 0, "estimated": "unknown", "region_count": n}


def analyze_karyotype_from_counts(total, pos21_count, sex_info) -> Dict:
    """Analyze karyotype based on CV counts."""
    analysis = {"total_chromosomes": total, "position_21_count": pos21_count, "sex_chromosomes": sex_info.get("estimated", "unknown"), "preliminary_diagnosis": "Unknown", "confidence": "low"}
    sex = sex_info.get("estimated", "unknown")
    if total == 46:
        if pos21_count == 2:
            if sex == "XY": analysis["preliminary_diagnosis"] = "46,XY (Normal male)"
            elif sex == "XX": analysis["preliminary_diagnosis"] = "46,XX (Normal female)"
            else: analysis["preliminary_diagnosis"] = "46,?? (Normal count, sex uncertain)"
            analysis["confidence"] = "medium"
        else:
            analysis["preliminary_diagnosis"] = f"46 total but pos21={pos21_count} (needs review)"
    elif total == 47:
        if pos21_count >= 3:
            analysis["preliminary_diagnosis"] = f"47,{sex},+21 (Likely Down syndrome)"
            analysis["confidence"] = "medium"
        elif pos21_count == 2:
            if sex in ["XXY", "XXX", "XYY"]:
                analysis["preliminary_diagnosis"] = f"47,{sex} (Sex chromosome abnormality)"
            else:
                analysis["preliminary_diagnosis"] = "47,?? (Trisomy, location uncertain)"
            analysis["confidence"] = "medium"
        else:
            analysis["preliminary_diagnosis"] = "47 total (abnormality location uncertain)"
    elif total == 45:
        analysis["preliminary_diagnosis"] = "45,? (Possible monosomy)"
    else:
        analysis["preliminary_diagnosis"] = f"{total} chromosomes (unusual count)"
    return analysis


# -- private helpers ----------------------------------------------------

def _normalize_profile(profile: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


def _iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0
