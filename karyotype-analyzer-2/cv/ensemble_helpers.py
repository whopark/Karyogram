"""Ensemble strategy implementations: Simple CNN, Siamese, SRAS,
VariFocal Fusion, and Multi-Task."""

import numpy as np
from typing import Dict, List, Optional

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2

from cv.classifier import CHROMOSOME_TEMPLATES


def strategy_simple_cnn(gray, contours, boxes, areas, total_area, bc) -> List[Dict]:
    """Lightweight feature-based classification (size + centromere only)."""
    results = []
    for contour, box, area in zip(contours, boxes, areas):
        size_ratio = (area / total_area) * 100
        ci = bc._extract_features(gray, contour, box, area, total_area, None, None)[
            "centromere_index"
        ] if hasattr(bc, '_extract_features') else 0.5
        ci = _estimate_ci(gray, contour, box, bc)
        x, y, w, h = box
        ar = h / w if w > 0 else 1.0
        best_class, best_score = "21", 0
        for chr_name, tmpl in CHROMOSOME_TEMPLATES.items():
            s_diff = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
            s_score = np.exp(-2.0 * s_diff ** 2)
            ci_diff = abs(ci - tmpl["ci"])
            c_score = max(0, 1.0 - ci_diff * 3.0)
            score = 0.6 * s_score + 0.4 * c_score
            if score > best_score:
                best_score = score
                best_class = chr_name
        results.append({
            "class": best_class, "confidence": best_score,
            "features": {"size_ratio": size_ratio, "centromere_index": ci, "aspect_ratio": ar},
        })
    return results


def strategy_siamese(gray, contours, boxes, areas, band_profiles, total_area, bc) -> List[Dict]:
    """Pairwise distance metric classification using banding profile cosine similarity."""
    results = []
    for i, (contour, box, area) in enumerate(zip(contours, boxes, areas)):
        size_ratio = (area / total_area) * 100
        profile = band_profiles[i] if i < len(band_profiles) else np.zeros(32)
        ci = _estimate_ci(gray, contour, box, bc)
        best_class, best_score = "21", 0
        for chr_name, tmpl in CHROMOSOME_TEMPLATES.items():
            size_dist = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
            size_sim = np.exp(-1.5 * size_dist ** 2)
            ci_sim = np.exp(-3.0 * (ci - tmpl["ci"]) ** 2)
            profile_score = _banding_profile_score(profile, tmpl)
            score = 0.40 * size_sim + 0.30 * ci_sim + 0.30 * profile_score
            if score > best_score:
                best_score = score
                best_class = chr_name
        results.append({"class": best_class, "confidence": best_score})
    return results


def strategy_sras(gray, contours, boxes, areas, total_area, bc) -> List[Dict]:
    """Super-resolution enhanced classification."""
    results = []
    target_h = 64
    for contour, box, area in zip(contours, boxes, areas):
        x, y, w, h = box
        size_ratio = (area / total_area) * 100
        roi = gray[y : y + h, x : x + w]
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
            blurred = cv2.GaussianBlur(roi_sr, (0, 0), 1.0)
            roi_sr = cv2.addWeighted(roi_sr, 1.5, blurred, -0.5, 0)
        else:
            roi_sr, mask_sr = roi, mask

        sr_h = roi_sr.shape[0]
        widths = np.array([np.sum(mask_sr[r, :] > 0) for r in range(sr_h)], dtype=np.float64)
        if len(widths) > 5:
            ks = min(5, len(widths) // 2) | 1
            widths = np.convolve(widths, np.ones(ks) / ks, mode="same")
        margin = max(1, int(sr_h * 0.15))
        search = widths[margin : sr_h - margin]
        ci = (np.argmin(search) + margin) / sr_h if len(search) > 0 else 0.5

        best_class, best_score = "21", 0
        for chr_name, tmpl in CHROMOSOME_TEMPLATES.items():
            s_diff = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
            s_score = np.exp(-2.0 * s_diff ** 2)
            c_score = np.exp(-3.0 * (ci - tmpl["ci"]) ** 2)
            from cv.classifier_helpers import type_compatibility
            t_score = type_compatibility(ci, tmpl["type"])
            score = 0.45 * s_score + 0.35 * c_score + 0.20 * t_score
            if score > best_score:
                best_score = score
                best_class = chr_name
        results.append({"class": best_class, "confidence": best_score})
    return results


def strategy_varifocal(
    gray, contours, boxes, areas, band_profiles, straightened, total_area, bc
) -> List[Dict]:
    """Global shape + local banding feature fusion."""
    results = []
    for i, (contour, box, area) in enumerate(zip(contours, boxes, areas)):
        x, y, w, h = box
        size_ratio = (area / total_area) * 100
        profile = band_profiles[i] if i < len(band_profiles) else np.zeros(32)
        ci = _estimate_ci(gray, contour, box, bc)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if len(profile) >= 32 and np.linalg.norm(profile) > 0:
            ci_bin = max(1, min(31, int(ci * 32)))
            band_contrast = float(np.max(profile) - np.min(profile))
            mean_val = np.mean(profile)
            dark_bands = int(np.sum(profile < mean_val * 0.8))
        else:
            band_contrast = 0.5
            dark_bands = 4

        best_class, best_score = "21", 0
        from cv.classifier_helpers import type_compatibility
        for chr_name, tmpl in CHROMOSOME_TEMPLATES.items():
            s_diff = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
            g_size = np.exp(-2.0 * s_diff ** 2)
            g_ci = np.exp(-3.0 * (ci - tmpl["ci"]) ** 2)
            global_score = 0.5 * g_size + 0.3 * g_ci + 0.2 * solidity
            t_compat = type_compatibility(ci, tmpl["type"])
            expected_bands = max(2, int(tmpl["size_pct"] * 1.5))
            band_diff = abs(dark_bands - expected_bands) / max(expected_bands, 1)
            local_band = max(0, 1.0 - band_diff * 0.5)
            local_score = 0.5 * t_compat + 0.3 * local_band + 0.2 * min(1.0, band_contrast)
            score = 0.55 * global_score + 0.45 * local_score
            if score > best_score:
                best_score = score
                best_class = chr_name
        results.append({"class": best_class, "confidence": best_score})
    return results


def strategy_multitask(
    gray, contours, boxes, areas, band_profiles, straightened, total_area, bc
) -> List[Dict]:
    """Multi-task learning simulation combining classification, segmentation
    quality, and banding pattern matching."""
    results = []
    for i, (contour, box, area) in enumerate(zip(contours, boxes, areas)):
        x, y, w, h = box
        size_ratio = (area / total_area) * 100
        ci = _estimate_ci(gray, contour, box, bc)
        profile = band_profiles[i] if i < len(band_profiles) else np.zeros(32)

        class_scores: Dict[str, float] = {}
        for chr_name, tmpl in CHROMOSOME_TEMPLATES.items():
            s_diff = abs(size_ratio - tmpl["size_pct"]) / max(tmpl["size_pct"], 0.1)
            s_score = np.exp(-2.0 * s_diff ** 2)
            c_score = np.exp(-3.0 * (ci - tmpl["ci"]) ** 2)
            class_scores[chr_name] = 0.55 * s_score + 0.45 * c_score

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        seg_quality = 1.0
        if solidity < 0.5 or solidity > 0.99:
            seg_quality *= 0.8
        if circularity > 0.6:
            seg_quality *= 0.7

        band_quality = 0.5
        if np.linalg.norm(profile) > 0:
            band_quality = min(1.0, float(np.var(profile)) * 5 + 0.3)

        curvature = 0.0
        if straightened and i < len(straightened):
            curvature = straightened[i].get("curvature", 0.0)
        straightness_bonus = max(0.8, 1.0 - curvature)
        quality_modifier = seg_quality * band_quality * straightness_bonus

        best_class = max(class_scores, key=class_scores.get)
        best_score = class_scores[best_class] * quality_modifier
        results.append({"class": best_class, "confidence": best_score})
    return results


# -- Internal helpers ---------------------------------------------------

def _estimate_ci(gray, contour, box, bc) -> float:
    """Use the base classifier to estimate centromere index."""
    from cv.classifier_helpers import estimate_centromere_index
    return estimate_centromere_index(gray, contour, box)


def _banding_profile_score(profile: np.ndarray, tmpl: Dict) -> float:
    """Score banding profile match against template type."""
    if np.linalg.norm(profile) > 0:
        profile_skew = float(np.mean(profile[:16]) - np.mean(profile[16:]))
        if tmpl["type"] == "metacentric" and abs(profile_skew) < 0.1:
            return 0.7
        if tmpl["type"] == "acrocentric" and abs(profile_skew) > 0.05:
            return 0.7
        if tmpl["type"] == "submetacentric":
            return 0.6
    return 0.5
