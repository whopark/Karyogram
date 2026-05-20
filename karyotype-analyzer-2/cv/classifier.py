"""24-class chromosome classifier using morphometric features.

Classifies each chromosome into one of 24 classes (chr1-22, X, Y) using
Gaussian similarity scoring on size, centromere index, banding, and aspect
ratio, with pair-based refinement enforcing autosome=2.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2

from cv.classifier_helpers import (
    estimate_centromere_index,
    super_resolve_banding,
    type_compatibility,
)

# Reference chromosome properties based on ISCN 2020 ideogram data.
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


class ChromosomeClassifier:
    """24-class chromosome classifier with pair-based refinement."""

    CHROMOSOME_TEMPLATES = CHROMOSOME_TEMPLATES

    def __init__(self):
        self.super_res_target = 64
        self.feature_weights = {
            "size": 0.45,
            "centromere": 0.30,
            "banding": 0.15,
            "aspect_ratio": 0.10,
        }
        self.smote_noise = {"size": 0.05, "centromere": 0.03, "banding": 0.08}

    def classify_all(
        self,
        gray: np.ndarray,
        contours: List,
        boxes: List[Tuple],
        areas: List[float],
        band_profiles: List[np.ndarray],
        straightened: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Classify each chromosome into one of 24 classes."""
        if not contours:
            return []
        total_area = sum(areas) if areas else 1
        features_list = []
        for i, (contour, box, area) in enumerate(zip(contours, boxes, areas)):
            features = self._extract_features(
                gray, contour, box, area, total_area,
                band_profiles[i] if i < len(band_profiles) else None,
                straightened[i] if straightened and i < len(straightened) else None,
            )
            features_list.append(features)

        results = []
        for features in features_list:
            scores = self._compute_class_scores(features)
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
        results = self._refine_with_pairing(results)
        return results

    def _extract_features(self, gray, contour, box, area, total_area,
                          band_profile, straightened_info) -> Dict:
        x, y, w, h = box
        size_ratio = (area / total_area) * 100 if total_area > 0 else 0
        aspect_ratio = h / w if w > 0 else 1.0
        ci = estimate_centromere_index(gray, contour, box)
        enhanced_profile = super_resolve_banding(
            gray, contour, box, band_profile, self.super_res_target
        )
        curvature = 0.0
        if straightened_info:
            curvature = straightened_info.get("curvature", 0.0)
        return {
            "size_ratio": size_ratio, "centromere_index": ci,
            "aspect_ratio": aspect_ratio, "banding_profile": enhanced_profile,
            "curvature": curvature, "area": area,
        }

    def _compute_class_scores(self, features: Dict) -> Dict[str, float]:
        scores = {}
        size_ratio = features["size_ratio"]
        ci = features["centromere_index"]
        ar = features["aspect_ratio"]
        for chr_name, template in self.CHROMOSOME_TEMPLATES.items():
            size_diff = abs(size_ratio - template["size_pct"]) / max(template["size_pct"], 0.1)
            size_score = np.exp(-2.0 * size_diff ** 2)
            ci_diff = abs(ci - template["ci"])
            ci_score = max(0, 1.0 - ci_diff * 3.0)
            t_score = type_compatibility(ci, template["type"])
            expected_ar = 2.0 + template["size_pct"] * 0.3
            ar_diff = abs(ar - expected_ar) / max(expected_ar, 1)
            ar_score = max(0, 1.0 - ar_diff * 0.5)
            total = (
                self.feature_weights["size"] * size_score
                + self.feature_weights["centromere"] * ci_score
                + self.feature_weights["banding"] * t_score
                + self.feature_weights["aspect_ratio"] * ar_score
            )
            if template["group"] in ("A", "B"):
                total *= 1.02
            scores[chr_name] = total
        return scores

    def _refine_with_pairing(self, results: List[Dict]) -> List[Dict]:
        """Enforce autosome=2 constraint via iterative reassignment."""
        if not results:
            return results
        class_counts: Dict[str, int] = {}
        for r in results:
            cls = r["predicted_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        for _ in range(10):
            changed = False
            order = sorted(range(len(results)), key=lambda i: results[i]["confidence"])
            for i in order:
                r = results[i]
                cls = r["predicted_class"]
                expected_max = 2
                if cls == "X":
                    expected_max = 3
                elif cls == "Y":
                    expected_max = 2
                if class_counts.get(cls, 0) > expected_max:
                    features = r.get("features", {})
                    candidates = (
                        sorted(self._compute_class_scores(features).items(),
                               key=lambda x: x[1], reverse=True)
                        if features else r["top3"]
                    )
                    for alt_cls, alt_score in candidates:
                        if alt_cls == cls:
                            continue
                        alt_max = 2 if alt_cls not in ("X", "Y") else 3
                        if class_counts.get(alt_cls, 0) < alt_max:
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
        class_counts: Dict[str, int] = {}
        for r in classifications:
            cls = r["predicted_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        x_count = class_counts.get("X", 0)
        y_count = class_counts.get("Y", 0)
        sex_map = {
            (2, 0): "XX", (1, 1): "XY", (1, 0): "X",
            (2, 1): "XXY", (3, 0): "XXX", (1, 2): "XYY",
        }
        sex_chr = sex_map.get((x_count, y_count), f"X{x_count}Y{y_count}")
        total = sum(class_counts.values())

        abnormalities = []
        for chr_num in [str(i) for i in range(1, 23)]:
            count = class_counts.get(chr_num, 0)
            if count > 2:
                abnormalities.append({
                    "type": "trisomy", "chromosome": chr_num, "count": count,
                    "description": f"Trisomy {chr_num}: {count} copies detected",
                })
            elif count < 2:
                abnormalities.append({
                    "type": "monosomy", "chromosome": chr_num, "count": count,
                    "description": f"Monosomy {chr_num}: {count} copies detected",
                })

        notation = f"{total},{sex_chr}"
        if abnormalities or sex_chr not in ("XX", "XY"):
            for ab in abnormalities:
                if ab["type"] == "trisomy":
                    notation += f",+{ab['chromosome']}"
                elif ab["type"] == "monosomy" and ab["count"] == 0:
                    notation += f",-{ab['chromosome']}"

        group_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}
        for r in classifications:
            grp = r.get("denver_group", "?")
            if grp in group_dist:
                group_dist[grp] += 1

        return {
            "total_count": total, "sex_chromosomes": sex_chr,
            "notation": notation, "class_counts": class_counts,
            "abnormalities": abnormalities, "group_distribution": group_dist,
            "avg_confidence": round(
                sum(r["confidence"] for r in classifications) / len(classifications), 3
            ) if classifications else 0,
        }
