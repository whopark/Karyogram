"""ChromosomeDetector: orchestrates the full CV pipeline from image input
to classified karyotype output."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2

from cv.preprocessor import DigitalPreprocessor
from cv.segmentation import SegmentationMatrix
from cv.cluster_router import ClusterRouter
from cv.ensemble import EnsembleClassifier
from cv.detector_helpers import (
    select_best_result,
    detect_from_threshold,
    extract_banding_profiles,
    two_stage_nms,
    split_overlapping_chromosomes,
    estimate_denver_groups,
    detect_sex_chromosome_region,
    analyze_karyotype_from_counts,
)


class ChromosomeDetector:
    """Enhanced CV-based chromosome detection with banding pattern mining,
    two-stage NMS, and overlap-aware splitting."""

    def __init__(self):
        self.base_min_area = 200
        self.base_max_area = 150000
        self.band_profile_bins = 32
        self.band_similarity_threshold = 0.85
        self.overlap_area_ratio = 2.5
        self.concavity_depth_ratio = 0.20

    def detect_chromosomes(self, image: Image.Image) -> Dict:
        """Full chromosome detection pipeline: denoise, threshold, route,
        segment, straighten, NMS, ensemble classify."""
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available", "count": 0}

        img_array = np.array(image)
        gray_raw = (
            img_array
            if len(img_array.shape) == 2
            else cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        )

        preprocessor = DigitalPreprocessor()
        gray, denoise_metadata = preprocessor.denoise(gray_raw)
        img_height, img_width = gray.shape
        img_area = img_height * img_width
        scale_factor = img_area / (1000 * 1000)
        min_area = max(100, int(self.base_min_area * scale_factor))
        max_area = int(self.base_max_area * scale_factor)
        kernel = np.ones((3, 3), np.uint8)
        kernel_small = np.ones((2, 2), np.uint8)

        # Multi-threshold contour extraction
        _, otsu_thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        all_results = [detect_from_threshold(otsu_thresh, kernel, min_area, max_area, "otsu")]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5,
        )
        all_results.append(detect_from_threshold(adaptive_thresh, kernel, min_area, max_area, "adaptive"))
        for tv in [100, 127, 150, 180]:
            _, st = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY_INV)
            all_results.append(detect_from_threshold(st, kernel_small, min_area, max_area, f"simple_{tv}"))
        small_min = max(50, min_area // 3)
        all_results.append(detect_from_threshold(otsu_thresh, kernel, small_min, max_area, "otsu_small"))
        best = select_best_result(all_results)

        # Cluster router + segmentation matrix
        segmenter = SegmentationMatrix()
        router = ClusterRouter()
        if best["count"] < 55:
            routed_c, routed_b, routed_a, router_meta = router.route_and_segment(
                gray, otsu_thresh, best["chromosomes"], best["boxes"], best["areas"],
                min_area, segmenter,
            )
        else:
            routed_c, routed_b, routed_a = best["chromosomes"], best["boxes"], best["areas"]
            router_meta = {"routing": "skipped_high_count"}

        if len(routed_c) < 50:
            split_c, split_b, split_a = split_overlapping_chromosomes(
                gray, otsu_thresh, routed_c, routed_b, routed_a, min_area, self,
            )
        else:
            split_c, split_b, split_a = routed_c, routed_b, routed_a

        straightened = preprocessor.straighten_all(gray, split_c, split_b)
        band_profiles = []
        for i, (contour, box) in enumerate(zip(split_c, split_b)):
            if i < len(straightened) and straightened[i]["strip"].size > 0:
                strip = straightened[i]["strip"]
                if strip.shape[0] > 0 and strip.shape[1] > 0:
                    col_means = np.mean(strip, axis=1).astype(np.float64)
                    profile = np.interp(
                        np.linspace(0, len(col_means) - 1, self.band_profile_bins),
                        np.arange(len(col_means)), col_means,
                    )
                    norm = np.linalg.norm(profile)
                    band_profiles.append(profile / norm if norm > 0 else profile)
                    continue
            fb = extract_banding_profiles(gray, [contour], [box], self.band_profile_bins)
            band_profiles.append(fb[0] if fb else np.zeros(self.band_profile_bins))

        final_idx = two_stage_nms(split_b, split_a, band_profiles, self.band_similarity_threshold)
        final_boxes = [split_b[i] for i in final_idx]
        final_areas = [split_a[i] for i in final_idx]
        final_contours = [split_c[i] for i in final_idx]
        final_profiles = [band_profiles[i] for i in final_idx]

        if final_areas:
            si = np.argsort(final_areas)[::-1]
            final_boxes = [final_boxes[i] for i in si]
            final_areas = [final_areas[i] for i in si]
            final_contours = [final_contours[i] for i in si]
            final_profiles = [final_profiles[i] for i in si]

        group_counts = estimate_denver_groups(final_areas, len(final_contours))
        sex_chr_info = detect_sex_chromosome_region(image, final_boxes)

        ensemble = EnsembleClassifier()
        final_straightened = [
            straightened[i] if i < len(straightened) else {"strip": np.array([]), "medial_axis": np.array([]), "curvature": 0.0}
            for i in final_idx
        ]
        if final_areas and len(final_straightened) == len(final_idx):
            si2 = list(np.argsort(final_areas)[::-1])
            final_straightened = [final_straightened[i] for i in si2]

        classifications, ens_meta = ensemble.classify_ensemble(
            gray, final_contours, final_boxes, final_areas, final_profiles, final_straightened,
        )
        karyotype_summary = ensemble.base_classifier.generate_karyotype_summary(classifications)

        return {
            "count": len(final_contours), "bounding_boxes": final_boxes,
            "areas": final_areas, "group_counts": group_counts,
            "sex_chromosome_region": sex_chr_info,
            "detection_method": best["method"] + "+preprocess+cluster_router+seg_matrix+24class",
            "band_profiles": final_profiles,
            "overlap_splits_performed": len(split_c) - len(routed_c),
            "cluster_routing": router_meta, "preprocessing": denoise_metadata,
            "straightening": {
                "total": len(straightened),
                "avg_curvature": float(np.mean([s["curvature"] for s in straightened])) if straightened else 0.0,
                "max_curvature": float(max((s["curvature"] for s in straightened), default=0.0)),
            },
            "classifications": classifications, "karyotype_summary": karyotype_summary,
            "ensemble": ens_meta,
        }

    def detect_karyogram_positions(self, image: Image.Image) -> Dict:
        """Detect chromosome positions in an arranged karyogram image."""
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available"}
        detection = self.detect_chromosomes(image)
        boxes = detection.get("bounding_boxes", [])
        total_count = detection.get("count", 0)
        if total_count == 0:
            return {"error": "No chromosomes detected", "total": 0}

        iw, ih = image.width, image.height
        row4_y = ih * 0.75
        pos21_x_s, pos21_x_e = iw * 0.55, iw * 0.70
        pos22_x_s, pos22_x_e = iw * 0.70, iw * 0.82
        sex_x_s = iw * 0.82
        pos21_count = pos22_count = sex_chr_count = 0
        g_boxes = []
        for box in boxes:
            x, y, w, h = box
            cx, cy = x + w / 2, y + h / 2
            if cy > row4_y:
                if pos21_x_s < cx < pos21_x_e:
                    pos21_count += 1
                    g_boxes.append(("21", box))
                elif pos22_x_s < cx < pos22_x_e:
                    pos22_count += 1
                    g_boxes.append(("22", box))
                elif cx >= sex_x_s:
                    sex_chr_count += 1
                    g_boxes.append(("sex", box))

        sex_region = detection.get("sex_chromosome_region", {})
        pos_counts = {
            "position_21": pos21_count, "position_22": pos22_count,
            "sex_chromosomes": sex_chr_count,
            "sex_chr_estimated": sex_region.get("estimated", "unknown"),
        }
        ka = analyze_karyotype_from_counts(total_count, pos21_count, sex_region)
        return {
            "total_count": total_count, "position_counts": pos_counts,
            "group_g_details": g_boxes, "karyotype_analysis": ka,
            "detection_method": "karyogram_grid_analysis",
        }

    def create_annotated_image(self, image: Image.Image, detection_result: Dict) -> Image.Image:
        """Create an annotated image with detected chromosomes highlighted."""
        if not CV2_AVAILABLE:
            return image
        img_array = np.array(image.convert("RGB"))
        for i, (x, y, w, h) in enumerate(detection_result.get("bounding_boxes", [])):
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_array, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        count = detection_result.get("count", 0)
        cv2.putText(img_array, f"Detected: {count} chromosomes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return Image.fromarray(img_array)
